"""Refit the clinical Gaussian-process filter with every continuous input standardized (19 October 2026).

The PHQ-9 recode rerun added five previous-session measures, two of them continuous (stimulation intensity as a
proportion of motor threshold and course number), while the Gaussian-process scaler still standardized only session
number and comorbidity count. BALL.py now declares the continuous features by name. For each schedule this script
fits three versions of the filter with the replay's data and evaluation endpoint:

- original: the two continuous features standardized before the fix, which must reproduce the rerun's predictions
- standardized: the corrected filter, which replaces the rerun's Gaussian-process predictions
- without_session_measures: the corrected filter with the five session measures removed, to show their contribution

Patient-level rows go only to the protected rerun folder. Aggregates go to this folder.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'empirical'))
from revise_uncertainty import E   # noqa: E402  (installs BALL's virtual package)

PROTECTED = Path(os.environ['BALL_PROTECTED_ROOT'])
RUN = PROTECTED / 'phq9_recode_20261017'
COMORBIDITY = PROTECTED / 'session_comorbidity_features.csv'
OUT_EVENTS = RUN / 'classical_gp_standardized'   # comparator events with the corrected Gaussian-process column
CADENCES = [7, 14, 21, 28]
GP_AUXILIARY_WEIGHT = 0.15   # the rerun's --gp-auxiliary-weight
REPRODUCTION_TOLERANCE = 2e-5   # replay_comparators.py's tolerance for refitted comparators
ORIGINAL_CONTINUOUS = ['treatment_session_number', 'dx_comorbidity_count']
BOOTSTRAP_SEED, BOOTSTRAP_DRAWS = 101, 2000   # aggregate_revision.py's patient bootstrap
FIXED_CONTINUOUS = list(E.EMPIRICAL_CONTINUOUS_FEATURE_NAMES)
SESSION_MEASURE_INDICES = [E.EMPIRICAL_FEATURE_NAMES.index(n) for n in E.PREVIOUS_SESSION_FEATURE_NAMES]


def build(cadence):
    """The replay's clinical dataset for one schedule (revise_uncertainty.setup without the neural batches)."""
    measurement = json.loads((RUN / 'fit/measurement_calibration.json').read_text())
    sessions = E._coerce_sessions(pd.read_csv(E.OUT / 'sessions.csv'))
    splits, _, _ = E._primary_empirical_partitions(sessions)
    features = E._load_comorbidity_sessions(COMORBIDITY)
    anchors = pd.read_csv(E.OUT / f'anchors_sparse_{cadence}d.csv')
    data, days = E._empirical_neural_dataset(anchors, sessions, None, features, measurement, patient_splits=splits)
    return data, days, anchors, sessions, measurement


def without_session_measures(data):
    daily = data.daily.copy()
    for j in SESSION_MEASURE_INDICES:
        daily[f'X{j}'] = 0.0
        daily[f'input_X{j}'] = False
        daily[f'obs_X{j}'] = False
    return E.SimulationData(config=data.config, individuals=data.individuals.copy(), daily=daily,
                            anchors=data.anchors.copy(), treatments=data.treatments.copy(),
                            components=data.components.copy(), metadata=dict(data.metadata))


def fit_gp(data, days, continuous):
    E.EMPIRICAL_CONTINUOUS_FEATURE_NAMES = list(continuous)
    try:
        fits = {ch: E._fit_empirical_gp_channel(data, name, days, fixed_auxiliary_weight=GP_AUXILIARY_WEIGHT)
                for ch, name in enumerate(['Y1', 'Y2'])}
    finally:
        E.EMPIRICAL_CONTINUOUS_FEATURE_NAMES = list(FIXED_CONTINUOUS)
    metadata = {ch: {k: m[k] for k in ('continuous_feature_indices', 'continuous_feature_means',
                                       'continuous_feature_standard_deviations')}
                | {k: m['kernel_parameters'][k] for k in ('slow_length_days', 'fast_length_days')}
                for ch, (_, m) in fits.items()}
    return {ch: gp for ch, (gp, _) in fits.items()}, metadata


def endpoint_values(events, gps, anchors, sessions, days, measurement):
    """Questionnaire-scale predictions averaged over the recall-window sessions through the target (replay endpoint)."""
    windows = anchors.set_index(['id', 'session', 'anchor'])[['window_start_day', 'window_end_day']]
    session_map = {int(pid): g.sort_values(['session', 'day']).session.to_numpy() for pid, g in sessions.groupby('id')
                   if int(pid) in days}
    values = []
    for r in events.itertuples():
        w = windows.loc[(r.id, r.session, r.anchor)]
        idx = np.flatnonzero((days[r.id] >= w.window_start_day) & (days[r.id] <= w.window_end_day))
        endpoint = int(np.flatnonzero(session_map[r.id] == r.session)[0])
        ch = measurement['instruments'][r.anchor]['channel']
        values.append(r.instrument_intercept + r.instrument_loading * float(gps[ch][r.id][idx[idx <= endpoint]].mean()))
    return np.asarray(values)


def patient_errors(events, column):
    se = ((events[column] - events.z_true) / events.pooling_sd_normalized) ** 2
    return se.groupby(events.id).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cadences', type=int, nargs='+', default=CADENCES)
    args = p.parse_args()
    OUT_EVENTS.mkdir(exist_ok=True)
    report, summary = {}, []
    for cadence in args.cadences:
        events = pd.read_csv(RUN / f'classical/corrected_comparator_events_{cadence}d.csv', low_memory=False)
        data, days, anchors, sessions, measurement = build(cadence)
        variants, metadata = {}, {}
        for name, frame, continuous in [('original', data, ORIGINAL_CONTINUOUS),
                                        ('standardized', data, FIXED_CONTINUOUS),
                                        ('without_session_measures', without_session_measures(data), FIXED_CONTINUOUS)]:
            gps, metadata[name] = fit_gp(frame, days, continuous)
            variants[name] = endpoint_values(events, gps, anchors, sessions, days, measurement)
            print(json.dumps({'cadence': cadence, 'variant': name, 'metadata': metadata[name]}, default=str), flush=True)
        reproduction = float(np.max(np.abs(variants['original'] - events.gp_causal_filter.to_numpy())))
        assert reproduction < REPRODUCTION_TOLERANCE, (cadence, reproduction)

        rows = events[['id', 'session', 'anchor', 'evaluation_role']].copy()
        for name, values in variants.items():
            rows[f'gp_{name}'] = values
        rows.to_csv(OUT_EVENTS / f'gp_variants_{cadence}d.csv', index=False)
        corrected = events.copy()
        corrected['gp_causal_filter'] = variants['standardized']
        corrected['gp_causal_filter_native'] = corrected.gp_causal_filter * corrected.scale_half_range + corrected.scale_center
        corrected['gp_causal_filter_native_clipped'] = np.clip(corrected.gp_causal_filter_native, 0,
                                                               2 * corrected.scale_half_range)
        corrected['gp_causal_filter_clipped'] = ((corrected.gp_causal_filter_native_clipped - corrected.scale_center)
                                                 / corrected.scale_half_range)
        corrected.to_csv(OUT_EVENTS / f'corrected_comparator_events_{cadence}d.csv', index=False)

        test = events[events.evaluation_role.eq('test')].copy()
        for name in variants:
            test[f'gp_{name}'] = rows.loc[test.index, f'gp_{name}']
        ball = patient_errors(test, 'ball')
        draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(ball), size=(BOOTSTRAP_DRAWS, len(ball)))
        for name in variants:
            other = patient_errors(test, f'gp_{name}').reindex(ball.index).to_numpy()
            delta = np.sqrt(ball.to_numpy()[draws].mean(axis=1)) - np.sqrt(other[draws].mean(axis=1))
            lo, hi = np.quantile(delta, [.025, .975])
            summary.append(dict(cadence=cadence, variant=name, rmse=float(np.sqrt(other.mean())),
                                ball_minus_gp=float(np.sqrt(ball.mean()) - np.sqrt(other.mean())), ci_lo=lo, ci_hi=hi,
                                n_patients=len(ball)))
        report[cadence] = {'original_reproduction_max_error': reproduction, 'kernel_and_scaling': metadata}
    pd.DataFrame(summary).to_csv(HERE / 'gp_variant_summary.csv', index=False)
    (HERE / 'gp_rerun_report.json').write_text(json.dumps(report, indent=2, default=str))
    print(pd.DataFrame(summary).round(4).to_string(), flush=True)

    if sorted(args.cadences) == CADENCES:
        # The corrected events then pass through the rerun's own aggregation, so only Gaussian-process rows change.
        subprocess.run([sys.executable, str(ROOT / 'empirical/aggregate_revision.py'), '--run', str(OUT_EVENTS),
                        '--interval-run', str(RUN / 'refinement'), '--frozen-run', str(RUN / 'fit'),
                        '--output', str(HERE / 'aggregates')], cwd=ROOT / 'empirical', check=True)


if __name__ == '__main__':
    main()
