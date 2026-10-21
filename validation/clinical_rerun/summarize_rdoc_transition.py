"""Aggregate the RDoC-transition refit (aggregates only).

For each schedule, the refit's withheld-questionnaire predictions are formed exactly as replay_comparators.py forms
every BALL variant: the member-averaged forward-student states, averaged over the recall-window sessions through the
target session and mapped through the instrument's fixed measurement equation. The same code applied to the primary
fit's checkpoints must reproduce the primary BALL predictions the paper reports before the refit is scored. The refit
is compared with the primary fit on the same test-patient events by patient-balanced RMSE and a paired patient
bootstrap (aggregate_revision.py's seed, 2,000 draws, and Bonferroni level for four schedules). The six transition
coefficients are read from each member's saved generative model.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'empirical'))
from revise_uncertainty import setup   # noqa: E402  (installs BALL's virtual package)

PROTECTED = Path(os.environ['BALL_PROTECTED_ROOT'])
RUN = PROTECTED / 'phq9_recode_20261017'
COMORBIDITY = PROTECTED / 'session_comorbidity_features.csv'
CADENCES = [7, 14, 21, 28]
DOMAINS = ['Negative valence systems', 'Positive valence systems', 'Cognitive systems', 'Social processes',
           'Arousal and regulatory systems', 'Sensorimotor systems']   # B0-B5, the order of RDOC_COLS
REPRODUCTION_TOLERANCE = 2e-5   # replay_comparators.py's tolerance
BOOTSTRAP_SEED, BOOTSTRAP_DRAWS, SCHEDULES = 101, 2000, 4
PERSISTENCE = 0.9               # clinical daily persistence
GAP_DAYS = 7                    # Equation 1 scales B'beta by a_phi(dt) = (1 - phi^dt)/(1 - phi) across a gap of dt days,
GAP_DRIFT = (1 - PERSISTENCE ** GAP_DAYS) / (1 - PERSISTENCE)   # so this is the exact one-gap effect at a fixed state
PHQ9_HALF_RANGE = 13.5          # PHQ-9 points per unit of the depression state
GAD7_HALF_RANGE = 10.5          # GAD-7 points per unit of the anxiety state


def student_arrays(folder):
    files = sorted(folder.glob('student_member_*.npz'))
    members = [np.load(f) for f in files]
    states = np.mean([np.stack([m['z_d'], m['z_p']], axis=-1) for m in members], axis=0)
    return states, np.stack([m['rdoc_drift_beta'] for m in members]), len(files)


def endpoint_predictions(rows, states, test_ids, anchors, sessions, days, measurement):
    """replay_comparators.py's endpoint: recall-window sessions through the target session."""
    windows = anchors.set_index(['id', 'session', 'anchor'])[['window_start_day', 'window_end_day']]
    session_map = {int(pid): g.sort_values(['session', 'day']).session.to_numpy() for pid, g in sessions.groupby('id')
                   if int(pid) in days}
    position = {pid: i for i, pid in enumerate(test_ids)}
    values = []
    for r in rows.itertuples():
        w = windows.loc[(r.id, r.session, r.anchor)]
        idx = np.flatnonzero((days[r.id] >= w.window_start_day) & (days[r.id] <= w.window_end_day))
        endpoint = int(np.flatnonzero(session_map[r.id] == r.session)[0])
        channel = measurement['instruments'][r.anchor]['channel']
        values.append(r.instrument_intercept + r.instrument_loading
                      * float(states[position[r.id], idx[idx <= endpoint], channel].mean()))
    return np.asarray(values)


def patient_errors(frame, column):
    return (((frame[column] - frame.z_true) / frame.pooling_sd_normalized) ** 2).groupby(frame.id).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cadences', type=int, nargs='+', default=CADENCES)
    p.add_argument('--refit-root', type=Path, default=RUN / 'rdoc_transition')
    p.add_argument('--output', type=Path, default=HERE / 'aggregates')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    context = SimpleNamespace(frozen_run=RUN / 'fit', comorbidity_features=COMORBIDITY)
    prediction_rows, coefficient_rows, report = [], [], {}
    for cadence in args.cadences:
        data, days, anchors, sessions, _, _, measurement, _, _, test, _ = setup(context, cadence)
        test_ids = list(map(int, test.ids))
        events = pd.read_csv(RUN / f'classical/corrected_comparator_events_{cadence}d.csv', low_memory=False)
        primary, _, _ = student_arrays(RUN / f'fit/member_checkpoints/{cadence}d_primary_ball')
        reproduced = endpoint_predictions(events, primary, test_ids, anchors, sessions, days, measurement)
        reproduction = float(np.max(np.abs(reproduced - events.ball.to_numpy())))
        assert reproduction < REPRODUCTION_TOLERANCE, (cadence, reproduction)
        refit, betas, members = student_arrays(args.refit_root / f'member_checkpoints/{cadence}d_ball_rdoc_transition')
        events['ball_rdoc_transition'] = endpoint_predictions(events, refit, test_ids, anchors, sessions, days,
                                                              measurement)

        test_events = events[events.evaluation_role.eq('test')]
        base, refit_errors = patient_errors(test_events, 'ball'), patient_errors(test_events, 'ball_rdoc_transition')
        refit_errors = refit_errors.reindex(base.index).to_numpy()
        draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(base), size=(BOOTSTRAP_DRAWS, len(base)))
        delta = np.sqrt(refit_errors[draws].mean(axis=1)) - np.sqrt(base.to_numpy()[draws].mean(axis=1))
        alpha = 0.05 / SCHEDULES
        prediction_rows.append(dict(
            cadence=cadence, rmse_primary=float(np.sqrt(base.mean())), rmse_rdoc_transition=float(np.sqrt(refit_errors.mean())),
            rdoc_minus_primary=float(np.sqrt(refit_errors.mean()) - np.sqrt(base.mean())),
            ci_lo=float(np.quantile(delta, .025)), ci_hi=float(np.quantile(delta, .975)),
            ci_lo_bonferroni_schedules=float(np.quantile(delta, alpha / 2)),
            ci_hi_bonferroni_schedules=float(np.quantile(delta, 1 - alpha / 2)),
            n_events=int(len(test_events)), n_patients=int(len(base)), members=members))
        for j, domain in enumerate(DOMAINS):
            b = betas[:, j]
            coefficient_rows.append(dict(
                cadence=cadence, domain=domain, beta_mean=float(b.mean()), beta_sd=float(b.std(ddof=1)) if members > 1 else np.nan,
                beta_min=float(b.min()), beta_max=float(b.max()), members_positive=int((b > 0).sum()), members=members,
                seven_day_gap_phq9_points=float(b.mean() * GAP_DRIFT * PHQ9_HALF_RANGE),
                seven_day_gap_gad7_points=float(b.mean() * GAP_DRIFT * GAD7_HALF_RANGE)))
        report[cadence] = {'primary_reproduction_max_error': reproduction, 'members': members}
        print(json.dumps({'cadence': cadence, **report[cadence], **prediction_rows[-1]}), flush=True)
    pd.DataFrame(prediction_rows).to_csv(args.output / 'rdoc_transition_prediction.csv', index=False)
    pd.DataFrame(coefficient_rows).to_csv(args.output / 'rdoc_transition_coefficients.csv', index=False)
    (args.output / 'rdoc_transition_report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
