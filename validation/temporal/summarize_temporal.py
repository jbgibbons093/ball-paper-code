"""Aggregate a fresh temporal-only cohort rerun without touching primary results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'empirical'))
from conditional_intervals import pmean


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frozen-run', type=Path, required=True)
    p.add_argument('--replay-run', type=Path, required=True)
    p.add_argument('--classical-run', type=Path, required=True)
    p.add_argument('--interval-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(args.classical_run/'corrected_comparator_events_14d.csv')
    events = events[events.evaluation_role.eq('test')]
    summary = pd.read_csv(args.frozen_run/'empirical_generalization_complete.csv')
    methods = ['ball', 's0_direct_lgssm', 'markov_direct_transition']
    if 'ball_direct_causal' in set(summary.method):
        # The direct transformer's event predictions come from the refit, on the same events as BALL's replay.
        methods.append('ball_direct_causal')
        direct = pd.read_csv(args.frozen_run/'generalization_forward_2025_events.csv',
                             usecols=['id', 'session', 'anchor', 'ball_direct_causal'])
        events = events.merge(direct, on=['id', 'session', 'anchor'], how='left', validate='one_to_one')
        if events.ball_direct_causal.isna().any():
            raise AssertionError('Every temporal test event needs a direct-transformer prediction.')
    if set(summary.method) != set(methods) or len(summary) != len(methods):
        raise AssertionError('Fresh temporal summary must have exactly one row per model.')
    per_patient = {}
    for method in methods:
        error = ((events[method]-events.z_true)/events.pooling_sd_normalized)**2
        per_patient[method] = error.groupby(events.id).mean()
        value = float(np.sqrt(per_patient[method].mean()))
        summary.loc[summary.method.eq(method), ['rmse', 'rmse_raw']] = value
        clipped = np.clip(events[method], -1, 1)
        clipped_error = ((clipped-events.z_true)/events.pooling_sd_normalized)**2
        summary.loc[summary.method.eq(method), 'rmse_support_clipped'] = float(
            np.sqrt(clipped_error.groupby(events.id).mean().mean()))
    intervals = pd.read_csv(args.interval_run/'revised_intervals_14d.csv')
    chosen = intervals[intervals.method.eq('conditional_quantiles')]
    summary.loc[summary.method.eq('ball'), 'patient_balanced_conformal_coverage'] = (
        chosen.groupby(['anchor', 'id']).covered.mean().groupby('anchor').mean().mean())
    summary.loc[summary.method.eq('ball'), 'mean_interval_width_fraction_of_legal_range'] = (
        chosen.groupby(['anchor', 'id']).width_fraction.mean().groupby('anchor').mean().mean())
    summary.loc[~summary.method.eq('ball'), ['patient_balanced_conformal_coverage',
        'mean_interval_width_fraction_of_legal_range']] = np.nan
    summary.to_csv(args.output/'empirical_generalization_complete.csv', index=False)

    ball = per_patient['ball']
    draw = np.random.default_rng(101).integers(0, len(ball), size=(2000, len(ball)))
    pairs = []
    for method in methods[1:]:
        other = per_patient[method].reindex(ball.index)
        delta = np.sqrt(ball.to_numpy()[draw].mean(axis=1))-np.sqrt(other.to_numpy()[draw].mean(axis=1))
        lo, hi = np.quantile(delta, [.025, .975])
        pairs.append(dict(comparator=method, rmse_difference=np.sqrt(ball.mean())-np.sqrt(other.mean()),
                          ci_lo=lo, ci_hi=hi, n_patients=len(ball)))
    pd.DataFrame(pairs).to_csv(args.output/'temporal_pairwise_bootstrap.csv', index=False)

    old = pd.read_csv(args.replay_run/'revised_intervals_14d.csv')
    old = old[old.method.eq('conditional_quantiles')].copy()
    old['method'] = 'previous_conditional_quantiles'
    combined = pd.concat([old, intervals], ignore_index=True)
    sensitivities = [dict(setting='temporal', cadence=14, method=method,
        coverage=pmean(g, g.covered), width_fraction=float(g.width_fraction.mean()),
        n_events=len(g), n_patients=g.id.nunique()) for method, g in combined.groupby('method')]
    pd.DataFrame(sensitivities).to_csv(args.output/'refinement_sensitivity_summary.csv', index=False)
    choices = json.loads((args.interval_run/'interval_choices_14d.json').read_text())
    pd.DataFrame([dict(setting='temporal', cadence=14, instrument=inst, **choice)
                  for inst, choice in choices.items()]).to_csv(
                      args.output/'refinement_selected_specifications.csv', index=False)
    matched = chosen.merge(old, on=['id', 'session', 'anchor'], suffixes=('', '_old'), validate='one_to_one')
    assert len(matched) == len(chosen) == len(old)
    np.testing.assert_allclose(matched.target, matched.target_old)
    np.testing.assert_allclose(matched.ball, matched.ball_old)
    changes = []
    for inst, g in matched.groupby('anchor'):
        per = g.groupby('id')[['width_fraction', 'width_fraction_old', 'covered', 'covered_old']].mean()
        index = np.random.default_rng(20260918).integers(0, len(per), size=(2000, len(per)))
        change = per.width_fraction-per.width_fraction_old
        lo, hi = np.quantile(change.to_numpy()[index].mean(axis=1), [.025, .975])
        clo, chi = np.quantile(per.covered.to_numpy()[index].mean(axis=1), [.025, .975])
        changes.append(dict(setting='temporal', cadence=14, instrument=inst,
            old_width_fraction=g.width_fraction_old.mean(), new_width_fraction=g.width_fraction.mean(),
            relative_width_change=g.width_fraction.mean()/g.width_fraction_old.mean()-1,
            patient_balanced_width_change=change.mean(), width_change_ci_lo=lo, width_change_ci_hi=hi,
            old_coverage=per.covered_old.mean(), new_coverage=per.covered.mean(),
            new_coverage_ci_lo=clo, new_coverage_ci_hi=chi, n_patients=len(per), n_events=len(g)))
    pd.DataFrame(changes).to_csv(args.output/'refinement_matched_changes.csv', index=False)
    print('Fresh temporal-only aggregates written; primary analysis outputs were not read or changed.')


if __name__ == '__main__':
    main()
