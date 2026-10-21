"""Re-analyses for the independent review (issues 1, 3, and 4), run on 17 October 2026.

Patient-level rows stay under the protected run folder. Only aggregate summaries are
written here or printed. Subcommands run in the order listed in the plan:

  check-fixed       With the transition SD fixed at 0.75, the revised comparator code must
                    reproduce the September 14-day comparator coefficients.
  calibrate-random  Tune the random-request probability so the mean request count matches
                    the September 50% uncertainty rule at the same schedule.
  run-random        Replay the final random-request replicates at the calibrated probability.
  aggregate         Rebuild the clinical aggregates; values outside the two comparators must not move.
  summarize-random  Compare the 50% rule with the expected error of random requests (test patients).
  manifest          Record source hashes, commands, and aggregate outputs.

The random-request subcommands accept --frozen-fit, --revision-run, --run, and --out so that the
17 October 2026 PHQ-9 recode rerun can replay them against its own fit. The defaults are the
runs used on 17 October, which keeps those results reproducible.
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROTECTED = Path(os.environ['BALL_PROTECTED_ROOT'])
FROZEN_FIT = PROTECTED / 'publication_ready_20260809_v12_uniform_cadence'
COMORBIDITY = PROTECTED / 'session_comorbidity_features.csv'
SEPTEMBER_CLASSICAL = PROTECTED / 'prospective_classical_20260919'
SEPTEMBER_REVISION = PROTECTED / 'uncertainty_revision_20260918'
RUN = PROTECTED / 'review_reanalysis_20261017'
REQUESTS = RUN / 'requests'
HERE = Path(__file__).resolve().parent
REPRODUCTION_TOLERANCE = 1e-9
RULE_POLICY = 'latent_gain_50pct'
CALIBRATION_REPLICATES = 4
CALIBRATION_TOLERANCE = 0.02      # mean request count within 2% of the rule's count
MAX_CALIBRATION_ROUNDS = 6
CALIBRATION_FIRST_SEED = 1000     # pilot replicates never share seeds with the final ones
FINAL_REPLICATES = 200

sys.path.insert(0, str(ROOT / 'empirical'))


def check_fixed(cadence):
    from revise_uncertainty import setup, E
    from prospective_classical import fit_models
    args = SimpleNamespace(frozen_run=FROZEN_FIT, comorbidity_features=COMORBIDITY, output=RUN, temporal=False)
    data, *_ = setup(args, cadence)
    models, settings, train_ids = fit_models(data, E, estimate_transition_variance=False)
    old_models, old_settings, old_ids = pickle.loads((SEPTEMBER_CLASSICAL / f'models_{cadence}d.pkl').read_bytes())
    assert train_ids == old_ids
    differences = {}
    for channel in models:
        for short in ['lg', 'markov']:
            new, old = models[channel][short], old_models[channel][short]
            assert set(new) == set(old), (channel, short, set(new) ^ set(old))
            differences[f'{channel}_{short}'] = float(np.max(np.abs(new['theta'] - old['theta'])))
    # Differences near 1e-12 come from floating-point summation order in multithreaded linear algebra.
    report = dict(cadence=cadence, training_patients=len(train_ids), max_abs_theta_difference=differences,
                  tolerance=REPRODUCTION_TOLERANCE,
                  reproduced=all(v < REPRODUCTION_TOLERANCE for v in differences.values()))
    (HERE / f'check_fixed_{cadence}d.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def replay_random(cadence, probability, replicates, folder, first):
    """Run random-request replicates and return each replicate's total request count."""
    with (HERE / 'logs' / f'random_{folder}.log').open('w') as log:
        subprocess.run([sys.executable, '-X', 'utf8', '-u', str(ROOT / 'empirical/adaptive_reassessment.py'),
                        '--frozen-run', str(FROZEN_FIT), '--comorbidity-features', str(COMORBIDITY),
                        '--output', str(REQUESTS), '--member-archives', str(SEPTEMBER_REVISION),
                        '--cadence', str(cadence), '--random-probability', repr(probability),
                        '--random-replicates', str(replicates), '--random-first-replicate', str(first),
                        '--random-folder', folder], check=True, cwd=ROOT, stdout=log, stderr=log)
    files = sorted((REQUESTS / folder).glob(f'adaptive_events_{cadence}d_random_matched_r*.csv'))
    assert len(files) == replicates, (folder, len(files))
    return [int(pd.read_csv(f, usecols=['query'])['query'].sum()) for f in files]


def calibrate_random(cadence):
    target = int(pd.read_csv(SEPTEMBER_REVISION / f'adaptive_events_{cadence}d_{RULE_POLICY}.csv',
                             usecols=['query'])['query'].sum())
    eligible = int(pd.read_csv(SEPTEMBER_REVISION / f'adaptive_events_{cadence}d_fixed_schedule.csv',
                               usecols=['eligible_after_cooldown'])['eligible_after_cooldown'].sum())
    probability, history = target / eligible, []
    for round_ in range(MAX_CALIBRATION_ROUNDS):
        counts = replay_random(cadence, probability, CALIBRATION_REPLICATES, f'calibration_{cadence}d_round{round_}',
                               CALIBRATION_FIRST_SEED + 10 * round_)
        history.append(dict(probability=probability, mean_requests=float(np.mean(counts)), requests=counts))
        print(json.dumps(history[-1]), flush=True)
        if abs(history[-1]['mean_requests'] / target - 1) <= CALIBRATION_TOLERANCE:
            break
        if len(history) >= 2:   # secant step through the last two rounds
            (p0, c0), (p1, c1) = [(h['probability'], h['mean_requests']) for h in history[-2:]]
            probability = p1 + (target - c1) * (p1 - p0) / (c1 - c0) if c1 != c0 else p1 * target / c1
        else:
            probability = probability * target / history[-1]['mean_requests']
        probability = float(np.clip(probability, 1e-3, 1.0))
    else:
        raise RuntimeError(f'{cadence}d random-request calibration did not reach the tolerance')
    record = dict(cadence=cadence, rule_policy=RULE_POLICY, target_requests=target, eligible_under_fixed_schedule=eligible,
                  tolerance=CALIBRATION_TOLERANCE, replicates_per_round=CALIBRATION_REPLICATES,
                  history=history, calibrated_probability=history[-1]['probability'])
    (HERE / f'random_calibration_{cadence}d.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


def run_random(cadence):
    calibration = json.loads((HERE / f'random_calibration_{cadence}d.json').read_text())
    counts = replay_random(cadence, calibration['calibrated_probability'], FINAL_REPLICATES, f'final_{cadence}d', 0)
    summary = dict(cadence=cadence, probability=calibration['calibrated_probability'],
                   target_requests=calibration['target_requests'], replicates=len(counts),
                   mean_requests=float(np.mean(counts)), sd_requests=float(np.std(counts, ddof=1)))
    (HERE / f'random_final_counts_{cadence}d.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def summarize_random(cadence):
    """Compare the 50% uncertainty rule with random requests at the same rate, on test patients.

    RMSE follows empirical/summarize_reassessment.py: squared errors are divided by each
    instrument's development SD, averaged within patient, then across patients. The random
    policy's expected per-patient error averages the replicates, and a paired patient
    bootstrap gives the interval for rule-minus-random RMSE.
    """
    measurement = json.loads((FROZEN_FIT / 'measurement_calibration.json').read_text())
    scales = measurement['development_data']['normalized_score_sd_by_instrument']

    def patient_errors(frame):
        test = frame[frame.role.eq('test')]
        error = ((test.ball - test.target) / test.anchor.map(scales))**2
        return error.groupby(test.id).mean(), int(test['query'].sum()), len(test)

    rule, rule_requests, n_test = patient_errors(
        pd.read_csv(SEPTEMBER_REVISION / f'adaptive_events_{cadence}d_{RULE_POLICY}.csv'))
    fixed, _, _ = patient_errors(pd.read_csv(SEPTEMBER_REVISION / f'adaptive_events_{cadence}d_fixed_schedule.csv'))
    files = sorted((REQUESTS / f'final_{cadence}d').glob(f'adaptive_events_{cadence}d_random_matched_r*.csv'))
    assert len(files) == FINAL_REPLICATES, len(files)
    replicate_errors, replicate_rmse, replicate_requests = [], [], []
    for f in files:
        errors, requests, n = patient_errors(pd.read_csv(f))
        assert n == n_test
        replicate_errors.append(errors.reindex(rule.index))
        replicate_rmse.append(float(np.sqrt(errors.mean())))
        replicate_requests.append(requests)
    expected_random = pd.concat(replicate_errors, axis=1).mean(axis=1)
    draws = np.random.default_rng(20260918).integers(0, len(rule), size=(2000, len(rule)))
    difference = (np.sqrt(rule.to_numpy()[draws].mean(axis=1))
                  - np.sqrt(expected_random.to_numpy()[draws].mean(axis=1)))
    replicate_rmse = np.asarray(replicate_rmse)
    rule_rmse = float(np.sqrt(rule.mean()))
    summary = dict(
        cadence=cadence, test_events=n_test, test_patients=len(rule),
        fixed_schedule_rmse=float(np.sqrt(fixed.mean())),
        rule_rmse=rule_rmse, rule_test_requests=rule_requests, rule_request_fraction=rule_requests / n_test,
        random_replicates=len(files), random_expected_rmse=float(np.sqrt(expected_random.mean())),
        random_rmse_mean=float(replicate_rmse.mean()), random_rmse_sd=float(replicate_rmse.std(ddof=1)),
        random_rmse_p025=float(np.quantile(replicate_rmse, .025)),
        random_rmse_p975=float(np.quantile(replicate_rmse, .975)),
        random_test_requests_mean=float(np.mean(replicate_requests)),
        random_test_requests_sd=float(np.std(replicate_requests, ddof=1)),
        random_request_fraction_mean=float(np.mean(replicate_requests)) / n_test,
        fraction_of_replicates_at_or_below_rule=float((replicate_rmse <= rule_rmse).mean()),
        rule_minus_random_rmse=float(rule_rmse - np.sqrt(expected_random.mean())),
        rule_minus_random_ci_lo=float(np.quantile(difference, .025)),
        rule_minus_random_ci_hi=float(np.quantile(difference, .975)),
        bootstrap_replicates=2000)
    (HERE / f'random_summary_{cadence}d.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


SEPTEMBER_AGGREGATES = ROOT / '_working/paper_revision_20260918/refinement/results'
SEPTEMBER_TEMPORAL_AGGREGATES = ROOT / '_working/analysis_alignment_20260919/results/temporal'
SEPTEMBER_REFINEMENT = PROTECTED / 'uncertainty_refinement_20260918'
TEMPORAL = PROTECTED / 'alignment_temporal_20260919'
CLASSICAL = RUN / 'classical'
AGGREGATES = HERE / 'aggregates'
REESTIMATED = ('s0_direct_lgssm', 'markov_direct_transition')


def aggregate():
    """Rebuild the clinical aggregates with the re-estimated comparators; everything else must not move."""
    import shutil
    AGGREGATES.mkdir(exist_ok=True); (AGGREGATES / 'temporal').mkdir(exist_ok=True)
    for source, target in [(SEPTEMBER_AGGREGATES, AGGREGATES), (SEPTEMBER_TEMPORAL_AGGREGATES, AGGREGATES / 'temporal')]:
        for f in source.glob('*.csv'):
            shutil.copy2(f, target / f.name)
    for cadence in (7, 14, 21, 28):   # patient-level copies stay inside the protected run folder
        shutil.copy2(CLASSICAL / 'by_cadence' / f'{cadence}d' / f'corrected_comparator_events_{cadence}d.csv',
                     CLASSICAL / f'corrected_comparator_events_{cadence}d.csv')
    python = [sys.executable, '-X', 'utf8', '-u']
    steps = [
        [str(ROOT / 'empirical/aggregate_revision.py'), '--run', str(CLASSICAL), '--frozen-run', str(FROZEN_FIT),
         '--interval-run', str(SEPTEMBER_REFINEMENT), '--output', str(AGGREGATES)],
        [str(ROOT / 'empirical/summarize_interval_refinement.py'), '--run', str(SEPTEMBER_REFINEMENT),
         '--previous-run', str(SEPTEMBER_REVISION), '--frozen-run', str(FROZEN_FIT), '--classical-run', str(CLASSICAL),
         '--output', str(AGGREGATES)],
        [str(ROOT / '_working/analysis_alignment_20260919/summarize_temporal.py'), '--frozen-run', str(TEMPORAL / 'fit'),
         '--replay-run', str(TEMPORAL / 'replay'), '--classical-run', str(CLASSICAL / 'temporal'),
         '--interval-run', str(TEMPORAL / 'intervals'), '--output', str(AGGREGATES / 'temporal')],
    ]
    with (HERE / 'logs' / 'aggregate.log').open('w') as log:
        for step in steps:
            subprocess.run(python + step, check=True, cwd=ROOT / 'empirical', stdout=log, stderr=log)

    # Every method other than the two re-estimated comparators must reproduce September.
    checks = {}
    for name, keys in [('empirical_recovery_overall_complete.csv', ['cadence', 'method']),
                       ('empirical_recovery_by_instrument_complete.csv', ['cadence', 'method', 'instrument']),
                       ('empirical_calibration_complete.csv', ['cadence', 'instrument'])]:
        old, new = pd.read_csv(SEPTEMBER_AGGREGATES / name), pd.read_csv(AGGREGATES / name)
        if 'method' in keys:
            old, new = old[~old.method.isin(REESTIMATED)], new[~new.method.isin(REESTIMATED)]
        merged = old.merge(new, on=keys, suffixes=('_old', '_new'), validate='one_to_one')
        assert len(merged) == len(old), name
        value = 'rmse' if 'rmse_old' in merged else 'coverage'
        checks[name] = float((merged[f'{value}_old'] - merged[f'{value}_new']).abs().max())
    old_t = pd.read_csv(SEPTEMBER_TEMPORAL_AGGREGATES / 'empirical_generalization_complete.csv').set_index('method')
    new_t = pd.read_csv(AGGREGATES / 'temporal/empirical_generalization_complete.csv').set_index('method')
    checks['temporal_ball_rmse'] = float(abs(old_t.loc['ball', 'rmse'] - new_t.loc['ball', 'rmse']))
    assert max(checks.values()) < 1e-9, checks
    overall = pd.read_csv(AGGREGATES / 'empirical_recovery_overall_complete.csv')
    report = dict(unchanged_max_abs_difference=checks,
                  comparator_rmse=overall[overall.method.isin(('ball', 'ball_direct_causal') + REESTIMATED)]
                  .pivot(index='cadence', columns='method', values='rmse').round(4).to_dict(),
                  temporal_rmse=new_t['rmse'].round(4).to_dict())
    (HERE / 'aggregate_check.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


CHANGED_SOURCES = ['BALL.py', 'empirical/prospective_classical.py', 'empirical/rerun_prospective_classical.py',
                   'empirical/adaptive_reassessment.py', 'empirical/aggregate_revision.py',
                   'empirical/summarize_interval_refinement.py', '_working/review_reanalysis_20261017/run_reanalysis.py']
COMMANDS = [
    'python -X utf8 _working/review_reanalysis_20261017/run_reanalysis.py check-fixed --cadence 14',
    'python -X utf8 empirical/rerun_prospective_classical.py --frozen-run <protected>/publication_ready_20260809_v12_uniform_cadence '
    '--comorbidity-features <protected>/session_comorbidity_features.csv --previous-run <protected>/uncertainty_revision_20260918 '
    '--output <protected>/review_reanalysis_20261017/classical/by_cadence/<C>d --cadences <C>   (C = 7, 14, 21, 28)',
    'python -X utf8 empirical/rerun_prospective_classical.py --frozen-run <protected>/alignment_temporal_20260919/fit '
    '--comorbidity-features <protected>/session_comorbidity_features.csv --previous-run <protected>/alignment_temporal_20260919/replay '
    '--output <protected>/review_reanalysis_20261017/classical/temporal --cadences 14 --temporal',
    'python -X utf8 _working/review_reanalysis_20261017/run_reanalysis.py calibrate-random --cadence <C>   (C = 14, 21, 28)',
    'python -X utf8 _working/review_reanalysis_20261017/run_reanalysis.py run-random --cadence <C>',
    'python -X utf8 _working/review_reanalysis_20261017/run_reanalysis.py aggregate',
    'python -X utf8 _working/review_reanalysis_20261017/run_reanalysis.py summarize-random --cadence <C>',
]


def manifest():
    """Record source hashes, commands, and the aggregate outcome files of this re-analysis."""
    import hashlib
    from datetime import datetime
    record = dict(
        written_at=datetime.now().astimezone().isoformat(timespec='seconds'),
        protected_run='<protected> = $BALL_PROTECTED_ROOT; patient-level rows stay there',
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in CHANGED_SOURCES},
        commands=COMMANDS,
        validation=dict(
            known_truth_estimator='scratch test: true daily SD 0.08 recovered as 0.0799 on 300 synthetic patients',
            fixed_sd_reproduction=json.loads((HERE / 'check_fixed_14d.json').read_text()),
            random_policy_zero_probability='replayed the September fixed schedule within 1.2e-7 at 28 days',
            unchanged_aggregates=json.loads((HERE / 'aggregate_check.json').read_text())['unchanged_max_abs_difference']),
        outputs=sorted(str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.suffix in ('.csv', '.json')))
    (HERE / 'RUN_MANIFEST.json').write_text(json.dumps(record, indent=2))
    print(f"manifest written with {len(record['outputs'])} aggregate outputs")


RANDOM_COMMANDS = ('calibrate-random', 'run-random', 'summarize-random')


def main():
    global FROZEN_FIT, SEPTEMBER_REVISION, RUN, REQUESTS, HERE
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen-fit', type=Path, default=FROZEN_FIT)
    parser.add_argument('--revision-run', type=Path, default=SEPTEMBER_REVISION,
                        help='Run holding the ensemble member archives and the uncertainty-rule replays.')
    parser.add_argument('--run', type=Path, default=RUN, help='Protected root; random replays go to <run>/requests.')
    parser.add_argument('--out', type=Path, default=HERE, help='Folder for aggregate JSON summaries and logs.')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['check-fixed', 'calibrate-random', 'run-random', 'summarize-random']:
        sub.add_parser(name).add_argument('--cadence', type=int, required=True)
    sub.add_parser('aggregate')
    sub.add_parser('manifest')
    args = parser.parse_args()
    moved = (args.frozen_fit, args.revision_run, args.run, args.out) != (FROZEN_FIT, SEPTEMBER_REVISION, RUN, HERE)
    if moved and args.command not in RANDOM_COMMANDS:
        raise SystemExit(f'{args.command} compares against the 17 October inputs; path options apply only to '
                         f'{", ".join(RANDOM_COMMANDS)}.')
    FROZEN_FIT, SEPTEMBER_REVISION, RUN, HERE = args.frozen_fit, args.revision_run, args.run, args.out
    REQUESTS = RUN / 'requests'
    HERE.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(parents=True, exist_ok=True)
    (HERE / 'logs').mkdir(exist_ok=True)
    if args.command in ('aggregate', 'manifest'):
        {'aggregate': aggregate, 'manifest': manifest}[args.command]()
        return
    {'check-fixed': check_fixed, 'calibrate-random': calibrate_random, 'run-random': run_random,
     'summarize-random': summarize_random}[args.command](args.cadence)


if __name__ == '__main__':
    main()
