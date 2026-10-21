"""Fresh temporal-only refit after the shared clinic-exclusivity correction.

This helper does not change shared source or primary clinical fits. Without
--fit it performs only the aggregate partition audit. Fitting and patient-level
outputs require a protected directory outside the repository.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO/'empirical'))
from audit_temporal_partition import audit
import revise_uncertainty as R
E = R.E


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def private_output(path):
    path = path.expanduser().resolve()
    if path.is_relative_to(REPO.resolve()):
        raise ValueError('Fitted models and patient-level outputs require a protected directory outside BARL-SAR.')
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--comorbidity-features', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--replay-output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--fit', action='store_true')
    p.add_argument('--reuse-checkpoints-from', type=Path, default=None,
                   help='Earlier temporal refit whose BALL member checkpoints should be restored rather than retrained.')
    args = p.parse_args()
    partition_audit = audit(require_live_fix=True)
    print(json.dumps(partition_audit, indent=2), flush=True)
    if not args.fit:
        print('Audit only. Pass --fit explicitly to refit temporal models.', flush=True)
        return

    output = private_output(args.output)
    replay_output = private_output(args.replay_output)
    if output == args.source_run.resolve() or output == replay_output:
        raise ValueError('Source, new fitted-model directory, and replay directory must be distinct.')
    source_manifest = json.loads((args.source_run/'fit_run_manifest.json').read_text())
    values = dict(source_manifest['fit_args'])
    values.update(max_patients=None, select_core_hyperparameters=False,
                  delta_phi=.9, anchor_weight=10., student_anchor_weight=2.,
                  teacher_epochs=120, student_epochs=120, anchor_warmup=60,
                  kl_warmup=60, members=5, d_model=96, n_heads=4, n_layers=3,
                  batch_size=32, seed_offset=0, device=args.device,
                  progress_every_epochs=10, diagnostics_every_epochs=5,
                  checkpoint_every_epochs=10, run_dir=str(output),
                  comorbidity_features=str(args.comorbidity_features.resolve()),
                  run_label=output.name, cadences=[14], publish_top_level=False)
    fit_args = SimpleNamespace(**values)
    specification = {
        'analysis': 'clinic-exclusive pre-2025 temporal refit',
        'source_sha256': {'BALL.py': digest(REPO/'BALL.py'),
                          'empirical/prospective_classical.py': digest(REPO/'empirical/prospective_classical.py')},
        'settings': values,
        'partition_audit': partition_audit,
        'hyperparameters': 'Earlier development settings fixed at persistence .9, teacher weight 10, student weight 2; no new point-model tuning',
        'primary_clinical_fits_changed': False,
        'intervals': 'The canonical temporal summary intervals are provisional; use fresh conditional-quantile refinement outputs for reporting.'
    }
    output.mkdir(parents=True, exist_ok=True)
    replay_output.mkdir(parents=True, exist_ok=True)
    spec_path = output/'temporal_refit_specification.json'
    if spec_path.exists():
        if json.loads(spec_path.read_text()) != specification:
            raise ValueError('Existing temporal output belongs to another source or specification.')
    else:
        if any(output.iterdir()):
            raise ValueError('Use an empty protected output directory for the first run.')
        spec_path.write_text(json.dumps(specification, indent=2)+'\n')
    snapshot = output/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    if not (snapshot/'BALL.py').exists():
        shutil.copy2(REPO/'BALL.py', snapshot/'BALL.py')
    if args.reuse_checkpoints_from is not None:
        # Checkpoint signatures cover the training source, data, and settings, so matching BALL members restore
        # unchanged and the new model is paired against the same BALL predictions.
        source = args.reuse_checkpoints_from/'member_checkpoints'/'14d_generalization_forward_2025'
        target = output/'member_checkpoints'/'14d_generalization_forward_2025'
        if not source.is_dir():
            raise FileNotFoundError(f'No temporal BALL checkpoints in {source}')
        if not target.exists():
            shutil.copytree(source, target)

    sessions = E._coerce_sessions(pd.read_csv(E.OUT/'sessions.csv'))
    anchors = pd.read_csv(E.OUT/'anchors_sparse_14d.csv')
    mapping = E._generalization_split_maps(sessions)['forward_2025']
    source_split = sessions.groupby('id')['split'].first().to_dict()
    observation_ids = {int(pid): 'train' for pid, role in mapping.items()
                       if role == 'train' and source_split[pid] == 'train'}
    measurement_file = output/'measurement_calibration.json'
    if measurement_file.exists():
        measurement = json.loads(measurement_file.read_text())
    else:
        measurement = E._fit_development_measurement_calibration(
            pd.read_csv(E.OUT/'questionnaire_events_all.csv'), observation_ids)
        measurement['fit_scope'] = 'All pre-2025 genuine questionnaires from primary clinic-exclusive source-training patients'
        measurement_file.write_text(json.dumps(measurement, indent=2)+'\n')

    summary_path = output/'empirical_generalization_complete.csv'
    if not summary_path.exists():
        features = E._load_comorbidity_sessions(args.comorbidity_features)
        summary = E._fit_generalization_sensitivities(
            anchors, sessions, sessions[['id', 'day']].drop_duplicates(), features,
            measurement, fit_args, output)
        required = {'ball', 's0_direct_lgssm', 'markov_direct_transition'}
        if not required <= set(summary.method) <= required | {'ball_direct_causal'}:
            raise AssertionError('Fresh temporal output must contain BALL and both classical comparators.')
        summary['hyperparameter_selection_scope'] = 'Settings fixed from earlier development selection; no temporal point-model reselection'
        summary.to_csv(summary_path, index=False)
    else:
        summary = pd.read_csv(summary_path)

    manifest = {
        'status': 'complete', 'run_id': output.name, 'fit_args': values,
        'source_snapshot': 'source_snapshot/BALL.py',
        'ball_py_sha256': specification['source_sha256']['BALL.py'],
        'partition_audit': partition_audit,
        'primary_clinical_fits_changed': False,
        'temporal_only': True
    }
    (output/'fit_run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')

    # Older replay verification used an untruncated same-day window. Fresh
    # canonical temporal summaries now use the correct prospective window.
    # Compare the fresh replay's current causal-window RMSE below instead.
    replay_args = SimpleNamespace(frozen_run=output,
        comorbidity_features=args.comorbidity_features, output=replay_output,
        cadences=[14], temporal=True, device=args.device, batch_size=32)
    R.replay(replay_args, verify_frozen=False)
    replay_measurement = json.loads((replay_output/'temporal_measurement_calibration.json').read_text())
    if replay_measurement['instruments'] != measurement['instruments']:
        raise AssertionError('Fresh replay measurement links differ from fresh fitted-model links.')
    rows = pd.read_csv(replay_output/'replayed_events_14d.csv')
    test = rows[rows.role.eq('test')]
    scales = test.anchor.map(measurement['development_data']['normalized_score_sd_by_instrument'])
    squared = ((test.ball-test.target)/scales)**2
    rmse = float(np.sqrt(squared.groupby(test.id).mean().mean()))
    expected = float(summary.loc[summary.method.eq('ball'), 'rmse'].iloc[0])
    error = abs(rmse-expected)
    if error >= 2e-5:
        raise AssertionError(f'Fresh causal-window replay differs from fresh BALL summary: {error}')
    replay_audit = {'reference': 'fresh canonical temporal BALL summary with causal recall windows',
                    'rmse': rmse, 'reference_rmse': expected,
                    'absolute_rmse_difference': error,
                    'test_patients': int(test.id.nunique()), 'test_events': len(test),
                    'measurement_links_equal': True}
    (replay_output/'fresh_causal_replay_audit.json').write_text(json.dumps(replay_audit, indent=2)+'\n')
    print(json.dumps(replay_audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
