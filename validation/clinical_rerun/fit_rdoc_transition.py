"""Refit clinical BALL with the note-based RDoC ratings in its transition model (author request, 20 October 2026).

The primary clinical fit zeroes the RDoC inputs and switches off the RDoC transition term. This refit keeps every
setting of the PHQ-9 recode rerun's primary BALL fit (same data, patient partitions, measurement calibration,
hyperparameters, and member seeds) and changes two things, as in the simulations: each session's ratings enter the
data, and the transition model gives them one coefficient per domain (Equation 1, B'beta). The ratings at a session
come from the most recent note strictly before that session's day (empirical/derived/sessions_with_rdoc.csv), so no
input is recorded at or after the session being estimated.

Patient-level checkpoints and predictions stay in the protected rerun folder. summarize_rdoc_transition.py reads them
and writes aggregates only.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import BALL  # noqa: E402

BALL._install_virtual_package()
E = sys.modules['empirical.fit_empirical']

PROTECTED = Path(os.environ['BALL_PROTECTED_ROOT'])
RUN = PROTECTED / 'phq9_recode_20261017'
OUT = RUN / 'rdoc_transition'
COMORBIDITY = PROTECTED / 'session_comorbidity_features.csv'
V12_ARGS = json.loads((PROTECTED / 'publication_ready_20260809_v12_uniform_cadence/fit_run_manifest.json')
                      .read_text(encoding='utf-8-sig'))['fit_args']   # the arguments the rerun's primary fit used
CADENCES = [14, 21, 28, 7]
ARM = 'ball_rdoc_transition'


def build(cadence, rdoc):
    measurement = json.loads((RUN / 'fit/measurement_calibration.json').read_text(encoding='utf-8-sig'))
    sessions_full = E._coerce_sessions(pd.read_csv(E.OUT / 'sessions.csv'))
    splits, _, _ = E._primary_empirical_partitions(sessions_full)
    anchors = pd.read_csv(E.OUT / f'anchors_sparse_{cadence}d.csv')
    data, _ = E._empirical_neural_dataset(anchors, sessions_full, rdoc, E._load_comorbidity_sessions(COMORBIDITY),
                                          measurement, max_patients=None, patient_splits=splits)
    signature = hashlib.sha256(json.dumps({
        'cadence': int(cadence),
        'target_set_sha256': E._dataframe_sha256(anchors, [c for c in ('id', 'session', 'day', 'anchor', 'role', 'value',
                                                                       'window_start_day', 'window_end_day')
                                                           if c in anchors.columns]),
        'measurement_calibration': measurement,
        'patient_splits': sorted((int(k), str(v)) for k, v in splits.items()),
        'rdoc_sessions_sha256': hashlib.sha256((E.OUT / 'sessions_with_rdoc.csv').read_bytes()).hexdigest(),
        'data_variant': 'rdoc_transition',
    }, sort_keys=True, default=str).encode('utf-8')).hexdigest()
    return data, signature


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cadences', type=int, nargs='+', default=CADENCES)
    p.add_argument('--smoke', action='store_true', help='two epochs per phase, one member, separate folder')
    args = p.parse_args()
    out = OUT / 'smoke' if args.smoke else OUT
    out.mkdir(parents=True, exist_ok=True)
    rdoc = E._load_rdoc_sessions()
    if rdoc is None or rdoc.empty:
        raise FileNotFoundError('sessions_with_rdoc.csv is required for the RDoC transition refit.')
    a = V12_ARGS
    epochs = 2 if args.smoke else None
    timing = {}
    for cadence in args.cadences:
        started = perf_counter()
        data, signature = build(cadence, rdoc)
        cfg = E.SSMConfig(
            d_model=a['d_model'], n_heads=a['n_heads'], n_layers=a['n_layers'],
            teacher_epochs=epochs or a['teacher_epochs'], student_epochs=epochs or a['student_epochs'],
            anchor_warmup_epochs=epochs or a['anchor_warmup'], kl_warmup_epochs=epochs or a['kl_warmup'],
            batch_size=a['batch_size'], ensemble_size=1 if args.smoke else a['members'], max_individuals=data.config.n,
            seed=E.EMPIRICAL_SEED + int(a['seed_offset']),   # the primary fit's member seeds
            rdoc_drift_head=True,                             # the change: RDoC coefficients in the transition
            use_alpha_slow=False, delta_phi=a['delta_phi'], anchor_weight=a['anchor_weight'],
            student_anchor_weight=a['student_anchor_weight'], progress_every_epochs=int(a['progress_every_epochs']),
            diagnostics_every_epochs=int(a['diagnostics_every_epochs']),
            state_checkpoint_every_epochs=int(a['checkpoint_every_epochs']))
        cfg = E._empirical_checkpointed_config(cfg, out / 'member_checkpoints', f'{cadence}d_{ARM}', signature)
        student = E.fit_ball_ssm(data, cfg, device=a['device'], causal=True, prediction_split='test',
                                 return_teacher=False)
        E._assert_result_training_valid(student, 'RDoC-transition BALL student', cfg.ensemble_size)
        timing[str(cadence)] = round(perf_counter() - started, 1)
        (out / 'fit_timing_seconds.json').write_text(json.dumps(timing, indent=2))
        print(json.dumps({'cadence': cadence, 'seconds': timing[str(cadence)]}), flush=True)


if __name__ == '__main__':
    main()
