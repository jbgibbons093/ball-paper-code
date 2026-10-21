"""Full clinical rerun after recoding PHQ-9 totals of 0 without a raw total as missing (17 October 2026).

Inputs. The anchors were rebuilt by `python BALL.py empirical-build-anchors --sweep` after the recode, and the
9 August anchors are archived in empirical/derived/old/anchors_before_phq9_recode_20261017/. At the author's
request the clinical models also receive five session measures from the previous recorded session (stimulation
intensity as a proportion of motor threshold, current medication status, the provider's two suicidal-ideation
notes, and course number).

Primary fit. v12's arguments (publication_ready_20260809_v12_uniform_cadence/fit_run_manifest.json), except
--no-session-balanced-anchors, because the equal-weight analysis left the paper.

Downstream. Commands follow empirical/README_uncertainty_revision.md, the 2025-cohort commands recorded in
_working/analysis_alignment_20260919/PIPELINE_STATE.json, and the 17 October 2026 comparator and random-request
commands. Patient-level outputs stay under the protected run folder. Only aggregate CSV and JSON files are
written next to this script.

Scheduling. One full-data training process needs most of the 8 GB GPU, and two concurrent training processes
stall in memory paging (tested 17 October), so heavy GPU jobs (training and full ensemble replays) run one at
a time. Light GPU jobs (request replays) run up to three at once while no heavy job runs, as on 17 October.
CPU jobs run alongside.

Usage:  python run_rerun.py [--resume] [--max-light-gpu-jobs N] [--max-cpu-jobs N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTECTED = Path(os.environ['BALL_PROTECTED_ROOT'])
V12 = PROTECTED / 'publication_ready_20260809_v12_uniform_cadence'
COMORBIDITY = PROTECTED / 'session_comorbidity_features.csv'
RUN = PROTECTED / 'phq9_recode_20261017'
EMPIRICAL = ROOT / 'empirical'
ALIGNMENT = ROOT / '_working/analysis_alignment_20260919'
REANALYSIS = ROOT / '_working/review_reanalysis_20261017/run_reanalysis.py'
PYTHON = [sys.executable, '-X', 'utf8', '-u']
CADENCES = [14, 21, 28, 7]          # v12 order
RANDOM_CADENCES = [14, 21, 28]      # the 17 October random-request schedules
THREADS_PER_SHARED_JOB = '4'        # light GPU and CPU jobs share 32 cores
GPU_SAMPLE_SECONDS = 30
POLL_SECONDS = 5

V12_FIT_ARGS = json.loads((V12 / 'fit_run_manifest.json').read_text(encoding='utf-8-sig'))['fit_args']
PER_RUN_KEYS = {'run_dir', 'run_label', 'comorbidity_features', 'cadences'}
FIT_OVERRIDES = {'session_balanced_anchors': False, 'publish_top_level': False}


def fit_command(run_dir: Path):
    settings = {k: v for k, v in V12_FIT_ARGS.items() if k not in PER_RUN_KEYS}
    settings.update(FIT_OVERRIDES)
    command = PYTHON + [str(ROOT / 'BALL.py'), 'empirical-fit', '--run-dir', str(run_dir), '--run-label',
                        'phq9_recode_20261017', '--comorbidity-features', str(COMORBIDITY),
                        '--cadences', *map(str, CADENCES)]
    for key, value in settings.items():
        flag = '--' + key.replace('_', '-')
        if value is None:
            continue
        if isinstance(value, bool):
            command.append(flag if value else '--no-' + key.replace('_', '-'))
        elif isinstance(value, list):
            command += [flag, *map(str, value)]
        else:
            command += [flag, str(value)]
    return command


def script(path: Path, *args):
    return PYTHON + [str(path), *map(str, args)]


@dataclass
class Job:
    name: str
    deps: list[str]
    gpu: str | None                 # 'heavy', 'light', or None for CPU-only jobs
    command: list[str] | None = None
    function: Callable[[], dict] | None = None
    cwd: Path = ROOT


def gather_classical(classical: Path, cadences) -> dict:
    """Same step as the 17 October aggregate: the per-schedule comparator events sit at the run root."""
    for cadence in cadences:
        shutil.copy2(classical / 'by_cadence' / f'{cadence}d' / f'corrected_comparator_events_{cadence}d.csv',
                     classical / f'corrected_comparator_events_{cadence}d.csv')
    return {'cadences': list(cadences)}


def build_jobs() -> list[Job]:
    fit, rev, ref, cls, tmp = RUN / 'fit', RUN / 'revision', RUN / 'refinement', RUN / 'classical', RUN / 'temporal'
    agg, agg_rev = HERE / 'aggregates', HERE / 'aggregates_revision'
    agg_ref_t = HERE / 'aggregates_refinement_temporal'   # README temporal refinement, kept apart as in September
    common = ['--comorbidity-features', COMORBIDITY]
    jobs = [
        Job('preflight', [], None, PYTHON + ['-m', 'unittest', 'validation.test_uncertainty_revision',
                                             'validation.test_interval_refinement', 'validation.test_prospective_classical',
                                             'validation.test_prospective_simulation_filter']),
        Job('fit', ['preflight'], 'heavy', fit_command(fit)),
    ]
    # Heavy jobs start in list order, so the replay that most later jobs need comes first.
    jobs += [
        Job('revise', ['fit'], 'heavy', script(EMPIRICAL / 'revise_uncertainty.py', '--frozen-run', fit, *common, '--output', rev)),
        Job('revise_temporal', ['fit'], 'heavy', script(EMPIRICAL / 'revise_uncertainty.py', '--frozen-run', fit, *common,
                                                        '--output', rev / 'temporal', '--cadences', 14, '--temporal')),
        Job('temporal_fit', ['fit'], 'heavy', script(ALIGNMENT / 'refit_temporal.py', '--source-run', fit, *common,
                                                     '--output', tmp / 'fit', '--replay-output', tmp / 'replay',
                                                     '--device', 'cuda', '--fit')),
        Job('crossfit', ['fit'], 'heavy', script(EMPIRICAL / 'crossfit_uncertainty.py', '--frozen-run', fit, *common,
                                                 '--output', ref / 'crossfit')),
        Job('crossfit_temporal', ['fit'], 'heavy', script(EMPIRICAL / 'crossfit_uncertainty.py', '--frozen-run', fit, *common,
                                                          '--output', ref / 'temporal/crossfit', '--cadences', 14, '--temporal')),
        Job('temporal_crossfit', ['temporal_fit'], 'heavy', script(EMPIRICAL / 'crossfit_uncertainty.py', '--frozen-run', tmp / 'fit',
                                                                   *common, '--output', tmp / 'crossfit', '--cadences', 14,
                                                                   '--temporal', '--device', 'cuda')),
    ]
    # Uncertainty revision (README lines 41-49 and 53-54).
    jobs += [
        Job('conditional', ['revise'], None, script(EMPIRICAL / 'conditional_intervals.py', '--run', rev, '--aggregate-output', agg_rev)),
        Job('replay_comparators', ['revise'], 'light', script(EMPIRICAL / 'replay_comparators.py', '--frozen-run', fit, *common,
                                                              '--output', rev)),
    ]
    jobs += [Job(f'adaptive_{c}', ['revise'], 'light', script(EMPIRICAL / 'adaptive_reassessment.py', '--frozen-run', fit,
                                                            *common, '--output', rev, '--cadence', c)) for c in CADENCES]
    jobs += [
        Job('summarize_reassessment', ['conditional', 'replay_comparators'] + [f'adaptive_{c}' for c in CADENCES], None,
            script(EMPIRICAL / 'summarize_reassessment.py', '--run', rev, '--frozen-run', fit, '--output', agg_rev)),
        Job('aggregate_revision', ['summarize_reassessment'], None,
            script(EMPIRICAL / 'aggregate_revision.py', '--run', rev, '--frozen-run', fit, '--output', agg_rev), cwd=EMPIRICAL),
        Job('conditional_temporal', ['revise_temporal'], None,
            script(EMPIRICAL / 'conditional_intervals.py', '--run', rev / 'temporal', '--aggregate-output', agg_rev / 'temporal')),
    ]
    # Interval refinement (README lines 100-108).
    jobs += [
        Job('refine', ['crossfit', 'conditional'], None,
            script(EMPIRICAL / 'refine_intervals.py', '--previous-run', rev, '--crossfit-run', ref / 'crossfit',
                   '--output', ref, '--aggregate-output', agg)),
        Job('refine_temporal', ['crossfit_temporal', 'conditional_temporal'], None,
            script(EMPIRICAL / 'refine_intervals.py', '--previous-run', rev / 'temporal', '--crossfit-run', ref / 'temporal/crossfit',
                   '--output', ref / 'temporal', '--aggregate-output', agg_ref_t, '--cadences', 14)),
        Job('aggregate_refinement', ['refine', 'refine_temporal', 'aggregate_revision'], None,
            script(EMPIRICAL / 'aggregate_revision.py', '--run', rev, '--interval-run', ref, '--frozen-run', fit, '--output', agg),
            cwd=EMPIRICAL),
        Job('summarize_reassessment_refined', ['refine', 'summarize_reassessment'], None,
            script(EMPIRICAL / 'summarize_reassessment.py', '--run', rev, '--interval-run', ref, '--refined',
                   '--frozen-run', fit, '--output', agg)),
    ]
    # 2025 cohort (commands recorded in the September alignment pipeline state).
    jobs += [
        Job('temporal_previous_intervals', ['temporal_fit'], None,
            script(EMPIRICAL / 'conditional_intervals.py', '--run', tmp / 'replay', '--aggregate-output',
                   agg / 'temporal/previous_conditional_quantiles')),
        Job('temporal_intervals', ['temporal_previous_intervals', 'temporal_crossfit'], None,
            script(EMPIRICAL / 'refine_intervals.py', '--previous-run', tmp / 'replay', '--crossfit-run', tmp / 'crossfit',
                   '--output', tmp / 'intervals', '--aggregate-output', agg / 'temporal', '--cadences', 14)),
        Job('temporal_classical', ['temporal_fit'], None,
            script(EMPIRICAL / 'rerun_prospective_classical.py', '--frozen-run', tmp / 'fit', *common, '--previous-run', tmp / 'replay',
                   '--output', cls / 'temporal', '--cadences', 14, '--temporal')),
        Job('temporal_summary', ['temporal_intervals', 'temporal_classical'], None,
            script(ALIGNMENT / 'summarize_temporal.py', '--frozen-run', tmp / 'fit', '--replay-run', tmp / 'replay',
                   '--classical-run', cls / 'temporal', '--interval-run', tmp / 'intervals', '--output', agg / 'temporal'),
            cwd=EMPIRICAL),
    ]
    # Statistical comparators with estimated transition variance (17 October commands), then final aggregates.
    jobs += [Job(f'classical_{c}', ['replay_comparators'], None,
                 script(EMPIRICAL / 'rerun_prospective_classical.py', '--frozen-run', fit, *common, '--previous-run', rev,
                        '--output', cls / 'by_cadence' / f'{c}d', '--cadences', c)) for c in CADENCES]
    jobs += [
        Job('classical_gather', [f'classical_{c}' for c in CADENCES], None, function=lambda: gather_classical(cls, CADENCES)),
        Job('aggregate_final', ['classical_gather', 'aggregate_refinement', 'summarize_reassessment_refined'], None,
            script(EMPIRICAL / 'aggregate_revision.py', '--run', cls, '--interval-run', ref, '--frozen-run', fit, '--output', agg),
            cwd=EMPIRICAL),
        Job('summarize_interval_refinement', ['aggregate_final', 'temporal_classical', 'refine_temporal'], None,
            script(EMPIRICAL / 'summarize_interval_refinement.py', '--run', ref, '--previous-run', rev, '--frozen-run', fit,
                   '--classical-run', cls, '--output', agg), cwd=EMPIRICAL),
    ]
    # Random requests matched on count (17 October procedure), against the new fit and revision run.
    paths = ['--frozen-fit', fit, '--revision-run', rev, '--run', RUN, '--out', HERE]
    for c in RANDOM_CADENCES:
        jobs += [
            Job(f'random_calibrate_{c}', [f'adaptive_{c}'], 'light', script(REANALYSIS, *paths, 'calibrate-random', '--cadence', c)),
            Job(f'random_run_{c}', [f'random_calibrate_{c}'], 'light', script(REANALYSIS, *paths, 'run-random', '--cadence', c)),
            Job(f'random_summary_{c}', [f'random_run_{c}'], None, script(REANALYSIS, *paths, 'summarize-random', '--cadence', c)),
        ]
    return jobs


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_gpu(stop: threading.Event, path: Path):
    with path.open('a') as out:
        while not stop.is_set():
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True)
            out.write(f"{datetime.now().isoformat(timespec='seconds')},{result.stdout.strip()}\n")
            out.flush()
            stop.wait(GPU_SAMPLE_SECONDS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-light-gpu-jobs', type=int, default=3)
    parser.add_argument('--max-cpu-jobs', type=int, default=4)
    parser.add_argument('--resume', action='store_true', help='skip jobs already recorded as complete')
    args = parser.parse_args()

    logs = HERE / 'logs' / 'full'
    logs.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(parents=True, exist_ok=True)
    state_path = HERE / 'RUN_STATE.json'
    state = json.loads(state_path.read_text()) if args.resume and state_path.exists() else {}

    # Freeze the training source: checkpoint signatures hash this snapshot, and every launch verifies BALL.py.
    snapshot = RUN / 'source_snapshot/BALL.py'
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        shutil.copy2(ROOT / 'BALL.py', snapshot)
    frozen_sha = sha256(snapshot)
    base_env = {**os.environ, 'BALL_EMPIRICAL_TRAINING_SOURCE_SNAPSHOT': str(snapshot), 'PYTHONIOENCODING': 'utf-8'}
    shared_env = {**base_env, 'OMP_NUM_THREADS': THREADS_PER_SHARED_JOB, 'MKL_NUM_THREADS': THREADS_PER_SHARED_JOB,
                  'OPENBLAS_NUM_THREADS': THREADS_PER_SHARED_JOB}
    state.setdefault('meta', {}).update(
        ball_py_sha256=frozen_sha, max_light_gpu_jobs=args.max_light_gpu_jobs, max_cpu_jobs=args.max_cpu_jobs,
        fit_overrides_of_v12=FIT_OVERRIDES,
        input_sha256={name: sha256(ROOT / 'empirical/derived' / name) for name in
                      [f'anchors_sparse_{c}d.csv' for c in CADENCES] + ['questionnaire_events_all.csv', 'sessions.csv']},
        anchor_build_manifest_phq9_excluded=json.loads((ROOT / 'empirical/derived/build_manifest.json').read_text())[
            'phq9_totals_excluded_without_raw_score'])
    jobs = {job.name: job for job in build_jobs()}
    for job in jobs.values():
        assert all(dep in jobs for dep in job.deps), (job.name, job.deps)
    records = state.setdefault('jobs', {})

    def save():
        state_path.write_text(json.dumps(state, indent=1, default=str))

    done = {n for n, r in records.items() if r.get('status') == 'complete'} if args.resume else set()
    running: dict[str, tuple[subprocess.Popen, object]] = {}
    failed = None
    stop = threading.Event()
    sampler = threading.Thread(target=sample_gpu, args=(stop, logs / 'gpu_samples.csv'), daemon=True)
    sampler.start()
    try:
        while len(done) < len(jobs) and failed is None:
            for name, (process, log) in list(running.items()):
                if process.poll() is None:
                    continue
                log.close()
                records[name].update(status='complete' if process.returncode == 0 else 'failed',
                                     returncode=process.returncode, finished=datetime.now().isoformat(timespec='seconds'),
                                     seconds=round(time.time() - records[name]['start_epoch']))
                del running[name]
                if process.returncode == 0:
                    done.add(name)
                else:
                    failed = name
                save()
            if failed:
                break
            ready = [j for n, j in jobs.items() if n not in done and n not in running and all(d in done for d in j.deps)]
            heavy_running = any(jobs[n].gpu == 'heavy' for n in running)
            light_running = sum(jobs[n].gpu == 'light' for n in running)
            cpu_running = sum(jobs[n].gpu is None for n in running)
            heavy_waiting = any(j.gpu == 'heavy' for j in ready)
            to_start = []
            for job in ready:
                if job.gpu == 'heavy':
                    if not heavy_running and not light_running and not any(s.gpu for s in to_start):
                        to_start.append(job)
                        heavy_running = True
                elif job.gpu == 'light':
                    # Light jobs never start while a heavy job runs or waits, so heavy jobs are not starved.
                    if not heavy_running and not heavy_waiting and light_running < args.max_light_gpu_jobs:
                        to_start.append(job)
                        light_running += 1
                elif cpu_running < args.max_cpu_jobs:
                    to_start.append(job)
                    cpu_running += 1
            for job in to_start:
                if sha256(ROOT / 'BALL.py') != frozen_sha:
                    raise SystemExit('BALL.py changed during the run; checkpoints would no longer match.')
                start = time.time()
                records[job.name] = dict(status='running', started=datetime.now().isoformat(timespec='seconds'),
                                         start_epoch=start, gpu=job.gpu, deps=job.deps)
                if job.function is not None:
                    try:
                        records[job.name].update(status='complete', result=job.function(), seconds=round(time.time() - start))
                        done.add(job.name)
                    except Exception as error:   # surface the failure and stop the run
                        records[job.name].update(status='failed', error=repr(error))
                        failed = job.name
                    save()
                    if failed:
                        break
                    continue
                records[job.name]['command'] = [str(c) for c in job.command]
                log = (logs / f'{job.name}.log').open('w', encoding='utf-8')
                env = base_env if job.gpu == 'heavy' else shared_env
                running[job.name] = (subprocess.Popen(job.command, cwd=job.cwd, stdout=log, stderr=subprocess.STDOUT,
                                                      env=env), log)
                save()
            time.sleep(POLL_SECONDS)
        for name, (process, log) in running.items():   # let running jobs finish after a failure elsewhere
            process.wait()
            log.close()
            records[name].update(status='complete' if process.returncode == 0 else 'failed', returncode=process.returncode,
                                 finished=datetime.now().isoformat(timespec='seconds'))
            if process.returncode == 0:
                done.add(name)
    finally:
        stop.set()
        state['meta']['finished'] = datetime.now().isoformat(timespec='seconds')
        state['meta']['status'] = 'complete' if len(done) == len(jobs) else f'stopped after failure in {failed}'
        save()
    print(state['meta']['status'])
    sys.exit(0 if len(done) == len(jobs) else 1)


if __name__ == '__main__':
    main()
