# BALL

BALL means Bayesian Anchored Latent Learning. The method estimates session-level
latent trajectories from sparse validated measurements and longitudinal signals,
then reports uncertainty for each estimate.

This repository contains the model, simulation, empirical analysis, and focused
validation code.

The October 2026 update matches the clinical analyses rerun from 17 to 20
October 2026. In that rerun, PHQ-9 totals recorded as 0 without an item-level
raw score were treated as missing, and five measures from the previous session
joined the clinical inputs. `validation/finalized_analysis_sources.json`
records SHA-256 hashes of the 35 published analysis source files and of the
`BALL.py` version that each clinical run used. The temporal orchestration
scripts are located under `validation/temporal/`, and the October run scripts
are located under `validation/clinical_rerun/`, both with their source bytes
preserved. The published run scripts read the protected clinical workspace from
the `BALL_PROTECTED_ROOT` environment variable, which is their only change from
the versions that ran.

## Layout

`BALL.py` is the canonical single-file codebase. It holds the modules and package stubs as embedded
source blocks and installs them at import time as a virtual `simulations.*` and
`empirical.*` package, so the file is self-contained and needs no accompanying
package directory. It covers the data-generating process, the anchor and
missingness models, the metrics and diagnostics, every method implementation,
the simulation runners, and the empirical pipeline.

The method implementations inside `BALL.py` include the BALL teacher and causal
student, a forward-only transformer trained directly on the measurements, a
causal Gaussian-process filter, an exponential-decay gated recurrent unit,
linear-Gaussian and Markov statistical comparators, and the
questionnaire-history baselines.

`simulations/config/` holds the simulation parameters. `default.yml` defines the
primary scenario and `scenario_grid.yml` defines the identifiability-critical and
worst-case scenarios.

`validation/` contains reproducibility analyses and focused checks that run
against `BALL.py`. Manuscript assembly and document-checking utilities are not
included.

`empirical/` contains the prospective statistical filters, questionnaire-interval
refinement, and sequential reassessment scripts used in the finalized analyses.
Only Python source files are allowed there in this public repository.

| Files | Purpose |
| --- | --- |
| `ball_validation_harness.py` and `anchor_identifiability_check.py` | Check the data-generating process and anchor design |
| `direct_rdoc_*.py` | Compare transition-coefficient recovery across models and controls |
| `audit_classical_*.py` and `select_classical_persistence.py` | Check classical-model fitting and persistence selection |
| `build_irt_calibration.py`, `test_irt_calibration.py`, and `irt_calibration.json` | Build and test the frozen questionnaire calibration |
| `run_*.py`, `analyze_*.py`, `empirical_phq_reversion.py`, and `postprocess_empirical_rdoc_transitions.py` | Run the scientific sensitivity analyses |
| `test_*.py` | Check core code behavior and analysis contracts |
| `check_repository_privacy.py` | Prevent data and generated artifacts from entering the repository |

RDoC means Research Domain Criteria, the National Institute of Mental Health
framework used for the simulation example of explicit transition-coefficient
recovery. The primary clinical model excludes RDoC scores. A separate clinical
refit adds note-based RDoC scores to BALL's encoder and transition model and
reports their transition coefficients. The scores measure how much the clinical
notes discuss each domain.
IRT means item response theory, used
for the graded-response observation model. PHQ-9 means the nine-item Patient
Health Questionnaire.

## Running

The analyses ran on Python 3.13. The code contains no syntax newer than Python
3.10, but only 3.13 was tested. Exact tested package versions are listed in
`requirements.txt`.

`BALL.py` dispatches by subcommand. List the available commands with:

```
python BALL.py
```

That prints the available commands. The simulation entry point is `pipeline`, the
replicate-parallel runner is `pilot-batch`, and the empirical pipeline runs
through the `empirical-*` commands. The `paper-*` commands run the simulation
analyses and create their tables and figures.

The `validation/` scripts run directly, for example:

```
python validation/direct_rdoc_benchmark.py --help
```

The empirical commands and empirical validation scripts require locally supplied
inputs, so they will not run against a fresh clone.

For the finalized primary simulation, the full arguments for seeds 2027, 2028,
and 2029 are in `validation/finalized_simulation_arguments.json`. Each seed runs
five conditions at two signal strengths, giving 30 generated datasets overall.
The output paths are relative, ignored simulation-output directories. Run these
from the repository root in an environment with the listed dependencies and a
CUDA GPU. These are full analyses rather than short tests.

```python
import json
import subprocess
import sys

with open("validation/finalized_simulation_arguments.json") as source:
    jobs = json.load(source)
for job in jobs:
    subprocess.run([sys.executable, *job["arguments"]], check=True)
```

For protected empirical work, export the source into an approved secure
workspace outside any Git working tree. The embedded empirical modules expect
prepared session and questionnaire inputs under `empirical/derived/` in that
export. Keep all clinical inputs, fitted models, patient-level predictions, and
derived outputs in the approved environment. They are not part of this public
distribution.

The finalized interval pipeline replays the saved prospective predictions,
fits patient-exclusive development folds, and selects conditional questionnaire
intervals using development data before reserved-patient calibration. The main
entry points are `empirical/revise_uncertainty.py`, `crossfit_uncertainty.py`,
`refine_intervals.py`, and `summarize_interval_refinement.py`. Use each script's
`--help` to specify the protected source-run and output directories.
`validation/temporal/refit_temporal.py` applies the pre-2025 development-clinic
restriction to the temporal sensitivity. `empirical/adaptive_reassessment.py`
evaluates sequential measurement requests and reveals a requested score only
from the next recorded session.

The October clinical rerun scripts in `validation/clinical_rerun/` record the
exact order and arguments of each run. They expect the project working layout,
in which `_working/analysis_alignment_20260919/` holds the scripts published
under `validation/temporal/` and `_working/review_reanalysis_20261017/` holds
`run_reanalysis.py`. Set `BALL_PROTECTED_ROOT` to the protected clinical
workspace before running them.

| File | Purpose |
| --- | --- |
| `run_rerun.py` | Runs the primary clinical fit with the frozen arguments, then the uncertainty, interval, 2025-cohort, statistical-comparator, and request analyses |
| `run_reanalysis.py` | Calibrates and runs random questionnaire requests matched on count, then aggregates the paired patient bootstrap comparisons |
| `summarize_session_measures.py` and `check_session_measures.py` | Summarize the five previous-session measures for the cohort table and check them against the source records |
| `rerun_gp.py` | Refits the Gaussian-process filter after standardizing every continuous input |
| `fit_rdoc_transition.py` and `summarize_rdoc_transition.py` | Refit BALL with note-based RDoC scores in its transition model, then score withheld questionnaires and report the coefficients |

These scripts write aggregate CSV and JSON files and run logs next to
themselves, and they keep patient-level checkpoints, predictions, and event
files under `BALL_PROTECTED_ROOT`. Run them from an export outside any Git
working tree.

The original `paper-*` entry points remain available, but their default
arguments alone do not reproduce every finalized analysis. Use the frozen
primary arguments and the interval and temporal entry points above.

The empirical comorbidity builder is `empirical-build-comorbidities`. It reads
the protected SAS extracts and writes a numeric session-level feature table to
an explicitly supplied location outside every Git working tree. A diagnosis is
available to a modeled session only when its recorded date is strictly earlier
than the session date. The empirical fit requires that protected file through
`--comorbidity-features`. Patient-level feature tables must never be copied into
this repository.

## Terminology

The BALL neural architecture is a teacher and student pair. A bidirectional
transformer teacher reads the full training record and estimates the latent
recovery posterior. A forward-only transformer student applies causal masking
and distills from the teacher, and the student is the deployable model.
Uncertainty combines a deep ensemble, a last-layer Laplace approximation, and
target-specific calibration. Simulation analyses can assess coverage of known
latent states. Clinical intervals predict withheld questionnaire scores using
development-fitted conditional quantiles and patient-weighted conformal
calibration. Questionnaire coverage does not establish latent-state coverage.

Prospective comparisons use the same patient partitions, eligible measurement
events, two-channel observation links, and information available at the target
session. Linear Gaussian and Markov models issue forward estimates without
revising earlier predictions using future observations. Retained questionnaires
enter from the next recorded session. Interpolation, questionnaire-history
averages, and last-observation carry-forward are not manuscript comparators.

Reported results label the evaluated role precisely. A result from the
bidirectional model is the teacher smoother. A result from the deployable causal
model is the student.

## Checks

Install `pytest` in addition to the analysis dependencies, then run:

```sh
python -m pytest validation -q
python validation/check_repository_privacy.py
```

The tests use synthetic inputs and require no clinical data. They check causal
information boundaries, instrument-specific measurement links, patient and
clinic partitions, uncertainty calculations, and frozen-source hashes. The
privacy guard scans both the working tree and reachable Git history for data
artifacts, credentials, and local user paths.
