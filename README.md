# BALL

BALL means Bayesian Anchored Latent Learning. The method estimates session-level
latent trajectories from sparse validated measurements and longitudinal signals,
then reports uncertainty for each estimate.

This repository contains the model, simulation, empirical analysis, and focused
validation code.

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
framework used as the transition target. IRT means item response theory, used
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
split conformal calibration on held-out anchor residuals.

Reported results label the evaluated role precisely. A result from the
bidirectional model is the teacher smoother. A result from the deployable causal
model is the student.
