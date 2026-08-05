# BALL

BALL means Bayesian Anchored Latent Learning. The method estimates session-level
latent trajectories from sparse validated measurements and longitudinal signals,
then reports uncertainty for each estimate.

This repository contains the model, simulation, empirical evaluation, validation,
table, and figure code for the manuscript.

## Layout

`BALL.py` is the canonical single-file codebase. It holds 37 modules and package stubs as embedded
source blocks and installs them at import time as a virtual `simulations.*` and
`empirical.*` package, so the file is self-contained and needs no accompanying
package directory. It covers the data-generating process, the anchor and
missingness models, the metrics and diagnostics, every method implementation,
the simulation runners, and the empirical pipeline.

The method implementations inside `BALL.py` include the BALL teacher and causal
student, a forward-only transformer trained directly on the measurements, a
causal Gaussian-process filter, a causal ordinary differential equation recurrent
neural network, linear-Gaussian and Markov statistical comparators, and the
questionnaire-history baselines.

`simulations/config/` holds the simulation parameters. `default.yml` defines the
primary scenario and `scenario_grid.yml` defines the identifiability-critical and
worst-case scenarios.

`validation/` holds the standalone validation, comparator, ablation, and
empirical-check scripts that run against the model in `BALL.py`.

| File | Purpose |
| --- | --- |
| `ball_validation_harness.py` | Validation harness for the data-generating process |
| `anchor_identifiability_check.py` | Anchor information-structure identifiability check |
| `direct_rdoc_benchmark.py` | Consolidated direct-RDoC benchmark, BALL against fair comparators |
| `direct_rdoc_common.py` | Shared diagnostics for the direct-RDoC scripts |
| `direct_rdoc_fair_comparator.py` | Fair classical comparator for direct-RDoC drift recovery |
| `direct_rdoc_s0_compare.py` | S0 direct comparator on direct-RDoC cells |
| `direct_rdoc_markov_compare.py` | Native Markov direct-RDoC comparator |
| `direct_rdoc_negative_controls.py` | Negative controls for direct-RDoC drift recovery |
| `direct_rdoc_transformer_ablation.py` | Transformer ablation for drift-parameter recovery |
| `transformer_pressure_ablation.py` | Transformer decomposition-pressure ablation |
| `build_irt_calibration.py` | Builds the frozen graded-response IRT calibration |
| `test_irt_calibration.py` | Deterministic tests for the calibrated IRT observation model |
| `irt_calibration.json` | Frozen IRT parameters from an independent synthetic calibration population |
| `run_publication_sensitivities.py` | Runs the publication sensitivity analyses |
| `analyze_rdoc_scorer_sensitivity.py` | Sensitivity of results to the RDoC scorer |
| `analyze_empirical_stability.py` | Within-patient and between-patient stability analysis |
| `build_direct_rdoc_manuscript_outputs.py` | Builds the aggregate manuscript tables and figures |
| `validate_publication_outputs.py` | Validates all completed simulation and empirical outputs |
| `validate_manuscript_output_package.py` | Checks the aggregate manuscript package for completeness and patient-level content |
| `run_final_simulation_chain_20260805.ps1` | Runs the frozen simulation suite |
| `run_final_empirical_and_publication_chain_20260805.ps1` | Runs the protected empirical suite and aggregate output build |
| `empirical_phq_reversion.py` | Empirical PHQ-9 fast-residual reversion check |

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

That prints 19 commands. The simulation entry point is `pipeline`, the
replicate-parallel runner is `pilot-batch`, and the empirical pipeline runs
through the `empirical-*` commands. The `paper-*` commands regenerate the
manuscript tables and figures from completed runs.

The `validation/` scripts run directly, for example:

```
python validation/direct_rdoc_benchmark.py --help
```

The empirical commands and the empirical validation scripts require the data
inputs described above, so they will not run against a fresh clone.

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
