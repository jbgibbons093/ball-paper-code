# BALL

BALL means Bayesian Anchored Latent Learning. The method recovers an
anchor-calibrated latent recovery index from sparse clinical anchors and dense,
noisy longitudinal signals, and it reports calibrated uncertainty so that a user
can tell when the available information is insufficient.

This repository contains the model, simulation, and empirical evaluation code
for the manuscript. It contains no data.

## Layout

`BALL.py` is the canonical single-file codebase. It holds 37 modules as embedded
source blocks and installs them at import time as a virtual `simulations.*` and
`empirical.*` package, so the file is self-contained and needs no accompanying
package directory. It covers the data-generating process, the anchor and
missingness models, the metrics and diagnostics, every method implementation,
the simulation runners, and the empirical pipeline.

The method implementations inside `BALL.py` are the BALL state-space model, the
BALL structural posterior, the S0 linear-Gaussian smoother, the Markov
pattern-mixture comparator, and the simple baselines.

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
| `empirical_phq_reversion.py` | Empirical PHQ-9 fast-residual reversion check |

RDoC means Research Domain Criteria, the National Institute of Mental Health
framework used as the transition target. IRT means item response theory, used
for the graded-response observation model. PHQ-9 means the nine-item Patient
Health Questionnaire.

## Running

The analyses ran on Python 3.13. The code contains no syntax newer than Python
3.10, but only 3.13 was tested. The dependencies are NumPy, pandas, SciPy,
scikit-learn, PyTorch, PyYAML, and Matplotlib.

`BALL.py` dispatches by subcommand. List the available commands with:

```
python BALL.py
```

That prints 18 commands. The simulation entry point is `pipeline`, the
replicate-parallel runner is `pilot-batch`, and the empirical pipeline runs
through the `empirical-*` commands. The `paper-*` commands regenerate the
manuscript tables and figures from completed runs.

The `validation/` scripts run directly, for example:

```
python validation/direct_rdoc_benchmark.py --help
```

The empirical commands and the empirical validation scripts require the data
inputs described above, so they will not run against a fresh clone.

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
