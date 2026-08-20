from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from validation.direct_rdoc_benchmark import fit_markov_direct, make_data
from validation.direct_rdoc_fair_comparator import fit_direct_map


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        n=48,
        t=42,
        slow_fraction=0.75,
        delta_ar=0.3,
        rdoc_active_dims=3,
        rdoc_beta_seed=31011,
        rdoc_min_abs=0.35,
        interaction_strength=1.5,
        nonlinear_strength=0.75,
        heterogeneity_strength=0.5,
        n_subtypes_for_scale=3,
        missing_daily_probability=0.55,
        missing_mnar_gamma=0.7,
        missing_proxy_noise=0.4,
        missing_density_multiplier=1.5,
        anchor_observation="gaussian",
        irt_n_items=9,
        irt_discrimination=1.5,
        daily_ridge=1.0,
        s0_basis="linear",
        iters=3,
        markov_iters=3,
        beta_ridge=10.0,
        markov_ridge=10.0,
        prior_sd=5.0,
        transition_sd=0.75,
        markov_strata=3,
        classical_persistence_grid=[0.3, 0.9],
        classical_min_iterations=2,
        classical_convergence_tol=1e-3,
        classical_innovation_floor=0.05,
        classical_min_stratum_validation_patients=2,
        classical_markov_selection_sweeps=2,
    )


def test_lgssm_selects_persistence_in_validation_patients() -> None:
    args = _args()
    data = make_data(args, 1729, 0.10, "linear")
    predictions, _ = fit_direct_map(data, args)

    assert predictions.attrs["persistence_selection_target"] == "measurement_selection"
    assert np.isfinite(predictions.attrs["validation_selection_rmse"])
    assert 1 <= predictions.attrs["iterations_completed"] <= args.iters
    assert set(predictions.attrs["stratum_persistence"]["depression"].values()).issubset(
        set(args.classical_persistence_grid)
    )
    assert set(predictions.attrs["stratum_persistence"]["anxiety"].values()).issubset(
        set(args.classical_persistence_grid)
    )


def test_markov_has_channel_by_stratum_persistence_and_shared_beta() -> None:
    args = _args()
    data = make_data(args, 2027, 0.10, "linear")
    predictions, theta, _ = fit_markov_direct(data, args, basis="linear")

    exact = predictions.attrs["stratum_persistence"]
    assert predictions.attrs["pattern_mixture"] is True
    assert len(exact["depression"]) == args.markov_strata
    assert len(exact["anxiety"]) == args.markov_strata
    assert set(exact["depression"].values()).issubset(set(args.classical_persistence_grid))
    assert set(exact["anxiety"].values()).issubset(set(args.classical_persistence_grid))
    assert np.allclose(theta[1 : 1 + data.config.q], predictions.attrs["shared_beta"])
    assert np.isfinite(predictions.attrs["validation_selection_rmse"])
    fallback_rows = [
        row
        for row in predictions.attrs["persistence_candidate_trace"]
        if row.get("validation_scope") == "channel_fallback_applied"
    ]
    assert all(row["selection_patients"] < args.classical_min_stratum_validation_patients for row in fallback_rows)


def test_frozen_persistence_runs_one_lgssm_candidate() -> None:
    args = _args()
    args.classical_persistence_grid = [0.6]
    data = make_data(args, 8111, 0.10, "linear")
    predictions, _ = fit_direct_map(data, args)

    assert predictions.attrs["persistence_selection_target"] == "separate_development_exercise"
    assert len(predictions.attrs["persistence_candidate_trace"]) == 1
    assert set(predictions.attrs["stratum_persistence"]["depression"].values()) == {0.6}
    assert set(predictions.attrs["stratum_persistence"]["anxiety"].values()) == {0.6}


def test_frozen_persistence_runs_one_markov_candidate() -> None:
    args = _args()
    args.classical_persistence_grid = [0.6]
    data = make_data(args, 8221, 0.10, "linear")
    predictions, _, _ = fit_markov_direct(data, args, basis="linear")

    assert predictions.attrs["persistence_selection_target"] == "separate_development_exercise"
    assert len(predictions.attrs["persistence_candidate_trace"]) == 1
    assert set(predictions.attrs["stratum_persistence"]["depression"].values()) == {0.6}
    assert set(predictions.attrs["stratum_persistence"]["anxiety"].values()) == {0.6}
