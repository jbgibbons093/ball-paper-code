'''Audit recovery of known transition coefficients by the classical models.'''

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


VALIDATION_DIR = Path(__file__).resolve().parent
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

import direct_rdoc_benchmark as benchmark  # noqa: E402
from direct_rdoc_fair_comparator import (  # noqa: E402
    _classical_gap_factors,
    fit_direct_map,
    impute_by_time,
    transition_features,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fit_args(n: int, t: int) -> SimpleNamespace:
    return SimpleNamespace(
        n=int(n),
        t=int(t),
        slow_fraction=0.75,
        delta_ar=0.3,
        rdoc_active_dims=3,
        rdoc_beta_seed=31011,
        rdoc_min_abs=0.35,
        interaction_strength=1.5,
        nonlinear_strength=0.75,
        heterogeneity_strength=0.50,
        n_subtypes_for_scale=3,
        missing_daily_probability=0.55,
        missing_mnar_gamma=0.7,
        missing_proxy_noise=0.40,
        missing_density_multiplier=1.5,
        anchor_observation="gaussian",
        irt_n_items=9,
        irt_discrimination=1.5,
        daily_ridge=1.0,
        s0_basis="linear",
        iters=25,
        markov_iters=25,
        beta_ridge=10.0,
        markov_ridge=10.0,
        prior_sd=5.0,
        transition_sd=0.75,
        markov_strata=3,
        classical_persistence_grid=[0.6],
        classical_min_iterations=3,
        classical_convergence_tol=1e-4,
        classical_innovation_floor=0.05,
        classical_min_stratum_validation_patients=3,
        classical_markov_selection_sweeps=3,
    )


def _one_active_fixture(
    args: SimpleNamespace,
    *,
    active_index: int,
    seed: int,
):
    """Create a two-channel AR process with one known positive coefficient."""

    cfg = benchmark._base_config(args, int(seed), 0.0, "linear")
    anchor_type = type(cfg.y1)
    cfg = dataclasses.replace(
        cfg,
        delta_ar=tuple([0.6] * int(cfg.n_subtypes)),
        y1=anchor_type("Y1", 1, 1, 1.0, 0.02, 0.0),
        y2=anchor_type("Y2", 1, 1, 1.0, 0.02, 0.0),
        rho_cross=0.0,
        rho_serial_y1=0.0,
        rho_serial_y2=0.0,
    )
    data = benchmark.H.generate_dataset(cfg)
    components = data.components.copy()
    daily = data.daily.copy()
    beta = np.zeros(int(cfg.q), dtype=float)
    beta[int(active_index)] = 0.55

    for patient_id, patient in components.groupby("id", sort=True):
        order = patient.sort_values("t").index
        count = len(order)
        rng = np.random.default_rng(int(seed) * 100_003 + int(patient_id))
        exposure = rng.normal(0.0, 1.0, size=(count, int(cfg.q)))
        z_d = np.zeros(count, dtype=float)
        z_a = np.zeros(count, dtype=float)
        z_d[0], z_a[0] = rng.normal(0.0, 0.05, size=2)
        for position in range(1, count):
            drift = float(exposure[position - 1] @ beta)
            z_d[position] = 0.6 * z_d[position - 1] + drift + rng.normal(0.0, 0.02)
            z_a[position] = 0.6 * z_a[position - 1] + drift + rng.normal(0.0, 0.02)

        components.loc[order, "a"] = -1
        components.loc[order, "recent_treatment"] = 0.0
        components.loc[order, "treatment_burden"] = 0.0
        components.loc[order, "dt"] = 1.0
        components.loc[order, "proxy_observed"] = True
        components.loc[order, "z_d"] = z_d
        components.loc[order, "z_p"] = z_a
        components.loc[order, "L"] = 0.5 * (z_d + z_a)
        components.loc[order, "S_joint"] = 0.5 * (z_d + z_a)
        components.loc[order, "delta"] = 0.5 * (z_d + z_a)
        components.loc[order, "delta_d"] = z_d
        components.loc[order, "delta_p"] = z_a
        for dimension in range(int(cfg.q)):
            components.loc[order, f"C{dimension}"] = exposure[:, dimension]
            components.loc[order, f"B{dimension}"] = exposure[:, dimension]

    # Dense EHR values are removed from this fixture so that the full estimator must.
    for column in daily.columns:
        if column.startswith("X"):
            daily[column] = 0.0
        if column.startswith("obs_X") or column.startswith("input_X"):
            daily[column] = False

    train_ids = set(
        data.individuals.loc[data.individuals["split"].eq("train"), "id"].astype(int)
    )
    anchors = benchmark.generate_anchors(components, cfg, train_ids=train_ids)
    anchors["measurement_selection"] = False
    anchors["measurement_eval"] = False
    anchors["measurement_embargo"] = False
    metadata = {
        **data.metadata,
        "rdoc_drift_beta_unit": (beta / np.linalg.norm(beta)).tolist(),
        "rdoc_drift_active_dims": 1,
        "coefficient_fixture": "one active coefficient with dense nearly noiseless one-day anchors",
    }
    return dataclasses.replace(
        data,
        config=cfg,
        components=components,
        daily=daily,
        anchors=anchors,
        metadata=metadata,
    )


def _publication_variant(data, *, true_profile: bool, dense_readout: bool):
    components = data.components.copy()
    daily = data.daily.copy()
    if true_profile:
        for dimension in range(int(data.config.q)):
            components[f"B{dimension}"] = components[f"C{dimension}"]
    if not dense_readout:
        for column in daily.columns:
            if column.startswith("obs_X") or column.startswith("input_X"):
                daily[column] = False
    return dataclasses.replace(data, components=components, daily=daily)


def _fit_method(data, args: SimpleNamespace, method: str):
    if method == "lgssm":
        predictions, _ = fit_direct_map(data, args)
    elif method == "markov":
        predictions, _, _ = benchmark.fit_markov_direct(data, args, basis="linear")
    else:
        raise ValueError(method)
    beta_hat = np.asarray(predictions.attrs["shared_beta"], dtype=float)
    beta_true = np.asarray(data.metadata["rdoc_drift_beta_unit"], dtype=float)
    denominator = float(np.linalg.norm(beta_hat) * np.linalg.norm(beta_true))
    cosine = float(beta_hat @ beta_true / denominator) if denominator > 0 else float("nan")
    return predictions, beta_hat, beta_true, cosine


def _oracle_shared_beta(data, args: SimpleNamespace, *, true_profile: bool):
    """Fit the shared transition regression to the known latent trajectories."""

    q = int(data.config.q)
    training_ids = set(
        data.individuals.loc[data.individuals["split"].eq("train"), "id"].astype(int)
    )
    channel_designs: list[list[np.ndarray]] = [[], []]
    channel_targets: list[list[np.ndarray]] = [[], []]
    for patient_id in sorted(training_ids):
        patient = (
            data.components.loc[data.components["id"].eq(patient_id)]
            .sort_values("t")
            .reset_index(drop=True)
        )
        profile = (
            patient[[f"C{dimension}" for dimension in range(q)]].to_numpy(dtype=float)
            if true_profile
            else impute_by_time(patient, [f"B{dimension}" for dimension in range(q)])
        )
        action = patient["a"].to_numpy(dtype=int)
        recent = patient["recent_treatment"].to_numpy(dtype=float)
        burden = patient["treatment_burden"].to_numpy(dtype=float)
        features = transition_features(
            profile,
            action,
            recent,
            burden,
            data.config.n_treatment_types,
            basis="linear",
            subtype=int(patient["subtype"].iloc[0]),
            n_subtypes=data.config.n_subtypes,
        )
        count = len(features)
        gaps = (
            pd.to_numeric(patient["dt"], errors="coerce")
            .fillna(1.0)
            .clip(lower=0.0)
            .to_numpy(dtype=float)[:count]
            if "dt" in patient
            else np.ones(count, dtype=float)
        )
        persistence, drift = _classical_gap_factors(0.6, gaps)
        valid = gaps > 0
        for channel_index, latent_column in enumerate(("z_d", "z_p")):
            latent = patient[latent_column].to_numpy(dtype=float)
            outcome = (
                latent[1 : count + 1] - persistence * latent[:count]
            ) / np.maximum(drift, 1e-8)
            channel_designs[channel_index].append(features[valid])
            channel_targets[channel_index].append(outcome[valid])

    designs = [np.vstack(blocks) for blocks in channel_designs]
    outcomes = [np.concatenate(blocks) for blocks in channel_targets]
    non_beta = np.asarray(
        [index for index in range(designs[0].shape[1]) if not 1 <= index < 1 + q],
        dtype=int,
    )
    stacked_designs: list[np.ndarray] = []
    for channel_index, design in enumerate(designs):
        shared = design[:, 1 : 1 + q]
        own = design[:, non_beta]
        zeros = np.zeros_like(own)
        stacked_designs.append(
            np.column_stack(
                [
                    shared,
                    own if channel_index == 0 else zeros,
                    zeros if channel_index == 0 else own,
                ]
            )
        )
    design = np.vstack(stacked_designs)
    outcome = np.concatenate(outcomes)
    penalty = float(args.beta_ridge) * np.eye(design.shape[1])
    for channel_index in range(2):
        penalty[q + channel_index * len(non_beta), q + channel_index * len(non_beta)] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ outcome,
    )
    beta_hat = coefficients[:q]
    beta_true = np.asarray(data.metadata["rdoc_drift_beta_unit"], dtype=float)
    denominator = float(np.linalg.norm(beta_hat) * np.linalg.norm(beta_true))
    cosine = float(beta_hat @ beta_true / denominator) if denominator > 0 else float("nan")
    return beta_hat, beta_true, cosine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=91001)
    parser.add_argument("--fixture-n", type=int, default=48)
    parser.add_argument("--fixture-t", type=int, default=36)
    parser.add_argument("--diagnostic-n", type=int, default=90)
    parser.add_argument("--diagnostic-t", type=int, default=84)
    parser.add_argument("--skip-fixtures", action="store_true")
    parser.add_argument("--skip-diagnostic", action="store_true")
    cli = parser.parse_args()
    cli.out.mkdir(parents=True, exist_ok=True)
    result_path = cli.out / "classical_coefficient_recovery_audit.csv"
    manifest_path = cli.out / "audit_manifest.json"
    rows: list[dict] = []

    if not cli.skip_fixtures:
        fixture_args = _fit_args(cli.fixture_n, cli.fixture_t)
        for active_index in range(6):
            data = _one_active_fixture(
                fixture_args,
                active_index=active_index,
                seed=cli.seed + active_index,
            )
            for method in ("lgssm", "markov"):
                predictions, beta_hat, beta_true, cosine = _fit_method(data, fixture_args, method)
                rows.append(
                    {
                        "audit_stage": "one_active_known_truth",
                        "variant": f"active_{active_index}",
                        "method": method,
                        "seed": cli.seed + active_index,
                        "active_index": active_index,
                        "beta_cosine": cosine,
                        "largest_absolute_index": int(np.argmax(np.abs(beta_hat))),
                        "active_sign_correct": bool(beta_hat[active_index] > 0),
                        "active_index_correct": bool(int(np.argmax(np.abs(beta_hat))) == active_index),
                        "converged": bool(predictions.attrs["converged"]),
                        "iterations_completed": int(predictions.attrs["iterations_completed"]),
                        "max_convergence_change": float(predictions.attrs["max_convergence_change"]),
                        **{f"beta_hat_{index}": float(beta_hat[index]) for index in range(6)},
                        **{f"beta_true_{index}": float(beta_true[index]) for index in range(6)},
                    }
                )
                pd.DataFrame(rows).to_csv(result_path, index=False)
                print(
                    f"fixture active={active_index} method={method} cosine={cosine:.4f}",
                    flush=True,
                )

    if not cli.skip_diagnostic:
        diagnostic_args = _fit_args(cli.diagnostic_n, cli.diagnostic_t)
        publication = benchmark.make_data(diagnostic_args, cli.seed, 0.10, "linear")
        for profile_label, true_profile in (
            ("observed_profile_oracle_latent", False),
            ("true_profile_oracle_latent", True),
        ):
            beta_hat, beta_true, cosine = _oracle_shared_beta(
                publication,
                diagnostic_args,
                true_profile=true_profile,
            )
            rows.append(
                {
                    "audit_stage": "publication_dgp_path_diagnostic",
                    "variant": profile_label,
                    "method": "oracle_true_latent_regression",
                    "seed": cli.seed,
                    "active_index": -1,
                    "beta_cosine": cosine,
                    "largest_absolute_index": int(np.argmax(np.abs(beta_hat))),
                    "active_sign_correct": np.nan,
                    "active_index_correct": np.nan,
                    "converged": np.nan,
                    "iterations_completed": np.nan,
                    "max_convergence_change": np.nan,
                    **{f"beta_hat_{index}": float(beta_hat[index]) for index in range(6)},
                    **{f"beta_true_{index}": float(beta_true[index]) for index in range(6)},
                }
            )
            pd.DataFrame(rows).to_csv(result_path, index=False)
            print(f"diagnostic variant={profile_label} method=oracle cosine={cosine:.4f}", flush=True)
        for variant, true_profile, dense_readout in (
            ("observed_profile_with_dense_readout", False, True),
            ("observed_profile_without_dense_readout", False, False),
            ("true_profile_with_dense_readout", True, True),
            ("true_profile_without_dense_readout", True, False),
        ):
            data = _publication_variant(
                publication,
                true_profile=true_profile,
                dense_readout=dense_readout,
            )
            for method in ("lgssm", "markov"):
                predictions, beta_hat, beta_true, cosine = _fit_method(data, diagnostic_args, method)
                rows.append(
                    {
                        "audit_stage": "publication_dgp_path_diagnostic",
                        "variant": variant,
                        "method": method,
                        "seed": cli.seed,
                        "active_index": -1,
                        "beta_cosine": cosine,
                        "largest_absolute_index": int(np.argmax(np.abs(beta_hat))),
                        "active_sign_correct": np.nan,
                        "active_index_correct": np.nan,
                        "converged": bool(predictions.attrs["converged"]),
                        "iterations_completed": int(predictions.attrs["iterations_completed"]),
                        "max_convergence_change": float(predictions.attrs["max_convergence_change"]),
                        **{f"beta_hat_{index}": float(beta_hat[index]) for index in range(6)},
                        **{f"beta_true_{index}": float(beta_true[index]) for index in range(6)},
                    }
                )
                pd.DataFrame(rows).to_csv(result_path, index=False)
                print(f"diagnostic variant={variant} method={method} cosine={cosine:.4f}", flush=True)

    manifest = {
        "status": "complete",
        "completed": datetime.now().isoformat(timespec="seconds"),
        "result": str(result_path.resolve()),
        "result_sha256": _sha256(result_path),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "benchmark_sha256": _sha256(VALIDATION_DIR / "direct_rdoc_benchmark.py"),
        "comparator_sha256": _sha256(VALIDATION_DIR / "direct_rdoc_fair_comparator.py"),
        "frozen_persistence": 0.6,
        "fixture_anchor_cadence_days": 1,
        "fixture_anchor_recall_days": 1,
        "fixture_anchor_error_sd": 0.02,
        "fixture_dense_readout_enabled": False,
        "fixture_profile_measurement_error": 0.0,
        "row_count": len(rows),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {result_path}", flush=True)
    print(f"wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
