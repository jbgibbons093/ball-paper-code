#!/usr/bin/env python
"""BALL DGP validation harness (validation step A).

Persisted, reproducible replacement for the temporary multi-seed sweep (written
to validation/, locked by BALL.py content hash and a run manifest; this repo is
not under git). It isolates each DGP knob (slow_fraction, delta_ar) across seeds
and scores the slow/fast RDoC decomposition with the leveled structural
comparator under two delta-AR prior settings:

  - matched:   the structural delta_phi is set to the DGP delta_ar (a mild
               simulation oracle: it hands the comparator the true residual
               time scale).
  - estimated: the structural delta_phi is selected with no oracle, over PHI_GRID,
               by minimum held-out validation-anchor RMSE (the only deployable
               signal). A residual-autocorrelation reading of the random-walk fit
               is also recorded per row as a diagnostic only (delta_phi_autocorr_rw).

The estimated arm is the honest test of whether the structural's decomposition
win survives when the residual time scale must be chosen from observables rather
than handed.

For each cell the harness also computes an oracle composite-noise floor (the
lowest composite RMSE any covariate-driven estimator can reach) and runs a
column-permutation safety check on the comparator.

No BALL.py defaults are changed and no model file is edited. S0 is reported as
UNMATCHED: its delta prior is a hardcoded random walk in BALL.py and is not
exposed, so it is excluded from the matched comparison rather than compared as
though it were tuned.

Outputs (under --out): per-seed rows CSV, aggregate mean and MCSE CSV, the
column-permutation CSV, and a JSON provenance manifest (BALL.py content hash and
mtime, cell knobs, command line, seeds, timestamp, output schema).

Usage:
    python validation/ball_validation_harness.py --quick
    python validation/ball_validation_harness.py --n 300 --seeds 1729 2027 2028 2029 4242 9001
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
BALL_PATH = REPO_ROOT / "BALL.py"


def _load_ball():
    """Exec BALL.py as a module and install its embedded virtual packages."""
    spec = importlib.util.spec_from_file_location("BALL", BALL_PATH)
    ball = importlib.util.module_from_spec(spec)
    sys.modules["BALL"] = ball
    spec.loader.exec_module(ball)
    ball._install_virtual_package()
    return ball


_BALL = _load_ball()
SimulationConfig = sys.modules["simulations.src.model_utils"].SimulationConfig
generate_dataset = sys.modules["simulations.src.dgp"].generate_dataset
_metrics = sys.modules["simulations.src.metrics"]
latent_rmse = _metrics.latent_rmse
active_set_metrics = _metrics.active_set_metrics
rdoc_recovery_fraction = _metrics.rdoc_recovery_fraction
identifiability_diagnostics = _metrics.identifiability_diagnostics
anchor_conformal_residuals = _metrics.anchor_conformal_residuals
_structural = sys.modules["simulations.src.methods.ball_structural"]
BallStructuralHyperparameters = _structural.BallStructuralHyperparameters
fit_ball_structural_posterior = _structural.fit_ball_structural_posterior


# --- Fixed harness configuration (named, not scattered as literals) -----------

CELLS = {
    "baseline": {"slow_fraction": 0.30, "delta_ar": 1.0},
    "slow-only": {"slow_fraction": 0.75, "delta_ar": 1.0},
    "delta-only": {"slow_fraction": 0.30, "delta_ar": 0.3},
    "empirical-cal": {"slow_fraction": 0.75, "delta_ar": 0.3},
}
CELL_ORDER = ["baseline", "slow-only", "delta-only", "empirical-cal"]
DEFAULT_SEEDS = [1729, 2027, 2028, 2029, 4242, 9001]
DEFAULT_N = 300
DEFAULT_T = 84
QUICK_N = 100
QUICK_SEEDS = [1729, 2027]
N_SUBTYPES = 3
EVAL_SPLIT = "test"
TRAIN_SPLIT = "train"
ACTIVE_TAU = 0.05
RIDGE_LAMBDA = 1.0
PERM_TOL = 1e-6
NEUTRAL_PHI = 1.0  # random-walk reference for the autocorrelation diagnostic
PHI_GRID = (0.0, 0.3, 0.5, 0.7, 0.9, 1.0)  # delta_phi grid for the invalid anchor-fit selector
# Manuscript-facing reverting-prior family justified by the empirical PHQ-9 artifact
# (validation/empirical_phq_reversion.py): central 0.5, sensitivity over the bracket.
EMPIRICAL_DELTA_PHI_FAMILY = (0.3, 0.5, 0.7)
PERM_CHECK_PHI = 0.5  # central empirical prior used for the column-permutation check
PRIOR_SOURCE_ORDER = ("simulation_matched", "empirical_phq_reversion", "invalid_anchor_fit_selector")

METRIC_KEYS = [
    "slow_recovery_corr",
    "fast_recovery_corr",
    "slow_to_fast_leakage",
    "fast_to_slow_leakage",
    "rho_rdoc",
    "active_f1",
    "active_topk_f1",
    "comp_rmse",
    "oracle_floor_rmse",
    "delta_phi_used",
    "val_anchor_rmse",
]


# --- DGP and fitting ----------------------------------------------------------

def build_config(cell_name: str, seed: int, n: int, t: int) -> "SimulationConfig":
    knobs = CELLS[cell_name]
    return dataclasses.replace(
        SimulationConfig(),
        seed=seed,
        n=n,
        t=t,
        slow_fraction=knobs["slow_fraction"],
        delta_ar=tuple([float(knobs["delta_ar"])] * N_SUBTYPES),
    )


def fit_structural(data, delta_phi: float):
    hyper = BallStructuralHyperparameters(delta_phi=float(delta_phi))
    return fit_ball_structural_posterior(data, hyper)


def score(data, result) -> dict:
    """The full decomposition metric set on the evaluation split."""
    preds = result.predictions
    ident = identifiability_diagnostics(preds, data, EVAL_SPLIT)
    active = active_set_metrics(
        preds, data.components, data.config.q, ACTIVE_TAU, data.individuals, EVAL_SPLIT
    )
    return {
        "slow_recovery_corr": ident["slow_recovery_corr"],
        "fast_recovery_corr": ident["fast_recovery_corr"],
        "slow_to_fast_leakage": ident["slow_to_fast_leakage"],
        "fast_to_slow_leakage": ident["fast_to_slow_leakage"],
        "rho_rdoc": rdoc_recovery_fraction(preds, data, EVAL_SPLIT),
        "active_f1": active["active_f1"],
        "active_topk_f1": active["active_topk_f1"],
        "comp_rmse": latent_rmse(preds, data.components, data.individuals, EVAL_SPLIT),
    }


def _lag1_ar_coef(preds, train_ids) -> float:
    """Pooled within-person lag-1 AR(1) OLS coefficient of fitted delta_hat."""
    df = preds.loc[preds["id"].isin(train_ids), ["id", "t", "delta_hat"]].sort_values(["id", "t"])
    num, den = 0.0, 0.0
    for _, g in df.groupby("id", sort=False):
        v = g["delta_hat"].to_numpy(dtype=float)
        if len(v) < 3:
            continue
        v = v - v.mean()
        num += float(np.sum(v[1:] * v[:-1]))
        den += float(np.sum(v[:-1] * v[:-1]))
    return num / den if den > 1e-12 else NEUTRAL_PHI


def val_anchor_rmse(data, preds) -> float:
    """Held-out anchor fit on the validation split (observable, no oracle).

    Root-mean-square of the appendix conformity residuals |y - f_hat(x)| over the
    observed validation anchors for both channels. This is the only ground-truth
    signal available at deployment, so it is the honest criterion for selecting a
    hyperparameter such as delta_phi.
    """
    zd = anchor_conformal_residuals(preds, data.anchors, data.individuals, "validation", "Y1", "z_d_hat")
    zp = anchor_conformal_residuals(preds, data.anchors, data.individuals, "validation", "Y2", "z_p_hat")
    res = np.concatenate([zd, zp]) if (len(zd) + len(zp)) > 0 else np.array([], dtype=float)
    return float(np.sqrt(np.mean(res ** 2))) if len(res) > 0 else float("nan")


def select_delta_phi(data, cached_fit) -> tuple[float, dict]:
    """Pick delta_phi by held-out validation-anchor fit over PHI_GRID, no oracle.

    Returns the selected phi and the full criterion curve. The decomposition
    quality is unobservable, so this selects on the only deployable signal. If the
    curve is monotone in phi, the residual time scale is not identifiable from
    observable fit and the decomposition cannot be recovered without an oracle.
    """
    curve = {phi: val_anchor_rmse(data, cached_fit(phi).predictions) for phi in PHI_GRID}
    finite = {k: v for k, v in curve.items() if np.isfinite(v)}
    selected = float(min(finite, key=finite.get)) if finite else NEUTRAL_PHI
    return selected, curve


def oracle_composite_floor(data) -> float:
    """Lowest composite RMSE a covariate-driven estimator could reach.

    Ridge regression of the TRUE composite latent L on the full daily covariate
    matrix (all p_daily features), fit on train, evaluated on test. The residual
    is irreducible feature measurement and innovation noise. Missing daily values
    are imputed with train-column means. This is an idealized lower bound: any
    real method that never sees the true L cannot beat it.
    """
    feat_cols = [f"X{j}" for j in range(data.config.p_daily)]
    df = (
        data.daily[["id", "t"] + feat_cols]
        .merge(data.components[["id", "t", "L"]], on=["id", "t"], how="inner")
        .merge(data.individuals[["id", "split"]], on="id", how="left")
    )
    train = df[df["split"] == TRAIN_SPLIT]
    test = df[df["split"] == EVAL_SPLIT]
    if train.empty or test.empty:
        return float("nan")
    x_tr = train[feat_cols].to_numpy(dtype=float)
    x_te = test[feat_cols].to_numpy(dtype=float)
    col_mean = np.nanmean(x_tr, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    x_tr = np.where(np.isfinite(x_tr), x_tr, col_mean)
    x_te = np.where(np.isfinite(x_te), x_te, col_mean)
    mu = x_tr.mean(axis=0)
    sd = x_tr.std(axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    x_tr = (x_tr - mu) / sd
    x_te = (x_te - mu) / sd
    x_tr = np.hstack([np.ones((len(x_tr), 1)), x_tr])
    x_te = np.hstack([np.ones((len(x_te), 1)), x_te])
    y_tr = train["L"].to_numpy(dtype=float)
    y_te = test["L"].to_numpy(dtype=float)
    reg = RIDGE_LAMBDA * np.eye(x_tr.shape[1])
    reg[0, 0] = 0.0  # leave the intercept unpenalized
    beta = np.linalg.solve(x_tr.T @ x_tr + reg, x_tr.T @ y_tr)
    pred = x_te @ beta
    return float(np.sqrt(np.mean((pred - y_te) ** 2)))


def permutation_check(data, delta_phi: float, base_comp_rmse: float, seed: int) -> dict:
    """Refit the comparator on a daily-feature column permutation.

    A comparator with no hard-coded column assignment relearns its readout after
    the daily features are relabeled, so test composite RMSE is invariant. A
    nonzero difference would expose a hidden dependence on specific column slots.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(data.config.p_daily)
    daily = data.daily.copy()
    rename = {}
    for new_j, old_j in enumerate(perm):
        rename[f"X{int(old_j)}"] = f"X{new_j}__tmp"
        rename[f"obs_X{int(old_j)}"] = f"obs_X{new_j}__tmp"
    daily = daily.rename(columns=rename)
    daily = daily.rename(columns={c: c.replace("__tmp", "") for c in daily.columns})
    permuted = dataclasses.replace(data, daily=daily)
    result = fit_structural(permuted, delta_phi)
    perm_rmse = latent_rmse(
        result.predictions, permuted.components, permuted.individuals, EVAL_SPLIT
    )
    diff = abs(perm_rmse - base_comp_rmse)
    return {
        "cell": None,
        "seed": seed,
        "delta_phi": delta_phi,
        "orig_comp_rmse": base_comp_rmse,
        "perm_comp_rmse": perm_rmse,
        "abs_diff": diff,
        "invariant": bool(diff < PERM_TOL),
    }


def run_cell_seed(cell_name: str, seed: int, n: int, t: int, do_perm: bool):
    config = build_config(cell_name, seed, n, t)
    data = generate_dataset(config)
    matched_phi = float(CELLS[cell_name]["delta_ar"])

    fit_cache: dict[float, object] = {}

    def cached_fit(phi: float):
        key = round(float(phi), 6)
        if key not in fit_cache:
            fit_cache[key] = fit_structural(data, key)
        return fit_cache[key]

    floor = oracle_composite_floor(data)
    common = {
        "cell": cell_name,
        "seed": seed,
        "slow_fraction": CELLS[cell_name]["slow_fraction"],
        "delta_ar": matched_phi,
        "n": n,
        "t": t,
        "oracle_floor_rmse": floor,
    }

    train_ids = set(
        int(i) for i in data.individuals.loc[data.individuals["split"] == TRAIN_SPLIT, "id"]
    )
    rows = []

    def make_row(prior_source, delta_phi, delta_phi_source, group_phi, result, extra=None):
        r = score(data, result)
        r.update(common)
        r.update(
            {
                "prior_source": prior_source,
                "delta_phi_used": float(delta_phi),
                "delta_phi_source": delta_phi_source,
                "group_phi": group_phi,
                "val_anchor_rmse": val_anchor_rmse(data, result.predictions),
                "delta_phi_autocorr_rw": float("nan"),
                "phi_crit_curve": "",
            }
        )
        if extra:
            r.update(extra)
        return r

    # 1. simulation_matched: diagnostic only, delta_phi set to the DGP's true value.
    rows.append(
        make_row("simulation_matched", matched_phi, "dgp_true_value", f"{matched_phi:.2f}", cached_fit(matched_phi))
    )

    # 2. empirical_phq_reversion: manuscript-facing reverting-prior family {0.3, 0.5, 0.7},
    #    justified by the empirical PHQ-9 fast-residual persistence bracket.
    for phi in EMPIRICAL_DELTA_PHI_FAMILY:
        rows.append(
            make_row("empirical_phq_reversion", phi, "empirical_phq_reversion_family", f"{phi:.2f}", cached_fit(phi))
        )

    # 3. invalid_anchor_fit_selector: NEGATIVE CONTROL, not a deployable selector.
    #    Selecting delta_phi by in-sample validation-anchor RMSE rewards the flexible
    #    random walk and collapses the decomposition (the anchors are used in each
    #    person's own fit, so scoring them is in-sample). Retained as a warning.
    sel_phi, crit_curve = select_delta_phi(data, cached_fit)
    autocorr_rw = float(np.clip(_lag1_ar_coef(cached_fit(NEUTRAL_PHI).predictions, train_ids), 0.0, 1.0))
    rows.append(
        make_row(
            "invalid_anchor_fit_selector",
            sel_phi,
            "in_sample_anchor_rmse_INVALID",
            "selected",
            cached_fit(sel_phi),
            extra={
                "delta_phi_autocorr_rw": autocorr_rw,
                "phi_crit_curve": ";".join(f"{k:.2f}:{v:.4f}" for k, v in crit_curve.items()),
            },
        )
    )

    perm_info = None
    if do_perm:
        # Column-permutation invariance at the central empirical prior (delta_phi 0.5).
        base = next(
            r for r in rows
            if r["prior_source"] == "empirical_phq_reversion" and abs(r["delta_phi_used"] - PERM_CHECK_PHI) < 1e-9
        )
        perm_info = permutation_check(data, PERM_CHECK_PHI, base["comp_rmse"], seed)
        perm_info["cell"] = cell_name
    return rows, perm_info


# --- Aggregation and provenance ----------------------------------------------

def aggregate(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = []
    for cell in CELL_ORDER:
        for prior_source in PRIOR_SOURCE_ORDER:
            sub = df[(df["cell"] == cell) & (df["prior_source"] == prior_source)]
            if sub.empty:
                continue
            for group_phi in sub["group_phi"].unique():
                g = sub[sub["group_phi"] == group_phi]
                rec = {
                    "cell": cell,
                    "prior_source": prior_source,
                    "group_phi": group_phi,
                    "n_seeds": int(len(g)),
                }
                for key in METRIC_KEYS:
                    vals = g[key].to_numpy(dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if len(vals) == 0:
                        rec[f"{key}_mean"] = float("nan")
                        rec[f"{key}_mcse"] = float("nan")
                    else:
                        rec[f"{key}_mean"] = float(np.mean(vals))
                        rec[f"{key}_mcse"] = (
                            float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                        )
                out.append(rec)
    return pd.DataFrame(out)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def build_manifest(seeds, n, t, do_perm, per_seed_cols, aggregate_cols) -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command_line": " ".join(sys.argv),
        "ball_py_path": str(BALL_PATH),
        "ball_py_sha256": file_sha256(BALL_PATH),
        "ball_py_mtime": datetime.fromtimestamp(BALL_PATH.stat().st_mtime).isoformat(timespec="seconds"),
        "seeds": list(seeds),
        "n_individuals": n,
        "t": t,
        "n_subtypes": N_SUBTYPES,
        "eval_split": EVAL_SPLIT,
        "active_tau": ACTIVE_TAU,
        "ridge_lambda": RIDGE_LAMBDA,
        "permutation_check": bool(do_perm),
        "permutation_tol": PERM_TOL,
        "cells": {name: CELLS[name] for name in CELL_ORDER},
        "comparator": "BALL structural posterior (leveled, train-only learned daily readout)",
        "s0_status": (
            "UNMATCHED: S0 delta prior is a hardcoded random walk in BALL.py and is "
            "not exposed; excluded from the matched comparison (exposing it requires a "
            "BALL.py change, deferred to keep the model file stable)."
        ),
        "phi_grid": list(PHI_GRID),
        "empirical_delta_phi_family": list(EMPIRICAL_DELTA_PHI_FAMILY),
        "prior_sources": {
            "simulation_matched": "diagnostic only; delta_phi set to the DGP true value (uses simulation truth)",
            "empirical_phq_reversion": (
                "manuscript-facing reverting-prior family {0.3, 0.5, 0.7}, justified by the empirical "
                "PHQ-9 fast-residual persistence bracket (validation/empirical_phq_reversion.py)"
            ),
            "invalid_anchor_fit_selector": (
                "NEGATIVE CONTROL, not deployable; selects delta_phi over PHI_GRID by minimum held-out "
                "validation-anchor RMSE, which is in-sample because each person's anchors are used in their "
                "own fit, so it rewards the flexible random walk and collapses the decomposition. "
                "delta_phi_autocorr_rw is a further diagnostic (lag-1 AR(1) OLS on the delta_phi=1.0 "
                "delta_hat, inflated by slow leakage)."
            ),
        },
        "oracle_floor": (
            "ridge(true composite L ~ daily X0..X{p-1}); train-fit, test RMSE; "
            "train-mean imputed; lambda=1.0; unpenalized intercept"
        ),
        "metric_keys": METRIC_KEYS,
        "per_seed_columns": list(per_seed_cols),
        "aggregate_columns": list(aggregate_cols),
    }


# --- Reporting ----------------------------------------------------------------

def print_summary(agg: pd.DataFrame) -> None:
    for prior_source in PRIOR_SOURCE_ORDER:
        sub = agg[agg["prior_source"] == prior_source]
        if sub.empty:
            continue
        print(f"\n=== structural decomposition | prior_source = {prior_source} (mean +- MCSE) ===")
        header = (
            f"{'cell':<16}{'group_phi':>10}{'phi_used':>14}{'slow_corr':>16}{'s2f_leak':>16}"
            f"{'rho_rdoc':>16}{'comp_rmse':>16}{'floor':>10}"
        )
        print(header)
        for cell in CELL_ORDER:
            cell_rows = sub[sub["cell"] == cell]
            for _, r in cell_rows.iterrows():

                def cell_str(key):
                    return f"{r[f'{key}_mean']:.3f}+-{r[f'{key}_mcse']:.3f}"

                print(
                    f"{cell:<16}{str(r['group_phi']):>10}{cell_str('delta_phi_used'):>14}"
                    f"{cell_str('slow_recovery_corr'):>16}{cell_str('slow_to_fast_leakage'):>16}"
                    f"{cell_str('rho_rdoc'):>16}{cell_str('comp_rmse'):>16}"
                    f"{r['oracle_floor_rmse_mean']:>10.3f}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="BALL DGP validation harness (step A).")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="individuals per cell")
    parser.add_argument("--t", type=int, default=DEFAULT_T, help="days per individual")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--cells", type=str, nargs="+", default=CELL_ORDER, choices=CELL_ORDER)
    parser.add_argument("--out", type=str, default=None, help="output directory")
    parser.add_argument("--no-perm", action="store_true", help="skip the column-permutation check")
    parser.add_argument(
        "--quick", action="store_true", help=f"fast smoke: n={QUICK_N}, seeds={QUICK_SEEDS}"
    )
    args = parser.parse_args()

    n = QUICK_N if args.quick else args.n
    seeds = QUICK_SEEDS if args.quick else args.seeds
    cells = [c for c in CELL_ORDER if c in set(args.cells)]
    do_perm = not args.no_perm

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else (REPO_ROOT / "validation" / "outputs" / f"run_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"BALL validation harness: n={n}, seeds={seeds}, cells={cells}, perm={do_perm}")
    print(f"output -> {out_dir}")

    per_seed_rows: list[dict] = []
    perm_rows: list[dict] = []
    for cell in cells:
        for seed in seeds:
            # Run the permutation check once per cell, on the first seed, to keep cost down.
            cell_perm = do_perm and seed == seeds[0]
            rows, perm_info = run_cell_seed(cell, seed, n, args.t, cell_perm)
            per_seed_rows.extend(rows)
            if perm_info is not None:
                perm_rows.append(perm_info)
            done = rows[0]  # simulation_matched (diagnostic) row
            print(
                f"  [{cell:<14} seed={seed}] sim_matched slow={done['slow_recovery_corr']:.3f} "
                f"rho={done['rho_rdoc']:.3f} comp={done['comp_rmse']:.3f} floor={done['oracle_floor_rmse']:.3f}"
            )

    per_seed_df = pd.DataFrame(per_seed_rows)
    agg_df = aggregate(per_seed_rows)
    perm_df = pd.DataFrame(perm_rows)

    per_seed_path = out_dir / "per_seed_metrics.csv"
    agg_path = out_dir / "aggregate_mcse.csv"
    perm_path = out_dir / "permutation_check.csv"
    manifest_path = out_dir / "manifest.json"

    per_seed_df.to_csv(per_seed_path, index=False)
    agg_df.to_csv(agg_path, index=False)
    if not perm_df.empty:
        perm_df.to_csv(perm_path, index=False)
    manifest = build_manifest(
        seeds, n, args.t, do_perm, per_seed_df.columns, agg_df.columns
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print_summary(agg_df)
    if not perm_df.empty:
        n_ok = int(perm_df["invariant"].sum())
        print(
            f"\ncolumn-permutation check: {n_ok}/{len(perm_df)} cells invariant "
            f"(max abs comp_RMSE diff {perm_df['abs_diff'].max():.2e})"
        )
    print(f"\nwrote:\n  {per_seed_path}\n  {agg_path}\n  {perm_path}\n  {manifest_path}")


if __name__ == "__main__":
    main()
