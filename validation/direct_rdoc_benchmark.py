#!/usr/bin/env python
"""Consolidated direct-RDoC benchmark: BALL transformer vs fair comparators.

This runner is the mirror-image comparison harness for the direct RDoC drift
estimand. For each generated dataset, it fits:

  - BALL teacher/student transformer: explicit neural transition beta head.
  - S0 direct LGSSM: explicit classical linear-Gaussian transition beta.
  - Markov direct transition: explicit beta plus observed-pattern Markov terms.

All methods see the same observed anchors, daily covariates, treatments, and
observed/noisy RDoC proxy B. Both are scored against the same true beta and
latent trajectory, with truth used only for evaluation.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ball_validation_harness as H  # noqa: E402
from direct_rdoc_common import beta_metric_fields, mcse  # noqa: E402
from direct_rdoc_fair_comparator import (  # noqa: E402
    anchor_constraint,
    fit_daily_readout,
    fit_direct_map,
    init_irt_state,
    irt_holdout_metrics,
    marginal_anchor_sd,
    person_arrays,
    transition_features,
    update_irt_zbar,
)

_ssm = sys.modules["simulations.src.methods.ball_ssm"]
SSMConfig = _ssm.SSMConfig
fit_ball_ssm = _ssm.fit_ball_ssm
_anchors = sys.modules["simulations.src.anchors"]
generate_anchors = _anchors.generate_anchors

DEFAULT_CELLS = ("linear", "interaction", "nonlinear", "heterogeneous", "missingness")

METRICS = [
    "beta_cosine",
    "beta_abs_cosine",
    "beta_topk_f1",
    "beta_hat_norm",
    "beta_hat_norm_trait",
    "latent_rmse",
    "latent_rmse_trait",
    "val_anchor_rmse",
    "irt_item_nll",
    "irt_expected_total_rmse",
    "elapsed_seconds",
]

PRIMARY_BALL_METHOD = "ball_student_causal"

COMPARATOR_LABELS = {
    "s0_direct_lgssm": "s0",
    "markov_direct_transition": "markov",
}


IRT_CALIBRATION_PATH = Path(__file__).resolve().parent / "irt_calibration.json"
_IRT_CALIBRATION_CACHE = {}


def load_irt_calibration() -> dict:
    """Load the frozen calibrated-instrument spec (validation/irt_calibration.json)."""
    if "spec" not in _IRT_CALIBRATION_CACHE:
        if not IRT_CALIBRATION_PATH.exists():
            raise FileNotFoundError(
                f"IRT calibration not found at {IRT_CALIBRATION_PATH}; run build_irt_calibration.py first")
        _IRT_CALIBRATION_CACHE["spec"] = json.loads(IRT_CALIBRATION_PATH.read_text(encoding="utf-8"))
    return _IRT_CALIBRATION_CACHE["spec"]


def _apply_irt_calibration(cfg):
    """Set the FROZEN calibrated-instrument constants on the config (identical everywhere)."""
    cal = load_irt_calibration()
    return dataclasses.replace(
        cfg,
        irt_n_items=int(cal["n_items"]),
        irt_n_categories=int(cal["n_categories"]),
        irt_loc=float(cal["affine"]["loc"]),
        irt_scale=float(cal["affine"]["scale"]),
        irt_item_thresholds=tuple(tuple(float(x) for x in b) for b in cal["thresholds"]),
        irt_item_discriminations=tuple(float(a) for a in cal["discriminations"]),
    )


def irt_calibration_provenance(args) -> dict | None:
    """Calibration JSON sha256 and full spec for the manifest/metadata, or None for Gaussian."""
    if str(getattr(args, "anchor_observation", "gaussian")).lower() != "irt":
        return None
    import hashlib
    raw = IRT_CALIBRATION_PATH.read_bytes()
    return {
        "path": IRT_CALIBRATION_PATH.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "spec": load_irt_calibration(),
    }


def _base_config(args, seed: int, share: float, cell: str):
    native_share = float(share) if cell in {"linear", "missingness"} else 0.0
    cfg = dataclasses.replace(
        H.SimulationConfig(),
        seed=seed,
        n=args.n,
        t=args.t,
        slow_fraction=args.slow_fraction,
        delta_ar=tuple([float(args.delta_ar)] * H.N_SUBTYPES),
        rdoc_drift_share=native_share,
        rdoc_drift_active_dims=int(args.rdoc_active_dims),
        rdoc_drift_beta_seed=int(args.rdoc_beta_seed),
        rdoc_drift_min_abs=float(args.rdoc_min_abs),
        anchor_observation=str(getattr(args, "anchor_observation", "gaussian")),
        irt_n_items=int(getattr(args, "irt_n_items", 9)),
        irt_discrimination=float(getattr(args, "irt_discrimination", 1.5)),
    )
    if cell == "missingness":
        cfg = dataclasses.replace(
            cfg,
            daily_base_missing_probability=max(float(cfg.daily_base_missing_probability), args.missing_daily_probability),
            mnar_gamma_l=max(float(cfg.mnar_gamma_l), args.missing_mnar_gamma),
            proxy_measurement_error_sd=max(float(cfg.proxy_measurement_error_sd), args.missing_proxy_noise),
            note_density_error_multiplier=max(float(cfg.note_density_error_multiplier), args.missing_density_multiplier),
        )
    if str(getattr(args, "anchor_observation", "gaussian")).lower() == "irt":
        cfg = _apply_irt_calibration(cfg)
    return cfg


def _cell_raw_signal(comp_i: pd.DataFrame, beta: np.ndarray, cell: str, args) -> np.ndarray:
    c = comp_i[[f"C{j}" for j in range(len(beta))]].to_numpy(dtype=float)
    proj = c @ beta
    proj = proj - float(np.mean(proj))
    if cell == "interaction":
        recent = comp_i["recent_treatment"].to_numpy(dtype=float)
        return proj * (1.0 + float(args.interaction_strength) * recent)
    if cell == "nonlinear":
        tanh_proj = np.tanh(c) @ beta
        tanh_proj = tanh_proj - float(np.mean(tanh_proj))
        return proj + float(args.nonlinear_strength) * tanh_proj
    if cell == "heterogeneous":
        subtype = int(comp_i["subtype"].iloc[0]) if "subtype" in comp_i.columns and len(comp_i) else 0
        midpoint = max((args.n_subtypes_for_scale - 1) / 2.0, 1.0)
        scale = 1.0 + float(args.heterogeneity_strength) * ((subtype - midpoint) / midpoint)
        return scale * proj
    return proj


def _scaled_signal_level(raw: np.ndarray, base_latent: np.ndarray, share: float) -> tuple[np.ndarray, np.ndarray, float]:
    raw = np.asarray(raw, dtype=float)
    raw = raw - float(np.mean(raw))
    raw_sd = float(np.std(raw))
    base_var = float(np.var(np.diff(base_latent)))
    if raw_sd <= 1e-8 or base_var <= 1e-12 or float(share) <= 0.0:
        return np.zeros_like(raw), np.zeros_like(raw), 0.0
    share = max(0.0, min(float(share), 0.95))
    signal_sd = np.sqrt(base_var * share / max(1.0 - share, 1e-8))
    signal = raw / raw_sd * signal_sd
    level = np.zeros_like(signal)
    if len(signal) > 1:
        level[1:] = np.cumsum(signal[:-1])
    return signal, level, signal_sd / raw_sd


def _inject_benchmark_cell(data, cell: str, share: float, args):
    if cell in {"linear", "missingness"}:
        data.metadata["direct_rdoc_cell"] = cell
        data.metadata["direct_rdoc_cell_note"] = "Native linear direct-RDoC DGP path."
        return data

    comp = data.components.copy()
    daily = data.daily.copy()
    beta = np.asarray(data.metadata.get("rdoc_drift_beta_unit", []), dtype=float)
    scales = []
    level_frames = []
    for pid, comp_i in comp.groupby("id", sort=False):
        comp_i = comp_i.sort_values("t")
        idx = comp_i.index
        raw = _cell_raw_signal(comp_i, beta, cell, args)
        base_latent = comp.loc[idx, "L"].to_numpy(dtype=float)
        signal, level, scale = _scaled_signal_level(raw, base_latent, share)
        scales.append(float(scale))
        comp.loc[idx, "rdoc_direct_signal"] = signal
        comp.loc[idx, "rdoc_direct_level"] = level
        for col in ["z_d", "z_p", "S_joint", "L", "delta", "delta_d", "delta_p"]:
            if col in comp.columns:
                comp.loc[idx, col] = comp.loc[idx, col].to_numpy(dtype=float) + level
        level_frames.append(pd.DataFrame({"id": int(pid), "t": comp_i["t"].to_numpy(dtype=int), "_rdoc_level": level}))

    levels = pd.concat(level_frames, ignore_index=True) if level_frames else pd.DataFrame(columns=["id", "t", "_rdoc_level"])
    daily = daily.merge(levels, on=["id", "t"], how="left")
    daily["_rdoc_level"] = daily["_rdoc_level"].fillna(0.0)
    for pid, d_i in daily.groupby("id", sort=False):
        order = d_i.sort_values("t").index
        level = daily.loc[order, "_rdoc_level"].to_numpy(dtype=float)
        roll_level = pd.Series(level).rolling(7, min_periods=1).mean().to_numpy()
        for j in range(4):
            col = f"X{j}"
            if col in daily.columns:
                daily.loc[order, col] = daily.loc[order, col].to_numpy(dtype=float) + level
            lag_col = f"X{4 + j}"
            if lag_col in daily.columns:
                daily.loc[order, lag_col] = daily.loc[order, lag_col].to_numpy(dtype=float) + roll_level
    daily = daily.drop(columns=["_rdoc_level"])

    train_ids = set(data.individuals.loc[data.individuals["split"] == "train", "id"].astype(int))
    anchors = generate_anchors(comp, data.config, train_ids=train_ids)
    metadata = dict(data.metadata)
    metadata.update(
        {
            "direct_rdoc_cell": cell,
            "rdoc_drift_share": float(share),
            "rdoc_drift_scale_mean": float(np.mean(scales)) if scales else 0.0,
            "direct_rdoc_cell_note": (
                "Benchmark-local direct-RDoC cell injection. The sparse beta direction is unchanged; "
                "cell-specific drift modifies the one-step RDoC signal before integration into latent level."
            ),
        }
    )
    return dataclasses.replace(data, components=comp, daily=daily, anchors=anchors, metadata=metadata)


def make_data(args, seed: int, share: float, cell: str):
    cfg = _base_config(args, seed, share, cell)
    data = H.generate_dataset(cfg)
    data = _inject_benchmark_cell(data, cell, share, args)
    prov = irt_calibration_provenance(args)
    if prov is not None:
        data.metadata["irt_calibration_sha256"] = prov["sha256"]
        data.metadata["irt_calibration"] = prov["spec"]
    return data


def ball_config(args, seed: int) -> "SSMConfig":
    return SSMConfig(
        seed=seed,
        ensemble_size=args.ensemble_size,
        teacher_epochs=args.teacher_epochs,
        student_epochs=args.student_epochs,
        anchor_warmup_epochs=args.anchor_warmup,
        kl_warmup_epochs=args.kl_warmup,
        batch_size=args.batch_size,
        alpha_lasso_xi=args.alpha_lasso_xi,
        rdoc_drift_l1=args.rdoc_drift_l1,
        rdoc_drift_adaptive=getattr(args, "rdoc_drift_adaptive", False),
        rdoc_drift_adaptive_gamma=getattr(args, "rdoc_drift_adaptive_gamma", 1.0),
        rdoc_drift_adaptive_eps=getattr(args, "rdoc_drift_adaptive_eps", 1e-3),
        delta_phi=args.delta_phi,
        max_individuals=args.n,
        rdoc_drift_head=True,
        use_alpha_slow=False,
        delta_drift_use_ehr=not args.ball_no_ehr_drift,
    )


def score(data, preds: pd.DataFrame, beta_hat: np.ndarray, *, method: str, cell: str, seed: int, share: float, elapsed: float, args) -> dict:
    beta_true = np.asarray(data.metadata.get("rdoc_drift_beta_unit", []), dtype=float)
    active_k = int(data.metadata.get("rdoc_drift_active_dims", args.rdoc_active_dims))
    irt = str(getattr(data.config, "anchor_observation", "gaussian")).lower() == "irt"
    latent_raw = H.latent_rmse(preds, data.components, data.individuals, H.EVAL_SPLIT)
    if irt:
        # Held-out fit goes through the calibrated link; latent RMSE and beta magnitude
        # are also reported on the declared trait scale (trait = (latent - loc) / scale,
        # so beta_trait = beta_raw / scale). The Gaussian val-anchor RMSE is undefined.
        scale = float(getattr(data.config, "irt_scale", 1.0)) or 1.0
        item_nll, etot_rmse = irt_holdout_metrics(data, preds, "validation")
        latent_trait = latent_raw / scale
        beta_norm_trait = float(np.linalg.norm(beta_hat)) / scale
        val_anchor = float("nan")
    else:
        item_nll = etot_rmse = latent_trait = beta_norm_trait = float("nan")
        val_anchor = H.val_anchor_rmse(data, preds)
    return {
        "share": float(share),
        "cell": cell,
        "seed": int(seed),
        "method": method,
        "n": int(args.n),
        "t": int(args.t),
        "latent_rmse": latent_raw,
        "latent_rmse_trait": latent_trait,
        "val_anchor_rmse": val_anchor,
        "irt_item_nll": item_nll,
        "irt_expected_total_rmse": etot_rmse,
        "beta_hat_norm_trait": beta_norm_trait,
        "elapsed_seconds": float(elapsed),
        **beta_metric_fields(beta_hat, beta_true, active_k),
    }


def _matched_s0_basis(cell: str) -> str:
    return {
        "linear": "linear",
        "missingness": "linear",
        "interaction": "interaction",
        "nonlinear": "nonlinear",
        "heterogeneous": "heterogeneous",
    }.get(cell, "linear")


def extract_s0_beta(theta: np.ndarray, q: int, n_actions: int, basis: str, n_subtypes: int) -> np.ndarray:
    """Map matched S0 transition coefficients back to the shared beta direction."""

    beta = np.asarray(theta[1 : 1 + q], dtype=float).copy()
    pos = 1 + q + int(n_actions) + 2
    if basis in {"interaction", "full"}:
        beta = beta + np.asarray(theta[pos : pos + q], dtype=float)
        pos += 2 * q
    if basis in {"nonlinear", "full"}:
        # The nonlinear benchmark cell uses beta both on B and tanh(B). The B^2
        # block is retained as a nuisance flexibility term and not counted as a
        # signed beta direction.
        pos += q
        beta = beta + np.asarray(theta[pos : pos + q], dtype=float)
        pos += q
    if basis in {"heterogeneous", "full"}:
        subtype_blocks = []
        for _ in range(int(n_subtypes)):
            subtype_blocks.append(np.asarray(theta[pos : pos + q], dtype=float))
            pos += q
        if subtype_blocks:
            beta = beta + np.mean(np.vstack(subtype_blocks), axis=0)
    return beta


def _anchor_pattern_strata(data, n_strata: int) -> dict[int, int]:
    """Observed-data pattern strata for the Markov comparator."""

    n_strata = max(1, int(n_strata))
    ids = sorted(data.individuals["id"].astype(int).tolist())
    observed = data.anchors.loc[data.anchors["observed"].astype(bool)]
    counts = observed.groupby("id").size()
    vals = np.asarray([float(counts.get(pid, 0.0)) for pid in ids], dtype=float)
    if n_strata == 1 or len(np.unique(vals)) <= 1:
        return {pid: 0 for pid in ids}
    edges = np.quantile(vals, np.linspace(0.0, 1.0, n_strata + 1)[1:-1])
    return {pid: int(min(n_strata - 1, max(0, np.searchsorted(edges, vals[i], side="right")))) for i, pid in enumerate(ids)}


def _markov_design(base_feats: np.ndarray, l_hat: np.ndarray, stratum: int, n_strata: int) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(base_feats), max(len(l_hat) - 1, 0))
    if n <= 0:
        return np.zeros((0, base_feats.shape[1] + 2 * n_strata), dtype=float), np.zeros(0, dtype=float)
    base = base_feats[:n]
    lag = np.asarray(l_hat[:n], dtype=float)
    onehot = np.zeros((n, n_strata), dtype=float)
    onehot[:, int(stratum)] = 1.0
    x = np.column_stack([base, onehot, onehot * lag[:, None]])
    y = np.diff(np.asarray(l_hat[: n + 1], dtype=float))
    return x, y


def solve_markov_person(
    data,
    pid: int,
    theta: np.ndarray,
    daily_readout,
    args,
    *,
    basis: str,
    stratum: int,
    n_strata: int,
    irt_state=None,
):
    comp_i, anchors_i, b, daily_pred, any_daily, daily_sd, action, recent, burden = person_arrays(data, pid, daily_readout)
    t_count = len(comp_i)
    if irt_state is not None:
        irt_state["pid"] = pid
    subtype = int(comp_i["subtype"].iloc[0]) if "subtype" in comp_i.columns and len(comp_i) else 0
    feats = transition_features(
        b,
        action,
        recent,
        burden,
        data.config.n_treatment_types,
        basis=basis,
        subtype=subtype,
        n_subtypes=data.config.n_subtypes,
    )
    n_base = feats.shape[1]
    base_theta = theta[:n_base]
    stratum_offsets = theta[n_base : n_base + n_strata]
    phis = np.clip(theta[n_base + n_strata : n_base + 2 * n_strata], -0.95, 0.95)
    phi = float(phis[int(stratum)]) if len(phis) else 0.0
    trans_mean = feats @ base_theta
    if len(stratum_offsets):
        trans_mean = trans_mean + float(stratum_offsets[int(stratum)])

    rows, targets = [], []

    def add(coeffs: list[tuple[int, float]], target: float, sd: float):
        row = np.zeros(t_count, dtype=float)
        for idx, value in coeffs:
            row[idx] = value / max(sd, 1e-8)
        rows.append(row)
        targets.append(float(target) / max(sd, 1e-8))

    add([(0, 1.0)], 0.0, args.prior_sd)
    for row in anchors_i.itertuples(index=False):
        constraint = anchor_constraint(row, t_count, data, args, irt_state)
        if constraint is not None:
            add(*constraint)

    for k, observed in enumerate(any_daily):
        if observed and np.isfinite(daily_pred[k]):
            add([(k, 1.0)], float(daily_pred[k]), daily_sd)

    for k in range(1, t_count):
        add([(k, 1.0), (k - 1, -(1.0 + phi))], float(trans_mean[k - 1]), args.transition_sd)

    design = np.vstack(rows)
    target = np.asarray(targets, dtype=float)
    sol = np.linalg.lstsq(design, target, rcond=None)[0]
    update_irt_zbar(irt_state, anchors_i, sol, t_count)
    pred = pd.DataFrame(
        {
            "id": comp_i["id"].to_numpy(dtype=int),
            "t": comp_i["t"].to_numpy(dtype=int),
            "L_hat": sol,
            "z_d_hat": sol,
            "z_p_hat": sol,
        }
    )
    return pred


def fit_markov_direct(data, args, *, basis: str) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Fit a direct-RDoC Markov/pattern comparator with native beta recovery."""

    train_ids = set(data.individuals.loc[data.individuals["split"] == "train", "id"].astype(int))
    ids = sorted(data.individuals["id"].astype(int).tolist())
    daily_readout = fit_daily_readout(data, train_ids, ridge=args.daily_ridge)
    init_args = argparse.Namespace(**vars(args))
    init_args.s0_basis = basis
    init_preds, base_theta = fit_direct_map(data, init_args)
    n_base = int(len(base_theta))
    n_strata = max(1, int(args.markov_strata))
    strata = _anchor_pattern_strata(data, n_strata)
    theta = np.zeros(n_base + 2 * n_strata, dtype=float)
    theta[:n_base] = base_theta
    pred_by_pid = {int(pid): frame.sort_values("t").reset_index(drop=True) for pid, frame in init_preds.groupby("id", sort=True)}
    irt_state = init_irt_state(data, args)

    for _ in range(max(1, int(args.markov_iters))):
        x_rows, y_rows = [], []
        for pid in ids:
            if pid not in train_ids or pid not in pred_by_pid:
                continue
            comp_i, _, b, _, _, _, action, recent, burden = person_arrays(data, pid, daily_readout)
            subtype = int(comp_i["subtype"].iloc[0]) if "subtype" in comp_i.columns and len(comp_i) else 0
            base_feats = transition_features(
                b,
                action,
                recent,
                burden,
                data.config.n_treatment_types,
                basis=basis,
                subtype=subtype,
                n_subtypes=data.config.n_subtypes,
            )
            l_hat = pred_by_pid[pid]["L_hat"].to_numpy(dtype=float)
            x, y = _markov_design(base_feats, l_hat, strata[pid], n_strata)
            if len(y):
                x_rows.append(x)
                y_rows.append(y)
        if not x_rows:
            break
        x = np.vstack(x_rows)
        y = np.concatenate(y_rows)
        penalty = float(args.markov_ridge) * np.eye(theta.size)
        penalty[0, 0] = 0.0
        theta = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        theta[n_base + n_strata :] = np.clip(theta[n_base + n_strata :], -0.95, 0.95)
        pred_by_pid = {
            pid: solve_markov_person(
                data,
                pid,
                theta,
                daily_readout,
                args,
                basis=basis,
                stratum=strata[pid],
                n_strata=n_strata,
                irt_state=irt_state,
            )
            for pid in ids
        }

    preds = pd.concat([pred_by_pid[pid] for pid in ids], ignore_index=True)
    return preds, theta, n_base


def run_s0(data, args, cell: str, seed: int, share: float) -> dict:
    s0_args = argparse.Namespace(**vars(args))
    s0_basis = _matched_s0_basis(cell) if args.s0_basis == "matched" else args.s0_basis
    s0_args.s0_basis = s0_basis
    start = perf_counter()
    preds, theta = fit_direct_map(data, s0_args)
    elapsed = perf_counter() - start
    q = data.config.q
    beta_hat = extract_s0_beta(theta, q, data.config.n_treatment_types, s0_basis, data.config.n_subtypes)
    return score(data, preds, beta_hat, method="s0_direct_lgssm", cell=cell, seed=seed, share=share, elapsed=elapsed, args=args)


def run_markov(data, args, cell: str, seed: int, share: float) -> dict:
    basis = _matched_s0_basis(cell) if args.s0_basis == "matched" else args.s0_basis
    start = perf_counter()
    preds, theta, n_base = fit_markov_direct(data, args, basis=basis)
    elapsed = perf_counter() - start
    q = data.config.q
    beta_hat = extract_s0_beta(theta[:n_base], q, data.config.n_treatment_types, basis, data.config.n_subtypes)
    return score(
        data,
        preds,
        beta_hat,
        method="markov_direct_transition",
        cell=cell,
        seed=seed,
        share=share,
        elapsed=elapsed,
        args=args,
    )


def _score_ball_result(data, result, *, method: str, cell: str, seed: int, share: float, elapsed: float, args) -> dict:
    beta_true = np.asarray(data.metadata.get("rdoc_drift_beta_unit", []), dtype=float)
    beta_hat = np.asarray(result.metadata.get("rdoc_drift_beta_hat", []), dtype=float)
    if beta_hat.size != beta_true.size:
        beta_hat = np.zeros_like(beta_true)
    return score(
        data,
        result.predictions,
        beta_hat,
        method=method,
        cell=cell,
        seed=seed,
        share=share,
        elapsed=elapsed,
        args=args,
    )


def _adaptive_drift_weights(beta_pilot: np.ndarray, gamma: float, eps: float) -> np.ndarray:
    """Fixed adaptive-Lasso weights w_j = 1/(|beta_pilot_j|+eps)^gamma (Zou 2006),
    normalized to mean 1 so the L1 rate keeps the same overall penalty scale."""
    beta = np.abs(np.asarray(beta_pilot, dtype=float))
    w = 1.0 / np.power(beta + float(eps), float(gamma))
    m = float(np.mean(w))
    return w / m if m > 1e-12 else np.ones_like(w)


def run_ball(data, args, cell: str, seed: int, share: float) -> list[dict]:
    start = perf_counter()
    cfg = ball_config(args, seed)
    beta_pilot = None
    weights = None
    if getattr(args, "rdoc_drift_adaptive", False):
        # Stage 1: unpenalized pilot fit to obtain beta_hat for the FIXED weights.
        pilot_cfg = dataclasses.replace(cfg, rdoc_drift_l1=0.0, rdoc_drift_adaptive=False, rdoc_drift_weights=())
        # The drift beta belongs to the shared generative model and is not updated
        # during student distillation, so a teacher-only pilot is identical for the
        # adaptive weights and avoids an unnecessary full student training pass.
        pilot = fit_ball_ssm(
            data, pilot_cfg, device=args.device, causal=False, prediction_split=None,
        )
        beta_pilot = np.asarray(pilot.metadata.get("rdoc_drift_beta_hat", []), dtype=float)
        weights = _adaptive_drift_weights(
            beta_pilot, args.rdoc_drift_adaptive_gamma, getattr(args, "rdoc_drift_adaptive_eps", 1e-3),
        )
        # Stage 2: refit with the L1 penalty reweighted by the frozen pilot weights.
        cfg = dataclasses.replace(cfg, rdoc_drift_weights=tuple(float(x) for x in weights))
    student, teacher = fit_ball_ssm(
        data,
        cfg,
        device=args.device,
        causal=True,
        prediction_split=None,
        return_teacher=True,
    )
    elapsed = perf_counter() - start
    scored = [
        _score_ball_result(
            data,
            teacher,
            method="ball_teacher_smoother",
            cell=cell,
            seed=seed,
            share=share,
            elapsed=elapsed,
            args=args,
        ),
        _score_ball_result(
            data,
            student,
            method="ball_student_causal",
            cell=cell,
            seed=seed,
            share=share,
            elapsed=elapsed,
            args=args,
        ),
    ]
    pilot_json = json.dumps(beta_pilot.tolist()) if beta_pilot is not None else None
    weights_json = json.dumps(weights.tolist()) if weights is not None else None
    for row in scored:
        row["adaptive_pilot_beta"] = pilot_json
        row["adaptive_weights"] = weights_json
    return scored


def aggregate(per: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in per.groupby(["cell", "share", "method"], sort=True):
        row = {"cell": keys[0], "share": float(keys[1]), "method": keys[2], "n_seeds": int(group["seed"].nunique())}
        for metric in METRICS:
            vals = group[metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{metric}_mcse"] = mcse(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_deltas(per: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cell, share, seed), group in per.groupby(["cell", "share", "seed"], sort=True):
        by_method = {row.method: row for row in group.itertuples(index=False)}
        if PRIMARY_BALL_METHOD not in by_method:
            continue
        ball = by_method[PRIMARY_BALL_METHOD]
        row = {"cell": cell, "share": float(share), "seed": int(seed)}
        any_comparator = False
        for method, label in COMPARATOR_LABELS.items():
            if method not in by_method:
                continue
            any_comparator = True
            comparator = by_method[method]
            for metric in METRICS:
                row[f"ball_minus_{label}_{metric}"] = float(getattr(ball, metric) - getattr(comparator, metric))
        if not any_comparator:
            continue
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_deltas(delta: pd.DataFrame) -> pd.DataFrame:
    if delta.empty:
        return delta
    rows = []
    for (cell, share), group in delta.groupby(["cell", "share"], sort=True):
        row = {"cell": cell, "share": float(share), "n_seeds": int(group["seed"].nunique())}
        for col in [c for c in delta.columns if c.startswith("ball_minus_")]:
            vals = group[col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{col}_mcse"] = mcse(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct RDoC benchmark: BALL transformer vs fair direct comparators.")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--t", type=int, default=84)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1729, 2027, 2028])
    parser.add_argument("--cells", type=str, nargs="+", default=["linear"], choices=list(DEFAULT_CELLS))
    parser.add_argument("--shares", type=float, nargs="+", default=[0.10, 0.25])
    parser.add_argument("--slow-fraction", type=float, default=0.75)
    parser.add_argument("--delta-ar", type=float, default=0.3)
    parser.add_argument("--rdoc-active-dims", type=int, default=3)
    parser.add_argument("--rdoc-beta-seed", type=int, default=31011)
    parser.add_argument("--rdoc-min-abs", type=float, default=0.35)
    parser.add_argument("--interaction-strength", type=float, default=1.5)
    parser.add_argument("--nonlinear-strength", type=float, default=0.75)
    parser.add_argument("--heterogeneity-strength", type=float, default=0.50)
    parser.add_argument("--n-subtypes-for-scale", type=int, default=3)
    parser.add_argument("--missing-daily-probability", type=float, default=0.55)
    parser.add_argument("--missing-mnar-gamma", type=float, default=0.7)
    parser.add_argument("--missing-proxy-noise", type=float, default=0.40)
    parser.add_argument("--missing-density-multiplier", type=float, default=1.5)

    parser.add_argument("--iters", type=int, default=5, help="S0 MAP coordinate-update iterations")
    parser.add_argument("--transition-sd", type=float, default=0.75)
    parser.add_argument("--prior-sd", type=float, default=5.0)
    parser.add_argument("--daily-ridge", type=float, default=1.0)
    parser.add_argument("--beta-ridge", type=float, default=10.0)
    parser.add_argument(
        "--s0-basis",
        choices=["matched", "linear", "interaction", "nonlinear", "heterogeneous", "full"],
        # Fair mirror-image default: the classical comparators use a FIXED linear B*beta
        # transition in every cell, so they face the same functional mis-specification the
        # transformer does in the interaction/nonlinear/heterogeneous cells (the transformer
        # is never told the cell form either). "matched" is retained only as a best-case
        # classical sensitivity, not the default.
        default="linear",
    )
    parser.add_argument("--skip-markov", action="store_true", help="Skip the native direct-transition Markov comparator")
    parser.add_argument("--markov-strata", type=int, default=3)
    parser.add_argument("--markov-iters", type=int, default=4)
    parser.add_argument("--markov-ridge", type=float, default=10.0)

    parser.add_argument("--teacher-epochs", type=int, default=240)
    parser.add_argument("--student-epochs", type=int, default=240)
    parser.add_argument("--anchor-warmup", type=int, default=60)
    parser.add_argument("--kl-warmup", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ensemble-size", type=int, default=5)  # deep ensemble K (spec Sec. 5)
    parser.add_argument("--alpha-lasso-xi", type=float, default=2.0)
    parser.add_argument("--rdoc-drift-l1", type=float, default=0.0)
    parser.add_argument("--rdoc-drift-adaptive", action="store_true",
                        help="adaptive-Lasso (Zou 2006): fixed pilot-derived per-coordinate weights on beta; requires --rdoc-drift-l1 > 0")
    parser.add_argument("--rdoc-drift-adaptive-gamma", type=float, default=1.0)
    parser.add_argument("--rdoc-drift-adaptive-eps", type=float, default=1e-3)
    parser.add_argument("--anchor-observation", choices=["gaussian", "irt"], default="gaussian",
                        help="irt: frozen calibrated graded-response instrument shared by BALL, S0, and Markov")
    parser.add_argument("--irt-n-items", type=int, default=9)
    parser.add_argument("--irt-discrimination", type=float, default=1.5)
    parser.add_argument("--delta-phi", type=float, default=0.3)
    parser.add_argument("--ball-no-ehr-drift", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="resume an interrupted run from per_run.csv in --out")
    args = parser.parse_args()
    if args.rdoc_drift_adaptive and args.rdoc_drift_l1 <= 0.0:
        parser.error("--rdoc-drift-adaptive requires --rdoc-drift-l1 > 0 (adaptive weights only act through the L1 penalty)")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = Path(args.out) if args.out else H.REPO_ROOT / "validation" / "outputs" / f"direct_rdoc_benchmark_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = out_dir / "per_run.csv"
    rows = []
    completed: set[tuple[str, float, int]] = set()
    if args.resume and checkpoint_path.exists():
        prior = pd.read_csv(checkpoint_path)
        rows = prior.to_dict("records")
        completed = {
            (str(row.cell), float(row.share), int(row.seed))
            for row in prior[["cell", "share", "seed"]].drop_duplicates().itertuples(index=False)
        }
        print(f"resuming {len(completed)} completed cell/share/seed combinations from {checkpoint_path}")
    for cell in args.cells:
        for share in args.shares:
            for seed in args.seeds:
                key = (str(cell), float(share), int(seed))
                if key in completed:
                    print(f"cell={cell} share={share:.2f} seed={seed} already complete; skipping")
                    continue
                print(f"cell={cell} share={share:.2f} seed={seed} building data")
                data = make_data(args, seed, share, cell)
                print("  s0_direct_lgssm")
                rows.append(run_s0(data, args, cell, seed, share))
                if not args.skip_markov:
                    print("  markov_direct_transition")
                    rows.append(run_markov(data, args, cell, seed, share))
                print("  ball_teacher_student")
                rows.extend(run_ball(data, args, cell, seed, share))
                pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

    per = pd.DataFrame(rows)
    agg = aggregate(per)
    delta = paired_deltas(per)
    delta_agg = aggregate_deltas(delta)

    per.to_csv(out_dir / "per_run.csv", index=False)
    agg.to_csv(out_dir / "aggregate.csv", index=False)
    delta.to_csv(out_dir / "paired_deltas.csv", index=False)
    delta_agg.to_csv(out_dir / "paired_delta_aggregate.csv", index=False)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "command_line": " ".join(sys.argv),
                "ball_py_sha256": H.file_sha256(H.BALL_PATH),
                "args": vars(args),
                "irt_calibration": irt_calibration_provenance(args),
                "methods": {
                    "ball_teacher_smoother": "BALL bidirectional transformer teacher smoother with rdoc_drift_head=True and use_alpha_slow=False",
                    "ball_student_causal": "BALL causal transformer student distilled from the teacher, with rdoc_drift_head=True and use_alpha_slow=False",
                    "s0_direct_lgssm": "Classical direct linear-Gaussian MAP smoother with matched direct-RDoC transition basis",
                    "markov_direct_transition": "Classical direct Markov/pattern smoother with matched direct-RDoC transition beta and observed-pattern Markov terms",
                },
                "note": "Mirror-image direct-RDoC benchmark; all methods estimate explicit drift beta on the same generated datasets. Nonlinear/interaction/heterogeneity cells retain the same sparse beta direction and add matched comparator basis terms.",
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    irt = str(getattr(args, "anchor_observation", "gaussian")).lower() == "irt"
    # Under IRT the held-out fit is item NLL + expected-total RMSE through the calibrated
    # link, and latent/beta magnitude are on the declared trait scale; the Gaussian
    # val-anchor RMSE is undefined and would print as NaN.
    if irt:
        head_metrics = ["beta_cosine_mean", "beta_topk_f1_mean", "beta_hat_norm_trait_mean",
                        "latent_rmse_trait_mean", "irt_item_nll_mean", "irt_expected_total_rmse_mean"]
        delta_metrics = ["beta_cosine", "beta_topk_f1", "latent_rmse_trait", "irt_item_nll", "irt_expected_total_rmse"]
    else:
        head_metrics = ["beta_cosine_mean", "beta_topk_f1_mean", "latent_rmse_mean", "val_anchor_rmse_mean"]
        delta_metrics = ["beta_cosine", "beta_topk_f1", "latent_rmse", "val_anchor_rmse"]

    print(f"\noutput -> {out_dir}")
    print("\naggregate:")
    cols = ["cell", "share", "method", "n_seeds"] + head_metrics + ["elapsed_seconds_mean"]
    print(agg[[c for c in cols if c in agg.columns]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if not delta_agg.empty:
        print("\npaired BALL - comparator deltas:")
        dcols = ["cell", "share", "n_seeds"]
        for label in COMPARATOR_LABELS.values():
            for metric in delta_metrics:
                col = f"ball_minus_{label}_{metric}_mean"
                if col in delta_agg.columns:
                    dcols.append(col)
        print(delta_agg[dcols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
