#!/usr/bin/env python
"""Consolidated direct-RDoC benchmark: BALL transformer vs fair comparators.

This runner is the mirror-image comparison harness for the direct RDoC drift
estimand. For each generated dataset, it fits:

  - BALL teacher/student transformer: explicit neural transition beta head.
  - Direct causal transformer: the student architecture trained directly on anchors.
  - Direct LGSSM: explicit classical linear-Gaussian transition beta.
  - Markov direct transition: explicit beta plus observed-pattern Markov terms.
  - Causal Gaussian-process estimator with two time scales.
  - Causal exponential-decay GRU: learned decay over elapsed time.

All methods see the same observed anchors, daily covariates, treatments, and
observed/noisy RDoC proxy B. Every trajectory estimator is scored against the
same latent truth. Methods with an explicit coefficient are also scored against
the same true beta, with truth used only for evaluation.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import platform
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn.functional as F
from scipy.linalg import cho_factor, cho_solve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ball_validation_harness as H  # noqa: E402
from direct_rdoc_common import beta_metric_fields, mcse  # noqa: E402
from direct_rdoc_fair_comparator import (  # noqa: E402
    anchor_constraint,
    causal_encoder_action_array,
    fit_daily_readout,
    fit_direct_map,
    fit_two_channel_direct_map,
    init_irt_state,
    irt_anchor_pseudo_observation,
    irt_holdout_metrics,
    marginal_anchor_sd,
    observed_treatment_history_arrays,
    person_arrays,
    transition_features,
    update_irt_zbar,
)

_ssm = sys.modules["simulations.src.methods.ball_ssm"]
SSMConfig = _ssm.SSMConfig
fit_ball_ssm = _ssm.fit_ball_ssm
fit_ball_ssm_direct_causal = _ssm.fit_ball_ssm_direct_causal
fit_ball_ssm_distillation_decomposition = _ssm.fit_ball_ssm_distillation_decomposition
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
    "depression_latent_rmse",
    "anxiety_latent_rmse",
    "latent_rmse_trait",
    "val_anchor_rmse",
    "y1_measurement_rmse",
    "y2_measurement_rmse",
    "irt_item_nll",
    "irt_expected_total_rmse",
    "elapsed_seconds",
]

PRIMARY_BALL_METHOD = "ball_student_causal"

REQUIRED_METHODS = {
    "s0_direct_lgssm",
    "markov_direct_transition",
    "ball_teacher_smoother",
    "ball_student_causal",
    "ball_direct_causal",
    "ball_direct_causal_compute_matched",
    "ball_inherited_dynamics_only",
    "ball_teacher_matching_only",
    "ball_full_decomposition",
    "ball_student_transition_only",
    "ball_student_transition_only_strict",
    "gp_causal_filter",
    "exponential_decay_gru",
}

COMPARATOR_LABELS = {
    "ball_direct_causal": "direct",
    "ball_direct_causal_compute_matched": "direct_compute_matched",
    "ball_inherited_dynamics_only": "inherited_dynamics",
    "ball_teacher_matching_only": "teacher_matching",
    "s0_direct_lgssm": "s0",
    "markov_direct_transition": "markov",
    "gp_causal_filter": "gp_causal",
    "exponential_decay_gru": "decay_gru",
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


def _apply_measurement_evaluation_mask(data):
    """Create prospective validation targets with recall-window embargoes."""

    anchors = data.anchors.copy()
    anchors["measurement_eval"] = False
    anchors["measurement_embargo"] = False
    validation_ids = set(
        data.individuals.loc[data.individuals["split"] == "validation", "id"].astype(int)
    )
    target_indices: list[int] = []
    for pid in sorted(validation_ids):
        patient = anchors.loc[
            anchors["id"].astype(int).eq(pid) & anchors["observed"].astype(bool)
        ]
        for anchor_name in ("Y1", "Y2"):
            candidates = patient.loc[patient["anchor"].eq(anchor_name)]
            if candidates.empty:
                continue
            target_index = int(
                candidates.sort_values(["window_end", "t"], kind="stable").index[-1]
            )
            target_indices.append(target_index)

    if target_indices:
        anchors.loc[target_indices, "measurement_eval"] = True
        for target_index in target_indices:
            target = anchors.loc[target_index]
            overlap = (
                anchors["id"].astype(int).eq(int(target["id"]))
                & anchors["observed"].astype(bool)
                & anchors["window_start"].le(int(target["window_end"]))
                & anchors["window_end"].ge(int(target["window_start"]))
            )
            anchors.loc[overlap, "measurement_embargo"] = True
        anchors.loc[anchors["measurement_embargo"].astype(bool), "observed"] = False

    metadata = dict(data.metadata)
    metadata.update(
        {
            "measurement_evaluation": "last observed validation anchor per patient and instrument",
            "measurement_evaluation_targets": int(anchors["measurement_eval"].sum()),
            "measurement_evaluation_embargoed_rows": int(anchors["measurement_embargo"].sum()),
            "measurement_evaluation_recall_overlap_removed": True,
        }
    )
    return dataclasses.replace(data, anchors=anchors, metadata=metadata)


def _measurement_anchor_rmse(
    data,
    preds: pd.DataFrame,
    *,
    anchor_name: str | None = None,
) -> float:
    """RMSE for questionnaire anchors hidden from every fitted estimator."""

    if "measurement_eval" not in data.anchors.columns:
        return H.val_anchor_rmse(data, preds)
    prediction_columns = {
        "Y1": "z_d_hat" if "z_d_hat" in preds.columns else "L_hat",
        "Y2": "z_p_hat" if "z_p_hat" in preds.columns else "L_hat",
    }
    by_column = {
        column: {
            int(pid): frame.set_index("t")[column]
            for pid, frame in preds[["id", "t", column]].groupby("id", sort=False)
        }
        for column in set(prediction_columns.values())
    }
    residuals: list[float] = []
    for row in data.anchors.loc[data.anchors["measurement_eval"].astype(bool)].itertuples(index=False):
        if anchor_name is not None and str(getattr(row, "anchor")) != str(anchor_name):
            continue
        value = float(getattr(row, "value", np.nan))
        loading = float(getattr(row, "loading", np.nan))
        if not np.isfinite(value) or not np.isfinite(loading):
            continue
        column = prediction_columns[str(getattr(row, "anchor"))]
        series = by_column[column].get(int(getattr(row, "id")))
        if series is None:
            continue
        window = series.loc[
            (series.index >= int(getattr(row, "window_start")))
            & (series.index <= int(getattr(row, "window_end")))
        ]
        if window.empty:
            continue
        residuals.append(value - loading * float(window.mean()))
    return float(np.sqrt(np.mean(np.square(residuals)))) if residuals else float("nan")


def _channel_latent_rmse(data, preds: pd.DataFrame, prediction: str, truth: str) -> float:
    test_ids = set(
        data.individuals.loc[data.individuals["split"].eq(H.EVAL_SPLIT), "id"].astype(int)
    )
    merged = preds.loc[preds["id"].astype(int).isin(test_ids), ["id", "t", prediction]].merge(
        data.components[["id", "t", truth]],
        on=["id", "t"],
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        return float("nan")
    patient_error = ((merged[prediction] - merged[truth]) ** 2).groupby(merged["id"]).mean()
    return float(np.sqrt(patient_error.mean()))


def make_data(args, seed: int, share: float, cell: str):
    cfg = _base_config(args, seed, share, cell)
    data = H.generate_dataset(cfg)
    data = _inject_benchmark_cell(data, cell, share, args)
    data = _apply_measurement_evaluation_mask(data)
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
        anchor_weight=args.anchor_weight,
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
    depression_latent_rmse = _channel_latent_rmse(
        data, preds, "z_d_hat", "z_d"
    )
    anxiety_latent_rmse = _channel_latent_rmse(
        data, preds, "z_p_hat", "z_p"
    )
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
        val_anchor = _measurement_anchor_rmse(data, preds)
    y1_measurement_rmse = (
        float("nan") if irt else _measurement_anchor_rmse(data, preds, anchor_name="Y1")
    )
    y2_measurement_rmse = (
        float("nan") if irt else _measurement_anchor_rmse(data, preds, anchor_name="Y2")
    )
    return {
        "share": float(share),
        "cell": cell,
        "seed": int(seed),
        "method": method,
        "n": int(args.n),
        "t": int(args.t),
        "latent_rmse": latent_raw,
        "depression_latent_rmse": depression_latent_rmse,
        "anxiety_latent_rmse": anxiety_latent_rmse,
        "latent_rmse_trait": latent_trait,
        "val_anchor_rmse": val_anchor,
        "y1_measurement_rmse": y1_measurement_rmse,
        "y2_measurement_rmse": y2_measurement_rmse,
        "irt_item_nll": item_nll,
        "irt_expected_total_rmse": etot_rmse,
        "beta_hat_norm_trait": beta_norm_trait,
        "elapsed_seconds": float(elapsed),
        "measurement_eval_n": int(data.metadata.get("measurement_evaluation_targets", 0)),
        "measurement_embargo_n": int(data.metadata.get("measurement_evaluation_embargoed_rows", 0)),
        **beta_metric_fields(beta_hat, beta_true, active_k),
    }


def _matched_s0_basis(cell: str) -> str:
    return {
        "linear": "linear",
        "missingness": "linear",
        "interaction": "interaction",
        "nonlinear": "nonlinear",
        # Patient subtype is latent in the heterogeneous data-generating
        # condition and is not supplied to BALL or the other comparators.
        # The classical estimators therefore use the observable linear basis.
        "heterogeneous": "linear",
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


def _markov_design(
    base_feats: np.ndarray,
    l_hat: np.ndarray,
    gaps: np.ndarray,
    stratum: int,
    n_strata: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(base_feats), max(len(l_hat) - 1, 0))
    if n <= 0:
        return np.zeros((0, base_feats.shape[1] + 2 * n_strata), dtype=float), np.zeros(0, dtype=float)
    positive = np.asarray(gaps[:n], dtype=float) > 0
    base = base_feats[:n][positive]
    lag = np.asarray(l_hat[:n], dtype=float)[positive]
    gaps_use = np.asarray(gaps[:n], dtype=float)[positive]
    if not len(gaps_use):
        return np.zeros((0, base_feats.shape[1] + 2 * n_strata), dtype=float), np.zeros(0, dtype=float)
    onehot = np.zeros((len(gaps_use), n_strata), dtype=float)
    onehot[:, int(stratum)] = 1.0
    x = np.column_stack([base, onehot, onehot * lag[:, None]])
    y = np.diff(np.asarray(l_hat[: n + 1], dtype=float))[positive] / gaps_use
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
    phis = np.clip(theta[n_base + n_strata : n_base + 2 * n_strata], -0.95, 0.05)
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

    next_gaps = (
        pd.to_numeric(comp_i["dt"], errors="coerce").fillna(1.0).clip(lower=0.0).to_numpy(dtype=float)
        if "dt" in comp_i.columns
        else np.ones(t_count, dtype=float)
    )
    for k in range(1, t_count):
        gap = float(next_gaps[k - 1])
        if gap <= 0:
            continue
        persistence = float(np.exp(phi * gap))
        drift_scale = gap if abs(phi) < 1e-8 else float(np.expm1(phi * gap) / phi)
        add(
            [(k, 1.0), (k - 1, -persistence)],
            float(trans_mean[k - 1]) * drift_scale,
            args.transition_sd * math.sqrt(gap),
        )

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
    """Fit two channel-specific pattern-mixture trajectories with one shared beta."""

    markov_args = argparse.Namespace(**vars(args))
    markov_args.s0_basis = basis
    markov_args.iters = max(int(args.iters), int(args.markov_iters))
    predictions, theta = fit_two_channel_direct_map(
        data,
        markov_args,
        pattern_mixture=True,
    )
    return predictions, theta, int(len(theta))


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


def _sum_kernel(
    times: np.ndarray,
    variance: float,
    *,
    slow_length: float = 28.0,
    fast_length: float = 3.0,
    slow_weight: float = 0.70,
) -> np.ndarray:
    """Slow radial-basis plus fast exponential covariance for one channel."""

    distance = np.abs(times[:, None] - times[None, :])
    slow = slow_weight * variance * np.exp(-0.5 * np.square(distance / slow_length))
    fast = (1.0 - slow_weight) * variance * np.exp(-distance / fast_length)
    return slow + fast + 1e-6 * np.eye(len(times))


def _gp_auxiliary_design(
    b: np.ndarray,
    action: np.ndarray,
    recent: np.ndarray,
    burden: np.ndarray,
    n_actions: int,
) -> np.ndarray:
    """Observed RDoC and treatment inputs used by the GP auxiliary kernel."""

    action_onehot = np.column_stack(
        [(np.asarray(action, dtype=int) == j).astype(float) for j in range(int(n_actions))]
    )
    return np.column_stack(
        [
            np.asarray(b, dtype=float),
            action_onehot,
            np.asarray(recent, dtype=float),
            np.asarray(burden, dtype=float),
        ]
    )


def _fit_gp_auxiliary_scaler(data, train_ids: set[int], daily_readout):
    """Fit training-only location and scale for the GP auxiliary kernel."""

    blocks: list[np.ndarray] = []
    n_actions = int(data.config.n_treatment_types)
    for pid in sorted(train_ids):
        comp_i, _, b, _, _, _, _, recent, burden = person_arrays(
            data, pid, daily_readout
        )
        action = causal_encoder_action_array(comp_i)
        blocks.append(_gp_auxiliary_design(b, action, recent, burden, n_actions))
    if not blocks:
        n_features = int(data.config.q) + n_actions + 2
        return np.zeros(n_features, dtype=float), np.ones(n_features, dtype=float)
    values = np.vstack(blocks)
    center = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return center, scale


def _gp_conditioned_mean(
    prior_mean: np.ndarray,
    kernel: np.ndarray,
    design: np.ndarray,
    targets: np.ndarray,
    noise_variance: np.ndarray,
    *,
    prediction_index: int,
) -> float:
    """Condition a Gaussian-process prior on windowed questionnaire measurements."""

    residual = targets - design @ prior_mean
    observed_covariance = design @ kernel @ design.T + np.diag(noise_variance)
    try:
        factor = cho_factor(observed_covariance, lower=True, check_finite=False)
        weights = cho_solve(factor, residual, check_finite=False)
    except np.linalg.LinAlgError:
        stabilized = observed_covariance + 1e-5 * np.eye(len(observed_covariance))
        weights = np.linalg.solve(stabilized, residual)
    cross_covariance = kernel[int(prediction_index)] @ design.T
    return float(prior_mean[int(prediction_index)] + cross_covariance @ weights)


def _gp_elapsed_times(comp_i: pd.DataFrame) -> np.ndarray:
    """Convert the stored next-session gaps into elapsed calendar time."""

    count = len(comp_i)
    elapsed = np.zeros(count, dtype=float)
    if count > 1:
        next_gap = pd.to_numeric(
            comp_i.get("dt", pd.Series(1.0, index=comp_i.index)), errors="coerce"
        ).fillna(1.0).clip(lower=0.0).to_numpy(dtype=float)
        elapsed[1:] = np.cumsum(next_gap[:-1])
    return elapsed


def _select_gp_time_lengths(
    data,
    train_ids: set[int],
    anchor_name: str,
    daily_readout,
    auxiliary_center: np.ndarray,
    auxiliary_scale: np.ndarray,
    variance: float,
) -> tuple[float, float]:
    """Select channel-specific time kernels from training patients only."""

    use_irt = str(getattr(data.config, "anchor_observation", "gaussian")).lower() == "irt"
    best = (28.0, 3.0)
    best_score = float("inf")
    for slow_length in (14.0, 28.0, 56.0):
        for fast_length in (1.0, 3.0, 7.0):
            score = 0.0
            event_count = 0
            for pid in sorted(train_ids):
                comp_i, anchors_i, b, daily_pred, _, _, _, recent, burden = person_arrays(
                    data, pid, daily_readout, anchor_name=anchor_name
                )
                times = _gp_elapsed_times(comp_i)
                count = len(times)
                action = causal_encoder_action_array(comp_i)
                auxiliary = _gp_auxiliary_design(
                    b, action, recent, burden, int(data.config.n_treatment_types)
                )
                auxiliary = (auxiliary - auxiliary_center) / auxiliary_scale
                distance = np.square(
                    auxiliary[:, None, :] - auxiliary[None, :, :]
                ).mean(axis=2)
                kernel = _sum_kernel(
                    times,
                    0.85 * variance,
                    slow_length=slow_length,
                    fast_length=fast_length,
                )
                kernel += 0.15 * variance * np.exp(-0.5 * distance)
                design_rows, targets, noise = [], [], []
                for row in anchors_i.itertuples(index=False):
                    if not bool(getattr(row, "observed", False)):
                        continue
                    start = max(0, int(getattr(row, "window_start")))
                    end = min(count - 1, int(getattr(row, "window_end")))
                    if end < start:
                        continue
                    h = np.zeros(count, dtype=float)
                    if use_irt:
                        summary = irt_anchor_pseudo_observation(row, data)
                        if summary is None:
                            continue
                        target, anchor_sd = summary
                        h[start : end + 1] = 1.0 / (end - start + 1)
                    else:
                        target = float(getattr(row, "value", np.nan))
                        loading = float(getattr(row, "loading", np.nan))
                        if not np.isfinite(target) or not np.isfinite(loading):
                            continue
                        h[start : end + 1] = loading / (end - start + 1)
                        spec = data.config.y1 if anchor_name == "Y1" else data.config.y2
                        serial_rho = data.config.rho_serial_y1 if anchor_name == "Y1" else data.config.rho_serial_y2
                        anchor_sd = marginal_anchor_sd(spec, serial_rho)
                    design_rows.append(h)
                    targets.append(float(target))
                    noise.append(float(anchor_sd) ** 2)
                if not design_rows:
                    continue
                design = np.vstack(design_rows)
                residual = np.asarray(targets) - design @ np.asarray(daily_pred, dtype=float)
                covariance = design @ kernel @ design.T + np.diag(noise)
                try:
                    factor = cho_factor(covariance, lower=True, check_finite=False)
                    quadratic = float(residual @ cho_solve(factor, residual, check_finite=False))
                    logdet = 2.0 * float(np.log(np.diag(factor[0])).sum())
                except np.linalg.LinAlgError:
                    continue
                score += 0.5 * (quadratic + logdet + len(residual) * np.log(2.0 * np.pi))
                event_count += len(residual)
            if event_count and score / event_count < best_score:
                best_score = score / event_count
                best = (slow_length, fast_length)
    return best


def fit_gp_causal_filter(data, args) -> pd.DataFrame:
    """Fit separate causal Gaussian-process filters for depression and anxiety."""

    train_ids = set(data.individuals.loc[data.individuals["split"] == "train", "id"].astype(int))
    daily_readouts = {
        anchor_name: fit_daily_readout(
            data,
            train_ids,
            ridge=args.daily_ridge,
            anchor_name=anchor_name,
        )
        for anchor_name in ("Y1", "Y2")
    }
    auxiliary_center, auxiliary_scale = _fit_gp_auxiliary_scaler(
        data, train_ids, daily_readouts["Y1"]
    )
    use_irt = str(getattr(data.config, "anchor_observation", "gaussian")).lower() == "irt"
    channel_frames = []
    for anchor_name, output_column in (("Y1", "z_d_hat"), ("Y2", "z_p_hat")):
        train_anchors = data.anchors.loc[
            data.anchors["observed"].astype(bool)
            & data.anchors["id"].astype(int).isin(train_ids)
            & data.anchors["anchor"].astype(str).eq(anchor_name)
        ].copy()
        if use_irt:
            variance_source = np.asarray([
                summary[0]
                for row in train_anchors.itertuples(index=False)
                for summary in [irt_anchor_pseudo_observation(row, data)]
                if summary is not None
            ], dtype=float)
        else:
            variance_source = (
                pd.to_numeric(train_anchors["value"], errors="coerce")
                / pd.to_numeric(train_anchors["loading"], errors="coerce").replace(0.0, np.nan)
            ).to_numpy(dtype=float)
        finite_source = variance_source[np.isfinite(variance_source)]
        variance = max(
            float(np.var(finite_source)) if len(finite_source) else 0.25,
            0.25,
        )
        slow_length, fast_length = _select_gp_time_lengths(
            data,
            train_ids,
            anchor_name,
            daily_readouts[anchor_name],
            auxiliary_center,
            auxiliary_scale,
            variance,
        )
        predictions = []
        for pid in sorted(data.individuals["id"].astype(int).tolist()):
            comp_i, anchors_i, b, daily_pred, _, _, _, recent, burden = person_arrays(
                data,
                pid,
                daily_readouts[anchor_name],
                anchor_name=anchor_name,
            )
            action = causal_encoder_action_array(comp_i)
            times = _gp_elapsed_times(comp_i)
            t_count = len(times)
            auxiliary = _gp_auxiliary_design(
                b, action, recent, burden, int(data.config.n_treatment_types)
            )
            auxiliary = (auxiliary - auxiliary_center) / auxiliary_scale
            auxiliary_distance = np.square(
                auxiliary[:, None, :] - auxiliary[None, :, :]
            ).mean(axis=2)
            kernel = _sum_kernel(
                times,
                0.85 * variance,
                slow_length=slow_length,
                fast_length=fast_length,
            )
            kernel += 0.15 * variance * np.exp(-0.5 * auxiliary_distance)
            design_rows: list[np.ndarray] = []
            targets: list[float] = []
            noise_variance: list[float] = []
            availability_times: list[int] = []
            for row in anchors_i.itertuples(index=False):
                if not bool(getattr(row, "observed", False)):
                    continue
                start = max(0, int(getattr(row, "window_start")))
                end = min(t_count - 1, int(getattr(row, "window_end")))
                if end < start:
                    continue
                design_row = np.zeros(t_count, dtype=float)
                if use_irt:
                    summary = irt_anchor_pseudo_observation(row, data)
                    if summary is None:
                        continue
                    target, anchor_sd = summary
                    design_row[start : end + 1] = 1.0 / (end - start + 1)
                else:
                    target = float(getattr(row, "value", np.nan))
                    loading = float(getattr(row, "loading", np.nan))
                    if not np.isfinite(target) or not np.isfinite(loading):
                        continue
                    design_row[start : end + 1] = loading / (end - start + 1)
                    spec = data.config.y1 if anchor_name == "Y1" else data.config.y2
                    serial_rho = data.config.rho_serial_y1 if anchor_name == "Y1" else data.config.rho_serial_y2
                    anchor_sd = marginal_anchor_sd(spec, serial_rho)
                design_rows.append(design_row)
                targets.append(float(target))
                noise_variance.append(float(anchor_sd) ** 2)
                availability_times.append(int(getattr(row, "t", end)))
            mean = np.asarray(daily_pred, dtype=float).copy()
            if design_rows:
                design_matrix = np.vstack(design_rows)
                target_vector = np.asarray(targets, dtype=float)
                noise = np.asarray(noise_variance, dtype=float)
                available = np.asarray(availability_times, dtype=int)
                for index, _current_time in enumerate(times):
                    observed = available < int(comp_i.iloc[index]["t"])
                    if observed.any():
                        mean[index] = _gp_conditioned_mean(
                            np.asarray(daily_pred, dtype=float),
                            kernel,
                            design_matrix[observed],
                            target_vector[observed],
                            noise[observed],
                            prediction_index=index,
                        )
            predictions.append(pd.DataFrame({
                "id": int(pid),
                "t": comp_i["t"].to_numpy(dtype=int),
                output_column: mean,
            }))
        channel_frames.append(pd.concat(predictions, ignore_index=True))
    combined = channel_frames[0].merge(channel_frames[1], on=["id", "t"], validate="one_to_one")
    combined["L_hat"] = 0.5 * (combined["z_d_hat"] + combined["z_p_hat"])
    return combined


def run_gp(data, args, cell: str, seed: int, share: float) -> dict:
    """Run the causal Gaussian-process benchmark."""

    start = perf_counter()
    predictions = fit_gp_causal_filter(data, args)
    elapsed = perf_counter() - start
    beta_none = np.zeros(data.config.q, dtype=float)
    row = score(
        data,
        predictions,
        beta_none,
        method="gp_causal_filter",
        cell=cell,
        seed=seed,
        share=share,
        elapsed=elapsed,
        args=args,
    )
    for name in ["beta_cosine", "beta_abs_cosine", "beta_topk_f1", "beta_hat_norm", "beta_hat_norm_trait"]:
        row[name] = float("nan")
    return row


class _CausalODERNN(torch.nn.Module):
    """Forward GRU with an exact diagonal exponential decay between updates."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.cell = torch.nn.GRUCell(input_dim, hidden_dim)
        self.log_decay = torch.nn.Parameter(torch.full((hidden_dim,), -2.0))
        self.readout = torch.nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = x.shape
        hidden = x.new_zeros((batch, self.cell.hidden_size))
        outputs = []
        decay = F.softplus(self.log_decay).view(1, -1)
        for step in range(steps):
            # This is the closed-form solution of dh/dt = -decay * h over the
            # observed gap, followed by a GRU measurement update. No future
            # token or backward recurrence is present.
            hidden = hidden * torch.exp(-decay * dt[:, step].view(-1, 1).clamp_min(0.0))
            hidden = self.cell(x[:, step], hidden)
            outputs.append(self.readout(hidden))
        return torch.stack(outputs, dim=1)


def _ode_rnn_tensors(data, ids: list[int], device: torch.device):
    t_values = np.sort(data.components["t"].unique().astype(int))
    t_count = len(t_values)
    p = int(data.config.p_daily)
    q = int(data.config.q)
    n_actions = int(data.config.n_treatment_types) + 1
    x_cols = [f"X{j}" for j in range(p)]
    obs_cols = [f"obs_X{j}" for j in range(p)]
    input_cols = [f"input_X{j}" for j in range(p)]
    b_cols = [f"B{j}" for j in range(q)]
    features = []
    gaps = []
    anchor_inputs = []
    anchor_flags = []
    valid_steps = []
    id_to_row = {int(pid): row for row, pid in enumerate(ids)}

    comp_by = {int(pid): g.set_index("t").reindex(t_values) for pid, g in data.components.groupby("id")}
    daily_by = {int(pid): g.set_index("t").reindex(t_values) for pid, g in data.daily.groupby("id")}
    observed_anchors = data.anchors.loc[data.anchors["observed"].astype(bool)].copy()
    anchors_by = {int(pid): g for pid, g in observed_anchors.groupby("id")}

    use_irt = str(getattr(data.config, "anchor_observation", "gaussian")).lower() == "irt"
    loss_rows: list[tuple[int, int, int, int, float, float, np.ndarray | None]] = []
    for pid in ids:
        ci = comp_by[int(pid)]
        di = daily_by[int(pid)]
        raw_x = di[x_cols].apply(pd.to_numeric, errors="coerce")
        mask_cols = input_cols if all(col in di.columns for col in input_cols) else obs_cols
        masks = (
            di[mask_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            if all(col in di.columns for col in mask_cols)
            else raw_x.notna().astype(float)
        )
        x_imputed = raw_x.ffill().fillna(0.0).to_numpy(dtype=np.float32)
        b_imputed = ci[b_cols].apply(pd.to_numeric, errors="coerce").ffill().fillna(0.0).to_numpy(dtype=np.float32)
        action = causal_encoder_action_array(ci) + 1
        action = np.clip(action, 0, n_actions - 1)
        action_onehot = np.eye(n_actions, dtype=np.float32)[action]
        recent, burden = observed_treatment_history_arrays(
            ci.reset_index(), di.reset_index(), data.config
        )
        ain = np.zeros((t_count, 2), dtype=np.float32)
        aflag = np.zeros((t_count, 2), dtype=np.float32)
        for anchor in anchors_by.get(int(pid), pd.DataFrame()).itertuples(index=False):
            step = int(getattr(anchor, "t"))
            if step not in set(t_values):
                continue
            pos = int(np.searchsorted(t_values, step))
            channel = 0 if str(getattr(anchor, "anchor")) == "Y1" else 1
            if use_irt:
                item_values = np.asarray(getattr(anchor, "irt_items", None), dtype=float)
                total = float(getattr(anchor, "irt_total", np.nan))
                if item_values.size == 0 or not np.isfinite(total) or not np.isfinite(item_values).all():
                    continue
                encoder_value = total
                loading = 1.0
                value = 0.0
                loss_items = item_values.astype(np.int64)
            else:
                value = float(getattr(anchor, "value", np.nan))
                loading = float(getattr(anchor, "loading", np.nan))
                if not np.isfinite(value) or not np.isfinite(loading):
                    continue
                encoder_value = value
                loss_items = None
            input_pos = pos + 1
            if input_pos < t_count:
                if aflag[input_pos, channel] > 0:
                    ain[input_pos, channel] = 0.5 * (
                        ain[input_pos, channel] + encoder_value
                    )
                else:
                    ain[input_pos, channel] = encoder_value
                aflag[input_pos, channel] = 1.0
            start = max(0, int(getattr(anchor, "window_start")))
            end = min(t_count - 1, int(getattr(anchor, "window_end")))
            if end >= start:
                loss_rows.append(
                    (
                        id_to_row[int(pid)],
                        channel,
                        start,
                        end,
                        loading,
                        value,
                        loss_items,
                    )
                )
        features.append(
            np.concatenate(
                [
                    x_imputed,
                    masks.to_numpy(dtype=np.float32),
                    b_imputed,
                    action_onehot,
                    recent[:, None].astype(np.float32),
                    burden[:, None].astype(np.float32),
                    ain,
                    aflag,
                ],
                axis=1,
            )
        )
        next_gap_source = ci["dt"] if "dt" in ci.columns else pd.Series(1.0, index=ci.index)
        next_gap = pd.to_numeric(next_gap_source, errors="coerce").fillna(1.0).clip(lower=0.0).to_numpy(dtype=np.float32)
        previous_gap = np.zeros(t_count, dtype=np.float32)
        if t_count > 1:
            previous_gap[1:] = next_gap[:-1]
        gaps.append(previous_gap)
        if "session_observed" in ci.columns:
            valid = ci["session_observed"].fillna(False).astype(bool).to_numpy()
        else:
            valid = np.ones(t_count, dtype=bool)
        valid_steps.append(valid)
        anchor_inputs.append(ain)
        anchor_flags.append(aflag)

    return (
        torch.as_tensor(np.stack(features), dtype=torch.float32, device=device),
        torch.as_tensor(np.stack(gaps), dtype=torch.float32, device=device),
        loss_rows,
        t_values,
        torch.as_tensor(np.stack(valid_steps), dtype=torch.bool, device=device),
    )


def _pack_ode_anchor_rows(
    rows: list[tuple[int, int, int, int, float, float, np.ndarray | None]],
    device: torch.device,
):
    if not rows:
        raise ValueError("Exponential-decay GRU comparator has no observed training anchors")
    array = np.asarray([row[:6] for row in rows], dtype=float)
    item_rows = [row[6] for row in rows]
    if all(items is not None for items in item_rows):
        items = torch.as_tensor(np.stack(item_rows), dtype=torch.long, device=device)
    elif all(items is None for items in item_rows):
        items = None
    else:
        raise ValueError("Exponential-decay GRU anchor rows mix Gaussian and item-response targets")
    return {
        "row": torch.as_tensor(array[:, 0], dtype=torch.long, device=device),
        "channel": torch.as_tensor(array[:, 1], dtype=torch.long, device=device),
        "start": torch.as_tensor(array[:, 2], dtype=torch.long, device=device),
        "end": torch.as_tensor(array[:, 3], dtype=torch.long, device=device),
        "loading": torch.as_tensor(array[:, 4], dtype=torch.float32, device=device),
        "value": torch.as_tensor(array[:, 5], dtype=torch.float32, device=device),
        "items": items,
    }


def _ode_anchor_loss(prediction: torch.Tensor, packed_rows, data) -> torch.Tensor:
    row = packed_rows["row"]
    channel = packed_rows["channel"]
    start = packed_rows["start"]
    end = packed_rows["end"]
    loading = packed_rows["loading"]
    value = packed_rows["value"]
    trajectories = prediction[row, :, channel]
    cumulative = F.pad(torch.cumsum(trajectories, dim=1), (1, 0))
    index = torch.arange(len(row), device=prediction.device)
    total = cumulative[index, end + 1] - cumulative[index, start]
    mean = total / (end - start + 1).to(prediction.dtype)
    items = packed_rows["items"]
    if items is not None:
        cfg = data.config
        loc = torch.as_tensor(float(cfg.irt_loc), dtype=prediction.dtype, device=prediction.device)
        scale = torch.as_tensor(float(cfg.irt_scale) or 1.0, dtype=prediction.dtype, device=prediction.device)
        disc = torch.as_tensor(cfg.irt_item_discriminations, dtype=prediction.dtype, device=prediction.device)
        thresholds = torch.as_tensor(cfg.irt_item_thresholds, dtype=prediction.dtype, device=prediction.device)
        trait = (mean - loc) / scale
        pstar = torch.sigmoid(
            disc.view(1, -1, 1)
            * (trait.view(-1, 1, 1) - thresholds.unsqueeze(0))
        )
        ones = torch.ones_like(pstar[..., :1])
        zeros = torch.zeros_like(pstar[..., :1])
        padded = torch.cat([ones, pstar, zeros], dim=-1)
        probabilities = (padded[..., :-1] - padded[..., 1:]).clamp_min(1e-9)
        selected = probabilities.gather(-1, items.unsqueeze(-1)).squeeze(-1)
        return -torch.log(selected).mean()
    return torch.square(loading * mean - value).mean()


def fit_ode_rnn_causal(data, args) -> pd.DataFrame:
    """Fit a modern continuous-time forward baseline on training patients only."""

    requested_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    train_ids = sorted(data.individuals.loc[data.individuals["split"] == "train", "id"].astype(int).tolist())
    all_ids = sorted(data.individuals["id"].astype(int).tolist())
    train_x, train_dt, train_rows, _, train_valid = _ode_rnn_tensors(data, train_ids, device)
    packed_train_rows = _pack_ode_anchor_rows(train_rows, device)
    torch.manual_seed(int(data.config.seed) + 88001)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(data.config.seed) + 88001)
    model = _CausalODERNN(train_x.shape[-1], int(args.ode_hidden)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.ode_lr))
    epochs = int(args.ode_epochs if args.ode_epochs is not None else args.teacher_epochs)
    model.train()
    for _ in range(max(1, epochs)):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(train_x, train_dt)
        anchor_loss = _ode_anchor_loss(prediction, packed_train_rows, data)
        transition_valid = train_valid[:, 1:] & train_valid[:, :-1]
        change = torch.square(prediction[:, 1:] - prediction[:, :-1]).mean(dim=-1)
        elapsed = train_dt[:, 1:].clamp_min(1.0)
        smoothness = (
            (change / elapsed)[transition_valid].mean()
            if bool(transition_valid.any())
            else change.new_zeros(())
        )
        loss = anchor_loss + float(args.ode_smoothness) * smoothness
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    eval_x, eval_dt, _, t_values, eval_valid = _ode_rnn_tensors(data, all_ids, device)
    model.eval()
    with torch.no_grad():
        predicted = model(eval_x, eval_dt).detach().cpu().numpy()
    frames = []
    for row, pid in enumerate(all_ids):
        keep = eval_valid[row].detach().cpu().numpy().astype(bool)
        frames.append(
            pd.DataFrame(
                {
                    "id": int(pid),
                    "t": t_values[keep],
                    "z_d_hat": predicted[row, keep, 0],
                    "z_p_hat": predicted[row, keep, 1],
                    "L_hat": predicted[row, keep].mean(axis=1),
                }
            )
        )
    del model, train_x, train_dt, train_valid, eval_x, eval_dt, eval_valid
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.concat(frames, ignore_index=True)


def run_ode_rnn(data, args, cell: str, seed: int, share: float) -> dict:
    start = perf_counter()
    preds = fit_ode_rnn_causal(data, args)
    elapsed = perf_counter() - start
    row = score(
        data,
        preds,
        np.zeros(data.config.q, dtype=float),
        method="exponential_decay_gru",
        cell=cell,
        seed=seed,
        share=share,
        elapsed=elapsed,
        args=args,
    )
    for name in ["beta_cosine", "beta_abs_cosine", "beta_topk_f1", "beta_hat_norm", "beta_hat_norm_trait"]:
        row[name] = float("nan")
    return row


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


def run_ball_direct_causal(data, args, cell: str, seed: int, share: float) -> dict:
    """Fit the same forward-only transformer architecture directly on anchors."""

    start = perf_counter()
    result = fit_ball_ssm_direct_causal(
        data,
        ball_config(args, seed),
        device=args.device,
        prediction_split=None,
    )
    elapsed = perf_counter() - start
    return _score_ball_result(
        data,
        result,
        method="ball_direct_causal",
        cell=cell,
        seed=seed,
        share=share,
        elapsed=elapsed,
        args=args,
    )


def run_ball_direct_causal_compute_matched(
    data, args, cell: str, seed: int, share: float
) -> dict:
    """Train the direct causal model for the teacher-plus-student update budget."""

    start = perf_counter()
    config = ball_config(args, seed)
    config = dataclasses.replace(
        config,
        teacher_epochs=int(config.teacher_epochs + config.student_epochs),
    )
    result = fit_ball_ssm_direct_causal(
        data,
        config,
        device=args.device,
        prediction_split=None,
    )
    elapsed = perf_counter() - start
    row = _score_ball_result(
        data,
        result,
        method="ball_direct_causal_compute_matched",
        cell=cell,
        seed=seed,
        share=share,
        elapsed=elapsed,
        args=args,
    )
    row["optimizer_update_budget"] = "teacher plus student epoch count"
    return row


def run_distillation_decomposition(
    data, args, cell: str, seed: int, share: float
) -> list[dict]:
    """Compare inherited dynamics, teacher matching, and re-anchored BALL."""

    start = perf_counter()
    results = fit_ball_ssm_distillation_decomposition(
        data,
        ball_config(args, seed),
        device=args.device,
        prediction_split=None,
    )
    elapsed = perf_counter() - start
    method_names = {
        "inherited_dynamics_only": "ball_inherited_dynamics_only",
        "teacher_matching_only": "ball_teacher_matching_only",
        "full_ball": "ball_full_decomposition",
    }
    return [
        _score_ball_result(
            data,
            result,
            method=method_names[arm],
            cell=cell,
            seed=seed,
            share=share,
            elapsed=elapsed,
            args=args,
        )
        for arm, result in results.items()
    ]


def run_ball_transition_only(
    data, args, cell: str, seed: int, share: float
) -> dict:
    """Remove the observed RDoC profile from the encoder while retaining correlated proxies."""

    start = perf_counter()
    config = dataclasses.replace(ball_config(args, seed), encoder_use_rdoc=False)
    student = fit_ball_ssm(
        data,
        config,
        device=args.device,
        causal=True,
        prediction_split=None,
        return_teacher=False,
    )
    elapsed = perf_counter() - start
    return _score_ball_result(
        data,
        student,
        method="ball_student_transition_only",
        cell=cell,
        seed=seed,
        share=share,
        elapsed=elapsed,
        args=args,
    )


def run_ball_transition_only_strict(
    data, args, cell: str, seed: int, share: float
) -> dict:
    """Reserve RDoC information for the explicit transition coefficient only."""

    strict_daily = data.daily.copy()
    proxy_columns = [f"X{index}" for index in range(8, 12)]
    for column in proxy_columns:
        if column in strict_daily:
            strict_daily[column] = 0.0
        for prefix in ("input_", "obs_"):
            flag = f"{prefix}{column}"
            if flag in strict_daily:
                strict_daily[flag] = False
    strict_data = type(data)(
        config=data.config,
        individuals=data.individuals.copy(),
        daily=strict_daily,
        anchors=data.anchors.copy(),
        treatments=data.treatments.copy(),
        components=data.components.copy(),
        metadata={
            **data.metadata,
            "transition_identification_arm": "strict",
            "encoder_observed_rdoc_profile": False,
            "encoder_rdoc_proxy_columns": [],
            "removed_proxy_columns": proxy_columns,
        },
    )
    start = perf_counter()
    config = dataclasses.replace(ball_config(args, seed), encoder_use_rdoc=False)
    student = fit_ball_ssm(
        strict_data,
        config,
        device=args.device,
        causal=True,
        prediction_split=None,
        return_teacher=False,
    )
    elapsed = perf_counter() - start
    return _score_ball_result(
        data,
        student,
        method="ball_student_transition_only_strict",
        cell=cell,
        seed=seed,
        share=share,
        elapsed=elapsed,
        args=args,
    )


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
    parser.add_argument("--skip-gp", action="store_true", help="Skip the slow-plus-fast sum-of-kernels Gaussian-process comparator")
    parser.add_argument("--skip-ode-rnn", action="store_true", help="Skip the causal exponential-decay GRU comparator")
    parser.add_argument("--markov-strata", type=int, default=3)
    parser.add_argument("--markov-iters", type=int, default=4)
    parser.add_argument("--markov-ridge", type=float, default=10.0)
    parser.add_argument(
        "--classical-persistence-grid",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.6, 0.9],
        help="Development-selected daily persistence candidates for each classical latent channel.",
    )

    parser.add_argument("--teacher-epochs", type=int, default=240)
    parser.add_argument("--student-epochs", type=int, default=240)
    parser.add_argument("--ode-epochs", type=int, default=None, help="Exponential-decay GRU epochs; defaults to --teacher-epochs")
    parser.add_argument("--ode-hidden", type=int, default=96)
    parser.add_argument("--ode-lr", type=float, default=1e-3)
    parser.add_argument("--ode-smoothness", type=float, default=0.01)
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
                        help="irt: frozen calibrated graded-response instrument shared by BALL, the direct causal transformer, S0, and Markov")
    parser.add_argument("--irt-n-items", type=int, default=9)
    parser.add_argument("--irt-discrimination", type=float, default=1.5)
    parser.add_argument("--delta-phi", type=float, default=0.3)
    parser.add_argument("--anchor-weight", type=float, default=40.0)
    parser.add_argument("--ball-no-ehr-drift", action="store_true")
    parser.add_argument(
        "--skip-direct-causal",
        action="store_true",
        help="Skip the forward-only student architecture trained directly on anchors.",
    )
    parser.add_argument("--skip-compute-matched-direct", action="store_true")
    parser.add_argument("--skip-distillation-decomposition", action="store_true")
    parser.add_argument("--skip-transition-identification", action="store_true")
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
    completed_methods: dict[tuple[str, float, int], set[str]] = {}
    if args.resume and checkpoint_path.exists():
        prior = pd.read_csv(checkpoint_path)
        rows = prior.to_dict("records")
        for row in prior[["cell", "share", "seed", "method"]].drop_duplicates().itertuples(index=False):
            key = (str(row.cell), float(row.share), int(row.seed))
            completed_methods.setdefault(key, set()).add(str(row.method))
        print(f"resuming {len(completed_methods)} cell/share/seed combinations from {checkpoint_path}")
    for cell in args.cells:
        for share in args.shares:
            for seed in args.seeds:
                key = (str(cell), float(share), int(seed))
                have = completed_methods.get(key, set())
                required = set(REQUIRED_METHODS)
                if args.skip_markov:
                    required.discard("markov_direct_transition")
                if args.skip_gp:
                    required.discard("gp_causal_filter")
                if args.skip_ode_rnn:
                    required.discard("exponential_decay_gru")
                if args.skip_direct_causal:
                    required.discard("ball_direct_causal")
                if args.skip_compute_matched_direct:
                    required.discard("ball_direct_causal_compute_matched")
                if args.skip_distillation_decomposition:
                    required.difference_update({
                        "ball_inherited_dynamics_only",
                        "ball_teacher_matching_only",
                        "ball_full_decomposition",
                    })
                if args.skip_transition_identification:
                    required.discard("ball_student_transition_only")
                    required.discard("ball_student_transition_only_strict")
                if required.issubset(have):
                    print(f"cell={cell} share={share:.2f} seed={seed} already complete; skipping")
                    continue
                print(f"cell={cell} share={share:.2f} seed={seed} building data")
                data = make_data(args, seed, share, cell)
                if "s0_direct_lgssm" not in have:
                    print("  s0_direct_lgssm")
                    rows.append(run_s0(data, args, cell, seed, share))
                if not args.skip_markov and "markov_direct_transition" not in have:
                    print("  markov_direct_transition")
                    rows.append(run_markov(data, args, cell, seed, share))
                if not args.skip_gp and "gp_causal_filter" not in have:
                    print("  gp_causal_filter")
                    rows.append(run_gp(data, args, cell, seed, share))
                if not args.skip_ode_rnn and "exponential_decay_gru" not in have:
                    print("  exponential_decay_gru")
                    rows.append(run_ode_rnn(data, args, cell, seed, share))
                if not {"ball_teacher_smoother", "ball_student_causal"}.issubset(have):
                    print("  ball_teacher_student")
                    rows.extend(run_ball(data, args, cell, seed, share))
                if not args.skip_direct_causal and "ball_direct_causal" not in have:
                    print("  ball_direct_causal")
                    rows.append(run_ball_direct_causal(data, args, cell, seed, share))
                if (
                    not args.skip_compute_matched_direct
                    and "ball_direct_causal_compute_matched" not in have
                ):
                    print("  ball_direct_causal_compute_matched")
                    rows.append(
                        run_ball_direct_causal_compute_matched(
                            data, args, cell, seed, share
                        )
                    )
                decomposition_methods = {
                    "ball_inherited_dynamics_only",
                    "ball_teacher_matching_only",
                    "ball_full_decomposition",
                }
                if (
                    not args.skip_distillation_decomposition
                    and not decomposition_methods.issubset(have)
                ):
                    print("  distillation_mechanistic_decomposition")
                    rows.extend(run_distillation_decomposition(data, args, cell, seed, share))
                if (
                    not args.skip_transition_identification
                    and "ball_student_transition_only" not in have
                ):
                    print("  ball_student_transition_only")
                    rows.append(run_ball_transition_only(data, args, cell, seed, share))
                if (
                    not args.skip_transition_identification
                    and "ball_student_transition_only_strict" not in have
                ):
                    print("  ball_student_transition_only_strict")
                    rows.append(run_ball_transition_only_strict(data, args, cell, seed, share))
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
                "benchmark_py_sha256": H.file_sha256(Path(__file__).resolve()),
                "fair_comparator_py_sha256": H.file_sha256(
                    Path(__file__).resolve().parent / "direct_rdoc_fair_comparator.py"
                ),
                "common_py_sha256": H.file_sha256(
                    Path(__file__).resolve().parent / "direct_rdoc_common.py"
                ),
                "harness_py_sha256": H.file_sha256(
                    Path(__file__).resolve().parent / "ball_validation_harness.py"
                ),
                "runtime_environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "scipy": scipy.__version__,
                    "torch": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "gpu_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else None,
                },
                "args": vars(args),
                "irt_calibration": irt_calibration_provenance(args),
                "methods": {
                    "ball_teacher_smoother": "BALL bidirectional transformer teacher smoother with rdoc_drift_head=True and use_alpha_slow=False",
                    "ball_student_causal": "BALL causal transformer student distilled from the teacher, with rdoc_drift_head=True and use_alpha_slow=False",
                    "ball_direct_causal": "Forward-only student architecture trained directly on anchors",
                    "ball_direct_causal_compute_matched": "Direct causal transformer trained for the combined teacher-plus-student optimizer-update budget",
                    "ball_inherited_dynamics_only": "Causal encoder with frozen teacher-fitted dynamics and measurement model, trained through the causal variational and questionnaire objectives",
                    "ball_teacher_matching_only": "Causal encoder trained against teacher posterior distributions without questionnaire re-anchoring",
                    "ball_full_decomposition": "Causal encoder trained with teacher-distribution matching and questionnaire re-anchoring against the same fitted teacher",
                    "ball_student_transition_only": "BALL student with the observed RDoC profile removed from the inference encoder while correlated structured proxies remain",
                    "ball_student_transition_only_strict": "BALL student with the observed RDoC profile and its four correlated structured proxies removed from the inference encoder and neural nuisance inputs while RDoC remains in the explicit transition",
                    "s0_direct_lgssm": f"Classical direct linear-Gaussian MAP smoother with {args.s0_basis} transition basis",
                    "markov_direct_transition": f"Classical direct Markov/pattern smoother with {args.s0_basis} transition basis and observed-pattern Markov terms",
                    "gp_causal_filter": (
                        "Causal Gaussian-process filter with 28-day radial-basis and 3-day Ornstein-Uhlenbeck time kernels, "
                        "a training-patient structured-input mean function, and only questionnaire measurements available by each estimation time"
                    ),
                    "exponential_decay_gru": (
                        "Forward-only exponential-decay GRU with an exact diagonal linear decay between sessions, GRU updates, the same observed structured inputs, "
                        "and the exact calibrated graded-response likelihood when IRT is active"
                    ),
                },
                "input_contract": {
                    "patient_partitions": "identical train, validation, calibration, and test patients",
                    "anchors": "identical observed questionnaire anchors and recall windows",
                    "structured_inputs": "daily covariates, daily observation indicators, observed RDoC proxies, treatment categories, observed treatment-history transforms, and elapsed time",
                    "causal_estimators": [
                        "ball_student_causal",
                        "ball_direct_causal",
                        "ball_direct_causal_compute_matched",
                        "ball_inherited_dynamics_only",
                        "ball_teacher_matching_only",
                        "ball_full_decomposition",
                        "ball_student_transition_only",
                        "ball_student_transition_only_strict",
                        "gp_causal_filter",
                        "exponential_decay_gru",
                    ],
                    "retrospective_smoothers": ["ball_teacher_smoother", "s0_direct_lgssm", "markov_direct_transition"],
                    "measurement_evaluation": "last observed validation anchor per patient and instrument, hidden with every overlapping recall window",
                    "latent_evaluation": "test-patient latent trajectories",
                    "simulated_questionnaire_mapping": {
                        "Y1": "depression channel with a 14-day recall window",
                        "Y2": "anxiety channel with a 7-day recall window",
                    },
                    "measurement_parameters": "true simulated instrument loadings and observation variances frozen identically for every estimator",
                    "prospective_questionnaire_timing": "causal encoders receive each questionnaire beginning at the next recorded session",
                },
                "note": (
                    "Direct-transition benchmark on identical generated datasets. "
                    f"The classical estimators use the {args.s0_basis} transition basis. "
                    "BALL, the direct causal transformer, S0, and Markov estimate an explicit transition coefficient."
                ),
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
