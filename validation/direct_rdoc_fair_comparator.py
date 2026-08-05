#!/usr/bin/env python
"""Fair old-school comparator for direct RDoC drift recovery.

The comparator is a linear-Gaussian MAP smoother with the same explicit target
as the transformer direct head:

    L[t] - L[t-1] = B[t-1] beta + nuisance[t-1] gamma + noise

It uses only observed anchors, observed daily covariates, treatments, and the
observed/noisy RDoC proxy B. It does not see the simulated latent L except for
evaluation. Estimation alternates:

  1. smooth per-person L trajectories given beta/gamma;
  2. refit global beta/gamma on train-patient smoothed transitions.

This is the fair S0-style comparator for the new direct-RDoC estimand. The old
S0/Markov arms remain useful severity comparators, but they are not fair beta
comparators because they do not estimate direct RDoC drift parameters.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ball_validation_harness as H  # noqa: E402

_model_utils = sys.modules["simulations.src.model_utils"]
marginal_anchor_sd = _model_utils.marginal_anchor_sd


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 1e-12 else float("nan")


def topk_f1(v: np.ndarray, truth: np.ndarray, k: int) -> float:
    k = max(1, min(int(k), len(truth)))
    true = set(np.flatnonzero(np.abs(truth) > 1e-12).tolist())
    pred = set(np.argsort(-np.abs(v))[:k].tolist())
    if not true or not pred:
        return float("nan")
    tp = len(true & pred)
    precision = tp / len(pred)
    recall = tp / len(true)
    return float(2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0


def mcse(vals: np.ndarray) -> float:
    vals = vals[np.isfinite(vals)]
    return float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0


# --------------------------------------------------------------------------- #
# Graded-response IRT anchor observation (calibrated-instrument sensitivity).
#
# When anchor_observation="irt", every method uses the SAME frozen calibrated
# instrument. The per-item ordered thresholds and discriminations and the affine
# trait map trait = (windowed_latent - loc) / scale are fixed in advance
# (validation/irt_calibration.json) and carried on data.config. Item parameters are
# NOT learned, which is what identifies the latent scale. Each observed anchor enters
# as the exact per-item graded-response likelihood, linearized at the current trait
# estimate by Fisher scoring and expressed on the raw latent scale so it composes
# with the transition and daily rows. The exact total-score distribution by
# convolution is available for totals-only inputs (grm_total_logpmf).
# --------------------------------------------------------------------------- #


def grm_category_probs(z: float, a: float, b: np.ndarray) -> np.ndarray:
    """P(response = c | trait z) for one graded item, c = 0..K-1. Returns (K,)."""
    pstar = 1.0 / (1.0 + np.exp(-a * (z - np.asarray(b, dtype=float))))   # (K-1,) P(>=k)
    pad = np.concatenate(([1.0], pstar, [0.0]))
    return np.clip(pad[:-1] - pad[1:], 1e-12, None)


def grm_total_logpmf(z: float, disc: np.ndarray, thresholds, max_total: int) -> np.ndarray:
    """Exact total-score log-pmf at trait z by convolving the per-item category pmfs."""
    pmf = np.zeros(max_total + 1)
    pmf[0] = 1.0
    cur = 1
    for j in range(len(thresholds)):
        cp = grm_category_probs(z, float(disc[j]), thresholds[j])
        pmf = np.convolve(pmf[:cur], cp)
        cur = len(pmf)
    return np.log(np.clip(pmf, 1e-300, None))


def _grm_score_info(z: float, items, disc: np.ndarray, thresholds) -> tuple[float, float]:
    """Score and Fisher information of sum_j log P(item_j | trait z) at z (observed items only)."""
    score = 0.0
    info = 0.0
    for j, r in enumerate(items):
        if not np.isfinite(r):
            continue
        a = float(disc[j])
        b = np.asarray(thresholds[j], dtype=float)
        pstar = 1.0 / (1.0 + np.exp(-a * (z - b)))            # (K-1,) P(>=k)
        padp = np.concatenate(([1.0], pstar, [0.0]))
        probs = np.clip(padp[:-1] - padp[1:], 1e-12, None)    # (K,) P(=c)
        dpstar = a * pstar * (1.0 - pstar)                    # d/dz P(>=k)
        dpad = np.concatenate(([0.0], dpstar, [0.0]))
        dprob = dpad[:-1] - dpad[1:]                          # (K,) d/dz P(=c)
        score += dprob[int(r)] / probs[int(r)]
        info += float(np.sum(dprob ** 2 / probs))             # item Fisher information
    return float(score), float(info)


def _anchor_trait_mle(items, disc: np.ndarray, thresholds, iters: int = 6) -> float:
    """Per-anchor trait estimate from item responses by Fisher scoring (unit-info prior)."""
    z = 0.0
    for _ in range(iters):
        score, info = _grm_score_info(z, items, disc, thresholds)
        z = float(np.clip(z + score / (info + 1.0), -4.0, 4.0))
    return z


def irt_anchor_pseudo_observation(row, data) -> tuple[float, float] | None:
    """Map one observed item-response anchor to the calibrated raw latent scale.

    The point estimate is the unit-information-prior Fisher-scoring estimate on
    the frozen trait scale.  The returned standard deviation uses the local
    posterior information and is transformed through the frozen affine map.
    This supplies an identical, prespecified item-response measurement bridge
    to trajectory comparators that do not optimize the graded-response
    likelihood directly.
    """

    items = getattr(row, "irt_items", None)
    if items is None:
        return None
    item_values = np.asarray(items, dtype=float)
    if item_values.size == 0 or not np.isfinite(item_values).any():
        return None
    cfg = data.config
    disc = np.asarray(cfg.irt_item_discriminations, dtype=float)
    thresholds = [np.asarray(b, dtype=float) for b in cfg.irt_item_thresholds]
    if len(disc) != len(item_values) or len(thresholds) != len(item_values):
        return None
    trait = _anchor_trait_mle(item_values, disc, thresholds)
    _, information = _grm_score_info(trait, item_values, disc, thresholds)
    loc = float(cfg.irt_loc)
    scale = float(cfg.irt_scale) or 1.0
    raw_target = trait * scale + loc
    raw_sd = scale / np.sqrt(max(information + 1.0, 1e-6))
    return float(raw_target), float(raw_sd)


def init_irt_state(data, args):
    """Per-anchor IRT state from the FROZEN calibration on data.config, or None for Gaussian."""
    if str(getattr(args, "anchor_observation", "gaussian")).lower() != "irt":
        return None
    cfg = data.config
    thr = cfg.irt_item_thresholds
    if not thr:
        raise ValueError("anchor_observation='irt' but no calibrated instrument on the config")
    return {
        "loc": float(cfg.irt_loc),
        "scale": float(cfg.irt_scale) or 1.0,
        "disc": np.asarray(cfg.irt_item_discriminations, dtype=float),
        "thresholds": [np.asarray(b, dtype=float) for b in thr],
        "zbar": {},     # (pid, anchor, window_end) -> current trait linearization point
        "pid": None,
    }


def anchor_constraint(row, t_count: int, data, args, irt_state):
    """One observed-anchor normal-equation row: Gaussian, or exact-GRM Fisher-scored on the raw scale."""
    if not bool(getattr(row, "observed", False)):
        return None
    start = max(0, int(getattr(row, "window_start")))
    end = min(t_count - 1, int(getattr(row, "window_end")))
    if end < start:
        return None
    window = list(range(start, end + 1))
    anchor_name = getattr(row, "anchor")

    if irt_state is None:
        value = float(getattr(row, "value"))
        loading = float(getattr(row, "loading"))
        if not np.isfinite(value) or not np.isfinite(loading) or abs(loading) < 1e-8:
            return None
        spec = data.config.y1 if anchor_name == "Y1" else data.config.y2
        rho = data.config.rho_serial_y1 if anchor_name == "Y1" else data.config.rho_serial_y2
        return [(k, loading / len(window)) for k in window], value, marginal_anchor_sd(spec, rho)

    items = getattr(row, "irt_items")
    if items is None or all(not np.isfinite(x) for x in items):
        return None
    loc, scale = irt_state["loc"], irt_state["scale"]
    z0 = irt_state["zbar"].get((irt_state["pid"], anchor_name, end), 0.0)   # trait point
    score, info = _grm_score_info(z0, items, irt_state["disc"], irt_state["thresholds"])
    if info < 1e-3:
        return None                                           # uninformative anchor (items saturated)
    trait_target = z0 + score / info                          # Fisher-scoring (likelihood-only) step
    # Express on the raw latent scale (u = trait * scale + loc) so the row composes with
    # the transition and daily rows: mean(u over window) = trait_target * scale + loc.
    # Information is the anchor likelihood only; the trajectory prior supplies shrinkage.
    raw_target = trait_target * scale + loc
    raw_sd = float(scale / np.sqrt(info))
    return [(k, 1.0 / len(window)) for k in window], raw_target, raw_sd


def update_irt_zbar(irt_state, anchors_i, sol: np.ndarray, t_count: int) -> None:
    """Refresh per-anchor trait linearization points from the latest raw smoothed latent."""
    if irt_state is None:
        return
    loc, scale = irt_state["loc"], irt_state["scale"]
    for row in anchors_i.itertuples(index=False):
        if not bool(getattr(row, "observed", False)):
            continue
        items = getattr(row, "irt_items")
        if items is None or all(not np.isfinite(x) for x in items):
            continue
        start = max(0, int(getattr(row, "window_start")))
        end = min(t_count - 1, int(getattr(row, "window_end")))
        if end < start:
            continue
        raw_mean = float(np.mean(sol[start:end + 1]))
        irt_state["zbar"][(irt_state["pid"], getattr(row, "anchor"), end)] = (raw_mean - loc) / scale


def irt_holdout_metrics(data, preds, split: str = "validation") -> tuple[float, float]:
    """Held-out IRT anchor fit through the FIXED calibrated link.

    The predicted windowed latent is mapped to the calibrated trait scale, then the
    held-out anchor's item responses are scored exactly. Returns the mean per-item
    negative log-likelihood and the expected-total RMSE on the given split's observed
    anchors. This replaces the Gaussian val-anchor RMSE, which is undefined under IRT.
    """

    cfg = data.config
    if str(getattr(cfg, "anchor_observation", "gaussian")).lower() != "irt":
        return float("nan"), float("nan")
    loc = float(cfg.irt_loc)
    scale = float(cfg.irt_scale) or 1.0
    disc = np.asarray(cfg.irt_item_discriminations, dtype=float)
    thr = [np.asarray(b, dtype=float) for b in cfg.irt_item_thresholds]
    cats = np.arange(len(thr[0]) + 1) if thr else np.arange(1)
    ind = data.individuals
    split_ids = set(ind.loc[ind["split"] == split, "id"].astype(int)) if "split" in ind.columns else None
    zcol_d = "z_d_hat" if "z_d_hat" in preds.columns else "L_hat"
    zcol_p = "z_p_hat" if "z_p_hat" in preds.columns else "L_hat"
    pred_d = preds.groupby(["id", "t"])[zcol_d].mean()
    pred_p = preds.groupby(["id", "t"])[zcol_p].mean()

    nlls, tot_err = [], []
    use_measurement_targets = "measurement_eval" in data.anchors.columns
    for row in data.anchors.itertuples(index=False):
        eligible = (
            bool(getattr(row, "measurement_eval", False))
            if use_measurement_targets
            else bool(getattr(row, "observed", False))
        )
        if not eligible:
            continue
        items = getattr(row, "irt_items")
        total = getattr(row, "irt_total")
        if items is None or not np.isfinite(total):
            continue
        pid = int(getattr(row, "id"))
        if split_ids is not None and pid not in split_ids:
            continue
        ch_pred = pred_d if getattr(row, "anchor") == "Y1" else pred_p
        start = int(getattr(row, "window_start"))
        end = int(getattr(row, "window_end"))
        vals = [float(ch_pred.loc[(pid, t)]) for t in range(start, end + 1) if (pid, t) in ch_pred.index]
        if not vals:
            continue
        trait = (float(np.mean(vals)) - loc) / scale
        nll = 0.0
        etot = 0.0
        for j in range(len(thr)):
            probs = grm_category_probs(trait, float(disc[j]), thr[j])
            nll -= float(np.log(max(probs[int(items[j])], 1e-12)))
            etot += float(np.dot(cats, probs))
        nlls.append(nll / len(thr))
        tot_err.append((float(total) - etot) ** 2)
    item_nll = float(np.mean(nlls)) if nlls else float("nan")
    etot_rmse = float(np.sqrt(np.mean(tot_err))) if tot_err else float("nan")
    return item_nll, etot_rmse


def impute_by_time(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    vals = frame[cols].astype(float).interpolate(limit_direction="both")
    vals = vals.fillna(vals.mean()).fillna(0.0)
    return vals.to_numpy(dtype=float)


def observed_treatment_history_arrays(comp_i: pd.DataFrame, daily_i: pd.DataFrame, config):
    """Construct treatment-history features from inputs available to every model."""

    action = pd.to_numeric(comp_i["a"], errors="coerce").fillna(-1).to_numpy(dtype=int)
    treated = action >= 0
    recent = np.zeros(len(action), dtype=float)
    for index in range(len(action)):
        recent[index] = float(np.any(treated[max(0, index - 6) : index + 1]))

    burden_columns = [
        column
        for column in ("X18", "X19")
        if column in daily_i.columns
    ]
    if burden_columns:
        burden_values = daily_i[burden_columns].apply(pd.to_numeric, errors="coerce")
        observed_columns = [f"obs_{column}" for column in burden_columns]
        if all(column in daily_i.columns for column in observed_columns):
            observed = daily_i[observed_columns].astype(bool).to_numpy()
            values = burden_values.to_numpy(dtype=float)
            values[~observed] = np.nan
            burden = pd.DataFrame(values).mean(axis=1, skipna=True).ffill().fillna(0.0).to_numpy(dtype=float)
        else:
            burden = burden_values.mean(axis=1, skipna=True).ffill().fillna(0.0).to_numpy(dtype=float)
    else:
        burden = np.zeros(len(action), dtype=float)
    return recent, burden


def fit_daily_readout(
    data,
    train_ids: set[int],
    ridge: float = 1.0,
    *,
    anchor_name: str | None = None,
):
    """Train-only daily-feature readout for one latent channel."""

    x_cols = [f"X{j}" for j in range(data.config.p_daily)]
    obs_cols = [f"obs_X{j}" for j in range(data.config.p_daily)]
    input_cols = [f"input_X{j}" for j in range(data.config.p_daily)]
    daily_by = dict(tuple(data.daily.groupby("id", sort=True)))
    comp_by = dict(tuple(data.components.groupby("id", sort=True)))
    anc_by = dict(tuple(data.anchors.groupby("id", sort=True)))
    x_blocks, o_blocks, y_blocks = [], [], []

    # Under IRT the readout targets the GRM-implied latent (per-anchor trait MLE on
    # the calibrated scale), not the standardized total, so the comparator still uses
    # the daily covariates with the same observed inputs as BALL.
    _daily_irt = None
    if str(getattr(data.config, "anchor_observation", "gaussian")).lower() == "irt":
        _daily_irt = (
            float(data.config.irt_loc),
            float(data.config.irt_scale) or 1.0,
            np.asarray(data.config.irt_item_discriminations, dtype=float),
            [np.asarray(b, dtype=float) for b in data.config.irt_item_thresholds],
        )

    for pid in sorted(train_ids):
        comp_i = comp_by.get(pid)
        daily_i = daily_by.get(pid)
        anc_i = anc_by.get(pid)
        if comp_i is None or daily_i is None or anc_i is None:
            continue
        points_x, points_y = [], []
        for row in anc_i.itertuples(index=False):
            if not bool(getattr(row, "observed", False)):
                continue
            if anchor_name is not None and str(getattr(row, "anchor")) != str(anchor_name):
                continue
            if _daily_irt is not None:
                items = getattr(row, "irt_items")
                if items is None or all(not np.isfinite(x) for x in items):
                    continue
                loc, scale, disc, thr = _daily_irt
                target = _anchor_trait_mle(items, disc, thr) * scale + loc   # GRM-implied raw latent
            else:
                value = float(getattr(row, "value"))
                loading = float(getattr(row, "loading"))
                if not np.isfinite(value) or not np.isfinite(loading) or abs(loading) < 1e-8:
                    continue
                target = value / loading
            points_x.append(float(getattr(row, "window_end")))
            points_y.append(target)
        if len(points_x) < 2:
            continue
        order = np.argsort(points_x)
        days = comp_i.sort_values("t")["t"].to_numpy(dtype=float)
        target = np.interp(days, np.asarray(points_x)[order], np.asarray(points_y)[order])
        di = daily_i.sort_values("t").reset_index(drop=True)
        if not set(x_cols).issubset(di.columns):
            continue
        x = di[x_cols].to_numpy(dtype=float)
        mask_cols = input_cols if set(input_cols).issubset(di.columns) else obs_cols
        obs = (
            di[mask_cols].to_numpy(dtype=float)
            if set(mask_cols).issubset(di.columns)
            else np.isfinite(x).astype(float)
        )
        n = min(len(target), len(x))
        x_blocks.append(x[:n])
        o_blocks.append(obs[:n])
        y_blocks.append(target[:n])

    if not x_blocks:
        return 0.0, np.zeros(data.config.p_daily, dtype=float), np.zeros(data.config.p_daily, dtype=float), 1.0

    x = np.concatenate(x_blocks)
    obs = np.concatenate(o_blocks)
    y = np.concatenate(y_blocks)
    means = np.array([x[obs[:, j] > 0.5, j].mean() if (obs[:, j] > 0.5).any() else 0.0 for j in range(x.shape[1])])
    x_imp = np.where(obs > 0.5, x, means[None, :])
    design = np.column_stack([np.ones(len(x_imp)), x_imp])
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    resid = y - design @ coef
    sigma = max(float(np.std(resid)), 0.35)
    return float(coef[0]), coef[1:].astype(float), means.astype(float), sigma


def person_arrays(data, pid: int, daily_readout, *, anchor_name: str | None = None):
    comp_i = data.components[data.components["id"] == pid].sort_values("t").reset_index(drop=True)
    daily_i = data.daily[data.daily["id"] == pid].sort_values("t").reset_index(drop=True)
    anchors_i = data.anchors[data.anchors["id"] == pid].sort_values(["anchor", "t"]).reset_index(drop=True)
    if anchor_name is not None:
        anchors_i = anchors_i[anchors_i["anchor"].astype(str).eq(str(anchor_name))].copy()
    q = data.config.q
    b = impute_by_time(comp_i, [f"B{j}" for j in range(q)])
    x_cols = [f"X{j}" for j in range(data.config.p_daily)]
    obs_cols = [f"obs_X{j}" for j in range(data.config.p_daily)]
    input_cols = [f"input_X{j}" for j in range(data.config.p_daily)]
    x = daily_i[x_cols].to_numpy(dtype=float)
    mask_cols = input_cols if set(input_cols).issubset(daily_i.columns) else obs_cols
    obs = (
        daily_i[mask_cols].to_numpy(dtype=float)
        if set(mask_cols).issubset(daily_i.columns)
        else np.isfinite(x).astype(float)
    )
    intercept, x_beta, x_means, daily_sd = daily_readout
    x_imp = np.where(obs > 0.5, x, x_means[None, :])
    daily_pred = intercept + x_imp @ x_beta
    any_daily = (obs > 0.5).any(axis=1)
    action = comp_i["a"].to_numpy(dtype=int)
    recent, burden = observed_treatment_history_arrays(comp_i, daily_i, data.config)
    return comp_i, anchors_i, b, daily_pred, any_daily, daily_sd, action, recent, burden


def causal_encoder_action_array(comp_i: pd.DataFrame) -> np.ndarray:
    """Treatment category available before each session-level prediction."""

    if "encoder_a" in comp_i.columns:
        return pd.to_numeric(comp_i["encoder_a"], errors="coerce").fillna(-1).to_numpy(dtype=int)
    current = pd.to_numeric(comp_i["a"], errors="coerce").fillna(-1).to_numpy(dtype=int)
    previous = np.full(len(current), -1, dtype=int)
    if len(current) > 1:
        previous[1:] = current[:-1]
    return previous


def transition_features(
    b: np.ndarray,
    action: np.ndarray,
    recent: np.ndarray,
    burden: np.ndarray,
    n_actions: int,
    *,
    basis: str = "linear",
    subtype: int | None = None,
    n_subtypes: int = 3,
) -> np.ndarray:
    t = len(b)
    rows = []
    for k in range(t - 1):
        feat = [1.0]
        b_k = b[k]
        feat.extend(b_k.tolist())
        a = int(action[k])
        for j in range(n_actions):
            feat.append(1.0 if a == j else 0.0)
        feat.extend([float(recent[k]), float(burden[k])])
        if basis in {"interaction", "full"}:
            feat.extend((b_k * float(recent[k])).tolist())
            feat.extend((b_k * float(burden[k])).tolist())
        if basis in {"nonlinear", "full"}:
            feat.extend((b_k**2).tolist())
            feat.extend(np.tanh(b_k).tolist())
        if basis in {"heterogeneous", "full"}:
            subtype_i = int(subtype) if subtype is not None else 0
            for s in range(int(n_subtypes)):
                feat.extend((b_k * (1.0 if subtype_i == s else 0.0)).tolist())
        rows.append(feat)
    return np.asarray(rows, dtype=float)


def solve_person(data, pid: int, theta: np.ndarray, daily_readout, args, irt_state=None):
    comp_i, anchors_i, b, daily_pred, any_daily, daily_sd, action, recent, burden = person_arrays(data, pid, daily_readout)
    t_count = len(comp_i)
    if irt_state is not None:
        irt_state["pid"] = pid
    basis = getattr(args, "s0_basis", "linear")
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
    trans_mean = feats @ theta

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
        add(
            [(k, 1.0), (k - 1, -1.0)],
            float(trans_mean[k - 1]) * gap,
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
    return pred, b, action, recent, burden, next_gaps


def _classical_gap_factors(rho: float, gaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Persistence and accumulated-drift factors for unequal elapsed days."""

    gaps = np.asarray(gaps, dtype=float)
    persistence = np.power(float(rho), gaps)
    if abs(float(rho) - 1.0) < 1e-8:
        drift = gaps.copy()
    else:
        drift = (1.0 - persistence) / (1.0 - float(rho))
    return persistence, drift


def _classical_pattern_strata(data, n_strata: int) -> dict[int, int]:
    ids = sorted(data.individuals["id"].astype(int).tolist())
    counts = data.anchors.loc[data.anchors["observed"].astype(bool)].groupby("id").size()
    values = np.asarray([float(counts.get(pid, 0.0)) for pid in ids], dtype=float)
    n_strata = max(1, int(n_strata))
    if n_strata == 1 or len(np.unique(values)) <= 1:
        return {pid: 0 for pid in ids}
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_strata + 1)[1:-1])
    return {
        pid: int(np.searchsorted(edges, values[index], side="right"))
        for index, pid in enumerate(ids)
    }


def _solve_two_channel_person(
    data,
    pid: int,
    theta: np.ndarray,
    daily_readout,
    args,
    *,
    anchor_name: str,
    persistence: float,
    innovation_sd: float,
    stratum: int,
    n_strata: int,
    irt_state=None,
):
    comp_i, anchors_i, b, daily_pred, any_daily, daily_sd, action, recent, burden = person_arrays(
        data, pid, daily_readout, anchor_name=anchor_name
    )
    t_count = len(comp_i)
    if irt_state is not None:
        irt_state["pid"] = pid
    basis = getattr(args, "s0_basis", "linear")
    subtype = int(comp_i["subtype"].iloc[0]) if "subtype" in comp_i.columns and len(comp_i) else 0
    base_features = transition_features(
        b,
        action,
        recent,
        burden,
        data.config.n_treatment_types,
        basis=basis,
        subtype=subtype,
        n_subtypes=data.config.n_subtypes,
    )
    transition_mean = base_features @ theta[: base_features.shape[1]]
    if n_strata > 0:
        transition_mean = transition_mean + float(theta[base_features.shape[1] + int(stratum)])

    rows: list[np.ndarray] = []
    targets: list[float] = []

    def add(coefficients: list[tuple[int, float]], target: float, sd: float) -> None:
        row = np.zeros(t_count, dtype=float)
        for index, value in coefficients:
            row[index] = value / max(float(sd), 1e-8)
        rows.append(row)
        targets.append(float(target) / max(float(sd), 1e-8))

    add([(0, 1.0)], 0.0, args.prior_sd)
    for row in anchors_i.itertuples(index=False):
        constraint = anchor_constraint(row, t_count, data, args, irt_state)
        if constraint is not None:
            add(*constraint)
    for index, observed in enumerate(any_daily):
        if observed and np.isfinite(daily_pred[index]):
            add([(index, 1.0)], float(daily_pred[index]), daily_sd)

    gaps = (
        pd.to_numeric(comp_i["dt"], errors="coerce").fillna(1.0).clip(lower=0.0).to_numpy(dtype=float)
        if "dt" in comp_i.columns
        else np.ones(t_count, dtype=float)
    )
    gap_persistence, drift_scale = _classical_gap_factors(persistence, gaps)
    innovation_multiplier = np.sqrt(
        np.maximum(
            (1.0 - np.power(float(persistence), 2.0 * gaps))
            / max(1.0 - float(persistence) ** 2, 1e-8),
            0.0,
        )
    )
    for index in range(1, t_count):
        if gaps[index - 1] <= 0:
            continue
        add(
            [(index, 1.0), (index - 1, -float(gap_persistence[index - 1]))],
            float(transition_mean[index - 1]) * float(drift_scale[index - 1]),
            float(innovation_sd) * float(innovation_multiplier[index - 1]),
        )
    design = np.vstack(rows)
    target = np.asarray(targets, dtype=float)
    solution = np.linalg.lstsq(design, target, rcond=None)[0]
    update_irt_zbar(irt_state, anchors_i, solution, t_count)
    return solution, b, action, recent, burden, gaps, base_features


def fit_two_channel_direct_map(data, args, *, pattern_mixture: bool = False):
    """Fit two channel-specific trajectories with one shared RDoC coefficient."""

    train_ids = set(data.individuals.loc[data.individuals["split"] == "train", "id"].astype(int))
    ids = sorted(data.individuals["id"].astype(int).tolist())
    anchor_names = ("Y1", "Y2")
    readouts = {
        name: fit_daily_readout(data, train_ids, ridge=args.daily_ridge, anchor_name=name)
        for name in anchor_names
    }
    q = int(data.config.q)
    template_features = transition_features(
        np.zeros((2, q), dtype=float),
        np.zeros(2, dtype=int),
        np.zeros(2, dtype=float),
        np.zeros(2, dtype=float),
        data.config.n_treatment_types,
        basis=getattr(args, "s0_basis", "linear"),
        subtype=0,
        n_subtypes=data.config.n_subtypes,
    )
    base_count = int(template_features.shape[1])
    n_strata = max(1, int(getattr(args, "markov_strata", 1))) if pattern_mixture else 0
    strata = _classical_pattern_strata(data, n_strata) if n_strata else {pid: 0 for pid in ids}
    theta = [np.zeros(base_count + n_strata), np.zeros(base_count + n_strata)]
    persistence_grid = tuple(
        float(value)
        for value in getattr(args, "classical_persistence_grid", (0.1, 0.3, 0.6, 0.9))
    )
    if any(value <= 0.0 or value >= 1.0 for value in persistence_grid):
        raise ValueError("Classical persistence candidates must lie strictly between zero and one.")
    persistence = [0.6, 0.6]
    innovation_sd = [float(args.transition_sd), float(args.transition_sd)]
    irt_states = [init_irt_state(data, args), init_irt_state(data, args)]
    fitted: dict[tuple[int, int], tuple] = {}

    for _ in range(max(1, int(args.iters))):
        fitted.clear()
        for channel_index, anchor_name in enumerate(anchor_names):
            for pid in ids:
                fitted[(channel_index, pid)] = _solve_two_channel_person(
                    data,
                    pid,
                    theta[channel_index],
                    readouts[anchor_name],
                    args,
                    anchor_name=anchor_name,
                    persistence=persistence[channel_index],
                    innovation_sd=innovation_sd[channel_index],
                    stratum=strata[pid],
                    n_strata=n_strata,
                    irt_state=irt_states[channel_index],
                )

        best = None
        non_beta_indices = np.asarray(
            [index for index in range(base_count + n_strata) if not 1 <= index < 1 + q],
            dtype=int,
        )
        for rho_d in persistence_grid:
            for rho_p in persistence_grid:
                channel_designs = []
                channel_targets = []
                for channel_index, rho in enumerate((rho_d, rho_p)):
                    design_blocks, target_blocks = [], []
                    for pid in ids:
                        if pid not in train_ids:
                            continue
                        solution, _, _, _, _, gaps, features = fitted[(channel_index, pid)]
                        count = min(len(features), len(solution) - 1)
                        if count <= 0:
                            continue
                        gaps_use = np.asarray(gaps[:count], dtype=float)
                        valid = gaps_use > 0
                        if not valid.any():
                            continue
                        gap_persistence, drift = _classical_gap_factors(rho, gaps_use)
                        outcome = (
                            solution[1 : count + 1]
                            - gap_persistence * solution[:count]
                        ) / np.maximum(drift, 1e-8)
                        channel_features = features[:count]
                        if n_strata:
                            pattern = np.zeros((count, n_strata), dtype=float)
                            pattern[:, strata[pid]] = 1.0
                            channel_features = np.column_stack([channel_features, pattern])
                        design_blocks.append(channel_features[valid])
                        target_blocks.append(outcome[valid])
                    channel_designs.append(np.vstack(design_blocks))
                    channel_targets.append(np.concatenate(target_blocks))

                rows = []
                outcomes = []
                for channel_index in range(2):
                    base = channel_designs[channel_index]
                    shared = base[:, 1 : 1 + q]
                    own = base[:, non_beta_indices]
                    zeros = np.zeros_like(own)
                    rows.append(
                        np.column_stack(
                            [shared, own if channel_index == 0 else zeros, zeros if channel_index == 0 else own]
                        )
                    )
                    outcomes.append(channel_targets[channel_index])
                design = np.vstack(rows)
                outcome = np.concatenate(outcomes)
                penalty = float(args.beta_ridge) * np.eye(design.shape[1])
                for position in (q, q + len(non_beta_indices)):
                    penalty[position, position] = 0.0
                coefficients = np.linalg.solve(
                    design.T @ design + penalty,
                    design.T @ outcome,
                )
                residual = outcome - design @ coefficients
                objective = float(np.mean(residual ** 2))
                if best is None or objective < best[0]:
                    best = (objective, (rho_d, rho_p), coefficients, channel_designs, channel_targets)
        if best is None:
            break
        _, selected_rho, coefficients, channel_designs, channel_targets = best
        persistence = [float(selected_rho[0]), float(selected_rho[1])]
        shared_beta = coefficients[:q]
        for channel_index in range(2):
            start = q + channel_index * len(non_beta_indices)
            channel_theta = np.zeros(base_count + n_strata, dtype=float)
            channel_theta[1 : 1 + q] = shared_beta
            channel_theta[non_beta_indices] = coefficients[start : start + len(non_beta_indices)]
            theta[channel_index] = channel_theta
            residual = channel_targets[channel_index] - channel_designs[channel_index] @ channel_theta
            innovation_sd[channel_index] = max(float(np.std(residual, ddof=1)), 0.05)

    final_frames = []
    for pid in ids:
        channel_values = []
        time_values = None
        for channel_index, anchor_name in enumerate(anchor_names):
            solution, *_ = _solve_two_channel_person(
                data,
                pid,
                theta[channel_index],
                readouts[anchor_name],
                args,
                anchor_name=anchor_name,
                persistence=persistence[channel_index],
                innovation_sd=innovation_sd[channel_index],
                stratum=strata[pid],
                n_strata=n_strata,
                irt_state=irt_states[channel_index],
            )
            channel_values.append(solution)
            if time_values is None:
                time_values = data.components[data.components["id"].eq(pid)].sort_values("t")["t"].to_numpy(dtype=int)
        final_frames.append(pd.DataFrame({
            "id": int(pid),
            "t": time_values,
            "z_d_hat": channel_values[0],
            "z_p_hat": channel_values[1],
            "L_hat": 0.5 * (channel_values[0] + channel_values[1]),
        }))
    predictions = pd.concat(final_frames, ignore_index=True)
    predictions.attrs["channel_persistence"] = {
        "depression": float(persistence[0]), "anxiety": float(persistence[1])
    }
    predictions.attrs["channel_innovation_sd"] = {
        "depression": float(innovation_sd[0]), "anxiety": float(innovation_sd[1])
    }
    predictions.attrs["shared_beta"] = theta[0][1 : 1 + q].copy()
    return predictions, theta[0]


def fit_direct_map(data, args):
    return fit_two_channel_direct_map(data, args, pattern_mixture=False)


def run_one(args, seed: int, share: float) -> dict:
    data = H.generate_dataset(
        dataclasses.replace(
            H.SimulationConfig(),
            seed=seed,
            n=args.n,
            t=args.t,
            slow_fraction=args.slow_fraction,
            delta_ar=tuple([float(args.delta_ar)] * H.N_SUBTYPES),
            rdoc_drift_share=float(share),
            rdoc_drift_active_dims=int(args.rdoc_active_dims),
            rdoc_drift_beta_seed=int(args.rdoc_beta_seed),
            rdoc_drift_min_abs=float(args.rdoc_min_abs),
        )
    )
    preds, theta = fit_direct_map(data, args)
    q = data.config.q
    beta_hat = theta[1 : 1 + q]
    beta_true = np.asarray(data.metadata.get("rdoc_drift_beta_unit", []), dtype=float)
    return {
        "share": float(share),
        "seed": int(seed),
        "beta_cosine": cosine(beta_hat, beta_true),
        "beta_abs_cosine": abs(cosine(beta_hat, beta_true)) if np.isfinite(cosine(beta_hat, beta_true)) else float("nan"),
        "beta_topk_f1": topk_f1(beta_hat, beta_true, args.rdoc_active_dims),
        "beta_hat_norm": float(np.linalg.norm(beta_hat)),
        "latent_rmse": H.latent_rmse(preds, data.components, data.individuals, H.EVAL_SPLIT),
        "val_anchor_rmse": H.val_anchor_rmse(data, preds),
        **{f"beta_hat_{j}": float(beta_hat[j]) for j in range(q)},
        **{f"beta_true_{j}": float(beta_true[j]) for j in range(q)},
    }


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for share, group in df.groupby("share", sort=True):
        row = {"share": float(share), "method": "classical_direct_rdoc_map", "n_seeds": int(group["seed"].nunique())}
        for metric in ["beta_cosine", "beta_abs_cosine", "beta_topk_f1", "beta_hat_norm", "latent_rmse", "val_anchor_rmse"]:
            vals = group[metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{metric}_mcse"] = mcse(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair classical direct-RDoC comparator.")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--t", type=int, default=84)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1729, 2027, 2028])
    parser.add_argument("--shares", type=float, nargs="+", default=[0.10, 0.25])
    parser.add_argument("--slow-fraction", type=float, default=0.75)
    parser.add_argument("--delta-ar", type=float, default=0.3)
    parser.add_argument("--rdoc-active-dims", type=int, default=3)
    parser.add_argument("--rdoc-beta-seed", type=int, default=31011)
    parser.add_argument("--rdoc-min-abs", type=float, default=0.35)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--transition-sd", type=float, default=0.75)
    parser.add_argument("--prior-sd", type=float, default=5.0)
    parser.add_argument("--daily-ridge", type=float, default=1.0)
    parser.add_argument("--beta-ridge", type=float, default=10.0)
    parser.add_argument("--s0-basis", choices=["linear", "interaction", "nonlinear", "heterogeneous", "full"], default="linear")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = Path(args.out) if args.out else H.REPO_ROOT / "validation" / "outputs" / f"direct_rdoc_fair_comparator_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for share in args.shares:
        for seed in args.seeds:
            print(f"share={share:.2f} seed={seed} classical_direct_rdoc_map")
            rows.append(run_one(args, seed, share))
    per = pd.DataFrame(rows)
    agg = aggregate(per)
    per.to_csv(out_dir / "per_run.csv", index=False)
    agg.to_csv(out_dir / "aggregate.csv", index=False)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "command_line": " ".join(sys.argv),
                "ball_py_sha256": H.file_sha256(H.BALL_PATH),
                "args": vars(args),
                "note": "Fair old-school direct-RDoC comparator with explicit B_t beta drift.",
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\noutput -> {out_dir}")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
