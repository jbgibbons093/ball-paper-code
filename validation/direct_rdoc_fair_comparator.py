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
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ball_validation_harness as H  # noqa: E402

_model_utils = sys.modules["simulations.src.model_utils"]
marginal_anchor_sd = _model_utils.marginal_anchor_sd


def _write_classical_training_progress(args, payload: dict) -> None:
    """Write an append-only convergence trace and a one-record heartbeat."""

    directory_value = getattr(args, "classical_diagnostics_dir", None)
    if not directory_value:
        return
    directory = Path(directory_value)
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "process_id": int(os.getpid()),
        "status": "running",
        **payload,
    }
    try:
        with (directory / "training_diagnostics.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(record, default=str) + "\n")
            handle.flush()
    except OSError as error:
        print(
            f"classical convergence trace could not be appended ({error!r})",
            flush=True,
        )
    destination = directory / "live_training_state.json"
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    for attempt in range(40):
        try:
            os.replace(temporary, destination)
            break
        except PermissionError as error:
            if attempt == 39:
                print(
                    f"classical training heartbeat could not be refreshed "
                    f"({error!r})",
                    flush=True,
                )
                try:
                    temporary.unlink()
                except OSError:
                    pass
                break
            time.sleep(min(0.05 * (attempt + 1), 0.5))


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
        anchor_weight = float(getattr(args, "classical_anchor_weight", 1.0))
        if anchor_weight <= 0.0:
            raise ValueError("The classical questionnaire weight must be positive.")
        observation_sd = marginal_anchor_sd(spec, rho) / math.sqrt(anchor_weight)
        return [(k, loading / len(window)) for k in window], value, observation_sd

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
    anchor_weight = float(getattr(args, "classical_anchor_weight", 1.0))
    if anchor_weight <= 0.0:
        raise ValueError("The classical questionnaire weight must be positive.")
    # Multiplying the item log likelihood by the same prespecified weight used
    # by BALL multiplies Fisher information by that weight.  Dividing the local
    # Gaussian standard deviation by its square root is the corresponding
    # iteratively reweighted least-squares update and leaves the frozen item
    # parameters unchanged.
    raw_sd = float(scale / np.sqrt(info * anchor_weight))
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
    target_column = (
        "measurement_selection"
        if split == "validation" and "measurement_selection" in data.anchors.columns
        else "measurement_eval"
    )
    use_measurement_targets = target_column in data.anchors.columns
    for row in data.anchors.itertuples(index=False):
        eligible = (
            bool(getattr(row, target_column, False))
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
    vals = frame[cols].astype(float).ffill().fillna(0.0)
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
        zeros = np.zeros(data.config.p_daily, dtype=float)
        return 0.0, zeros.copy(), zeros.copy(), zeros.copy(), 1.0

    x = np.concatenate(x_blocks)
    obs = np.concatenate(o_blocks)
    y = np.concatenate(y_blocks)
    means = np.array([x[obs[:, j] > 0.5, j].mean() if (obs[:, j] > 0.5).any() else 0.0 for j in range(x.shape[1])])
    x_imp = np.where(obs > 0.5, x, means[None, :])
    # Availability indicators are observed inputs in their own right.  They are
    # included beside the imputed values so the classical comparator receives
    # the same informative missingness signal supplied to the neural and
    # Gaussian-process comparators in the missingness stress condition.
    design = np.column_stack([np.ones(len(x_imp)), x_imp, obs])
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    resid = y - design @ coef
    sigma = max(float(np.std(resid)), 0.35)
    feature_count = x_imp.shape[1]
    return (
        float(coef[0]),
        coef[1 : 1 + feature_count].astype(float),
        coef[1 + feature_count :].astype(float),
        means.astype(float),
        sigma,
    )


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
    intercept, x_beta, availability_beta, x_means, daily_sd = daily_readout
    x_imp = np.where(obs > 0.5, x, x_means[None, :])
    daily_pred = intercept + x_imp @ x_beta + obs @ availability_beta
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
    rho = np.asarray(rho, dtype=float)
    persistence = np.power(rho, gaps)
    drift = np.divide(1.0-persistence, 1.0-rho,
        out=np.asarray(gaps, dtype=float).copy(), where=np.abs(1.0-rho)>=1e-8)
    return persistence, drift


def _classical_pattern_strata(
    data,
    n_strata: int,
    *,
    fit_ids: set[int] | None = None,
) -> dict[int, int]:
    from prospective_simulation_filter import history_strata
    if fit_ids is None:
        raise ValueError("History strata require explicit development patients.")
    return history_strata(data, n_strata, fit_ids)


def _classical_measurement_score(
    data,
    predictions: pd.DataFrame,
    *,
    target_column: str,
    anchor_name: str | None = None,
    strata: dict[int, int] | None = None,
    stratum: int | None = None,
) -> float:
    """Patient-balanced questionnaire RMSE for a prespecified target partition."""

    if target_column not in data.anchors.columns:
        return float("nan")
    targets = data.anchors.loc[data.anchors[target_column].astype(bool)].copy()
    if anchor_name is not None:
        targets = targets.loc[targets["anchor"].astype(str).eq(str(anchor_name))]
    if strata is not None and stratum is not None:
        from prospective_simulation_filter import event_strata
        targets = targets.loc[event_strata(targets, strata).eq(int(stratum))]
    if targets.empty:
        return float("nan")

    prediction_columns = {
        "Y1": "z_d_hat" if "z_d_hat" in predictions.columns else "L_hat",
        "Y2": "z_p_hat" if "z_p_hat" in predictions.columns else "L_hat",
    }
    indexed = {
        column: {
            int(pid): frame.set_index("t")[column]
            for pid, frame in predictions[["id", "t", column]].groupby("id", sort=False)
        }
        for column in set(prediction_columns.values())
    }
    use_irt = str(getattr(data.config, "anchor_observation", "gaussian")).lower() == "irt"
    squared_errors: list[tuple[int, float]] = []
    for row in targets.itertuples(index=False):
        instrument = str(getattr(row, "anchor"))
        column = prediction_columns[instrument]
        pid = int(getattr(row, "id"))
        series = indexed[column].get(pid)
        if series is None:
            continue
        start = int(getattr(row, "window_start"))
        end = int(getattr(row, "window_end"))
        window = series.loc[(series.index >= start) & (series.index <= end)]
        if window.empty:
            continue
        latent_mean = float(window.mean())
        if use_irt:
            total = float(getattr(row, "irt_total", np.nan))
            if not np.isfinite(total):
                continue
            loc = float(data.config.irt_loc)
            scale = float(data.config.irt_scale) or 1.0
            trait = (latent_mean - loc) / scale
            expected_total = 0.0
            for discrimination, thresholds in zip(
                data.config.irt_item_discriminations,
                data.config.irt_item_thresholds,
            ):
                probabilities = grm_category_probs(
                    trait,
                    float(discrimination),
                    np.asarray(thresholds, dtype=float),
                )
                expected_total += float(
                    np.dot(np.arange(len(probabilities), dtype=float), probabilities)
                )
            residual = total - expected_total
        else:
            value = float(getattr(row, "value", np.nan))
            loading = float(getattr(row, "loading", np.nan))
            intercept = float(getattr(row, "intercept", 0.0))
            if not np.isfinite(value) or not np.isfinite(loading):
                continue
            residual = value - (intercept + loading * latent_mean)
        squared_errors.append((pid, float(residual) ** 2))
    if not squared_errors:
        return float("nan")
    errors = pd.DataFrame(squared_errors, columns=["id", "squared_error"])
    patient_mse = errors.groupby("id", sort=False)["squared_error"].mean()
    return float(np.sqrt(patient_mse.mean()))


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
        groups = np.broadcast_to(np.asarray(stratum, int), (t_count,))[1:]
        transition_mean = transition_mean + theta[base_features.shape[1] + groups]

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
    rho = np.broadcast_to(np.asarray(persistence, float), (t_count,))
    # Transition k -> k+1 uses the history available at k+1.
    rho = np.r_[rho[1:], rho[-1]]
    gap_persistence, drift_scale = _classical_gap_factors(rho, gaps)
    innovation_multiplier = np.sqrt(
        np.maximum(
            (1.0 - np.power(rho, 2.0 * gaps))
            / np.maximum(1.0 - rho ** 2, 1e-8),
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
    precision = design.T @ design
    right_hand_side = design.T @ target
    try:
        solution = np.linalg.solve(precision, right_hand_side)
        posterior_covariance = np.linalg.solve(
            precision,
            np.eye(t_count, dtype=float),
        )
    except np.linalg.LinAlgError:
        posterior_covariance = np.linalg.pinv(precision, rcond=1e-10)
        solution = posterior_covariance @ right_hand_side
    posterior_variance = np.clip(
        np.diag(posterior_covariance).astype(float),
        0.0,
        None,
    )
    adjacent_posterior_covariance = np.diag(
        posterior_covariance,
        k=1,
    ).astype(float)
    update_irt_zbar(irt_state, anchors_i, solution, t_count)
    return (
        solution,
        b,
        action,
        recent,
        burden,
        gaps,
        base_features,
        posterior_variance,
        adjacent_posterior_covariance,
    )


def fit_two_channel_direct_map(data, args, *, pattern_mixture: bool = False):
    """Fit two channel-specific trajectories with one shared RDoC coefficient.

    Transition coefficients are estimated from training patients.  Daily
    persistence is selected by patient-balanced questionnaire prediction in the
    validation patients, whose selection questionnaires remain hidden from the
    trajectory fit.  The pattern-mixture comparator assigns a separate
    persistence value and transition intercept to each questionnaire-frequency
    stratum and latent channel.
    """

    train_ids = set(
        data.individuals.loc[data.individuals["split"] == "train", "id"].astype(int)
    )
    selection_ids = set(
        data.individuals.loc[data.individuals["split"] == "validation", "id"].astype(int)
    )
    if not train_ids or not selection_ids:
        raise ValueError("Classical fitting requires separate training and validation patients.")
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
    persistence_strata = n_strata if n_strata else 1
    strata = (
        _classical_pattern_strata(data, n_strata, fit_ids=train_ids)
        if n_strata
        else {pid: 0 for pid in ids}
    )
    theta = [np.zeros(base_count + n_strata), np.zeros(base_count + n_strata)]
    persistence_grid = tuple(
        sorted(
            set(
                float(value)
                for value in getattr(
                    args,
                    "classical_persistence_grid",
                    (0.1, 0.3, 0.6, 0.9),
                )
            )
        )
    )
    if not persistence_grid or any(value <= 0.0 or value >= 1.0 for value in persistence_grid):
        raise ValueError("Classical persistence candidates must lie strictly between zero and one.")
    initial_persistence = min(persistence_grid, key=lambda value: (abs(value - 0.6), value))
    persistence = np.full((2, persistence_strata), float(initial_persistence), dtype=float)
    innovation_floor = float(getattr(args, "classical_innovation_floor", 0.05))
    innovation_sd = np.full(2, float(args.transition_sd), dtype=float)
    maximum_iterations = max(1, int(args.iters))
    minimum_iterations = min(
        maximum_iterations,
        max(1, int(getattr(args, "classical_min_iterations", 3))),
    )
    convergence_tolerance = float(getattr(args, "classical_convergence_tol", 1e-4))
    minimum_stratum_selection_patients = max(
        1,
        int(getattr(args, "classical_min_stratum_validation_patients", 3)),
    )
    candidate_trace: list[dict[str, float | int | str]] = []
    fit_counter = 0
    non_beta_indices = np.asarray(
        [index for index in range(base_count + n_strata) if not 1 <= index < 1 + q],
        dtype=int,
    )

    def estimate_parameters(fitted_state, candidate_persistence):
        channel_designs: list[np.ndarray] = []
        channel_targets: list[np.ndarray] = []
        channel_transition_moments: list[list[dict[str, np.ndarray]]] = []
        for channel_index in range(2):
            design_blocks: list[np.ndarray] = []
            target_blocks: list[np.ndarray] = []
            transition_moments: list[dict[str, np.ndarray]] = []
            for pid in sorted(train_ids):
                (
                    solution,
                    _,
                    _,
                    _,
                    _,
                    gaps,
                    features,
                    posterior_variance,
                    adjacent_posterior_covariance,
                ) = fitted_state[(channel_index, pid)]
                count = min(len(features), len(solution) - 1)
                if count <= 0:
                    continue
                gaps_use = np.asarray(gaps[:count], dtype=float)
                valid = gaps_use > 0
                if not valid.any():
                    continue
                groups = np.broadcast_to(np.asarray(strata[pid], int), (len(solution),))[1:count+1]
                rho = candidate_persistence[channel_index, groups]
                gap_persistence, drift = _classical_gap_factors(rho, gaps_use)
                raw_difference = (
                    solution[1 : count + 1]
                    - gap_persistence * solution[:count]
                )
                outcome = raw_difference / np.maximum(drift, 1e-8)
                posterior_residual_variance = np.maximum(
                    posterior_variance[1 : count + 1]
                    + np.square(gap_persistence) * posterior_variance[:count]
                    - 2.0
                    * gap_persistence
                    * adjacent_posterior_covariance[:count],
                    0.0,
                )
                innovation_multiplier = np.sqrt(
                    np.maximum(
                        (1.0 - np.power(rho, 2.0 * gaps_use))
                        / np.maximum(1.0 - rho**2, 1e-8),
                        0.0,
                    )
                )
                channel_features = features[:count]
                if n_strata:
                    pattern = np.zeros((count, n_strata), dtype=float)
                    pattern[np.arange(count), groups] = 1.0
                    channel_features = np.column_stack([channel_features, pattern])
                design_blocks.append(channel_features[valid])
                target_blocks.append(outcome[valid])
                transition_moments.append(
                    {
                        "raw_difference": raw_difference[valid],
                        "drift": drift[valid],
                        "innovation_multiplier": innovation_multiplier[valid],
                        "posterior_residual_variance": posterior_residual_variance[valid],
                        "features": channel_features[valid],
                    }
                )
            if not design_blocks:
                raise RuntimeError("No positive-gap training transitions were available.")
            channel_designs.append(np.vstack(design_blocks))
            channel_targets.append(np.concatenate(target_blocks))
            channel_transition_moments.append(transition_moments)

        stacked_designs: list[np.ndarray] = []
        stacked_targets: list[np.ndarray] = []
        for channel_index in range(2):
            channel_design = channel_designs[channel_index]
            shared = channel_design[:, 1 : 1 + q]
            own = channel_design[:, non_beta_indices]
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
            stacked_targets.append(channel_targets[channel_index])
        design = np.vstack(stacked_designs)
        outcome = np.concatenate(stacked_targets)
        penalty = float(args.beta_ridge) * np.eye(design.shape[1])
        for channel_index in range(2):
            own_start = q + channel_index * len(non_beta_indices)
            for offset, original_index in enumerate(non_beta_indices):
                position = own_start + offset
                if original_index == 0:
                    penalty[position, position] = 0.0
                elif n_strata and original_index >= base_count:
                    penalty[position, position] = float(
                        getattr(args, "markov_ridge", args.beta_ridge)
                    )
        normal = design.T @ design + penalty
        rhs = design.T @ outcome
        try:
            coefficients = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(normal, rhs, rcond=None)[0]

        candidate_theta: list[np.ndarray] = []
        candidate_innovation = np.zeros(2, dtype=float)
        shared_beta = coefficients[:q]
        for channel_index in range(2):
            start = q + channel_index * len(non_beta_indices)
            channel_theta = np.zeros(base_count + n_strata, dtype=float)
            channel_theta[1 : 1 + q] = shared_beta
            channel_theta[non_beta_indices] = coefficients[
                start : start + len(non_beta_indices)
            ]
            candidate_theta.append(channel_theta)
            expected_base_innovations: list[np.ndarray] = []
            for moment in channel_transition_moments[channel_index]:
                predicted_drift = moment["features"] @ channel_theta
                residual = (
                    moment["raw_difference"]
                    - moment["drift"] * predicted_drift
                )
                denominator = np.square(moment["innovation_multiplier"])
                eligible = denominator > 1e-10
                if eligible.any():
                    expected_base_innovations.append(
                        (
                            np.square(residual[eligible])
                            + moment["posterior_residual_variance"][eligible]
                        )
                        / denominator[eligible]
                    )
            expected_variance = (
                float(np.mean(np.concatenate(expected_base_innovations)))
                if expected_base_innovations
                else 0.0
            )
            candidate_innovation[channel_index] = max(
                math.sqrt(max(expected_variance, 0.0)),
                innovation_floor,
            )
        residual = outcome - design @ coefficients
        objective = float(np.mean(np.square(residual)))
        return candidate_theta, candidate_innovation, objective

    def solve_ids(
        patient_ids,
        candidate_theta,
        candidate_persistence,
        candidate_innovation,
        *,
        irt_states=None,
        prospective=False,
    ):
        states = irt_states or [init_irt_state(data, args), init_irt_state(data, args)]
        fitted_state: dict[tuple[int, int], tuple] = {}
        frames: list[pd.DataFrame] = []
        for pid in sorted(patient_ids):
            channel_values: list[np.ndarray] = []
            time_values = None
            for channel_index, anchor_name in enumerate(anchor_names):
                from prospective_simulation_filter import filter_person
                causal = prospective or pid not in train_ids
                solver = filter_person if causal else _solve_two_channel_person
                extra = {"helpers": sys.modules[__name__]} if causal else {"irt_state": states[channel_index]}
                solution_and_inputs = solver(
                    data,
                    pid,
                    candidate_theta[channel_index],
                    readouts[anchor_name],
                    args,
                    anchor_name=anchor_name,
                    persistence=candidate_persistence[channel_index, strata[pid]],
                    innovation_sd=float(candidate_innovation[channel_index]),
                    stratum=strata[pid],
                    n_strata=n_strata,
                    **extra,
                )
                fitted_state[(channel_index, pid)] = solution_and_inputs
                solution = solution_and_inputs[0]
                channel_values.append(solution)
                if time_values is None:
                    time_values = (
                        data.components.loc[data.components["id"].eq(pid)]
                        .sort_values("t")["t"]
                        .to_numpy(dtype=int)
                    )
            frames.append(
                pd.DataFrame(
                    {
                        "id": int(pid),
                        "t": time_values,
                        "z_d_hat": channel_values[0],
                        "z_p_hat": channel_values[1],
                        "L_hat": 0.5 * (channel_values[0] + channel_values[1]),
                    }
                )
            )
        predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return fitted_state, predictions, states

    def score_validation_predictions(
        validation_predictions,
        *,
        anchor_name=None,
        stratum=None,
    ):
        local_score = _classical_measurement_score(
            data,
            validation_predictions,
            target_column="measurement_selection",
            anchor_name=anchor_name,
            strata=strata if stratum is not None else None,
            stratum=stratum,
        )
        overall_score = _classical_measurement_score(
            data,
            validation_predictions,
            target_column="measurement_selection",
        )
        return local_score, overall_score

    def fit_fixed_persistence(candidate_persistence):
        """Fit one prespecified persistence candidate without using validation outcomes."""

        nonlocal fit_counter
        fit_counter += 1
        fit_identifier = int(fit_counter)

        candidate_theta = [
            np.zeros(base_count + n_strata, dtype=float),
            np.zeros(base_count + n_strata, dtype=float),
        ]
        candidate_innovation = np.full(2, float(args.transition_sd), dtype=float)
        states = [init_irt_state(data, args), init_irt_state(data, args)]
        fitted_state, training_predictions, states = solve_ids(
            train_ids,
            candidate_theta,
            candidate_persistence,
            candidate_innovation,
            irt_states=states,
        )
        candidate_converged = False
        candidate_max_change = float("inf")
        candidate_objective = float("nan")
        candidate_iterations = 0
        iteration_trace: list[dict[str, object]] = []
        for iteration in range(1, maximum_iterations + 1):
            previous_theta = [values.copy() for values in candidate_theta]
            previous_innovation = candidate_innovation.copy()
            previous_predictions = training_predictions[
                ["id", "t", "z_d_hat", "z_p_hat"]
            ].copy()
            candidate_theta, candidate_innovation, candidate_objective = estimate_parameters(
                fitted_state,
                candidate_persistence,
            )
            fitted_state, training_predictions, states = solve_ids(
                train_ids,
                candidate_theta,
                candidate_persistence,
                candidate_innovation,
                irt_states=states,
            )
            aligned = previous_predictions.merge(
                training_predictions[["id", "t", "z_d_hat", "z_p_hat"]],
                on=["id", "t"],
                suffixes=("_old", "_new"),
                validate="one_to_one",
            )
            trajectory_change = max(
                float(
                    np.max(
                        np.abs(aligned[f"{column}_new"] - aligned[f"{column}_old"])
                        / (1.0 + np.abs(aligned[f"{column}_old"]))
                    )
                )
                for column in ("z_d_hat", "z_p_hat")
            )
            theta_change = max(
                float(np.max(np.abs(new - old) / (1.0 + np.abs(old))))
                for new, old in zip(candidate_theta, previous_theta)
            )
            innovation_change = float(
                np.max(
                    np.abs(candidate_innovation - previous_innovation)
                    / (1.0 + np.abs(previous_innovation))
                )
            )
            candidate_max_change = max(
                trajectory_change,
                theta_change,
                innovation_change,
            )
            candidate_iterations = iteration
            iteration_trace.append(
                {
                    "fit_identifier": fit_identifier,
                    "iteration": int(iteration),
                    "maximum_iterations": int(maximum_iterations),
                    "transition_mse": float(candidate_objective),
                    "trajectory_relative_change": float(trajectory_change),
                    "coefficient_relative_change": float(theta_change),
                    "innovation_relative_change": float(innovation_change),
                    "maximum_relative_change": float(candidate_max_change),
                    "depression_innovation_sd": float(candidate_innovation[0]),
                    "anxiety_innovation_sd": float(candidate_innovation[1]),
                    "shared_coefficient_l2_norm": float(
                        np.linalg.norm(candidate_theta[0][1 : 1 + q])
                    ),
                    "persistence": np.asarray(
                        candidate_persistence, dtype=float
                    ).tolist(),
                }
            )
            _write_classical_training_progress(
                args,
                {
                    "role": "markov_pattern_mixture"
                    if pattern_mixture
                    else "linear_gaussian_state_space",
                    "phase": "alternating_trajectory_and_transition_fit",
                    **iteration_trace[-1],
                    "total_iterations": int(maximum_iterations),
                    "convergence_tolerance": float(convergence_tolerance),
                },
            )
            if (
                iteration >= minimum_iterations
                and candidate_max_change <= convergence_tolerance
            ):
                candidate_converged = True
                break

        _, validation_predictions, _ = solve_ids(
            selection_ids,
            candidate_theta,
            candidate_persistence,
            candidate_innovation,
        )
        validation_rmse, _ = score_validation_predictions(validation_predictions)
        return {
            "theta": candidate_theta,
            "innovation_sd": candidate_innovation,
            "persistence": candidate_persistence.copy(),
            "training_transition_mse": float(candidate_objective),
            "converged": bool(candidate_converged),
            "iterations_completed": int(candidate_iterations),
            "max_convergence_change": float(candidate_max_change),
            "validation_predictions": validation_predictions,
            "validation_rmse": float(validation_rmse),
            "iteration_trace": iteration_trace,
        }

    def candidate_rank(result, *tie_breakers):
        return (
            0 if result["converged"] else 1,
            float(result["validation_rmse"]),
            *tuple(float(value) for value in tie_breakers),
        )

    persistence_selection_converged = True
    if len(persistence_grid) == 1:
        # Definitive analyses use one persistence value selected once in a
        # separate development exercise and frozen before the evaluation
        # seeds are generated.  A fixed value turns each classical comparator
        # into one converged fit per replicate instead of repeating the full
        # latent-trajectory fit for a persistence grid inside every replicate.
        frozen_persistence = float(persistence_grid[0])
        selected = np.full(
            (2, persistence_strata),
            frozen_persistence,
            dtype=float,
        )
        selected_result = fit_fixed_persistence(selected)
        candidate_trace.append(
            {
                "selection_stage": "development_frozen_persistence",
                "depression_persistence": frozen_persistence,
                "anxiety_persistence": frozen_persistence,
                "validation_rmse": float(selected_result["validation_rmse"]),
                "training_transition_mse": float(
                    selected_result["training_transition_mse"]
                ),
                "fit_converged": bool(selected_result["converged"]),
                "iterations_completed": int(
                    selected_result["iterations_completed"]
                ),
                "max_convergence_change": float(
                    selected_result["max_convergence_change"]
                ),
            }
        )
    elif not pattern_mixture:
        best = None
        for rho_d in persistence_grid:
            for rho_p in persistence_grid:
                candidate_persistence = np.asarray([[rho_d], [rho_p]], dtype=float)
                result = fit_fixed_persistence(candidate_persistence)
                candidate_trace.append(
                    {
                        "selection_stage": "joint_channel_grid",
                        "depression_persistence": float(rho_d),
                        "anxiety_persistence": float(rho_p),
                        "validation_rmse": float(result["validation_rmse"]),
                        "training_transition_mse": float(
                            result["training_transition_mse"]
                        ),
                        "fit_converged": bool(result["converged"]),
                        "iterations_completed": int(result["iterations_completed"]),
                        "max_convergence_change": float(
                            result["max_convergence_change"]
                        ),
                    }
                )
                if not np.isfinite(result["validation_rmse"]):
                    continue
                rank = candidate_rank(
                    result,
                    rho_d + rho_p,
                    rho_d,
                    rho_p,
                )
                if best is None or rank < best[0]:
                    best = (rank, result)
        if best is None:
            raise RuntimeError("No classical persistence candidate produced a validation score.")
        selected_result = best[1]
    else:
        selection_targets = data.anchors.loc[
            data.anchors["measurement_selection"].astype(bool)
        ].copy()
        best = None
        for rho_d in persistence_grid:
            for rho_p in persistence_grid:
                candidate_persistence = np.vstack(
                    [
                        np.full(n_strata, float(rho_d), dtype=float),
                        np.full(n_strata, float(rho_p), dtype=float),
                    ]
                )
                result = fit_fixed_persistence(candidate_persistence)
                candidate_trace.append(
                    {
                        "selection_stage": "channel_fallback_grid",
                        "selection_sweep": 0,
                        "channel": "joint",
                        "stratum": -1,
                        "depression_persistence": float(rho_d),
                        "anxiety_persistence": float(rho_p),
                        "validation_rmse": float(result["validation_rmse"]),
                        "training_transition_mse": float(
                            result["training_transition_mse"]
                        ),
                        "fit_converged": bool(result["converged"]),
                        "iterations_completed": int(result["iterations_completed"]),
                        "max_convergence_change": float(
                            result["max_convergence_change"]
                        ),
                    }
                )
                if not np.isfinite(result["validation_rmse"]):
                    continue
                rank = candidate_rank(
                    result,
                    rho_d + rho_p,
                    rho_d,
                    rho_p,
                )
                if best is None or rank < best[0]:
                    best = (rank, result)
        if best is None:
            raise RuntimeError(
                "No channel-level Markov persistence candidate produced a validation score."
            )
        selected = best[1]["persistence"].copy()
        channel_fallback = selected[:, 0].copy()
        maximum_selection_sweeps = max(
            1,
            int(getattr(args, "classical_markov_selection_sweeps", 3)),
        )
        persistence_selection_converged = False
        for sweep in range(1, maximum_selection_sweeps + 1):
            previous_selected = selected.copy()
            for channel_index, anchor_name in enumerate(anchor_names):
                for stratum_index in range(n_strata):
                    stratum_patient_count = int(
                        selection_targets.loc[
                            selection_targets["anchor"].astype(str).eq(anchor_name)
                            & __import__('prospective_simulation_filter').event_strata(selection_targets, strata).eq(stratum_index),
                            "id",
                        ].nunique()
                    )
                    if stratum_patient_count < minimum_stratum_selection_patients:
                        selected[channel_index, stratum_index] = channel_fallback[
                            channel_index
                        ]
                        candidate_trace.append(
                            {
                                "selection_stage": "stratum_coordinate",
                                "selection_sweep": sweep,
                                "channel": "depression"
                                if channel_index == 0
                                else "anxiety",
                                "stratum": int(stratum_index),
                                "candidate_persistence": float(
                                    selected[channel_index, stratum_index]
                                ),
                                "validation_rmse": float("nan"),
                                "validation_scope": "channel_fallback_applied",
                                "selection_patients": stratum_patient_count,
                            }
                        )
                        continue
                    coordinate_best = None
                    for candidate_rho in persistence_grid:
                        candidate_persistence = selected.copy()
                        candidate_persistence[channel_index, stratum_index] = float(
                            candidate_rho
                        )
                        result = fit_fixed_persistence(candidate_persistence)
                        local_score, overall_score = score_validation_predictions(
                            result["validation_predictions"],
                            anchor_name=anchor_name,
                            stratum=stratum_index,
                        )
                        candidate_trace.append(
                            {
                                "selection_stage": "stratum_coordinate",
                                "selection_sweep": sweep,
                                "channel": "depression"
                                if channel_index == 0
                                else "anxiety",
                                "stratum": int(stratum_index),
                                "candidate_persistence": float(candidate_rho),
                                "validation_rmse": float(local_score),
                                "overall_validation_rmse": float(overall_score),
                                "validation_scope": "channel_stratum",
                                "selection_patients": stratum_patient_count,
                                "training_transition_mse": float(
                                    result["training_transition_mse"]
                                ),
                                "fit_converged": bool(result["converged"]),
                                "iterations_completed": int(
                                    result["iterations_completed"]
                                ),
                                "max_convergence_change": float(
                                    result["max_convergence_change"]
                                ),
                            }
                        )
                        if not np.isfinite(local_score):
                            continue
                        rank = (
                            0 if result["converged"] else 1,
                            float(local_score),
                            float(overall_score)
                            if np.isfinite(overall_score)
                            else float("inf"),
                            float(candidate_rho),
                        )
                        if coordinate_best is None or rank < coordinate_best[0]:
                            coordinate_best = (rank, float(candidate_rho))
                    if coordinate_best is None:
                        raise RuntimeError(
                            "No pattern-mixture persistence candidate produced a validation score."
                        )
                    selected[channel_index, stratum_index] = coordinate_best[1]
            if np.array_equal(selected, previous_selected):
                persistence_selection_converged = True
                break
        selected_result = fit_fixed_persistence(selected)

    persistence = selected_result["persistence"].copy()
    theta = [values.copy() for values in selected_result["theta"]]
    innovation_sd = selected_result["innovation_sd"].copy()
    final_objective = float(selected_result["training_transition_mse"])
    final_selection_score = float(selected_result["validation_rmse"])
    converged = bool(selected_result["converged"])
    iterations_completed = int(selected_result["iterations_completed"])
    max_change = float(selected_result["max_convergence_change"])

    final_states = None
    final_repetitions = (
        4
        if str(getattr(data.config, "anchor_observation", "gaussian")).lower()
        == "irt"
        else 1
    )
    predictions = pd.DataFrame()
    for _ in range(final_repetitions):
        _, predictions, final_states = solve_ids(
            ids,
            theta,
            persistence,
            innovation_sd,
            irt_states=final_states,
            prospective=True,
        )

    stratum_persistence = {
        "depression": {
            f"stratum_{index}": float(persistence[0, index])
            for index in range(persistence_strata)
        },
        "anxiety": {
            f"stratum_{index}": float(persistence[1, index])
            for index in range(persistence_strata)
        },
    }
    predictions.attrs["channel_persistence"] = {
        "depression": float(np.mean(persistence[0])),
        "anxiety": float(np.mean(persistence[1])),
    }
    predictions.attrs["stratum_persistence"] = stratum_persistence
    predictions.attrs["channel_innovation_sd"] = {
        "depression": float(innovation_sd[0]),
        "anxiety": float(innovation_sd[1]),
    }
    predictions.attrs["innovation_update"] = (
        "posterior_expected_squared_transition_error_with_adjacent_covariance"
    )
    predictions.attrs["shared_beta"] = theta[0][1 : 1 + q].copy()
    predictions.attrs["pattern_mixture"] = bool(pattern_mixture)
    predictions.attrs["information_set"] = "prospective; prior sessions only; no backward evaluation pass"
    predictions.attrs["pattern_definition"] = "prior observed questionnaire count at each session"
    predictions.attrs["converged"] = bool(converged)
    predictions.attrs["iterations_completed"] = int(iterations_completed)
    predictions.attrs["maximum_iterations"] = int(maximum_iterations)
    predictions.attrs["convergence_tolerance"] = float(convergence_tolerance)
    predictions.attrs["max_convergence_change"] = float(max_change)
    predictions.attrs["validation_selection_rmse"] = float(final_selection_score)
    predictions.attrs["training_transition_mse"] = float(final_objective)
    predictions.attrs["persistence_selection_target"] = (
        "separate_development_exercise"
        if len(persistence_grid) == 1
        else "measurement_selection"
    )
    predictions.attrs["persistence_selection_converged"] = bool(
        persistence_selection_converged
    )
    predictions.attrs["persistence_candidate_trace"] = candidate_trace
    predictions.attrs["convergence_iteration_trace"] = selected_result[
        "iteration_trace"
    ]
    predictions.attrs["innovation_floor"] = float(innovation_floor)
    predictions.attrs["minimum_stratum_selection_patients"] = int(
        minimum_stratum_selection_patients
    )
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
