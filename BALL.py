#!/usr/bin/env python
"""Canonical one-file BALL codebase.

All active simulation, empirical, table, and figure code lives in this file.
Archived split files are loaded from the source blocks below as an in-memory package.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import types

REPO_ROOT = Path(__file__).resolve().parent

# --- simulations (simulations/__init__.py) ---
SRC_SIMULATIONS = r'''
"""BALL simulation package."""
'''

# --- simulations.src (simulations/src/__init__.py) ---
SRC_SIMULATIONS_SRC = r'''
"""BALL simulation package."""
'''

# --- simulations.src.methods (simulations/src/methods/__init__.py) ---
SRC_SIMULATIONS_SRC_METHODS = r'''
"""Simulation method implementations."""
'''

# --- simulations.paper (simulations/paper/__init__.py) ---
SRC_SIMULATIONS_PAPER = r'''

'''

# --- empirical (empirical/__init__.py) ---
SRC_EMPIRICAL = r'''
'''

# --- simulations.src.model_utils (simulations/src/model_utils.py) ---
SRC_SIMULATIONS_SRC_MODEL_UTILS = r'''
"""Shared utilities and schemas for BALL simulations.

All simulation modules should depend on this file for common dataclasses,
RNG handling, and dataframe schema helpers. This keeps the DGP, diagnostics,
baselines, and S0 prototype connected to one implementation contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
import json
import math
import random
import time

import numpy as np
import pandas as pd


DEFAULT_SUBTYPE_PROBS = (0.40, 0.35, 0.25)


@dataclass(frozen=True)
class AnchorSpec:
    name: str
    cadence: int
    recall_window: int
    loading: float
    error_sd: float
    missing_probability: float


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 1729
    n: int = 1000
    t: int = 84
    q: int = 6
    p_daily: int = 24
    n_subtypes: int = 3
    subtype_probs: tuple[float, ...] = DEFAULT_SUBTYPE_PROBS
    baseline_means: tuple[float, ...] = (0.5, 0.0, -0.5)
    baseline_sd: float = 0.75
    r_ts: float = 0.03
    double_anchor_latents: bool = True
    slow_fraction: float = 0.30
    active_loading_prob: float = 0.30
    loading_drift_sd: float = 0.02
    alpha_action_drift: float = 0.04
    support_switch_prob: float = 0.04
    c_process_sd: float = 0.05
    c_reversion_rate: float = 0.03          # kappa_B: slow-process mean reversion per day
    c_treatment_drift: float = 0.02         # H_B: treatment nudge on the slow process
    # Optional direct RDoC drift term for parameter-recovery scenarios:
    # z_{t+1} - z_t contains C_t beta in addition to nuisance dynamics.
    # Default 0.0 preserves the original DGP.
    rdoc_drift_share: float = 0.0
    rdoc_drift_active_dims: int = 3
    rdoc_drift_beta_seed: int = 31011
    rdoc_drift_min_abs: float = 0.35
    # Residual (fast) persistence per subtype. PDF dynamics are a unit-coefficient
    # random walk (phi = 1); subtype-specific AR < 1 is available for scenarios
    # that need stationary, subtype-heterogeneous residual autocorrelation.
    delta_ar: tuple[float, ...] = (1.0, 1.0, 1.0)
    delta_innovation_sd: float = 0.35
    shock_probability: float = 0.02
    shock_sd: float = 1.0
    proxy_measurement_error_sd: float = 0.25
    # Moderate/primary scenario is clean (no site bias); proxy-fragility scenarios
    # set this > 0. The decomposition machinery is always active.
    site_bias_sd: float = 0.0               # systematic per-site documentation bias
    note_density_error_multiplier: float = 1.0   # extra proxy noise when records are sparse
    endogenous_proxy_observation: bool = True
    n_treatment_types: int = 12
    min_treatment_count: int = 10
    max_treatment_count: int = 15
    assignment: str = "weakly_adaptive"
    y1: AnchorSpec = field(
        default_factory=lambda: AnchorSpec("Y1", 7, 14, 1.0, 0.95, 0.10)
    )
    y2: AnchorSpec = field(
        default_factory=lambda: AnchorSpec("Y2", 7, 7, 0.75, 0.85, 0.20)
    )
    rho_cross: float = 0.30
    rho_serial_y1: float = 0.20
    rho_serial_y2: float = 0.20
    # Anchor observation model (spec): "gaussian" on the total score (default) or
    # "irt", a graded-response item response model generating item-level data.
    anchor_observation: str = "gaussian"
    irt_n_items: int = 9
    irt_n_categories: int = 4
    irt_discrimination: float = 1.5
    # Frozen calibrated-instrument parameters (validation/irt_calibration.json), set by
    # the benchmark loader for the IRT observation-model sensitivity. The model latent is
    # expressed on the calibrated trait scale via trait = (windowed_latent - irt_loc) /
    # irt_scale. The per-item ordered thresholds and discriminations are FIXED (not
    # learned) and identical for BALL, S0, and Markov. Empty -> legacy unit calibration.
    irt_loc: float = 0.0
    irt_scale: float = 1.0
    irt_item_thresholds: tuple = ()        # (n_items, n_categories-1), ordered per item, trait scale
    irt_item_discriminations: tuple = ()   # (n_items,)
    missingness_mechanism: str = "mar"
    daily_base_missing_probability: float = 0.30
    block_probability: float = 0.02
    mnar_gamma_l: float = 0.0
    train_fraction: float = 0.60
    validation_fraction: float = 0.10
    conformal_fraction: float = 0.10
    test_fraction: float = 0.20


@dataclass
class SimulationData:
    config: SimulationConfig
    individuals: pd.DataFrame
    daily: pd.DataFrame
    anchors: pd.DataFrame
    treatments: pd.DataFrame
    components: pd.DataFrame
    metadata: Dict[str, Any]


@dataclass
class MethodResult:
    method: str
    predictions: pd.DataFrame
    intervals: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def combine_raw_ensemble(
    member_results: Iterable[MethodResult],
    method: str,
    *,
    mean_col: str = "L_hat",
    interval_scale: float = 1.96,
    member_sd_source: str = "raw_sd",
    metadata: Optional[Mapping[str, Any]] = None,
) -> MethodResult:
    """Moment-match member predictions and raw Gaussian intervals.

    This is the shared deep-ensemble scaffold used before Laplace or final
    calibration. The interval variance follows the law of total variance:

        mean_k(raw_sd_k^2 + mean_k^2) - mean_ensemble^2

    When member_sd_source is "interval_width", the member standard deviation is
    recovered from lower/upper bounds as half_width / interval_scale. This lets
    existing oracle- or anchor-conformal member intervals flow through the same
    ensemble moment-matching rule.
    """

    members = list(member_results)
    if not members:
        raise ValueError("combine_raw_ensemble requires at least one member result.")
    if member_sd_source not in {"raw_sd", "interval_width"}:
        raise ValueError("member_sd_source must be 'raw_sd' or 'interval_width'.")

    pred_frames: list[pd.DataFrame] = []
    interval_frames: list[pd.DataFrame] = []
    for member, result in enumerate(members):
        pred = result.predictions.copy()
        pred.insert(0, "member", member)
        pred_frames.append(pred)
        if result.intervals is not None:
            if "raw_sd" not in result.intervals.columns:
                raise ValueError(f"Member {member} has intervals but no raw_sd column.")
            interval_cols = ["id", "t", "raw_sd"]
            if "interval_scale" in result.intervals.columns:
                interval_cols.append("interval_scale")
            for col in [
                "z_d_raw_sd",
                "z_p_raw_sd",
                "z_d_lower",
                "z_d_upper",
                "z_p_lower",
                "z_p_upper",
            ]:
                if col in result.intervals.columns:
                    interval_cols.append(col)
            if member_sd_source == "interval_width":
                missing = {"lower", "upper"}.difference(result.intervals.columns)
                if missing:
                    raise ValueError(
                        f"Member {member} interval_width source is missing columns: {sorted(missing)}."
                    )
                interval_cols.extend(["lower", "upper"])
            interval = result.intervals[interval_cols].copy()
            interval.insert(0, "member", member)
            interval_frames.append(interval)

    predictions = pd.concat(pred_frames, ignore_index=True)
    numeric_cols = [
        col
        for col in predictions.columns
        if col not in {"member", "id", "t"} and pd.api.types.is_numeric_dtype(predictions[col])
    ]
    ensemble_pred = (
        predictions.groupby(["id", "t"], sort=True)[numeric_cols]
        .mean()
        .reset_index()
        .sort_values(["id", "t"])
        .reset_index(drop=True)
    )

    intervals = None
    if interval_frames:
        member_interval = pd.concat(interval_frames, ignore_index=True)
        member_mean_cols = ["member", "id", "t", mean_col]
        for col in ["z_d_hat", "z_p_hat"]:
            if col in predictions.columns and col not in member_mean_cols:
                member_mean_cols.append(col)
        member_mean = predictions[member_mean_cols].merge(
            member_interval,
            on=["member", "id", "t"],
            how="inner",
        )
        # Vectorized Gaussian-mixture moment matching over members, per (id, t).
        # total_var = E_members[sd^2 + mean^2] - E_members[mean]^2 (law of total
        # variance); epistemic = between-member spread.
        mm = member_mean.copy()
        scale_col = (
            mm["interval_scale"].to_numpy(dtype=float)
            if "interval_scale" in mm.columns
            else np.full(len(mm), interval_scale, dtype=float)
        )
        if member_sd_source == "interval_width":
            mm["_member_sd"] = (mm["upper"].to_numpy(dtype=float) - mm["lower"].to_numpy(dtype=float)) / (2.0 * scale_col)
        else:
            mm["_member_sd"] = mm["raw_sd"].to_numpy(dtype=float)
        mm["_sq"] = mm["_member_sd"] ** 2 + mm[mean_col] ** 2
        grouped = mm.groupby(["id", "t"], sort=True)
        mean = grouped[mean_col].mean()
        total_sd = np.sqrt((grouped["_sq"].mean() - mean**2).clip(lower=1e-8))
        intervals = pd.DataFrame(
            {
                "id": mean.index.get_level_values("id").astype(int),
                "t": mean.index.get_level_values("t").astype(int),
                "lower": (mean - interval_scale * total_sd).to_numpy(),
                "upper": (mean + interval_scale * total_sd).to_numpy(),
                "raw_sd": total_sd.to_numpy(),
                "ensemble_total_sd": total_sd.to_numpy(),
                "ensemble_epistemic_sd": grouped[mean_col].std(ddof=1).fillna(0.0).to_numpy(),
                "mean_member_raw_sd": grouped["raw_sd"].mean().to_numpy(),
                "mean_member_effective_sd": grouped["_member_sd"].mean().to_numpy(),
                "n_members": grouped.size().to_numpy().astype(int),
                "interval_scale": float(interval_scale),
            }
        )
        for channel in ["z_d", "z_p"]:
            channel_mean_col = f"{channel}_hat"
            if channel_mean_col not in mm.columns:
                continue
            if member_sd_source == "interval_width" and {f"{channel}_lower", f"{channel}_upper"}.issubset(mm.columns):
                channel_member_sd = (
                    mm[f"{channel}_upper"].to_numpy(dtype=float) - mm[f"{channel}_lower"].to_numpy(dtype=float)
                ) / (2.0 * scale_col)
            elif f"{channel}_raw_sd" in mm.columns:
                channel_member_sd = mm[f"{channel}_raw_sd"].to_numpy(dtype=float)
            else:
                channel_member_sd = mm["_member_sd"].to_numpy(dtype=float)
            mm["_csq"] = channel_member_sd ** 2 + mm[channel_mean_col] ** 2
            cgrouped = mm.groupby(["id", "t"], sort=True)
            channel_mean = cgrouped[channel_mean_col].mean()
            channel_sd = np.sqrt((cgrouped["_csq"].mean() - channel_mean**2).clip(lower=1e-8))
            intervals[f"{channel}_lower"] = (channel_mean - interval_scale * channel_sd).to_numpy()
            intervals[f"{channel}_upper"] = (channel_mean + interval_scale * channel_sd).to_numpy()
            intervals[f"{channel}_raw_sd"] = channel_sd.to_numpy()
            intervals[f"{channel}_total_sd"] = channel_sd.to_numpy()
            intervals[f"{channel}_ensemble_total_sd"] = channel_sd.to_numpy()
            intervals[f"{channel}_ensemble_epistemic_sd"] = cgrouped[channel_mean_col].std(ddof=1).fillna(0.0).to_numpy()

    merged_metadata = {
        "status": "raw_ensemble_moment_matched",
        "not_manuscript_result": True,
        "n_members": len(members),
        "mean_col": mean_col,
        "interval_scale": interval_scale,
        "member_sd_source": member_sd_source,
        "interval_combination_rule": (
            f"Gaussian mixture moment match using {member_sd_source}: "
            "mean(member_sd^2 + member_mean^2) - ensemble_mean^2"
        ),
    }
    if metadata:
        merged_metadata.update(dict(metadata))

    return MethodResult(method, ensemble_pred, intervals, metadata=merged_metadata)


def make_rng(seed: int, *keys: Any) -> np.random.Generator:
    """Create a deterministic RNG from a base seed and arbitrary keys."""

    key_text = "|".join(str(k) for k in keys)
    digest = hashlib.sha256(f"{int(seed)}|{key_text}".encode("utf-8")).digest()
    entropy = int.from_bytes(digest[:8], "big") % (2**32)
    return np.random.default_rng(entropy)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_run_id(prefix: str) -> str:
    """Create a timestamped run id that is safe for concurrent local runs."""

    clean_prefix = prefix.rstrip("_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{clean_prefix}_{timestamp}_pid{os.getpid()}"


def logistic(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def standardize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return (arr - arr.mean(axis=0)) / (arr.std(axis=0) + eps)


def weighted_window_average(values: np.ndarray, start: int, end: int) -> float:
    """Uniform recall-window average over inclusive day bounds."""

    start = max(int(start), 0)
    end = min(int(end), len(values) - 1)
    if end < start:
        return float("nan")
    return float(np.mean(values[start : end + 1]))


def marginal_anchor_sd(spec: AnchorSpec, rho_serial: float) -> float:
    """Stationary marginal SD of the AR(1) anchor error process.

    The anchor errors are generated as e_k = rho * e_{k-1} + N(0, error_sd^2),
    so the marginal error SD is error_sd / sqrt(1 - rho^2). Methods that fix a
    Gaussian anchor observation SD should use this rather than the raw
    innovation SD to avoid an overconfident anchor likelihood.
    """

    rho = float(rho_serial)
    return float(spec.error_sd) / math.sqrt(max(1.0 - rho**2, 1e-8))


def conformal_quantile(scores: np.ndarray, alpha: float, default: float) -> float:
    """Split-conformal finite-sample quantile for absolute residual scores."""

    clean = np.asarray(scores, dtype=float)
    clean = clean[np.isfinite(clean)]
    n = len(clean)
    if n == 0:
        return float(default)
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    # Finite-sample split conformal: when the required rank exceeds the number of
    # calibration scores, the (1-alpha) quantile is +inf (interval (-inf, +inf)).
    # Clipping to max(scores) would silently break the >= 1-alpha guarantee on
    # small calibration folds.
    if rank > n:
        return float("inf")
    return float(np.sort(clean)[max(rank, 1) - 1])


def split_individuals(
    individual_ids: Iterable[int], config: SimulationConfig
) -> pd.DataFrame:
    """Assign individuals to train/validation/conformal/test splits."""

    ids = np.array(list(individual_ids), dtype=int)
    rng = make_rng(config.seed, "splits")
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(config.train_fraction * n))
    n_val = int(round(config.validation_fraction * n))
    n_conf = int(round(config.conformal_fraction * n))
    labels = np.empty(n, dtype=object)
    labels[:n_train] = "train"
    labels[n_train : n_train + n_val] = "validation"
    labels[n_train + n_val : n_train + n_val + n_conf] = "conformal"
    labels[n_train + n_val + n_conf :] = "test"
    return pd.DataFrame({"id": shuffled, "split": labels}).sort_values("id")


def config_from_mapping(mapping: Mapping[str, Any]) -> SimulationConfig:
    """Build SimulationConfig from a nested dict, tolerating missing fields."""

    data = mapping.get("data", {})
    latent = mapping.get("latent", {})
    proxy = mapping.get("proxy", {})
    treatment = mapping.get("treatment", {})
    anchors = mapping.get("anchors", {})
    missing = mapping.get("missingness", {})
    splits = mapping.get("splits", {})
    y1 = anchors.get("y1", {})
    y2 = anchors.get("y2", {})
    return SimulationConfig(
        seed=int(mapping.get("seed", 1729)),
        n=int(data.get("n", 1000)),
        t=int(data.get("t", 84)),
        q=int(data.get("q", 6)),
        p_daily=int(data.get("p_daily", 24)),
        n_subtypes=int(data.get("n_subtypes", 3)),
        subtype_probs=tuple(data.get("subtype_probs", DEFAULT_SUBTYPE_PROBS)),
        baseline_means=tuple(latent.get("baseline_means", (0.5, 0.0, -0.5))),
        baseline_sd=float(latent.get("baseline_sd", 0.75)),
        r_ts=float(latent.get("r_ts", 0.03)),
        double_anchor_latents=bool(latent.get("double_anchor_latents", True)),
        slow_fraction=float(latent.get("slow_fraction", 0.30)),
        active_loading_prob=float(latent.get("active_loading_prob", 0.30)),
        loading_drift_sd=float(latent.get("loading_drift_sd", 0.02)),
        alpha_action_drift=float(latent.get("alpha_action_drift", 0.04)),
        support_switch_prob=float(latent.get("support_switch_prob", 0.04)),
        c_process_sd=float(latent.get("c_process_sd", 0.05)),
        c_reversion_rate=float(latent.get("c_reversion_rate", 0.03)),
        c_treatment_drift=float(latent.get("c_treatment_drift", 0.02)),
        rdoc_drift_share=float(latent.get("rdoc_drift_share", 0.0)),
        rdoc_drift_active_dims=int(latent.get("rdoc_drift_active_dims", 3)),
        rdoc_drift_beta_seed=int(latent.get("rdoc_drift_beta_seed", 31011)),
        rdoc_drift_min_abs=float(latent.get("rdoc_drift_min_abs", 0.35)),
        delta_ar=tuple(latent.get("delta_ar", (1.0, 1.0, 1.0))),
        delta_innovation_sd=float(latent.get("delta_innovation_sd", 0.35)),
        shock_probability=float(latent.get("shock_probability", 0.02)),
        shock_sd=float(latent.get("shock_sd", 1.0)),
        proxy_measurement_error_sd=float(proxy.get("measurement_error_sd", 0.25)),
        site_bias_sd=float(proxy.get("site_bias_sd", 0.0)),
        note_density_error_multiplier=float(proxy.get("note_density_error_multiplier", 1.0)),
        endogenous_proxy_observation=bool(proxy.get("endogenous_observation", True)),
        n_treatment_types=int(treatment.get("n_types", 12)),
        min_treatment_count=int(treatment.get("min_count", 10)),
        max_treatment_count=int(treatment.get("max_count", 15)),
        assignment=str(treatment.get("assignment", "weakly_adaptive")),
        y1=AnchorSpec("Y1", int(y1.get("cadence", 7)), int(y1.get("recall_window", 14)), float(y1.get("loading", 1.0)), float(y1.get("error_sd", 0.95)), float(y1.get("missing_probability", 0.10))),
        y2=AnchorSpec("Y2", int(y2.get("cadence", 7)), int(y2.get("recall_window", 7)), float(y2.get("loading", 0.75)), float(y2.get("error_sd", 0.85)), float(y2.get("missing_probability", 0.20))),
        rho_cross=float(anchors.get("rho_cross", 0.30)),
        rho_serial_y1=float(anchors.get("rho_serial_y1", 0.20)),
        rho_serial_y2=float(anchors.get("rho_serial_y2", 0.20)),
        anchor_observation=str(anchors.get("observation", "gaussian")),
        irt_n_items=int(anchors.get("irt_n_items", 9)),
        irt_n_categories=int(anchors.get("irt_n_categories", 4)),
        irt_discrimination=float(anchors.get("irt_discrimination", 1.5)),
        missingness_mechanism=str(missing.get("mechanism", "mar")),
        daily_base_missing_probability=float(missing.get("daily_base_probability", 0.30)),
        block_probability=float(missing.get("block_probability", 0.02)),
        mnar_gamma_l=float(missing.get("mnar_gamma_l", 0.0)),
        train_fraction=float(splits.get("train", 0.60)),
        validation_fraction=float(splits.get("validation", 0.10)),
        conformal_fraction=float(splits.get("conformal", 0.10)),
        test_fraction=float(splits.get("test", 0.20)),
    )


def load_config(path: str | Path) -> SimulationConfig:
    """Load YAML if PyYAML is available, otherwise JSON."""

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load YAML configs.") from exc
        mapping = yaml.safe_load(text)
    else:
        mapping = json.loads(text)
    return config_from_mapping(mapping or {})


def _is_file_lock_error(exc: OSError) -> bool:
    """Return True for common Windows/Dropbox transient file-lock failures."""

    return isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in {32, 33}


def publish_latest_artifact(
    source: str | Path,
    target: str | Path,
    retries: int = 5,
    delay_seconds: float = 0.25,
) -> bool:
    """Copy a run artifact to a latest path without making file locks fatal.

    Timestamped run artifacts are the durable record. The latest paths are
    convenience pointers for agents/reviewers and can be temporarily locked by
    Dropbox, Excel, or another process on Windows.
    """

    source_path = Path(source)
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(max(1, retries)):
        try:
            shutil.copy2(source_path, target_path)
            return True
        except OSError as exc:
            if not _is_file_lock_error(exc) or attempt == retries - 1:
                if _is_file_lock_error(exc):
                    return False
                raise
            time.sleep(delay_seconds * (attempt + 1))
    return False


def publish_latest_text(
    text: str,
    target: str | Path,
    retries: int = 5,
    delay_seconds: float = 0.25,
    encoding: str = "utf-8",
) -> bool:
    """Write a latest text artifact, tolerating transient file locks."""

    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(max(1, retries)):
        try:
            target_path.write_text(text, encoding=encoding)
            return True
        except OSError as exc:
            if not _is_file_lock_error(exc) or attempt == retries - 1:
                if _is_file_lock_error(exc):
                    return False
                raise
            time.sleep(delay_seconds * (attempt + 1))
    return False
'''

# --- simulations.src.missingness (simulations/src/missingness.py) ---
SRC_SIMULATIONS_SRC_MISSINGNESS = r'''
"""Missingness mechanisms for daily observed variables and proxy timing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model_utils import SimulationConfig, logistic, make_rng


def daily_observation_mask(
    latent: np.ndarray,
    config: SimulationConfig,
    person_id: int,
    x: np.ndarray | None = None,
) -> np.ndarray:
    """Return observed-mask matrix with shape (T, p_daily).

    Under "mar", observation propensity at day t depends on the previous day's
    OBSERVED strong-contemporaneous features (X0-X3), so the conditioning set is
    exactly the analyst-visible record: missing at random by construction. The
    MCAR block process is drawn first and folded into each day's realized mask
    before that mask drives the next day, so the dependence never conditions on
    values the analyst cannot see. Under "mnar", propensity depends directly on
    the true latent.
    """

    rng = make_rng(config.seed, "missingness", person_id)
    base_obs = 1.0 - config.daily_base_missing_probability
    base_logit = np.log(base_obs / (1.0 - base_obs))
    # Per-group observation propensities (6 groups of 4 columns): strong
    # contemporaneous features are documented more often than slow proxies,
    # treatment/burden, or pure-noise channels, so the 24 variables do not all
    # share a single missingness rate.
    group_offsets = np.array([0.6, 0.6, 0.3, 0.0, 0.2, -0.4])
    col_offset = np.repeat(group_offsets, 4)[: config.p_daily]
    mech = config.missingness_mechanism.lower()
    if mech == "mar" and x is None:
        raise ValueError(
            "MAR missingness conditions on observed daily values; pass x to "
            "daily_observation_mask."
        )

    # MCAR block dropout, independent of everything else; drawn up-front so the
    # MAR dependence below conditions on the post-block (analyst-visible) mask.
    blocked = np.zeros((config.t, config.p_daily), dtype=bool)
    if config.block_probability > 0:
        for j in range(config.p_daily):
            for t in range(config.t):
                if rng.random() < config.block_probability:
                    block_len = int(rng.integers(2, 8))
                    blocked[t : min(config.t, t + block_len), j] = True

    mask = np.zeros((config.t, config.p_daily), dtype=bool)
    for t in range(config.t):
        logits_t = base_logit + col_offset
        if mech == "mnar":
            logits_t = logits_t + config.mnar_gamma_l * latent[t]
        elif mech == "mar" and t > 0:
            prev_obs = mask[t - 1, 0:4]
            if prev_obs.any():
                driver = float(np.mean(x[t - 1, 0:4][prev_obs]))
                if np.isfinite(driver):
                    logits_t = logits_t + 0.20 * driver
        mask[t] = (rng.random(config.p_daily) < logistic(logits_t)) & ~blocked[t]
    return mask


def proxy_observation_mask(
    latent: np.ndarray,
    recent_treatment: np.ndarray,
    site: int,
    config: SimulationConfig,
    person_id: int,
) -> np.ndarray:
    """Observation timing for B(t), optionally endogenous in L(t)."""

    rng = make_rng(config.seed, "proxy_observation", person_id)
    if not config.endogenous_proxy_observation:
        return np.ones(config.t, dtype=bool)
    logits = -0.7 + 0.25 * latent + 0.35 * recent_treatment + 0.05 * site
    probs = logistic(logits)
    return rng.random(config.t) < probs


def apply_daily_missingness(
    daily: pd.DataFrame, config: SimulationConfig
) -> pd.DataFrame:
    """Set X columns to NA where their observed mask is false."""

    out = daily.copy()
    x_cols = [c for c in out.columns if c.startswith("X")]
    for col in x_cols:
        mask_col = f"obs_{col}"
        if mask_col in out.columns:
            out.loc[~out[mask_col], col] = np.nan
    return out
'''

# --- simulations.src.anchors (simulations/src/anchors.py) ---
SRC_SIMULATIONS_SRC_ANCHORS = r'''
"""Sparse recall-window anchor generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model_utils import AnchorSpec, SimulationConfig, make_rng, weighted_window_average


def scheduled_anchor_days(config: SimulationConfig, spec: AnchorSpec) -> list[int]:
    return list(range(spec.cadence - 1, config.t, spec.cadence))


def _grm_item_thresholds(config: SimulationConfig) -> np.ndarray:
    """Item thresholds b_{j,k} for the graded-response model, on the latent scale.

    Each item has n_categories-1 increasing thresholds spread across the
    standardized latent range, with mild per-item jitter so items differ.
    """

    rng = make_rng(config.seed, "irt_items")
    n_cat = max(int(config.irt_n_categories), 2)
    base = np.linspace(-1.5, 1.5, n_cat - 1)
    thresholds = base[None, :] + rng.normal(0.0, 0.3, size=(int(config.irt_n_items), n_cat - 1))
    return np.sort(thresholds, axis=1)


def _grm_total_score(window_mean: float, thresholds: np.ndarray, discrimination: float, rng) -> float:
    """Sample a graded-response total score for a recall-windowed latent value.

    For item j and category cut k, P(response_j >= k) = logit^{-1}(a (z - b_{j,k})).
    Category probabilities are successive differences; responses are summed.
    """

    if not np.isfinite(window_mean):
        return float("nan")
    p_ge = 1.0 / (1.0 + np.exp(-discrimination * (float(window_mean) - thresholds)))  # (J, K-1)
    total = 0
    for j in range(thresholds.shape[0]):
        pg = np.concatenate(([1.0], p_ge[j], [0.0]))  # P(>=0)=1 ... P(>=K)=0
        probs = pg[:-1] - pg[1:]
        probs = np.clip(probs, 1e-9, None)
        probs = probs / probs.sum()
        total += int(rng.choice(len(probs), p=probs))
    return float(total)


def _calibrated_irt_params(config: SimulationConfig):
    """Frozen calibrated-instrument parameters for the IRT observation model.

    Returns the affine trait map (loc, scale) and the FIXED per-item discriminations
    and ordered thresholds on the trait scale. When the config carries a calibration
    (set from validation/irt_calibration.json), those exact constants are used for
    every seed, share, method, and control. Without a calibration the function falls
    back to a deterministic legacy instrument, used only by the structural gate test.
    """

    thr = config.irt_item_thresholds
    disc = config.irt_item_discriminations
    if thr and disc:
        thresholds = np.asarray(thr, dtype=float)                       # (J, K-1)
        discriminations = np.asarray(disc, dtype=float)                 # (J,)
        scale = float(config.irt_scale) or 1.0
        return float(config.irt_loc), scale, discriminations, thresholds
    thresholds = _grm_item_thresholds(config)
    discriminations = np.full(thresholds.shape[0], float(config.irt_discrimination))
    return 0.0, 1.0, discriminations, thresholds


def _grm_item_responses(trait_value: float, discriminations: np.ndarray,
                        thresholds: np.ndarray, rng) -> np.ndarray:
    """Sample per-item graded responses for a windowed trait value (one anchor).

    For item j and cut k, P(response_j >= k) = logit^{-1}(a_j (trait - b_{j,k})).
    Returns an integer array (J,) of category responses.
    """

    p_ge = 1.0 / (1.0 + np.exp(-discriminations[:, None] * (float(trait_value) - thresholds)))  # (J, K-1)
    responses = np.empty(thresholds.shape[0], dtype=int)
    for j in range(thresholds.shape[0]):
        pg = np.concatenate(([1.0], p_ge[j], [0.0]))
        probs = np.clip(pg[:-1] - pg[1:], 1e-9, None)
        responses[j] = int(rng.choice(len(probs), p=probs / probs.sum()))
    return responses


def generate_anchors(
    components: pd.DataFrame,
    config: SimulationConfig,
    train_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Generate two sparse recall-windowed anchors.

    In the PDF-faithful default, Y1 is a PHQ-style anchor on z_d and Y2 is a
    PCL-style anchor on z_p. The legacy scalar columns are retained by storing
    the channel-specific recall target in window_mean_L.

    Note on the error process: anchor errors are AR(1) with coefficient
    rho_serial, so the stationary marginal error SD is
    error_sd / sqrt(1 - rho^2), slightly larger than error_sd. Methods that fix
    the anchor observation SD should use model_utils.marginal_anchor_sd rather
    than the raw error_sd to avoid overconfident anchor likelihoods.

    Under the IRT observation model the linear/Gaussian methods remain
    deliberately misspecified (they keep the loading * window-mean Gaussian
    working model); only the SSM has an IRT-aware likelihood. That is a
    misspecification-robustness scenario, not an oversight.
    """

    use_irt = str(config.anchor_observation).lower() == "irt"
    if use_irt:
        irt_loc, irt_scale, irt_disc, irt_thresholds = _calibrated_irt_params(config)
    else:
        irt_loc = irt_scale = 0.0
        irt_disc = irt_thresholds = None

    rows: list[dict] = []
    for person_id, comp_i in components.groupby("id", sort=True):
        rng = make_rng(config.seed, "anchors", int(person_id))
        comp_i = comp_i.sort_values("t")
        latent_joint = comp_i["L"].to_numpy()
        latent_d = comp_i["z_d"].to_numpy() if "z_d" in comp_i else latent_joint
        latent_p = comp_i["z_p"].to_numpy() if "z_p" in comp_i else latent_joint
        prior_errors = {"Y1": 0.0, "Y2": 0.0}
        y1_days = set(scheduled_anchor_days(config, config.y1))
        y2_days = set(scheduled_anchor_days(config, config.y2))
        for day in sorted(y1_days | y2_days):
            if day in y1_days and day in y2_days:
                noise_by_anchor = dict(
                    zip(
                        ["Y1", "Y2"],
                        rng.multivariate_normal(
                            mean=[0.0, 0.0],
                            cov=[
                                [
                                    config.y1.error_sd**2,
                                    config.rho_cross * config.y1.error_sd * config.y2.error_sd,
                                ],
                                [
                                    config.rho_cross * config.y1.error_sd * config.y2.error_sd,
                                    config.y2.error_sd**2,
                                ],
                            ],
                        ),
                    )
                )
            else:
                noise_by_anchor = {}
                if day in y1_days:
                    noise_by_anchor["Y1"] = rng.normal(0.0, config.y1.error_sd)
                if day in y2_days:
                    noise_by_anchor["Y2"] = rng.normal(0.0, config.y2.error_sd)
            for spec in (config.y1, config.y2):
                if spec.name not in noise_by_anchor:
                    continue
                noise = float(noise_by_anchor[spec.name])
                start = day - spec.recall_window + 1
                channel_values = latent_d if spec.name == "Y1" else latent_p
                window_mean = weighted_window_average(channel_values, start, day)
                window_mean_joint = weighted_window_average(latent_joint, start, day)
                window_mean_zd = weighted_window_average(latent_d, start, day)
                window_mean_zp = weighted_window_average(latent_p, start, day)
                serial_rho = config.rho_serial_y1 if spec.name == "Y1" else config.rho_serial_y2
                error = serial_rho * prior_errors[spec.name] + float(noise)
                prior_errors[spec.name] = error
                observed = rng.random() >= spec.missing_probability
                if use_irt:
                    # Express the windowed latent on the calibrated trait scale, then
                    # sample per-item graded responses from the FROZEN instrument.
                    # Unobserved anchors yield missing items and total (no leakage).
                    trait = (window_mean - irt_loc) / irt_scale if np.isfinite(window_mean) else np.nan
                    if observed and np.isfinite(trait):
                        items = _grm_item_responses(trait, irt_disc, irt_thresholds, rng)
                        irt_items = items.astype(float).tolist()
                        irt_total = float(items.sum())
                    else:
                        irt_items = [float("nan")] * int(config.irt_n_items)
                        irt_total = float("nan")
                    irt_trait_truth = float(trait) if np.isfinite(trait) else float("nan")
                    value = float("nan")   # IRT uses item responses, not a scalar readout
                else:
                    irt_total = float("nan")
                    irt_items = None
                    irt_trait_truth = float("nan")
                    value = spec.loading * window_mean + error if observed else np.nan
                rows.append(
                    {
                        "id": int(person_id),
                        "anchor": spec.name,
                        "t": int(day),
                        "window_start": int(max(start, 0)),
                        "window_end": int(day),
                        "window_mean_L": window_mean,
                        "window_mean_joint": window_mean_joint,
                        "window_mean_zd": window_mean_zd,
                        "window_mean_zp": window_mean_zp,
                        "latent_channel": "z_d" if spec.name == "Y1" else "z_p",
                        "value": value,
                        "observed": bool(observed),
                        "loading": spec.loading,
                        "irt_total": irt_total,
                        "irt_items": irt_items,
                        "irt_trait_truth": irt_trait_truth,
                    }
                )

    return pd.DataFrame(rows)
'''

# --- simulations.src.dgp (simulations/src/dgp.py) ---
SRC_SIMULATIONS_SRC_DGP = r'''
"""Data-generating process for the BALL sparse-anchor simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .anchors import generate_anchors
from .missingness import daily_observation_mask, proxy_observation_mask, apply_daily_missingness
from .model_utils import SimulationConfig, SimulationData, make_rng, split_individuals


REFERENCE_R_TS = 0.03

# Strength of the severity dependence in treatment assignment (the
# confounding-by-indication knob). "random" assigns days and protocol types
# uniformly; the adaptive modes up-weight treatment days and aggressive
# protocols when current severity is high.
ASSIGNMENT_GAMMA = {
    "random": 0.0,
    "weakly_adaptive": 0.4,
    "strongly_adaptive": 1.2,
}


def _alpha_innovation_sd(config: SimulationConfig) -> float:
    """Loading-drift innovation scale for the slow-process time-scale knob.

    `r_ts` controls relative loading innovation variance around the default
    reference value. `loading_drift_sd` is the absolute loading random-walk SD at
    the reference `r_ts`. The later slow/fast marginal rescaling is controlled
    separately by `slow_fraction`.
    """

    r_ts_scale = np.sqrt(max(float(config.r_ts), 1e-8) / REFERENCE_R_TS)
    return max(1e-4, float(config.loading_drift_sd) * r_ts_scale)


def _build_treatment_kernels(n_types: int, d_max: int = 20) -> tuple[np.ndarray, int]:
    """Fixed per-type response kernels K_a(d) over days-since-treatment.

    The 12 default types span five response shapes the model must learn:
    immediate-transient (3), delayed-transient (3), sustained (3), null/weak (2),
    and one harmful-for-subtype type. Negative amplitudes improve severity;
    the harmful type carries a positive amplitude applied only to its vulnerable
    subtype (handled at the call site). Returns the kernel matrix and the index
    of the harmful type (-1 if there are too few types to include one).
    """

    d = np.arange(d_max + 1, dtype=float)
    immediate = lambda amp, tau=2.0: amp * np.exp(-d / tau)
    delayed = lambda amp, tau=5.0: amp * (d / tau) * np.exp(1.0 - d / tau)   # peak ~tau
    sustained = lambda amp, tau=3.0: amp * (1.0 - np.exp(-d / tau))
    specs = [
        immediate(-0.30), immediate(-0.22), immediate(-0.38),
        delayed(-0.28), delayed(-0.20), delayed(-0.34),
        sustained(-0.18), sustained(-0.12), sustained(-0.24),
        np.zeros_like(d), np.zeros_like(d),                 # null / weak
        sustained(0.30, tau=4.0),                           # harmful (sign applied per subtype)
    ]
    kernels = np.zeros((n_types, d_max + 1), dtype=float)
    harmful_idx = 11 if n_types >= 12 else -1
    for a in range(n_types):
        kernels[a] = specs[a] if a < len(specs) else immediate(-0.15)
    return kernels, harmful_idx


def _make_rdoc_drift_beta(config: SimulationConfig) -> np.ndarray:
    """Sparse unit vector for direct RDoC drift parameter-recovery scenarios."""

    q = int(config.q)
    active_dims = max(1, min(int(config.rdoc_drift_active_dims), q))
    rng = make_rng(int(config.rdoc_drift_beta_seed), "rdoc_direct_drift_beta")
    active = rng.choice(q, size=active_dims, replace=False)
    beta = np.zeros(q, dtype=float)
    min_abs = max(0.0, min(float(config.rdoc_drift_min_abs), 0.95))
    signs = rng.choice(np.array([-1.0, 1.0]), size=active_dims)
    magnitudes = rng.uniform(min_abs, 1.0, size=active_dims)
    beta[active] = signs * magnitudes
    beta = beta / max(float(np.linalg.norm(beta)), 1e-8)
    first = int(active[np.argmin(active)])
    if beta[first] < 0:
        beta = -beta
    return beta


def _rdoc_drift_signal(c: np.ndarray, beta_unit: np.ndarray, base_latent: np.ndarray, share: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Scale C_t beta to a requested one-step latent-drift variance share."""

    share = max(0.0, min(float(share), 0.95))
    signal = np.zeros(len(base_latent), dtype=float)
    level = np.zeros(len(base_latent), dtype=float)
    if share <= 0.0 or len(base_latent) < 3:
        return signal, level, 0.0
    raw = c @ beta_unit
    raw = raw - float(np.mean(raw))
    raw_sd = float(np.std(raw))
    base_var = float(np.var(np.diff(base_latent)))
    if raw_sd <= 1e-8 or base_var <= 1e-12:
        return signal, level, 0.0
    signal_sd = np.sqrt(base_var * share / max(1.0 - share, 1e-8))
    signal = (raw / raw_sd) * signal_sd
    if len(signal) > 1:
        level[1:] = np.cumsum(signal[:-1])
    return signal, level, signal_sd / raw_sd


def generate_dataset(config: SimulationConfig) -> SimulationData:
    rng = make_rng(config.seed, "dgp")
    split_df = split_individuals(range(config.n), config)
    if config.assignment not in ASSIGNMENT_GAMMA:
        raise ValueError(
            f"Unknown treatment assignment '{config.assignment}'; "
            f"expected one of {sorted(ASSIGNMENT_GAMMA)}."
        )
    assignment_gamma = ASSIGNMENT_GAMMA[config.assignment]

    # Action-conditional drift directions for the loading dynamics (spec
    # f_theta(alpha, a)). Each rTMS protocol nudges the recovery recipe along a
    # fixed unit direction; the magnitude is config.alpha_action_drift. This is
    # what the model's neural drift has to learn.
    action_drift_dirs = rng.normal(0.0, 1.0, size=(config.n_treatment_types, config.q))
    action_drift_dirs /= np.maximum(np.linalg.norm(action_drift_dirs, axis=1, keepdims=True), 1e-8)

    # Fixed treatment response kernels (shared across patients; the model learns
    # them) and the harmful-for-subtype type.
    treatment_kernels, harmful_type = _build_treatment_kernels(config.n_treatment_types)
    kernel_dmax = treatment_kernels.shape[1] - 1
    vulnerable_subtype = config.n_subtypes - 1
    # Per-type aggressiveness (peak severity improvement) used by the adaptive
    # assignment modes: sicker patients are steered toward aggressive protocols.
    improvement = np.maximum(0.0, -treatment_kernels.min(axis=1))
    type_aggressiveness = (improvement - improvement.mean()) / (improvement.std() + 1e-8)
    # Subtype-specific slow-process long-run profiles mu_B,u and per-site
    # documentation-bias profiles, fixed across patients.
    c_subtype_profiles = rng.normal(0.0, 1.0, size=(config.n_subtypes, config.q))
    site_profiles = rng.normal(0.0, 1.0, size=(5, config.q))
    rdoc_drift_beta_unit = _make_rdoc_drift_beta(config)
    rdoc_drift_scales: list[float] = []

    individuals: list[dict] = []
    component_rows: list[dict] = []
    daily_rows: list[dict] = []
    treatment_rows: list[dict] = []

    for person_id in range(config.n):
        rng_i = make_rng(config.seed, "person", person_id)
        subtype = int(rng_i.choice(config.n_subtypes, p=np.array(config.subtype_probs)))
        site = int(rng_i.integers(0, 5))
        baseline = rng_i.normal(config.baseline_means[subtype], config.baseline_sd)

        # Per-patient, time-varying sparse support for the recovery recipe (spec
        # adaptive Lasso): most RDoC dimensions are silent at any session, a few
        # are active, the support is drawn per patient, and it can shift over the
        # course of treatment as dimensions toggle on and off.
        active_td = np.zeros((config.t, config.q), dtype=bool)
        init_active = rng_i.random(config.q) < config.active_loading_prob
        if not init_active.any():
            init_active[rng_i.integers(0, config.q)] = True
        active_td[0] = init_active

        # Treatment assignment is sequential: at each candidate day the treatment
        # hazard (and, in adaptive modes, the protocol choice) depends on the
        # patient's current severity state, so config.assignment controls
        # confounding by indication. The total count is forced into
        # [min_treatment_count, max_treatment_count] by the remaining/days-left
        # base hazard, which reduces to uniform random assignment when the
        # severity tilt gamma is zero.
        n_candidates = max(config.t - 2, 0)
        min_count = min(config.min_treatment_count, n_candidates)
        max_count = min(config.max_treatment_count, n_candidates)
        target_count = int(rng_i.integers(min_count, max_count + 1)) if n_candidates > 0 else 0
        remaining = target_count
        treatment_type_by_day = np.full(config.t, -1, dtype=int)
        treatment_burden_by_day = np.zeros(config.t, dtype=float)
        treatment_days: list[int] = []
        trt_level_d = np.zeros(config.t)
        trt_level_p = np.zeros(config.t)

        c = np.zeros((config.t, config.q))
        b = np.zeros((config.t, config.q))
        alpha = np.zeros((config.t, config.q))
        residual = np.zeros((config.t, 2))
        c[0] = rng_i.normal(0.0, 1.0, size=config.q)
        alpha[0] = init_active * rng_i.normal(0.6, 0.25, size=config.q)
        # Residual is zero-mean here; the per-patient level (baseline) and the
        # treatment response are added after the variance rescaling so neither is
        # washed out by standardization.
        residual[0, 0] = rng_i.normal(0.0, 0.25)
        residual[0, 1] = rng_i.normal(0.0, 0.25)

        delta_sd = config.delta_innovation_sd
        alpha_sd = _alpha_innovation_sd(config)
        # Subtype residual persistence (phi = 1 => PDF random walk).
        phi_delta_u = float(config.delta_ar[subtype]) if subtype < len(config.delta_ar) else 1.0
        innovation_cov = np.array(
            [
                [delta_sd**2, 0.35 * delta_sd**2],
                [0.35 * delta_sd**2, delta_sd**2],
            ]
        )
        for t in range(1, config.t):
            # Treatment decision for day t (candidate days 1..T-2), made from the
            # day t-1 state BEFORE the day-t dynamics so the assignment is
            # adapted but never anticipatory. The severity proxy uses the
            # pre-rescaling components plus the running treatment response, so
            # adaptive assignment reacts to shocks, drift, and prior response.
            if remaining > 0 and t <= config.t - 2:
                days_left = (config.t - 2) - t + 1
                # Dynamic (within-person) state drives treatment TIMING; the full
                # severity including the stable baseline drives the PROTOCOL
                # choice below. Folding the baseline into the timing tilt would
                # saturate tanh for severe patients and erase the within-person
                # dependence the adaptive scenarios are meant to create.
                s_dyn = (
                    float(np.dot(alpha[t - 1], c[t - 1]))
                    + 0.5 * (residual[t - 1, 0] + residual[t - 1, 1])
                    + 0.5 * (trt_level_d[t - 1] + trt_level_p[t - 1])
                )
                s_prev = baseline + s_dyn
                if remaining >= days_left:
                    treat_today = True
                else:
                    p_treat = (remaining / days_left) * float(
                        np.exp(assignment_gamma * np.tanh(s_dyn))
                    )
                    treat_today = bool(rng_i.random() < min(p_treat, 1.0))
                if treat_today:
                    if assignment_gamma > 0:
                        logits = assignment_gamma * np.tanh(s_prev) * type_aggressiveness
                        weights = np.exp(logits - logits.max())
                        weights = weights / weights.sum()
                        treatment_type = int(
                            rng_i.choice(config.n_treatment_types, p=weights)
                        )
                    else:
                        treatment_type = int(rng_i.integers(0, config.n_treatment_types))
                    treatment_type_by_day[t] = treatment_type
                    treatment_burden = 0.50 + 0.50 * (
                        treatment_type / max(config.n_treatment_types - 1, 1)
                    )
                    treatment_burden_by_day[t] = treatment_burden
                    treatment_days.append(t)
                    treatment_rows.append(
                        {
                            "id": person_id,
                            "t": int(t),
                            "a": treatment_type,
                            "treatment_burden": float(treatment_burden),
                        }
                    )
                    # Treatment response as a per-type kernel contribution (spec
                    # Stage-3 structure), accumulated as the trajectory unfolds.
                    # The harmful type only worsens its vulnerable subtype; for
                    # other subtypes it is effectively null.
                    end = min(config.t, t + kernel_dmax + 1)
                    seg = treatment_kernels[treatment_type][: end - t]
                    if treatment_type == harmful_type and subtype != vulnerable_subtype:
                        seg = seg * 0.0
                    trt_level_d[t:end] += seg
                    trt_level_p[t:end] += 0.8 * seg
                    remaining -= 1

            # Slow process: subtype-specific mean reversion toward mu_B,u plus a
            # slow treatment effect H_B along the delivered protocol's direction.
            prev_a = treatment_type_by_day[t - 1]
            h_b = config.c_treatment_drift * action_drift_dirs[prev_a] if prev_a >= 0 else 0.0
            c[t] = (
                c[t - 1]
                + config.c_reversion_rate * (c_subtype_profiles[subtype] - c[t - 1])
                + h_b
                + rng_i.normal(0.0, config.c_process_sd, size=config.q)
            )
            # Evolve the sparse support with an asymmetric on/off toggle whose
            # stationary active fraction equals active_loading_prob, so the recipe
            # shifts over time but stays sparse (most dimensions silent).
            prev_active = active_td[t - 1]
            p_active = max(min(config.active_loading_prob, 0.999), 1e-3)
            rate_on = config.support_switch_prob
            rate_off = config.support_switch_prob * (1.0 - p_active) / p_active
            turn_on = (~prev_active) & (rng_i.random(config.q) < rate_on)
            turn_off = prev_active & (rng_i.random(config.q) < rate_off)
            cur_active = (prev_active | turn_on) & ~turn_off
            if not cur_active.any():
                cur_active[rng_i.integers(0, config.q)] = True
            active_td[t] = cur_active
            # Action-conditional drift on the loading (spec f_theta(alpha, a)):
            # the delivered protocol nudges the recipe along its drift direction.
            a_t = treatment_type_by_day[t]
            drift = config.alpha_action_drift * action_drift_dirs[a_t] if a_t >= 0 else 0.0
            random_walk = alpha[t - 1] + drift + rng_i.normal(0.0, alpha_sd, size=config.q)
            fresh = rng_i.normal(0.6, 0.25, size=config.q)
            # continuing-active: random walk; newly-active: fresh loading; inactive: exactly zero.
            alpha[t] = np.where(
                cur_active & prev_active,
                random_walk,
                np.where(cur_active & ~prev_active, fresh, 0.0),
            )
            shock = np.zeros(2)
            if rng_i.random() < config.shock_probability:
                shock = rng_i.normal(0.0, config.shock_sd, size=2)
            # Residual dynamics: delta_{n+1} = phi * delta_n + shock + noise. The
            # treatment response is NOT in the increment (it is added as a level
            # after rescaling, via the per-type kernel contribution), so treatment
            # heterogeneity survives the variance normalization.
            residual[t] = (
                phi_delta_u * residual[t - 1]
                + shock
                + rng_i.multivariate_normal(np.zeros(2), innovation_cov)
            )

        recent_treatment = np.zeros(config.t)
        for day in treatment_days:
            recent_treatment[day : min(config.t, day + 7)] = 1.0

        # Center each proxy dimension per person over time so that the
        # RDoC-explained component carries no per-person level (the level lives
        # in `baseline`/delta, matching how the structural and neural fits
        # estimate an intercept). The loadings `alpha`, not the product, are
        # then rescaled to the target variance, so the appendix identity
        # slow = alpha' RDoC holds exactly for the stored alpha. Without this,
        # rescaling the product decoupled the stored alpha from slow and made
        # rho_RDoC, alpha-recovery, and active-set metrics ill-defined.
        c = c - c.mean(axis=0, keepdims=True)
        target_slow_sd = np.sqrt(max(config.slow_fraction, 1e-6))
        target_fast_sd = np.sqrt(max(1.0 - config.slow_fraction, 1e-6))
        raw_slow = np.sum(alpha * c, axis=1)
        slow_scale = target_slow_sd / (raw_slow.std() + 1e-8)
        alpha = alpha * slow_scale
        slow = np.sum(alpha * c, axis=1)
        # Standardize the zero-mean residual walk to the target fast SD, then add
        # the deterministic treatment response as a LEVEL (so it survives the
        # rescaling) and the per-patient baseline. The dynamic fast residual
        # (walk + treatment, zero per-patient level) is what the fast-residual
        # proxies X12-15 measure; the spec delta that carries the per-patient
        # level is baseline + dynamic residual.
        # Combine the zero-mean residual walk with the (zero-mean) treatment
        # response, then scale the joint dynamic fast component to the fast
        # variance budget. Treatment-type shape is preserved within the fast
        # signal while the WITHIN-PERSON slow/fast variance split stays at
        # slow_fraction (the rho_RDoC metric is within-person). Note the
        # per-patient baseline added below inflates the *population* fast
        # variance, so pooled slow share is lower than slow_fraction.
        resid_d = residual[:, 0] - residual[:, 0].mean()
        resid_p = residual[:, 1] - residual[:, 1].mean()
        fast_dyn_d = resid_d + (trt_level_d - trt_level_d.mean())
        fast_dyn_p = resid_p + (trt_level_p - trt_level_p.mean())
        joint_fast = 0.5 * (fast_dyn_d + fast_dyn_p)
        joint_scale = target_fast_sd / (joint_fast.std() + 1e-8)
        fast_dyn_d = fast_dyn_d * joint_scale
        fast_dyn_p = fast_dyn_p * joint_scale
        # Spec decomposition z = alpha'RDoC + delta with no separate intercept:
        # the per-patient level lives inside delta.
        fast_d = baseline + fast_dyn_d
        fast_p = baseline + fast_dyn_p
        z_d = slow + fast_d
        z_p = slow + fast_p if config.double_anchor_latents else z_d.copy()
        latent = 0.5 * (z_d + z_p)
        rdoc_direct_signal, rdoc_direct_level, rdoc_direct_scale = _rdoc_drift_signal(
            c,
            rdoc_drift_beta_unit,
            latent,
            config.rdoc_drift_share,
        )
        rdoc_drift_scales.append(float(rdoc_direct_scale))
        if float(config.rdoc_drift_share) > 0.0:
            # Direct RDoC parameter-recovery scenario: the same C_t beta drift
            # shifts both anchor channels, while the legacy decomposition columns
            # remain algebraically consistent by carrying the level in delta.
            fast_d = fast_d + rdoc_direct_level
            fast_p = fast_p + rdoc_direct_level
            z_d = z_d + rdoc_direct_level
            z_p = z_p + rdoc_direct_level if config.double_anchor_latents else z_d.copy()
            latent = 0.5 * (z_d + z_p)
        fast = 0.5 * (fast_d + fast_p)
        fast_dyn = 0.5 * (fast_dyn_d + fast_dyn_p)

        proxy_obs = proxy_observation_mask(latent, recent_treatment, site, config, person_id)
        # Proxy error decomposition m(t) = m_site + m_density(t) + m_random(t):
        #  - m_site: systematic per-site documentation bias (a fixed direction),
        #  - m_density(t): noise that grows when records are sparse (low recent
        #    proxy-observation density),
        #  - m_random(t): baseline measurement noise.
        m_site = config.site_bias_sd * site_profiles[site]
        density = pd.Series(proxy_obs.astype(float)).rolling(7, min_periods=1).mean().to_numpy()
        density_factor = 1.0 + config.note_density_error_multiplier * (1.0 - density)
        noise_sd = config.proxy_measurement_error_sd * density_factor
        m_random = rng_i.normal(0.0, 1.0, size=(config.t, config.q)) * noise_sd[:, None]
        b = c + m_site[None, :] + m_random
        b_observed = b.copy()
        b_observed[~proxy_obs] = np.nan
        treatment_risk = np.where(
            treatment_type_by_day >= 0,
            0.10 + 0.20 * ((treatment_type_by_day % 4) / 3.0),
            0.0,
        )
        side_effect_burden_by_day = np.maximum(
            0.0,
            treatment_risk + 0.10 * recent_treatment + rng_i.normal(0.0, 0.05, size=config.t),
        )

        x = np.zeros((config.t, config.p_daily))
        # Six groups of four variables.
        for j in range(4):
            if config.double_anchor_latents:
                severity_stream = z_d if j < 2 else z_p
                # Fast-residual proxies track the zero-baseline dynamic residual
                # (recent shocks + treatment response), not the static per-patient
                # level, which would otherwise contaminate the fast channel.
                fast_stream = fast_dyn_d if j < 2 else fast_dyn_p
            else:
                severity_stream = latent
                fast_stream = fast_dyn
            x[:, j] = severity_stream + rng_i.normal(0.0, 0.5, size=config.t)
            x[:, 4 + j] = pd.Series(severity_stream).rolling(7, min_periods=1).mean().to_numpy() + rng_i.normal(0.0, 0.7, size=config.t)
            # Noise recalibrated (0.6 -> 0.45) for the per-person-centered proxy
            # c: centering removes the slowly-decaying initial level, so the
            # slow-proxy streams need a lower noise floor to remain informative
            # about C(t)'s within-person variation.
            x[:, 8 + j] = c[:, j % config.q] + rng_i.normal(0.0, 0.45, size=config.t)
            x[:, 12 + j] = fast_stream + rng_i.normal(0.0, 0.6, size=config.t)
            # Treatment/burden group: recent-treatment indicator and side-effect
            # burden (a documented treatment-burden signal), both noisy.
            if j < 2:
                x[:, 16 + j] = recent_treatment + rng_i.normal(0.0, 0.4, size=config.t)
            else:
                x[:, 16 + j] = side_effect_burden_by_day + rng_i.normal(0.0, 0.4, size=config.t)
            x[:, 20 + j] = rng_i.normal(0.0, 1.0, size=config.t)

        # The observation mask is drawn AFTER the daily values exist so the MAR
        # mechanism can condition on the previous day's observed features.
        obs_mask = daily_observation_mask(latent, config, person_id, x=x)

        individuals.append(
            {
                "id": person_id,
                "subtype": subtype,
                "site": site,
                "baseline": baseline,
                "split": split_df.loc[split_df["id"] == person_id, "split"].iloc[0],
            }
        )

        for t in range(config.t):
            comp_row = {
                "id": person_id,
                "t": t,
                "subtype": subtype,
                "slow": slow[t],
                "delta": fast[t],
                "delta_d": fast_d[t],
                "delta_p": fast_p[t],
                "rdoc_direct_signal": rdoc_direct_signal[t],
                "rdoc_direct_level": rdoc_direct_level[t],
                "z_d": z_d[t],
                "z_p": z_p[t],
                "S_joint": latent[t],
                "L": latent[t],
                "proxy_observed": bool(proxy_obs[t]),
                "recent_treatment": recent_treatment[t],
                "a": int(treatment_type_by_day[t]),
                "treatment_burden": float(treatment_burden_by_day[t]),
                "side_effect_burden": float(side_effect_burden_by_day[t]),
            }
            for d in range(config.q):
                comp_row[f"C{d}"] = c[t, d]
                comp_row[f"B{d}"] = b_observed[t, d]
                comp_row[f"alpha{d}"] = alpha[t, d]
                comp_row[f"active{d}"] = bool(active_td[t, d])
            component_rows.append(comp_row)

            daily_row = {"id": person_id, "t": t}
            for j in range(config.p_daily):
                daily_row[f"X{j}"] = x[t, j]
                daily_row[f"obs_X{j}"] = bool(obs_mask[t, j])
            daily_rows.append(daily_row)

    individuals_df = pd.DataFrame(individuals)
    components_df = pd.DataFrame(component_rows)
    daily_df = apply_daily_missingness(pd.DataFrame(daily_rows), config)
    train_ids = set(split_df.loc[split_df["split"] == "train", "id"].astype(int))
    anchors_df = generate_anchors(components_df, config, train_ids=train_ids)
    treatments_df = pd.DataFrame(treatment_rows)
    metadata = {
        "seed": config.seed,
        "r_ts": config.r_ts,
        "r_ts_role": "loading_innovation_time_scale; marginal slow/fast variance is controlled by slow_fraction",
        "loading_drift_sd": config.loading_drift_sd,
        "alpha_innovation_sd": _alpha_innovation_sd(config),
        "alpha_innovation_sd_note": (
            "pre-rescaling scale; the realized loading innovations are multiplied "
            "by the per-person slow_scale applied to enforce slow_fraction"
        ),
        "alpha_action_drift": config.alpha_action_drift,
        "rdoc_drift_share": float(config.rdoc_drift_share),
        "rdoc_drift_active_dims": int(config.rdoc_drift_active_dims),
        "rdoc_drift_beta_seed": int(config.rdoc_drift_beta_seed),
        "rdoc_drift_beta_unit": rdoc_drift_beta_unit.tolist(),
        "rdoc_drift_scale_mean": float(np.mean(rdoc_drift_scales)) if rdoc_drift_scales else 0.0,
        "rdoc_drift_note": (
            "direct C_t beta contribution to latent drift; default share=0 preserves the original DGP"
        ),
        "support_switch_prob": config.support_switch_prob,
        "slow_fraction": config.slow_fraction,
        "support": "per-patient, time-varying sparse active set (adaptive-Lasso analogue)",
        "assignment": config.assignment,
        "assignment_gamma": assignment_gamma,
        "assignment_note": (
            "sequential severity-dependent hazard and protocol choice; gamma=0 "
            "reduces to uniform assignment with the configured count range"
        ),
    }
    return SimulationData(
        config=config,
        individuals=individuals_df,
        daily=daily_df,
        anchors=anchors_df,
        treatments=treatments_df,
        components=components_df,
        metadata=metadata,
    )
'''

# --- simulations.src.diagnostics (simulations/src/diagnostics.py) ---
SRC_SIMULATIONS_SRC_DIAGNOSTICS = r'''
"""Programmatic DGP diagnostics and sanity checks."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .model_utils import SimulationData


@dataclass
class DiagnosticResult:
    name: str
    value: float
    passed: bool
    message: str


def _corr(a: pd.Series, b: pd.Series) -> float:
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 3:
        return float("nan")
    return float(df.iloc[:, 0].corr(df.iloc[:, 1]))


def _within(value: float, lo: float, hi: float) -> bool:
    return bool(np.isfinite(value) and lo <= value <= hi)


def _add(
    results: list[DiagnosticResult],
    name: str,
    value: float,
    passed: bool,
    message: str,
) -> None:
    results.append(DiagnosticResult(name, float(value), bool(passed), message))


def run_diagnostics(data: SimulationData) -> list[DiagnosticResult]:
    config = data.config
    results: list[DiagnosticResult] = []

    expected_person_days = config.n * config.t
    _add(
        results,
        "component_row_count",
        len(data.components),
        len(data.components) == expected_person_days,
        "Component table should have one row per person-day.",
    )
    _add(
        results,
        "daily_row_count",
        len(data.daily),
        len(data.daily) == expected_person_days,
        "Daily table should have one row per person-day.",
    )
    component_duplicates = data.components.duplicated(["id", "t"]).sum()
    daily_duplicates = data.daily.duplicated(["id", "t"]).sum()
    anchor_duplicates = data.anchors.duplicated(["id", "anchor", "t"]).sum()
    _add(
        results,
        "component_duplicate_keys",
        component_duplicates,
        component_duplicates == 0,
        "Component keys should be unique by id,t.",
    )
    _add(
        results,
        "daily_duplicate_keys",
        daily_duplicates,
        daily_duplicates == 0,
        "Daily keys should be unique by id,t.",
    )
    _add(
        results,
        "anchor_duplicate_keys",
        anchor_duplicates,
        anchor_duplicates == 0,
        "Anchor keys should be unique by id,anchor,t.",
    )

    split_counts = data.individuals["split"].value_counts(normalize=True)
    target_splits = {
        "train": config.train_fraction,
        "validation": config.validation_fraction,
        "conformal": config.conformal_fraction,
        "test": config.test_fraction,
    }
    for split, target in target_splits.items():
        observed = float(split_counts.get(split, 0.0))
        tolerance = 1.0 / max(config.n, 1) + 0.02
        _add(
            results,
            f"split_{split}_fraction",
            observed,
            abs(observed - target) <= tolerance,
            f"{split} split fraction should be near configured target {target}.",
        )

    anchor_obs = data.anchors[data.anchors["observed"]].copy()
    for anchor in ["Y1", "Y2"]:
        sub = anchor_obs[anchor_obs["anchor"] == anchor]
        all_sub = data.anchors[data.anchors["anchor"] == anchor]
        # Under IRT the scalar value is missing by design; the integer total is the
        # observed readout that should track the recall window.
        anchor_signal = sub["value"] if str(config.anchor_observation).lower() == "gaussian" else sub["irt_total"]
        corr = _corr(anchor_signal, sub["window_mean_L"])
        # Bands reflect the spec anchor observation (unit loading). The upper bound
        # accommodates both the Gaussian total-score form and the slightly tighter
        # graded-response IRT form. Y2 has the shorter recall window and smaller
        # error_sd, so it tracks its window mean more tightly than Y1.
        lo, hi = (0.30, 0.96) if anchor == "Y1" else (0.30, 0.97)
        _add(
            results,
            f"{anchor}_latent_window_corr",
            corr,
            _within(corr, lo, hi),
            f"{anchor} anchor/window-L correlation should be between {lo} and {hi}",
        )
        observed_rate = float(all_sub["observed"].mean()) if len(all_sub) else 0.0
        expected_observed = 1.0 - getattr(config, anchor.lower()).missing_probability
        _add(
            results,
            f"{anchor}_observed_rate",
            observed_rate,
            abs(observed_rate - expected_observed) <= 0.08,
            f"{anchor} observed rate should be near configured missingness.",
        )

    both_anchor = anchor_obs.pivot_table(
        index=["id", "t"], columns="anchor", values=["value", "window_mean_L", "loading"]
    )
    # The cross-anchor residual correlation only has the configured Gaussian
    # structure under the Gaussian observation model; under graded-response IRT the
    # "residual" is item-sampling discretization with no rho_cross structure.
    if (
        str(config.anchor_observation).lower() == "gaussian"
        and ("value", "Y1") in both_anchor.columns
        and ("value", "Y2") in both_anchor.columns
    ):
        y1_error = both_anchor[("value", "Y1")] - both_anchor[("loading", "Y1")] * both_anchor[
            ("window_mean_L", "Y1")
        ]
        y2_error = both_anchor[("value", "Y2")] - both_anchor[("loading", "Y2")] * both_anchor[
            ("window_mean_L", "Y2")
        ]
        cross_corr = _corr(y1_error, y2_error)
        _add(
            results,
            "anchor_error_cross_corr",
            cross_corr,
            np.isfinite(cross_corr) and abs(cross_corr - config.rho_cross) <= 0.25,
            "Cross-anchor residual correlation should be near configured rho_cross.",
        )

    missing_cols = [c for c in data.daily.columns if c.startswith("obs_X")]
    observed_rate = float(data.daily[missing_cols].to_numpy(dtype=float).mean())
    target_observed = 1.0 - config.daily_base_missing_probability
    _add(
        results,
        "daily_observed_rate",
        observed_rate,
        abs(observed_rate - target_observed) <= 0.10,
        "Daily observed rate should be within 10 pp of target in the scaffold.",
    )

    daily_truth = data.daily.merge(data.components[["id", "t", "L", "delta", "C0"]], on=["id", "t"])
    strong_proxy_corr = _corr(daily_truth["X0"], daily_truth["L"])
    slow_proxy_corr = _corr(daily_truth["X8"], daily_truth["C0"])
    # Fast-residual proxies track the zero-baseline dynamic residual (recent
    # shocks + treatment response), not the per-person level, so compare against
    # the within-person-centered delta rather than the level-laden delta.
    daily_truth["delta_within"] = daily_truth.groupby("id")["delta"].transform(lambda s: s - s.mean())
    fast_proxy_corr = _corr(daily_truth["X12"], daily_truth["delta_within"])
    noise_corr = abs(_corr(daily_truth["X20"], daily_truth["L"]))
    _add(
        results,
        "daily_strong_proxy_corr",
        strong_proxy_corr,
        _within(strong_proxy_corr, 0.50, 0.95),
        "Strong daily proxy variables should correlate with latent state.",
    )
    _add(
        results,
        "daily_slow_proxy_corr",
        slow_proxy_corr,
        _within(slow_proxy_corr, 0.45, 0.95),
        "Slow-proxy daily variables should track C(t), not just B(t).",
    )
    _add(
        results,
        "daily_fast_proxy_corr",
        fast_proxy_corr,
        _within(fast_proxy_corr, 0.45, 0.95),
        "Fast-proxy daily variables should track delta(t).",
    )
    _add(
        results,
        "daily_noise_abs_corr",
        noise_corr,
        np.isfinite(noise_corr) and noise_corr <= 0.12,
        "Noise daily variables should be nearly uncorrelated with latent state.",
    )

    proxy_rate = float(data.components["proxy_observed"].mean())
    _add(
        results,
        "proxy_observed_rate",
        proxy_rate,
        0.05 <= proxy_rate <= 1.00,
        "Proxy observation timing should produce at least some observed proxy values.",
    )
    c_cols = [f"C{d}" for d in range(config.q)]
    b_cols = [f"B{d}" for d in range(config.q)]
    proxy_corrs = []
    for c_col, b_col in zip(c_cols, b_cols):
        proxy_corrs.append(_corr(data.components[c_col], data.components[b_col]))
    mean_proxy_corr = float(np.nanmean(proxy_corrs))
    _add(
        results,
        "proxy_reliability_mean_corr",
        mean_proxy_corr,
        _within(mean_proxy_corr, 0.35, 0.98),
        "Observed B(t) should be a noisy but useful proxy for C(t).",
    )

    # slow_fraction controls the within-person dynamic decomposition. Because the
    # residual delta now carries the per-person level (spec: z = alpha'RDoC +
    # delta, no separate intercept), the ratio is computed on within-person
    # (mean-removed) variances so the between-person level does not inflate delta.
    slow_within = data.components.groupby("id")["slow"].transform(lambda s: s - s.mean())
    delta_within = data.components.groupby("id")["delta"].transform(lambda s: s - s.mean())
    slow_var = float(slow_within.var())
    fast_var = float(delta_within.var())
    r_empirical = slow_var / max(fast_var, 1e-8)
    target_ratio = config.slow_fraction / max(1.0 - config.slow_fraction, 1e-8)
    _add(
        results,
        "slow_fast_variance_ratio",
        r_empirical,
        np.isfinite(r_empirical) and abs(r_empirical - target_ratio) <= 0.05,
        "Within-person slow and fast component variance ratio should match configured slow fraction.",
    )
    alpha_cols = [f"alpha{d}" for d in range(config.q)]
    active_cols = [f"active{d}" for d in range(config.q)]
    inactive_values = []
    for alpha_col, active_col in zip(alpha_cols, active_cols):
        inactive_values.append(data.components.loc[~data.components[active_col], alpha_col].abs().max())
    max_inactive_alpha = float(np.nanmax(inactive_values))
    _add(
        results,
        "inactive_alpha_max_abs",
        max_inactive_alpha,
        max_inactive_alpha <= 1e-10,
        "Inactive alpha dimensions should remain exactly zero in the DGP.",
    )

    treatment_counts = data.treatments.groupby("id").size()
    min_count = float(treatment_counts.min()) if len(treatment_counts) else 0.0
    max_count = float(treatment_counts.max()) if len(treatment_counts) else 0.0
    _add(
        results,
        "treatment_count_min",
        min_count,
        min_count >= config.min_treatment_count,
        "Each individual should meet the configured minimum treatment count.",
    )
    _add(
        results,
        "treatment_count_max",
        max_count,
        max_count <= config.max_treatment_count,
        "Each individual should not exceed the configured maximum treatment count.",
    )
    return results


def diagnostics_frame(results: list[DiagnosticResult]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])


def assert_diagnostics_pass(results: list[DiagnosticResult]) -> None:
    failed = [r for r in results if not r.passed]
    if failed:
        messages = "; ".join(f"{r.name}={r.value:.3g}: {r.message}" for r in failed)
        raise AssertionError(f"DGP diagnostics failed: {messages}")
'''

# --- simulations.src.metrics (simulations/src/metrics.py) ---
SRC_SIMULATIONS_SRC_METRICS = r'''
"""Recovery, coverage, and interval metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _with_split(truth: pd.DataFrame, individuals: pd.DataFrame | None = None) -> pd.DataFrame:
    if individuals is None or "split" in truth.columns:
        return truth
    return truth.merge(individuals[["id", "split"]], on="id", how="left")


def _filter_split(df: pd.DataFrame, split: str | None) -> pd.DataFrame:
    if split is None:
        return df
    if "split" not in df.columns:
        raise ValueError("split filtering requested but no split column is available.")
    return df[df["split"] == split]


def _rmse(estimate: pd.Series, truth: pd.Series) -> float:
    return float(np.sqrt(np.mean((estimate - truth) ** 2)))


def latent_rmse(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    individuals: pd.DataFrame | None = None,
    split: str | None = None,
) -> float:
    truth = _with_split(truth, individuals)
    merged = predictions.merge(truth[["id", "t", "L"] + (["split"] if "split" in truth.columns else [])], on=["id", "t"], how="inner")
    merged = _filter_split(merged, split)
    return _rmse(merged["L_hat"], merged["L"])


def latent_mae(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    individuals: pd.DataFrame | None = None,
    split: str | None = None,
) -> float:
    truth = _with_split(truth, individuals)
    merged = predictions.merge(truth[["id", "t", "L"] + (["split"] if "split" in truth.columns else [])], on=["id", "t"], how="inner")
    merged = _filter_split(merged, split)
    return float(np.mean(np.abs(merged["L_hat"] - merged["L"])))


def component_rmse(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    estimate_col: str,
    truth_col: str,
    individuals: pd.DataFrame | None = None,
    split: str | None = None,
) -> float:
    if estimate_col not in predictions.columns:
        return float("nan")
    truth = _with_split(truth, individuals)
    cols = ["id", "t", truth_col] + (["split"] if "split" in truth.columns else [])
    merged = predictions.merge(truth[cols], on=["id", "t"], how="inner")
    merged = _filter_split(merged, split)
    return _rmse(merged[estimate_col], merged[truth_col])


def double_anchor_latent_metrics(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    individuals: pd.DataFrame | None = None,
    split: str | None = None,
) -> dict[str, float]:
    """Metrics for PDF-style PHQ/PCL latent channels when available."""

    required = {"z_d_hat", "z_p_hat"}
    truth_required = {"z_d", "z_p"}
    if not required.issubset(predictions.columns) or not truth_required.issubset(truth.columns):
        return {
            "z_d_rmse": float("nan"),
            "z_p_rmse": float("nan"),
            "z_joint_rmse": float("nan"),
            "delta_d_rmse": float("nan"),
            "delta_p_rmse": float("nan"),
        }
    truth = _with_split(truth, individuals)
    cols = ["id", "t", "z_d", "z_p"] + (["split"] if "split" in truth.columns else [])
    for col in ["delta_d", "delta_p"]:
        if col in truth.columns:
            cols.append(col)
    pred_cols = ["id", "t", "z_d_hat", "z_p_hat"]
    for col in ["delta_d_hat", "delta_p_hat"]:
        if col in predictions.columns:
            pred_cols.append(col)
    merged = predictions[pred_cols].merge(truth[cols], on=["id", "t"], how="inner")
    merged = _filter_split(merged, split)
    out = {
        "z_d_rmse": _rmse(merged["z_d_hat"], merged["z_d"]),
        "z_p_rmse": _rmse(merged["z_p_hat"], merged["z_p"]),
        "z_joint_rmse": _rmse(0.5 * (merged["z_d_hat"] + merged["z_p_hat"]), 0.5 * (merged["z_d"] + merged["z_p"])),
        "delta_d_rmse": float("nan"),
        "delta_p_rmse": float("nan"),
    }
    if {"delta_d_hat", "delta_d"}.issubset(merged.columns):
        out["delta_d_rmse"] = _rmse(merged["delta_d_hat"], merged["delta_d"])
    if {"delta_p_hat", "delta_p"}.issubset(merged.columns):
        out["delta_p_rmse"] = _rmse(merged["delta_p_hat"], merged["delta_p"])
    return out


def leakage_corr(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    estimate_col: str,
    truth_col: str,
    individuals: pd.DataFrame | None = None,
    split: str | None = None,
) -> float:
    if estimate_col not in predictions.columns:
        return float("nan")
    truth = _with_split(truth, individuals)
    cols = ["id", "t", truth_col] + (["split"] if "split" in truth.columns else [])
    merged = predictions.merge(truth[cols], on=["id", "t"], how="inner")
    merged = _filter_split(merged, split).dropna(subset=[estimate_col, truth_col])
    if len(merged) < 3:
        return float("nan")
    # Center within person before correlating: a pooled correlation conflates
    # between-person level differences with within-person day-to-day leakage, so
    # a method that nails person means but mislocates daily variation would
    # spuriously pass. Leakage is a within-person dynamic property.
    est = merged[estimate_col].to_numpy(dtype=float) - merged.groupby("id")[estimate_col].transform("mean").to_numpy(dtype=float)
    tru = merged[truth_col].to_numpy(dtype=float) - merged.groupby("id")[truth_col].transform("mean").to_numpy(dtype=float)
    if np.std(est) < 1e-12 or np.std(tru) < 1e-12:
        return float("nan")
    return float(np.corrcoef(est, tru)[0, 1])


def coverage(
    intervals: pd.DataFrame,
    truth: pd.DataFrame,
    individuals: pd.DataFrame | None = None,
    split: str | None = None,
) -> float:
    truth = _with_split(truth, individuals)
    cols = ["id", "t", "L"] + (["split"] if "split" in truth.columns else [])
    merged = intervals.merge(truth[cols], on=["id", "t"], how="inner")
    merged = _filter_split(merged, split)
    if merged.empty:
        return float("nan")
    covered = (merged["lower"] <= merged["L"]) & (merged["L"] <= merged["upper"])
    # Replicate-level coverage: aggregate per person first (equal weight per
    # patient), then average across patients. Person-day pooling is prohibited
    # in primary tables because patients with more observed sessions would
    # otherwise dominate the statistic.
    per_person = covered.groupby(merged["id"]).mean()
    return float(per_person.mean())


def mean_width(
    intervals: pd.DataFrame,
    individuals: pd.DataFrame | None = None,
    split: str | None = None,
) -> float:
    data = intervals
    if split is not None:
        if individuals is None:
            raise ValueError("split filtering requested but no individuals table was supplied.")
        data = intervals.merge(individuals[["id", "split"]], on="id", how="left")
        data = _filter_split(data, split)
    if data.empty:
        return float("nan")
    # Replicate-level width: per-person mean first, then average across patients,
    # consistent with coverage() and interval_score() so the three columns
    # aggregate over the same units.
    widths = (data["upper"] - data["lower"]).to_numpy(dtype=float)
    per_person = pd.Series(widths, index=data["id"].to_numpy()).groupby(level=0).mean()
    return float(per_person.mean())


def interval_score(
    intervals: pd.DataFrame,
    truth: pd.DataFrame,
    alpha: float = 0.05,
    individuals: pd.DataFrame | None = None,
    split: str | None = None,
) -> float:
    """Mean interval (Winkler) score for the latent composite L.

    IS_alpha = width + (2/alpha)(lower - L) I[L < lower] + (2/alpha)(L - upper) I[L > upper].

    Rewards intervals that are simultaneously narrow and calibrated; reported
    alongside coverage and width because coverage alone is gameable by widening.
    Aggregated per person first (replicate-level), then averaged across patients.
    """

    truth = _with_split(truth, individuals)
    cols = ["id", "t", "L"] + (["split"] if "split" in truth.columns else [])
    merged = intervals.merge(truth[cols], on=["id", "t"], how="inner")
    merged = _filter_split(merged, split)
    if merged.empty:
        return float("nan")
    lower = merged["lower"].to_numpy(dtype=float)
    upper = merged["upper"].to_numpy(dtype=float)
    L = merged["L"].to_numpy(dtype=float)
    width = upper - lower
    below = (L < lower).astype(float)
    above = (L > upper).astype(float)
    score = width + (2.0 / alpha) * (lower - L) * below + (2.0 / alpha) * (L - upper) * above
    per_person = pd.Series(score, index=merged["id"].to_numpy()).groupby(level=0).mean()
    return float(per_person.mean())


def channel_interval_metrics(
    intervals: pd.DataFrame,
    truth: pd.DataFrame,
    individuals: pd.DataFrame | None = None,
    split: str | None = None,
) -> dict[str, float]:
    """Coverage and width for PDF disorder-specific severity intervals."""

    required = {"z_d_lower", "z_d_upper", "z_p_lower", "z_p_upper"}
    truth_required = {"z_d", "z_p"}
    if intervals is None or not required.issubset(intervals.columns) or not truth_required.issubset(truth.columns):
        return {
            "z_d_coverage": float("nan"),
            "z_p_coverage": float("nan"),
            "z_d_mean_width": float("nan"),
            "z_p_mean_width": float("nan"),
        }
    truth = _with_split(truth, individuals)
    cols = ["id", "t", "z_d", "z_p"] + (["split"] if "split" in truth.columns else [])
    merged = intervals[
        ["id", "t", "z_d_lower", "z_d_upper", "z_p_lower", "z_p_upper"]
    ].merge(truth[cols], on=["id", "t"], how="inner")
    merged = _filter_split(merged, split)
    z_d_covered = (merged["z_d_lower"] <= merged["z_d"]) & (merged["z_d"] <= merged["z_d_upper"])
    z_p_covered = (merged["z_p_lower"] <= merged["z_p"]) & (merged["z_p"] <= merged["z_p_upper"])
    # Replicate-level (per-person then average), consistent with coverage(); not
    # person-day pooled. Widths use the same aggregation as mean_width().
    ids = merged["id"]
    z_d_width = merged["z_d_upper"] - merged["z_d_lower"]
    z_p_width = merged["z_p_upper"] - merged["z_p_lower"]
    return {
        "z_d_coverage": float(z_d_covered.groupby(ids).mean().mean()),
        "z_p_coverage": float(z_p_covered.groupby(ids).mean().mean()),
        "z_d_mean_width": float(z_d_width.groupby(ids).mean().mean()),
        "z_p_mean_width": float(z_p_width.groupby(ids).mean().mean()),
    }


def active_set_metrics(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    q: int,
    tau: float = 0.05,
    individuals: pd.DataFrame | None = None,
    split: str | None = None,
) -> dict[str, float]:
    """Per-session active-set recovery for the time-varying sparse support.

    Because the recovery recipe's support shifts over time (spec adaptive Lasso),
    recovery is scored per session-dimension: at each (patient, session, RDoC
    dimension), the dimension is truly active when its loading is nonzero, and is
    predicted active when the fitted loading exceeds the threshold tau. The top-k
    variant uses, per session, the true number of active dimensions.
    """

    alpha_cols = [f"alpha_hat{d}" for d in range(q)]
    active_cols = [f"active{d}" for d in range(q)]
    if not all(col in predictions.columns for col in alpha_cols):
        return {
            "active_precision": float("nan"),
            "active_recall": float("nan"),
            "active_f1": float("nan"),
            "active_accuracy": float("nan"),
        }
    truth = _with_split(truth, individuals)
    truth_cols = ["id", "t"] + active_cols + (["split"] if "split" in truth.columns else [])
    merged = predictions[["id", "t"] + alpha_cols].merge(truth[truth_cols], on=["id", "t"], how="inner")
    merged = _filter_split(merged, split)
    if merged.empty:
        return {
            "active_precision": float("nan"),
            "active_recall": float("nan"),
            "active_f1": float("nan"),
            "active_accuracy": float("nan"),
        }
    alpha_abs = merged[alpha_cols].abs().to_numpy(dtype=float)  # (rows, q)
    pred_active = alpha_abs >= tau
    true_active = merged[active_cols].to_numpy(dtype=bool)       # (rows, q)
    tp = float(np.logical_and(pred_active, true_active).sum())
    fp = float(np.logical_and(pred_active, ~true_active).sum())
    fn = float(np.logical_and(~pred_active, true_active).sum())
    tn = float(np.logical_and(~pred_active, ~true_active).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
    recall = tp / (tp + fn) if tp + fn > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0 else float("nan")
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1.0)
    # Top-k per session: predict the k strongest loadings active, k = true count.
    # Vectorized: rank each dimension within its row by descending |alpha| and
    # mark it active when its rank is below that row's true active count.
    ranks = np.argsort(np.argsort(-alpha_abs, axis=1), axis=1)  # rank 0 = strongest
    k_per_row = true_active.sum(axis=1, keepdims=True)
    topk_pred = ranks < k_per_row
    topk_tp = float(np.logical_and(topk_pred, true_active).sum())
    topk_fp = float(np.logical_and(topk_pred, ~true_active).sum())
    topk_fn = float(np.logical_and(~topk_pred, true_active).sum())
    topk_tn = float(np.logical_and(~topk_pred, ~true_active).sum())
    topk_precision = topk_tp / (topk_tp + topk_fp) if topk_tp + topk_fp > 0 else float("nan")
    topk_recall = topk_tp / (topk_tp + topk_fn) if topk_tp + topk_fn > 0 else float("nan")
    topk_f1 = (
        2 * topk_precision * topk_recall / (topk_precision + topk_recall)
        if np.isfinite(topk_precision) and np.isfinite(topk_recall) and topk_precision + topk_recall > 0
        else float("nan")
    )
    topk_accuracy = (topk_tp + topk_tn) / max(topk_tp + topk_fp + topk_fn + topk_tn, 1.0)
    return {
        "active_precision": float(precision),
        "active_recall": float(recall),
        "active_f1": float(f1),
        "active_accuracy": float(accuracy),
        "active_topk_precision": float(topk_precision),
        "active_topk_recall": float(topk_recall),
        "active_topk_f1": float(topk_f1),
        "active_topk_accuracy": float(topk_accuracy),
    }


def proxy_state_metrics(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    q: int,
    individuals: pd.DataFrame | None = None,
    split: str | None = None,
) -> dict[str, float]:
    """Summarize denoised C_hat recovery across proxy dimensions."""

    estimate_cols = [f"C_hat{d}" for d in range(q)]
    truth_cols = [f"C{d}" for d in range(q)]
    if not all(col in predictions.columns for col in estimate_cols):
        return {"c_proxy_rmse": float("nan"), "c_proxy_corr": float("nan")}
    truth = _with_split(truth, individuals)
    cols = ["id", "t"] + truth_cols + (["split"] if "split" in truth.columns else [])
    merged = predictions[["id", "t"] + estimate_cols].merge(truth[cols], on=["id", "t"], how="inner")
    merged = _filter_split(merged, split)
    rmses = []
    corrs = []
    for d in range(q):
        estimate = merged[f"C_hat{d}"]
        target = merged[f"C{d}"]
        rmses.append(_rmse(estimate, target))
        corr = estimate.corr(target)
        if np.isfinite(corr):
            corrs.append(float(corr))
    return {
        "c_proxy_rmse": float(np.mean(rmses)) if rmses else float("nan"),
        "c_proxy_corr": float(np.mean(corrs)) if corrs else float("nan"),
    }


def anchor_conformal_residuals(
    predictions: pd.DataFrame,
    anchors: pd.DataFrame,
    individuals: pd.DataFrame,
    split: str,
    anchor_name: str,
    channel_hat_col: str,
) -> np.ndarray:
    """Absolute residuals between observed anchors and recall-windowed predictions.

    This is the appendix Sec. 5 conformity score R_i = |y_i - f_hat(x_i)|, where
    f_hat is the model's prediction of the *observable anchor* (recall-windowed
    mean of the predicted channel times the anchor loading), NOT the unobservable
    latent. Calibrating on these scores is what makes the coverage guarantee
    deployable: in the empirical study the latent is never measured, so conformal
    residuals can only be formed against anchors. Latent coverage against truth is
    reported separately as a simulation-only diagnostic.
    """

    if channel_hat_col not in predictions.columns:
        return np.array([], dtype=float)
    split_ids = set(individuals.loc[individuals["split"] == split, "id"]) if split is not None else set(individuals["id"])
    anc = anchors[
        (anchors["anchor"] == anchor_name)
        & (anchors["observed"])
        & (anchors["id"].isin(split_ids))
    ]
    if anc.empty:
        return np.array([], dtype=float)
    pred = predictions[["id", "t", channel_hat_col]]
    by_id = {pid: frame.set_index("t")[channel_hat_col] for pid, frame in pred.groupby("id", sort=False)}
    residuals: list[float] = []
    for row in anc.itertuples(index=False):
        series = by_id.get(int(row.id))
        if series is None:
            continue
        window = series[(series.index >= int(row.window_start)) & (series.index <= int(row.window_end))]
        if window.empty or not np.isfinite(row.value):
            continue
        predicted = float(row.loading) * float(window.mean())
        residuals.append(abs(float(row.value) - predicted))
    return np.asarray(residuals, dtype=float)


def rdoc_recovery_fraction(
    predictions: pd.DataFrame,
    data,
    split: str | None = None,
) -> float:
    """Estimate rho_RDoC = Var(alpha'RDoC) / (Var(z_d) + Var(z_p)) from fitted outputs.

    This is the falsifiable latent-model claim from the appendix ("What this
    buys"), the share of recovery variance explained by RDoC features. It is a
    section-3 latent-recovery quantity.
    """

    # Numerator is the slow component variance Var(alpha'RDoC). The fitted
    # estimate of that quantity is the proxy slow component alpha_hat dot B
    # (proxy_slow_hat), NOT slow_hat = z - delta: the latter inherits the full z
    # signal and double-counts against the z denominator. Fall back to slow_hat
    # only if the proxy column is unavailable.
    slow_col = "proxy_slow_hat" if "proxy_slow_hat" in predictions.columns else "slow_hat"
    required = {slow_col, "z_d_hat", "z_p_hat"}
    if not required.issubset(predictions.columns):
        return float("nan")
    frame = predictions[["id", "t", slow_col, "z_d_hat", "z_p_hat"]].merge(
        data.individuals[["id", "split"]],
        on="id",
        how="left",
    )
    if split is not None:
        frame = frame[frame["split"] == split]
    if frame.empty:
        return float("nan")
    # Within-person variance: the per-patient level lives in delta (spec
    # z = alpha'RDoC + delta, no intercept), so a pooled variance would inflate
    # the denominator with between-patient level and deflate rho. rho_RDoC is a
    # statement about the within-person dynamic decomposition, so center each
    # quantity per patient before taking variances.
    def _within_var(column: str) -> float:
        centered = frame[column].to_numpy(dtype=float) - frame.groupby("id")[column].transform("mean").to_numpy(dtype=float)
        return float(np.var(centered, ddof=0))

    numerator = _within_var(slow_col)
    denominator = _within_var("z_d_hat") + _within_var("z_p_hat")
    return numerator / denominator if denominator > 1e-12 else float("nan")


def identifiability_diagnostics(
    predictions: pd.DataFrame,
    data,
    split: str | None = None,
) -> dict[str, float]:
    """Post-fit structural-identifiability diagnostics.

    The decomposition claim (slow alpha'RDoC vs fast delta) requires more than
    good anchor fit: the recovered slow component must track the TRUE slow
    component and not the fast one, and vice versa. These within-person
    correlations expose signal swapping. A model that predicts anchors well but
    assigns fast shocks to the slow channel fails the structural claim even with
    low anchor RMSE.

    Reported: recovery correlations (want high), cross leakage correlations (want
    low), within-person component-variance recovery ratios (want ~1), and the
    mean per-step loading change (loading stability).
    """

    truth = data.components[["id", "t", "slow", "delta"]]
    ind = data.individuals
    slow_col = "proxy_slow_hat" if "proxy_slow_hat" in predictions.columns else "slow_hat"
    out: dict[str, float] = {
        "slow_recovery_corr": leakage_corr(predictions, truth, slow_col, "slow", ind, split),
        "fast_recovery_corr": leakage_corr(predictions, truth, "delta_hat", "delta", ind, split),
        "slow_to_fast_leakage": leakage_corr(predictions, truth, slow_col, "delta", ind, split),
        "fast_to_slow_leakage": leakage_corr(predictions, truth, "delta_hat", "slow", ind, split),
    }
    merged = predictions.merge(truth, on=["id", "t"], how="inner").merge(
        ind[["id", "split"]], on="id", how="left"
    )
    if split is not None:
        merged = merged[merged["split"] == split]

    def _wv(col: str) -> float:
        x = merged[col].to_numpy(dtype=float)
        m = merged.groupby("id")[col].transform("mean").to_numpy(dtype=float)
        return float(np.var(x - m))

    out["slow_var_recovery"] = (
        _wv(slow_col) / max(_wv("slow"), 1e-9) if slow_col in merged.columns and not merged.empty else float("nan")
    )
    out["fast_var_recovery"] = (
        _wv("delta_hat") / max(_wv("delta"), 1e-9) if "delta_hat" in merged.columns and not merged.empty else float("nan")
    )
    alpha_cols = [c for c in predictions.columns if c.startswith("alpha_hat")]
    if alpha_cols:
        am = predictions[["id", "t"] + alpha_cols].merge(ind[["id", "split"]], on="id", how="left")
        if split is not None:
            am = am[am["split"] == split]
        steps = []
        for _, g in am.sort_values("t").groupby("id"):
            a = g[alpha_cols].to_numpy(dtype=float)
            if len(a) > 1:
                steps.append(float(np.mean(np.abs(np.diff(a, axis=0)))))
        out["loading_step_mean_abs"] = float(np.mean(steps)) if steps else float("nan")
    else:
        out["loading_step_mean_abs"] = float("nan")
    return out


def latent_relative_width(
    intervals: pd.DataFrame,
    truth: pd.DataFrame,
    individuals: pd.DataFrame | None = None,
    split: str | None = None,
) -> float:
    """Mean interval width divided by the marginal SD of the latent state.

    Reports width on an interpretable scale (units of latent standard
    deviation), so coverage near 1.0 at a given width can be judged for
    informativeness. The denominator is the marginal SD of L on the evaluated
    split.
    """

    truth = _with_split(truth, individuals)
    cols = ["id", "t", "L"] + (["split"] if "split" in truth.columns else [])
    merged = intervals.merge(truth[cols], on=["id", "t"], how="inner")
    merged = _filter_split(merged, split)
    if merged.empty:
        return float("nan")
    sd = float(np.std(merged["L"].to_numpy(dtype=float)))
    if sd < 1e-8:
        return float("nan")
    widths = (merged["upper"] - merged["lower"]).to_numpy(dtype=float)
    per_person = pd.Series(widths, index=merged["id"].to_numpy()).groupby(level=0).mean()
    return float(per_person.mean() / sd)


def _nearest_anchor_distance(intervals: pd.DataFrame, anchors: pd.DataFrame) -> pd.Series:
    """Days from each (id, t) to the nearest observed anchor of any channel."""

    obs = anchors[anchors["observed"]] if "observed" in anchors.columns else anchors
    anchor_days = {pid: np.sort(g["t"].to_numpy()) for pid, g in obs.groupby("id")}
    out = np.full(len(intervals), np.nan)
    ids = intervals["id"].to_numpy()
    ts = intervals["t"].to_numpy()
    for r in range(len(intervals)):
        days = anchor_days.get(int(ids[r]))
        if days is None or len(days) == 0:
            continue
        out[r] = float(np.min(np.abs(days - ts[r])))
    return pd.Series(out, index=intervals.index)


def coverage_by_anchor_distance(
    intervals: pd.DataFrame,
    truth: pd.DataFrame,
    anchors: pd.DataFrame,
    individuals: pd.DataFrame | None = None,
    split: str | None = None,
    bins: tuple[int, ...] = (0, 1, 2, 4, 7, 1000),
) -> pd.DataFrame:
    """Latent coverage and relative width stratified by days to nearest anchor.

    Tests the overconfidence-under-weak-information risk. Days far from any
    observed anchor carry less direct information, so honest intervals should
    widen and coverage should hold there. Returns one row per distance bin with
    per-person-aggregated coverage and width in latent SD units.
    """

    truth = _with_split(truth, individuals)
    cols = ["id", "t", "L"] + (["split"] if "split" in truth.columns else [])
    merged = intervals.merge(truth[cols], on=["id", "t"], how="inner")
    merged = _filter_split(merged, split).reset_index(drop=True)
    if merged.empty:
        return pd.DataFrame(columns=["bin", "coverage", "rel_width", "n"])
    merged["dist"] = _nearest_anchor_distance(merged, anchors).to_numpy()
    merged = merged.dropna(subset=["dist"])
    sd = float(np.std(merged["L"].to_numpy(dtype=float))) or 1.0
    covered = (merged["lower"] <= merged["L"]) & (merged["L"] <= merged["upper"])
    width = (merged["upper"] - merged["lower"]).to_numpy(dtype=float)
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
    binned = pd.cut(merged["dist"], bins=list(bins), labels=labels, include_lowest=True, right=False)
    rows = []
    for lab in labels:
        mask = (binned == lab).to_numpy()
        if not mask.any():
            rows.append({"bin": lab, "coverage": float("nan"), "rel_width": float("nan"), "n": 0})
            continue
        ids = merged.loc[mask, "id"].to_numpy()
        cov = pd.Series(covered.to_numpy()[mask], index=ids).groupby(level=0).mean().mean()
        wid = pd.Series(width[mask] / sd, index=ids).groupby(level=0).mean().mean()
        rows.append({"bin": lab, "coverage": float(cov), "rel_width": float(wid), "n": int(mask.sum())})
    return pd.DataFrame(rows)


def coverage_by_daily_missingness(
    intervals: pd.DataFrame,
    truth: pd.DataFrame,
    daily: pd.DataFrame,
    individuals: pd.DataFrame | None = None,
    split: str | None = None,
    bins: tuple[float, ...] = (0.0, 0.5, 0.7, 0.85, 1.01),
) -> pd.DataFrame:
    """Latent coverage and relative width stratified by daily missingness.

    For each (id, t) the daily missingness fraction is one minus the mean of the
    obs_X observation flags. Days with more missing daily covariates carry less
    information, so honest intervals should widen there. Returns one row per
    missingness bin.
    """

    obs_cols = [c for c in daily.columns if c.startswith("obs_X")]
    if not obs_cols:
        return pd.DataFrame(columns=["bin", "coverage", "rel_width", "n"])
    miss = daily[["id", "t"]].copy()
    miss["missing_frac"] = 1.0 - daily[obs_cols].to_numpy(dtype=float).mean(axis=1)
    truth = _with_split(truth, individuals)
    cols = ["id", "t", "L"] + (["split"] if "split" in truth.columns else [])
    merged = intervals.merge(truth[cols], on=["id", "t"], how="inner").merge(miss, on=["id", "t"], how="inner")
    merged = _filter_split(merged, split).reset_index(drop=True)
    if merged.empty:
        return pd.DataFrame(columns=["bin", "coverage", "rel_width", "n"])
    sd = float(np.std(merged["L"].to_numpy(dtype=float))) or 1.0
    covered = (merged["lower"] <= merged["L"]) & (merged["L"] <= merged["upper"])
    width = (merged["upper"] - merged["lower"]).to_numpy(dtype=float)
    labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins) - 1)]
    binned = pd.cut(merged["missing_frac"], bins=list(bins), labels=labels, include_lowest=True, right=False)
    rows = []
    for lab in labels:
        mask = (binned == lab).to_numpy()
        if not mask.any():
            rows.append({"bin": lab, "coverage": float("nan"), "rel_width": float("nan"), "n": 0})
            continue
        ids = merged.loc[mask, "id"].to_numpy()
        cov = pd.Series(covered.to_numpy()[mask], index=ids).groupby(level=0).mean().mean()
        wid = pd.Series(width[mask] / sd, index=ids).groupby(level=0).mean().mean()
        rows.append({"bin": lab, "coverage": float(cov), "rel_width": float(wid), "n": int(mask.sum())})
    return pd.DataFrame(rows)


def summarize_mcse(values: pd.Series) -> dict[str, float]:
    clean = values.dropna().astype(float)
    n = int(len(clean))
    if n == 0:
        return {"mean": float("nan"), "sd": float("nan"), "mcse": float("nan"), "n": 0.0}
    sd = float(clean.std(ddof=1)) if n > 1 else 0.0
    return {
        "mean": float(clean.mean()),
        "sd": sd,
        "mcse": sd / np.sqrt(n) if n > 0 else float("nan"),
        "n": float(n),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }
'''

# --- simulations.src.methods.baselines (simulations/src/methods/baselines.py) ---
SRC_SIMULATIONS_SRC_METHODS_BASELINES = r'''
"""Cheap baseline methods for sparse anchors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..model_utils import MethodResult, SimulationData


def _anchor_series(data: SimulationData, anchor: str = "Y1") -> pd.DataFrame:
    obs = data.anchors[(data.anchors["anchor"] == anchor) & (data.anchors["observed"])].copy()
    return obs[["id", "t", "value"]].sort_values(["id", "t"])


def linear_interpolation(data: SimulationData, anchor: str = "Y1") -> MethodResult:
    rows: list[pd.DataFrame] = []
    anchors = _anchor_series(data, anchor)
    for person_id, comp in data.components.groupby("id", sort=True):
        days = comp["t"].to_numpy()
        sub = anchors[anchors["id"] == person_id]
        if len(sub) == 0:
            pred = np.repeat(np.nan, len(days))
        elif len(sub) == 1:
            pred = np.repeat(float(sub["value"].iloc[0]), len(days))
        else:
            pred = np.interp(days, sub["t"].to_numpy(), sub["value"].to_numpy())
        rows.append(pd.DataFrame({"id": person_id, "t": days, "L_hat": pred}))
    return MethodResult("linear_interpolation", pd.concat(rows, ignore_index=True))


def locf(data: SimulationData, anchor: str = "Y1") -> MethodResult:
    """Strictly causal last-observation-carried-forward on the anchor.

    Days before the first observed anchor are filled with 0.0, the population
    prior mean on the standardized anchor scale; backfilling with the first
    anchor would leak future information into a baseline that is meant to be a
    filter.
    """

    rows: list[pd.DataFrame] = []
    anchors = _anchor_series(data, anchor)
    for person_id, comp in data.components.groupby("id", sort=True):
        days = comp["t"].to_numpy()
        sub = anchors[anchors["id"] == person_id].set_index("t")["value"]
        values = []
        last = 0.0
        for day in days:
            if day in sub.index:
                last = float(sub.loc[day])
            values.append(last)
        rows.append(pd.DataFrame({"id": person_id, "t": days, "L_hat": values}))
    return MethodResult("locf", pd.concat(rows, ignore_index=True))


def anchor_only_windowed_mean(data: SimulationData) -> MethodResult:
    """Simple windowed-anchor baseline: assign each day mean covering anchor value.

    Each anchor value is divided by its loading before averaging so the two
    anchors contribute on a common latent scale (Y2's 0.75 loading would
    otherwise bias the level of every day its window covers).
    """

    rows = []
    for person_id, comp in data.components.groupby("id", sort=True):
        pred = np.full(len(comp), np.nan)
        counts = np.zeros(len(comp))
        obs = data.anchors[(data.anchors["id"] == person_id) & (data.anchors["observed"])]
        for _, row in obs.iterrows():
            start = int(row["window_start"])
            end = int(row["window_end"])
            loading = float(row["loading"]) if np.isfinite(row["loading"]) and row["loading"] != 0 else 1.0
            pred[start : end + 1] = np.nan_to_num(pred[start : end + 1], nan=0.0) + float(row["value"]) / loading
            counts[start : end + 1] += 1.0
        has = counts > 0
        pred[has] = pred[has] / counts[has]
        if np.any(~has) and np.any(has):
            pred[~has] = np.interp(comp["t"].to_numpy()[~has], comp["t"].to_numpy()[has], pred[has])
        rows.append(pd.DataFrame({"id": person_id, "t": comp["t"].to_numpy(), "L_hat": pred}))
    return MethodResult("anchor_only_windowed_mean", pd.concat(rows, ignore_index=True))
'''

# --- simulations.src.methods.s0 (simulations/src/methods/s0.py) ---
SRC_SIMULATIONS_SRC_METHODS_S0 = r'''
"""S0 structural-prior linear Gaussian smoother.

S0 is the non-transformer prototype comparator from the simulation plan. It is
not BALL: there is no teacher/student distillation, ensemble, Laplace layer,
or conformal calibration here. It is a per-person MAP smoother with:

- recall-window Gaussian anchor rows;
- observed daily proxy rows for latent and fast residual streams;
- structural rows tying L(t) to alpha'B(t) + delta(t) (no intercept; level in delta);
- random-walk smoothing on L(t) and delta AR smoothing;
- adaptive ridge rows that approximate sparse loading shrinkage.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

from ..model_utils import MethodResult, SimulationData, marginal_anchor_sd


@dataclass(frozen=True)
class S0Hyperparameters:
    xi: float = 0.15
    structural_sd: float = 0.35
    latent_smooth_sd: float = 1.25
    # daily_latent_sd / daily_delta_sd retained for compatibility but unused:
    # the daily-covariate likelihood now uses a learned readout (no oracle
    # group/channel hints) whose per-covariate residual sd replaces these.
    daily_latent_sd: float = 0.70
    daily_delta_sd: float = 0.80
    daily_readout_ridge: float = 1.0
    daily_readout_sigma_floor: float = 0.30
    baseline_prior_sd: float = 5.0
    alpha_weight_eps: float = 1e-3
    max_irls_iter: int = 2
    lsqr_atol: float = 1e-6
    lsqr_btol: float = 1e-6
    lsqr_iter_lim: int = 2000
    interval_inflation: float = 1.28


def _impute_proxy(comp_i: pd.DataFrame, q: int) -> np.ndarray:
    cols = [f"B{d}" for d in range(q)]
    proxy = comp_i[cols].astype(float).reset_index(drop=True)
    proxy = proxy.interpolate(limit_direction="both")
    proxy = proxy.fillna(proxy.mean()).fillna(0.0)
    arr = proxy.to_numpy(dtype=float)
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True)
    return arr / np.where(scale > 1e-8, scale, 1.0)


def _observed_daily_rows(daily_i: pd.DataFrame, prefix: str, start: int, stop: int) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    for t_pos, row in enumerate(daily_i.itertuples(index=False)):
        row_dict = row._asdict()
        for j in range(start, stop):
            col = f"{prefix}{j}"
            obs_col = f"obs_{col}"
            value = row_dict.get(col)
            observed = row_dict.get(obs_col, False)
            if bool(observed) and value is not None and np.isfinite(value):
                rows.append((t_pos, float(value)))
    return rows


def _solve_person(
    comp_i: pd.DataFrame,
    daily_i: pd.DataFrame,
    anchors_i: pd.DataFrame,
    data: SimulationData,
    hyper: S0Hyperparameters,
    daily_readout: dict[str, tuple[np.ndarray, float, float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    comp_i = comp_i.sort_values("t").reset_index(drop=True)
    daily_i = daily_i.sort_values("t").reset_index(drop=True)
    anchors_i = anchors_i.sort_values(["anchor", "t"]).reset_index(drop=True)

    t_count = len(comp_i)
    q = data.config.q
    # Match the spec: unit-coefficient random-walk delta and no separate
    # intercept (the level is carried by delta), so the comparator is scored on
    # the same decomposition convention as the truth and BALL.
    phi_delta = 1.0

    l_start = 0
    delta_start = l_start + t_count
    alpha_start = delta_start + t_count
    n_vars = alpha_start + q

    b_proxy = _impute_proxy(comp_i, q)
    alpha_group_weights = np.ones(q, dtype=float)
    last_result = None
    last_precision_diag = None
    last_residual_norm = np.nan

    def build_system() -> tuple[sparse.csr_matrix, np.ndarray]:
        row_ids: list[int] = []
        col_ids: list[int] = []
        values: list[float] = []
        targets: list[float] = []
        row = 0

        def add(coeffs: list[tuple[int, float]], target: float, sd: float) -> None:
            nonlocal row
            weight = 1.0 / max(float(sd), 1e-8)
            for col, value in coeffs:
                if value != 0.0:
                    row_ids.append(row)
                    col_ids.append(col)
                    values.append(float(value) * weight)
            targets.append(float(target) * weight)
            row += 1

        for anchor_row in anchors_i.itertuples(index=False):
            if not bool(anchor_row.observed) or not np.isfinite(anchor_row.value):
                continue
            start = int(anchor_row.window_start)
            end = int(anchor_row.window_end)
            window = list(range(max(0, start), min(t_count - 1, end) + 1))
            if not window:
                continue
            if anchor_row.anchor == "Y1":
                spec, rho = data.config.y1, data.config.rho_serial_y1
            else:
                spec, rho = data.config.y2, data.config.rho_serial_y2
            coeff = spec.loading / len(window)
            # Marginal AR(1) anchor error SD (error_sd / sqrt(1 - rho^2)).
            add([(l_start + t_pos, coeff) for t_pos in window], float(anchor_row.value), marginal_anchor_sd(spec, rho))

        # Daily-covariate likelihood via a LEARNED readout (no oracle group/channel
        # hints). Each covariate X_j is a learned linear function of the latent
        # coordinates [L, delta], with loadings beta_j, intercept b_j and residual
        # sd sigma_j fit on the training split only. One row per observed value:
        # beta_j_L * L(t) + beta_j_delta * delta(t) = X_j(t) - b_j, weighted 1/sigma_j.
        # The preliminary pass (daily_readout=None) uses anchors + proxy + smoothness
        # only, supplying the coordinates the readout is regressed on.
        if daily_readout:
            x_cols = list(daily_readout.keys())
            for t_pos, drow in enumerate(daily_i.itertuples(index=False)):
                rd = drow._asdict()
                for col in x_cols:
                    if not bool(rd.get(f"obs_{col}", False)):
                        continue
                    value = rd.get(col)
                    if value is None or not np.isfinite(value):
                        continue
                    beta, intercept, sigma = daily_readout[col]
                    coeffs = []
                    if beta[0] != 0.0:
                        coeffs.append((l_start + t_pos, float(beta[0])))
                    if beta[1] != 0.0:
                        coeffs.append((delta_start + t_pos, float(beta[1])))
                    if coeffs:
                        add(coeffs, float(value) - intercept, sigma)

        for t_pos in range(t_count):
            coeffs = [
                (l_start + t_pos, 1.0),
                (delta_start + t_pos, -1.0),
            ]
            coeffs.extend((alpha_start + d, -float(b_proxy[t_pos, d])) for d in range(q))
            add(coeffs, 0.0, hyper.structural_sd)

        for t_pos in range(1, t_count):
            add([(l_start + t_pos, 1.0), (l_start + t_pos - 1, -1.0)], 0.0, hyper.latent_smooth_sd)
            add(
                [(delta_start + t_pos, 1.0), (delta_start + t_pos - 1, -phi_delta)],
                0.0,
                data.config.delta_innovation_sd,
            )

        for d in range(q):
            shrink_sd = 1.0 / math.sqrt(max(hyper.xi * alpha_group_weights[d], 1e-8))
            add([(alpha_start + d, 1.0)], 0.0, shrink_sd)

        matrix = sparse.coo_matrix((values, (row_ids, col_ids)), shape=(row, n_vars)).tocsr()
        return matrix, np.asarray(targets, dtype=float)

    for _ in range(max(1, hyper.max_irls_iter)):
        matrix, target = build_system()
        solution = lsqr(
            matrix,
            target,
            atol=hyper.lsqr_atol,
            btol=hyper.lsqr_btol,
            iter_lim=hyper.lsqr_iter_lim,
        )
        vector = solution[0]
        last_result = vector
        last_residual_norm = float(solution[3])
        alpha = vector[alpha_start:]
        alpha_group_scale = np.abs(alpha)
        alpha_group_weights = 1.0 / np.maximum(alpha_group_scale, hyper.alpha_weight_eps)
        last_precision_diag = np.asarray(matrix.power(2).sum(axis=0)).ravel()

    if last_result is None or last_precision_diag is None:
        raise RuntimeError("S0 failed to solve a person-level system.")

    l_hat = last_result[l_start:delta_start]
    delta_hat = last_result[delta_start:alpha_start]
    alpha_hat = last_result[alpha_start:]
    proxy_slow_hat = b_proxy @ alpha_hat
    # No intercept: slow = L - delta; the level lives in the random-walk delta.
    slow_hat = l_hat - delta_hat
    precision_l = np.maximum(last_precision_diag[l_start:delta_start], 1e-8)
    raw_sd = hyper.interval_inflation / np.sqrt(precision_l)

    pred = pd.DataFrame({"id": comp_i["id"], "t": comp_i["t"], "L_hat": l_hat})
    pred["slow_hat"] = slow_hat
    pred["proxy_slow_hat"] = proxy_slow_hat
    pred["delta_hat"] = delta_hat
    for d in range(q):
        pred[f"alpha_hat{d}"] = alpha_hat[d]

    intervals = pd.DataFrame(
        {
            "id": comp_i["id"],
            "t": comp_i["t"],
            "lower": l_hat - 1.96 * raw_sd,
            "upper": l_hat + 1.96 * raw_sd,
            "raw_sd": raw_sd,
        }
    )
    metadata = {
        "residual_norm": last_residual_norm,
        "phi_delta": phi_delta,
        "mean_alpha_abs": float(np.mean(np.abs(alpha_hat))),
        "active_alpha_fraction_tau_0_05": float(np.mean(np.abs(alpha_hat) > 0.05)),
    }
    return pred, intervals, metadata


def _estimate_s0_daily_readout(work, train_ids, data, hyper, p_daily):
    """Learn each daily covariate's loading on [L, delta] on the TRAIN split only.

    Prelim per-person solves (no daily rows) supply the latent coordinates the
    24 covariates are ridge-regressed on. No oracle group/channel assignment, so
    permuting the covariate columns and refitting changes nothing.
    """

    x_cols = [f"X{j}" for j in range(p_daily)]
    obs_cols = [f"obs_X{j}" for j in range(p_daily)]
    coord_rows, x_rows, obs_rows = [], [], []
    for comp_i, daily_i, anchors_i in work:
        if int(comp_i["id"].iloc[0]) not in train_ids:
            continue
        pred_i, _, _ = _solve_person(comp_i, daily_i, anchors_i, data, hyper, daily_readout=None)
        coords = np.stack([pred_i["L_hat"].to_numpy(dtype=float), pred_i["delta_hat"].to_numpy(dtype=float)], axis=1)
        di = daily_i.sort_values("t").reset_index(drop=True)
        if not set(x_cols).issubset(di.columns):
            continue
        xv = di[x_cols].to_numpy(dtype=float)
        xo = di[obs_cols].to_numpy(dtype=float) if set(obs_cols).issubset(di.columns) else np.isfinite(xv).astype(float)
        n = min(len(coords), len(xv))
        coord_rows.append(coords[:n]); x_rows.append(xv[:n]); obs_rows.append(xo[:n])
    readout: dict[str, tuple[np.ndarray, float, float]] = {}
    if not coord_rows:
        return readout
    coords_all = np.concatenate(coord_rows, axis=0)
    x_all = np.concatenate(x_rows, axis=0)
    obs_all = np.concatenate(obs_rows, axis=0)
    design = np.concatenate([np.ones((coords_all.shape[0], 1)), coords_all], axis=1)  # [1, L, delta]
    ridge = hyper.daily_readout_ridge * np.eye(3)
    ridge[0, 0] = 0.0
    for j in range(p_daily):
        mask = (obs_all[:, j] > 0.5) & np.isfinite(x_all[:, j])
        if int(mask.sum()) < 5:
            continue
        dj = design[mask]; y = x_all[mask, j]
        theta = np.linalg.solve(dj.T @ dj + ridge, dj.T @ y)
        sigma = max(float((y - dj @ theta).std()), hyper.daily_readout_sigma_floor)
        readout[f"X{j}"] = (theta[1:3].astype(float), float(theta[0]), sigma)
    return readout


def fit_s0_structural_prior_ssm(
    data: SimulationData,
    xi: float = 0.15,
    hyperparameters: S0Hyperparameters | None = None,
) -> MethodResult:
    """Fit S0 structural-prior linear Gaussian smoother.

    The returned intervals are raw diagonal-precision Gaussian intervals. They
    are useful for plumbing and rough behavior checks, but they are not split
    conformal intervals and should not be used as BALL uncertainty claims.
    """

    hyper = hyperparameters or S0Hyperparameters(xi=xi)
    predictions: list[pd.DataFrame] = []
    intervals: list[pd.DataFrame] = []
    residual_norms: list[float] = []
    active_alpha: list[float] = []

    daily_groups = dict(tuple(data.daily.groupby("id", sort=True)))
    anchor_groups = dict(tuple(data.anchors.groupby("id", sort=True)))

    work: list[tuple] = []
    for person_id, comp_i in data.components.groupby("id", sort=True):
        daily_i = daily_groups.get(person_id)
        anchors_i = anchor_groups.get(person_id)
        if daily_i is None:
            daily_i = pd.DataFrame({"id": comp_i["id"], "t": comp_i["t"]})
        if anchors_i is None:
            anchors_i = pd.DataFrame(columns=data.anchors.columns)
        work.append((comp_i, daily_i, anchors_i))

    # Two-pass fair daily use: learn the readout on train, then solve all patients.
    train_ids = set(int(i) for i in data.individuals.loc[data.individuals["split"] == "train", "id"])
    readout = _estimate_s0_daily_readout(work, train_ids, data, hyper, data.config.p_daily)
    for comp_i, daily_i, anchors_i in work:
        pred_i, interval_i, meta_i = _solve_person(comp_i, daily_i, anchors_i, data, hyper, daily_readout=readout)
        predictions.append(pred_i)
        intervals.append(interval_i)
        residual_norms.append(meta_i["residual_norm"])
        active_alpha.append(meta_i["active_alpha_fraction_tau_0_05"])

    return MethodResult(
        "S0_structural_prior_lgssm",
        pd.concat(predictions, ignore_index=True),
        pd.concat(intervals, ignore_index=True),
        metadata={
            "status": "prototype_structural_prior_lgssm",
            "not_ball": True,
            "xi": hyper.xi,
            "structural_sd": hyper.structural_sd,
            "latent_smooth_sd": hyper.latent_smooth_sd,
            "daily_latent_sd": hyper.daily_latent_sd,
            "daily_delta_sd": hyper.daily_delta_sd,
            "alpha_parameterization": "static_sparse_person_loading",
            "slow_hat_definition": "L_hat - delta_hat (no intercept; level carried by delta)",
            "proxy_slow_hat_definition": "alpha_hat dot imputed_standardized_B; diagnostic only, not scored as slow recovery",
            "max_irls_iter": hyper.max_irls_iter,
            "interval_type": "raw_diagonal_precision_gaussian",
            "mean_person_residual_norm": float(np.mean(residual_norms)),
            "mean_active_alpha_fraction_tau_0_05": float(np.mean(active_alpha)),
        },
    )
'''

# --- simulations.src.methods.ball_structural (simulations/src/methods/ball_structural.py) ---
SRC_SIMULATIONS_SRC_METHODS_BALL_STRUCTURAL = r'''
"""Two-channel BALL structural posterior smoother.

This is a linear-Gaussian posterior scaffold for the PDF appendix structure:
shared dynamic RDoC loadings alpha(t), disorder-specific residual channels
delta_d(t)/delta_p(t), and recall-window PHQ/PCL anchors for z_d/z_p.

It is not the transformer student. It is a direct structural posterior used as
a smoke-stage target and ceiling check for the neural BALL path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr, splu

from ..metrics import anchor_conformal_residuals
from ..model_utils import (
    MethodResult,
    SimulationConfig,
    SimulationData,
    conformal_quantile,
    marginal_anchor_sd,
)


@dataclass(frozen=True)
class BallStructuralHyperparameters:
    xi: float = 0.08
    structural_sd: float = 0.30
    latent_smooth_sd: float = 1.10
    # daily_latent_sd / daily_lag_sd / daily_delta_sd are retained for backward
    # compatibility but no longer used: the daily-covariate likelihood now uses a
    # learned population readout (no oracle group/channel hints) whose per-feature
    # residual sd replaces these fixed scales.
    daily_latent_sd: float = 0.50
    daily_lag_sd: float = 0.70
    daily_delta_sd: float = 0.45
    # Learned daily-covariate readout: ridge strength for the per-covariate
    # latent->feature regression, and a floor on the estimated residual sd so a
    # spuriously well-fit covariate cannot dominate the calibrated anchors.
    daily_readout_ridge: float = 1.0
    daily_readout_sigma_floor: float = 0.30
    delta_phi: float = 1.0
    baseline_prior_sd: float = 5.0
    alpha_smooth_sd: float = 0.12
    alpha_weight_eps: float = 1e-3
    max_irls_iter: int = 3
    lsqr_atol: float = 1e-6
    lsqr_btol: float = 1e-6
    lsqr_iter_lim: int = 4000
    # Raw intervals now use the true marginal posterior variance, so no fudge
    # factor; split conformal performs the calibration step.
    interval_inflation: float = 1.0


def _with_split(truth: pd.DataFrame, individuals: pd.DataFrame) -> pd.DataFrame:
    if "split" in truth.columns:
        return truth
    return truth.merge(individuals[["id", "split"]], on="id", how="left")


def _impute_proxy(comp_i: pd.DataFrame, q: int) -> np.ndarray:
    cols = [f"B{d}" for d in range(q)]
    proxy = comp_i[cols].astype(float).reset_index(drop=True)
    proxy = proxy.interpolate(limit_direction="both")
    proxy = proxy.fillna(proxy.mean()).fillna(0.0)
    arr = proxy.to_numpy(dtype=float)
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True)
    return arr / np.where(scale > 1e-8, scale, 1.0)


def _observed_daily_rows(daily_i: pd.DataFrame, columns: list[str]) -> list[tuple[int, str, float]]:
    rows: list[tuple[int, str, float]] = []
    for t_pos, row in enumerate(daily_i.itertuples(index=False)):
        row_dict = row._asdict()
        for col in columns:
            value = row_dict.get(col)
            observed = row_dict.get(f"obs_{col}", False)
            if bool(observed) and value is not None and np.isfinite(value):
                rows.append((t_pos, col, float(value)))
    return rows


def _window(start: int, end: int, t_count: int) -> list[int]:
    return list(range(max(0, start), min(t_count - 1, end) + 1))


def _marginal_posterior_variances(
    matrix: sparse.csr_matrix,
    n_vars: int,
    zd_start: int,
    zp_start: int,
    dd_start: int,
    t_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-session marginal posterior variances for z_d, z_p and their
    cross-covariance.

    The weighted design ``matrix`` (every row already scaled by 1/sd) makes
    ``A^T A`` the Gaussian posterior *precision*, so the posterior covariance is
    ``(A^T A)^{-1}``. We need only the z_d / z_p diagonal (and the per-session
    z_d-z_p cross term for the joint composite), so we solve the precision system
    against the corresponding unit vectors via a single sparse LU factorization
    rather than inverting the full matrix.
    """

    ata = (matrix.T @ matrix).tocsc()
    ata = ata + 1e-8 * sparse.identity(n_vars, format="csc")
    lu = splu(ata)
    cols = np.arange(zd_start, dd_start)            # z_d block then z_p block
    rhs = np.zeros((n_vars, cols.size))
    rhs[cols, np.arange(cols.size)] = 1.0
    inv_cols = lu.solve(rhs)                         # (n_vars, 2*t_count)
    zd_rows = np.arange(zd_start, zp_start)
    zp_rows = np.arange(zp_start, dd_start)
    var_zd = inv_cols[zd_rows, np.arange(t_count)]
    var_zp = inv_cols[zp_rows, np.arange(t_count, 2 * t_count)]
    cov = inv_cols[zp_rows, np.arange(t_count)]      # Cov(z_d[t], z_p[t])
    var_zd = np.clip(var_zd, 1e-10, None)
    var_zp = np.clip(var_zp, 1e-10, None)
    return var_zd, var_zp, cov


def _solve_person(
    comp_i: pd.DataFrame,
    daily_i: pd.DataFrame,
    anchors_i: pd.DataFrame,
    config: SimulationConfig,
    hyper: BallStructuralHyperparameters,
    daily_readout: dict[str, tuple[np.ndarray, float, float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    comp_i = comp_i.sort_values("t").reset_index(drop=True)
    daily_i = daily_i.sort_values("t").reset_index(drop=True)
    anchors_i = anchors_i.sort_values(["anchor", "t"]).reset_index(drop=True)

    t_count = len(comp_i)
    q = config.q
    # Delta smoothing-prior AR coefficient. Default 1.0 matches the spec's
    # unit-coefficient random walk; setting it below 1 lets the comparator's delta
    # prior match a mean-reverting DGP (used to test slow/fast identifiability).
    phi_delta = float(hyper.delta_phi)

    # Spec decomposition has no separate intercept (z = alpha'RDoC + delta); the
    # per-patient level is carried by the random-walk residual delta, so the
    # linear system solves only for z_d, z_p, delta_d, delta_p, and alpha.
    zd_start = 0
    zp_start = zd_start + t_count
    dd_start = zp_start + t_count
    dp_start = dd_start + t_count
    alpha_start = dp_start + t_count
    n_vars = alpha_start + t_count * q

    def alpha_idx(t_pos: int, d: int) -> int:
        return alpha_start + t_pos * q + d

    b_proxy = _impute_proxy(comp_i, q)
    alpha_group_weights = np.ones(q, dtype=float)
    last_result: np.ndarray | None = None
    last_matrix: sparse.csr_matrix | None = None
    last_residual_norm = np.nan

    def build_system() -> tuple[sparse.csr_matrix, np.ndarray]:
        row_ids: list[int] = []
        col_ids: list[int] = []
        values: list[float] = []
        targets: list[float] = []
        row = 0

        def add(coeffs: list[tuple[int, float]], target: float, sd: float) -> None:
            nonlocal row
            weight = 1.0 / max(float(sd), 1e-8)
            for col, value in coeffs:
                if value != 0.0:
                    row_ids.append(row)
                    col_ids.append(col)
                    values.append(float(value) * weight)
            targets.append(float(target) * weight)
            row += 1

        for anchor_row in anchors_i.itertuples(index=False):
            if not bool(anchor_row.observed) or not np.isfinite(anchor_row.value):
                continue
            channel_start = zd_start if anchor_row.anchor == "Y1" else zp_start
            if anchor_row.anchor == "Y1":
                spec, rho = config.y1, config.rho_serial_y1
            else:
                spec, rho = config.y2, config.rho_serial_y2
            days = _window(int(anchor_row.window_start), int(anchor_row.window_end), t_count)
            if not days:
                continue
            coeff = spec.loading / len(days)
            # Marginal AR(1) anchor error SD (error_sd / sqrt(1 - rho^2)).
            add([(channel_start + t_pos, coeff) for t_pos in days], float(anchor_row.value), marginal_anchor_sd(spec, rho))

        # Daily-covariate likelihood via a LEARNED population readout (no oracle
        # group/channel hints). The readout supplies, for each daily covariate
        # X_j, loadings beta_j on the same-day latent coordinates
        # [z_d, z_p, delta_d, delta_p], an intercept b_j, and a residual sd
        # sigma_j, all estimated on the training split only (the linear-solver
        # mirror of the neural daily_readout, which learns which covariates load
        # on the latent and which are noise). Each observed value contributes one
        # row sum_c beta_{j,c} * coord_c(t) = X_j(t) - b_j, weighted by 1/sigma_j.
        # The preliminary pass (daily_readout=None) fits anchors + proxy +
        # smoothness only, supplying the coordinates the readout is regressed on.
        if daily_readout:
            coord_starts = (zd_start, zp_start, dd_start, dp_start)
            for t_pos, col, value in _observed_daily_rows(daily_i, list(daily_readout.keys())):
                beta, intercept, sigma = daily_readout[col]
                coeffs = [(coord_starts[c] + t_pos, float(beta[c])) for c in range(4) if beta[c] != 0.0]
                if coeffs:
                    add(coeffs, value - intercept, sigma)

        for t_pos in range(t_count):
            alpha_coeffs = [(alpha_idx(t_pos, d), -float(b_proxy[t_pos, d])) for d in range(q)]
            add(
                [(zd_start + t_pos, 1.0), (dd_start + t_pos, -1.0)] + alpha_coeffs,
                0.0,
                hyper.structural_sd,
            )
            add(
                [(zp_start + t_pos, 1.0), (dp_start + t_pos, -1.0)] + alpha_coeffs,
                0.0,
                hyper.structural_sd,
            )

        for t_pos in range(1, t_count):
            add([(zd_start + t_pos, 1.0), (zd_start + t_pos - 1, -1.0)], 0.0, hyper.latent_smooth_sd)
            add([(zp_start + t_pos, 1.0), (zp_start + t_pos - 1, -1.0)], 0.0, hyper.latent_smooth_sd)
            add([(dd_start + t_pos, 1.0), (dd_start + t_pos - 1, -phi_delta)], 0.0, config.delta_innovation_sd)
            add([(dp_start + t_pos, 1.0), (dp_start + t_pos - 1, -phi_delta)], 0.0, config.delta_innovation_sd)
            for d in range(q):
                add([(alpha_idx(t_pos, d), 1.0), (alpha_idx(t_pos - 1, d), -1.0)], 0.0, hyper.alpha_smooth_sd)

        for d in range(q):
            shrink_sd = 1.0 / math.sqrt(max(hyper.xi * alpha_group_weights[d], 1e-8))
            for t_pos in range(t_count):
                add([(alpha_idx(t_pos, d), 1.0)], 0.0, shrink_sd)

        matrix = sparse.coo_matrix((values, (row_ids, col_ids)), shape=(row, n_vars)).tocsr()
        return matrix, np.asarray(targets, dtype=float)

    for _ in range(max(1, hyper.max_irls_iter)):
        matrix, target = build_system()
        solution = lsqr(
            matrix,
            target,
            atol=hyper.lsqr_atol,
            btol=hyper.lsqr_btol,
            iter_lim=hyper.lsqr_iter_lim,
        )
        vector = solution[0]
        last_result = vector
        last_matrix = matrix
        last_residual_norm = float(solution[3])
        alpha = vector[alpha_start:].reshape(t_count, q)
        alpha_group_scale = np.mean(np.abs(alpha), axis=0)
        alpha_group_weights = 1.0 / np.maximum(alpha_group_scale, hyper.alpha_weight_eps)

    if last_result is None or last_matrix is None:
        raise RuntimeError("BALL structural posterior failed to solve a person-level system.")

    z_d = last_result[zd_start:zp_start]
    z_p = last_result[zp_start:dd_start]
    delta_d = last_result[dd_start:dp_start]
    delta_p = last_result[dp_start:alpha_start]
    alpha = last_result[alpha_start:].reshape(t_count, q)
    proxy_slow = np.sum(alpha * b_proxy, axis=1)
    joint = 0.5 * (z_d + z_p)
    delta = 0.5 * (delta_d + delta_p)
    # No intercept in the spec decomposition: slow = z - delta carries no level,
    # and the level is held by delta.
    slow_d = z_d - delta_d
    slow_p = z_p - delta_p
    slow = 0.5 * (slow_d + slow_p)

    var_zd, var_zp, cov_zdzp = _marginal_posterior_variances(
        last_matrix, n_vars, zd_start, zp_start, dd_start, t_count
    )
    # Joint composite L = 0.5 (z_d + z_p): Var(L) = 0.25 (Var z_d + Var z_p + 2 Cov).
    var_joint = np.clip(0.25 * (var_zd + var_zp + 2.0 * cov_zdzp), 1e-10, None)
    raw_sd = hyper.interval_inflation * np.sqrt(var_joint)
    raw_sd_zd = hyper.interval_inflation * np.sqrt(var_zd)
    raw_sd_zp = hyper.interval_inflation * np.sqrt(var_zp)

    pred = pd.DataFrame({"id": comp_i["id"], "t": comp_i["t"], "L_hat": joint})
    pred["slow_hat"] = slow
    pred["slow_d_hat"] = slow_d
    pred["slow_p_hat"] = slow_p
    pred["proxy_slow_hat"] = proxy_slow
    pred["delta_hat"] = delta
    pred["z_d_hat"] = z_d
    pred["z_p_hat"] = z_p
    pred["delta_d_hat"] = delta_d
    pred["delta_p_hat"] = delta_p
    for d in range(q):
        pred[f"alpha_hat{d}"] = alpha[:, d]

    intervals = pd.DataFrame(
        {
            "id": comp_i["id"],
            "t": comp_i["t"],
            "lower": joint - 1.96 * raw_sd,
            "upper": joint + 1.96 * raw_sd,
            "z_d_lower": z_d - 1.96 * raw_sd_zd,
            "z_d_upper": z_d + 1.96 * raw_sd_zd,
            "z_p_lower": z_p - 1.96 * raw_sd_zp,
            "z_p_upper": z_p + 1.96 * raw_sd_zp,
            "raw_sd": raw_sd,
            "total_sd": raw_sd,
            "z_d_raw_sd": raw_sd_zd,
            "z_d_total_sd": raw_sd_zd,
            "z_p_raw_sd": raw_sd_zp,
            "z_p_total_sd": raw_sd_zp,
            "interval_scale": 1.96,
            "interval_source": "raw_structural_gaussian",
        }
    )
    metadata = {
        "residual_norm": last_residual_norm,
        "phi_delta": phi_delta,
        "mean_alpha_abs": float(np.mean(np.abs(alpha))),
        "active_alpha_fraction_tau_0_05": float(np.mean(np.abs(alpha) > 0.05)),
    }
    return pred, intervals, metadata


def _estimate_daily_readout(
    work: list[tuple],
    train_ids: set[int],
    config: SimulationConfig,
    hyper: BallStructuralHyperparameters,
    p_daily: int,
) -> dict[str, tuple[np.ndarray, float, float]]:
    """Learn a per-covariate latent->feature readout on the TRAIN split.

    Each daily covariate X_j is ridge-regressed on the preliminary latent
    coordinates [z_d, z_p, delta_d, delta_p] (estimated from anchors + proxy +
    smoothness only, with no daily rows) pooled over training patient-days. The
    result is the loadings, intercept, and residual sd used as the covariate's
    observation model in the final solve. Fit on TRAIN only, so the readout never
    sees the conformal or test split (no leakage), mirroring the neural
    daily_readout trained on the train batch.
    """

    x_cols = [f"X{j}" for j in range(p_daily)]
    obs_cols = [f"obs_X{j}" for j in range(p_daily)]
    coord_rows: list[np.ndarray] = []
    x_rows: list[np.ndarray] = []
    x_obs_rows: list[np.ndarray] = []
    for comp_i, daily_i, anchors_i in work:
        if int(comp_i["id"].iloc[0]) not in train_ids:
            continue
        pred_i, _, _ = _solve_person(comp_i, daily_i, anchors_i, config, hyper, daily_readout=None)
        coords = np.stack(
            [pred_i["z_d_hat"].to_numpy(dtype=float), pred_i["z_p_hat"].to_numpy(dtype=float),
             pred_i["delta_d_hat"].to_numpy(dtype=float), pred_i["delta_p_hat"].to_numpy(dtype=float)],
            axis=1,
        )
        di = daily_i.sort_values("t").reset_index(drop=True)
        if not set(x_cols).issubset(di.columns):
            continue
        xv = di[x_cols].to_numpy(dtype=float)
        xo = di[obs_cols].to_numpy(dtype=float) if set(obs_cols).issubset(di.columns) else np.isfinite(xv).astype(float)
        n = min(len(coords), len(xv))
        coord_rows.append(coords[:n])
        x_rows.append(xv[:n])
        x_obs_rows.append(xo[:n])

    readout: dict[str, tuple[np.ndarray, float, float]] = {}
    if not coord_rows:
        return readout
    coords_all = np.concatenate(coord_rows, axis=0)
    x_all = np.concatenate(x_rows, axis=0)
    obs_all = np.concatenate(x_obs_rows, axis=0)
    design = np.concatenate([np.ones((coords_all.shape[0], 1)), coords_all], axis=1)  # [1, z_d, z_p, dd, dp]
    ridge = hyper.daily_readout_ridge * np.eye(5)
    ridge[0, 0] = 0.0  # do not shrink the intercept
    for j in range(p_daily):
        mask = (obs_all[:, j] > 0.5) & np.isfinite(x_all[:, j])
        if int(mask.sum()) < 5:
            continue
        design_j = design[mask]
        y = x_all[mask, j]
        theta = np.linalg.solve(design_j.T @ design_j + ridge, design_j.T @ y)
        sigma = max(float((y - design_j @ theta).std()), hyper.daily_readout_sigma_floor)
        readout[f"X{j}"] = (theta[1:5].astype(float), float(theta[0]), sigma)
    return readout


def fit_ball_structural_posterior(
    data: SimulationData,
    hyperparameters: BallStructuralHyperparameters | None = None,
) -> MethodResult:
    """Fit the two-channel BALL structural posterior per person."""

    hyper = hyperparameters or BallStructuralHyperparameters()
    config = data.config

    daily_groups = dict(tuple(data.daily.groupby("id", sort=True)))
    anchor_groups = dict(tuple(data.anchors.groupby("id", sort=True)))

    # Build the per-patient work items; the solves are independent, so they run
    # in parallel across CPU cores (the per-person linear system is the main
    # cost at scale).
    work: list[tuple] = []
    for person_id, comp_i in data.components.groupby("id", sort=True):
        daily_i = daily_groups.get(person_id)
        anchors_i = anchor_groups.get(person_id)
        if daily_i is None:
            daily_i = pd.DataFrame({"id": comp_i["id"], "t": comp_i["t"]})
        if anchors_i is None:
            anchors_i = pd.DataFrame(columns=data.anchors.columns)
        work.append((comp_i, daily_i, anchors_i))

    # Two-pass fair daily-covariate use: learn the population readout on train,
    # then solve every patient with it (no oracle group/channel hints).
    train_ids = set(int(i) for i in data.individuals.loc[data.individuals["split"] == "train", "id"])
    readout = _estimate_daily_readout(work, train_ids, config, hyper, config.p_daily)
    solved = [_solve_person(comp_i, daily_i, anchors_i, config, hyper, daily_readout=readout)
              for comp_i, daily_i, anchors_i in work]

    predictions = [pred_i for pred_i, _, _ in solved]
    intervals = [interval_i for _, interval_i, _ in solved]
    residual_norms = [meta_i["residual_norm"] for _, _, meta_i in solved]
    active_alpha = [meta_i["active_alpha_fraction_tau_0_05"] for _, _, meta_i in solved]

    return MethodResult(
        "BALL-structural-posterior",
        pd.concat(predictions, ignore_index=True),
        pd.concat(intervals, ignore_index=True),
        metadata={
            "status": "two_channel_structural_posterior",
            "ball_math_appendix": True,
            "xi": hyper.xi,
            "structural_sd": hyper.structural_sd,
            "latent_smooth_sd": hyper.latent_smooth_sd,
            "daily_covariate_model": (
                "learned population readout over all daily covariates "
                "(train-only ridge of each X_j on [z_d,z_p,delta_d,delta_p]); "
                "no oracle group or channel assignment"
            ),
            "daily_readout_covariates_used": int(len(readout)),
            "daily_readout_ridge": hyper.daily_readout_ridge,
            "daily_readout_sigma_floor": hyper.daily_readout_sigma_floor,
            "alpha_parameterization": "dynamic_shared_loading",
            "slow_hat_definition": "0.5 * ((z_d_hat - delta_d_hat) + (z_p_hat - delta_p_hat)); no intercept (spec z = alpha'RDoC + delta)",
            "proxy_slow_hat_definition": "alpha_hat dot imputed_standardized_B; diagnostic only, not scored as slow recovery",
            "max_irls_iter": hyper.max_irls_iter,
            "interval_type": "marginal_posterior_gaussian",
            "mean_person_residual_norm": float(np.mean(residual_norms)),
            "mean_active_alpha_fraction_tau_0_05": float(np.mean(active_alpha)),
        },
    )


def conformal_calibrate_structural_posterior(
    result: MethodResult,
    data: SimulationData,
    *,
    alpha: float = 0.05,
    split: str = "conformal",
) -> MethodResult:
    """Apply post-lock split-conformal calibration using anchor residuals.

    This follows the appendix Sec. 5 split-conformal interval form
    C(x) = [f_hat(x) - q_(1-eps), f_hat(x) + q_(1-eps)], with conformity scores
    computed on the *observable anchors* (R_i = |y_i - f_hat(x_i)|), not on the
    unobservable latent. The latent is never measured in the empirical study, so
    calibrating on true-latent residuals would be an oracle that cannot exist at
    deployment. Latent coverage against truth is reported separately as a
    simulation-only diagnostic.

    Because the anchor residuals include anchor measurement noise, the resulting
    latent intervals overcover and are wide. Per the masterclass conformal
    pitfall box, that is the documented honest cost of calibrating on
    observables, not a tuning failure. Do not rescale q through the recall
    window kernel norm: a 2026-05-31 revision did exactly that (h = q * sqrt(n)
    / loading, added on top of the raw interval) and produced degenerate
    intervals about 15x wider than the raw posterior. The raw posterior arm and
    the simulation-only oracle-latent-conformal arm are reported alongside this
    arm to expose the latent-calibration gap.

    The model result must already contain predictions for the calibration and
    test patients; no model parameters are changed here.
    """

    zd_scores = anchor_conformal_residuals(
        result.predictions, data.anchors, data.individuals, split, "Y1", "z_d_hat"
    )
    zp_scores = anchor_conformal_residuals(
        result.predictions, data.anchors, data.individuals, split, "Y2", "z_p_hat"
    )
    default = 0.5 * float(np.nanmean(result.intervals["upper"] - result.intervals["lower"]))
    q_zd = conformal_quantile(zd_scores, alpha, default=default)
    q_zp = conformal_quantile(zp_scores, alpha, default=default)
    # The joint composite severity is 0.5 * (z_d + z_p); its calibrated
    # half-width is the average of the per-channel anchor-calibrated half-widths.
    q_hat = 0.5 * (q_zd + q_zp)

    intervals = result.predictions[["id", "t", "L_hat", "z_d_hat", "z_p_hat"]].copy()
    intervals["lower"] = intervals["L_hat"] - q_hat
    intervals["upper"] = intervals["L_hat"] + q_hat
    intervals["z_d_lower"] = intervals["z_d_hat"] - q_zd
    intervals["z_d_upper"] = intervals["z_d_hat"] + q_zd
    intervals["z_p_lower"] = intervals["z_p_hat"] - q_zp
    intervals["z_p_upper"] = intervals["z_p_hat"] + q_zp
    intervals["raw_sd"] = q_hat / 1.96
    intervals["total_sd"] = q_hat / 1.96
    intervals["z_d_raw_sd"] = q_zd / 1.96
    intervals["z_d_total_sd"] = q_zd / 1.96
    intervals["z_p_raw_sd"] = q_zp / 1.96
    intervals["z_p_total_sd"] = q_zp / 1.96
    intervals["interval_scale"] = 1.96
    intervals["interval_source"] = "anchor_residual_conformal"
    intervals["conformal_q"] = q_hat
    intervals["z_d_conformal_q"] = q_zd
    intervals["z_p_conformal_q"] = q_zp
    intervals = intervals.drop(columns=["L_hat", "z_d_hat", "z_p_hat"])

    metadata = dict(result.metadata)
    metadata.update(
        {
            "interval_type": "anchor_residual_split_conformal",
            "conformal_score": "anchor_residual",
            "conformal_alpha": float(alpha),
            "conformal_split": split,
            "conformal_q": q_hat,
            "z_d_conformal_q": q_zd,
            "z_p_conformal_q": q_zp,
            "conformal_n_scores_zd": int(len(zd_scores)),
            "conformal_n_scores_zp": int(len(zp_scores)),
            "severity_interval_targets": ["S_joint", "z_d", "z_p"],
            "composite_interval_note": (
                "the conformal guarantee attaches to the per-channel anchor "
                "predictions; the S_joint half-width is the heuristic average "
                "of the channel half-widths and carries no separate guarantee"
            ),
        }
    )
    return MethodResult(
        f"{result.method}-conformal",
        result.predictions.copy(),
        intervals,
        metadata=metadata,
    )


def oracle_latent_conformal_intervals(
    result: MethodResult,
    data: SimulationData,
    *,
    alpha: float = 0.05,
    split: str = "conformal",
) -> MethodResult:
    """Simulation-only ORACLE: split conformal calibrated on TRUE-latent residuals.

    This is an upper-bound diagnostic on achievable latent calibration and is NOT
    available in a real application (the latent is never observed). Reported as
    the third latent-coverage row alongside the raw posterior interval and the
    anchor-residual split-conformal interval, to expose the latent-calibration gap.
    """

    comp = data.components[["id", "t", "z_d", "z_p"]].copy()
    comp["L"] = 0.5 * (comp["z_d"] + comp["z_p"])
    merged = result.predictions[["id", "t", "L_hat", "z_d_hat", "z_p_hat"]].merge(
        comp, on=["id", "t"], how="inner"
    )
    split_ids = set(data.individuals.loc[data.individuals["split"] == split, "id"])
    cal = merged[merged["id"].isin(split_ids)]
    default = 0.5 * float(np.nanmean(result.intervals["upper"] - result.intervals["lower"]))
    q_L = conformal_quantile(np.abs(cal["L_hat"] - cal["L"]).to_numpy(), alpha, default=default)
    q_zd = conformal_quantile(np.abs(cal["z_d_hat"] - cal["z_d"]).to_numpy(), alpha, default=default)
    q_zp = conformal_quantile(np.abs(cal["z_p_hat"] - cal["z_p"]).to_numpy(), alpha, default=default)

    intervals = result.predictions[["id", "t", "L_hat", "z_d_hat", "z_p_hat"]].copy()
    intervals["lower"] = intervals["L_hat"] - q_L
    intervals["upper"] = intervals["L_hat"] + q_L
    intervals["z_d_lower"] = intervals["z_d_hat"] - q_zd
    intervals["z_d_upper"] = intervals["z_d_hat"] + q_zd
    intervals["z_p_lower"] = intervals["z_p_hat"] - q_zp
    intervals["z_p_upper"] = intervals["z_p_hat"] + q_zp
    # Defensive sd columns (some consumers, e.g. ensemble combine, require them).
    intervals["raw_sd"] = q_L / 1.96
    intervals["total_sd"] = q_L / 1.96
    intervals["interval_source"] = "oracle_latent_conformal"
    intervals = intervals.drop(columns=["L_hat", "z_d_hat", "z_p_hat"])

    metadata = dict(result.metadata)
    metadata.update(
        {
            "interval_type": "oracle_latent_conformal",
            "conformal_score": "true_latent_residual_SIMULATION_ONLY",
            "conformal_alpha": float(alpha),
            "conformal_split": split,
            "oracle_q_L": float(q_L),
        }
    )
    return MethodResult(
        f"{result.method}-oracle-latent-conformal",
        result.predictions.copy(),
        intervals,
        metadata=metadata,
    )
'''

# --- simulations.src.methods.markov_pattern_mixture (simulations/src/methods/markov_pattern_mixture.py) ---
SRC_SIMULATIONS_SRC_METHODS_MARKOV_PATTERN_MIXTURE = r'''
"""Markov pattern-mixture latent-trajectory comparator.

This is the classical "old-school statistics" arm of the comparison required by
the project specification: old-school statistics versus the transformer
teacher/student model for latent recovery. It is a separate baseline. It is
never used as a pseudo-target for BALL.

The math-spec appendix (the BALL specification PDF, Sec. 4) names the classical analogue of
the inference problem explicitly: "the closest familiar analogue is a Kalman
filter on a linear-Gaussian state-space model." This module is that analogue,
extended in two classical directions that the transformer is not:

1. Markov dynamics. The per-day latent severity is an unobserved-components
   state-space model, L(t) = level(t) + fast(t), where level(t) is a random
   walk (the slow component) and fast(t) is a stationary AR(1) residual. This is
   a first-order Markov latent process with linear-Gaussian transitions.

2. Pattern mixture. The cohort is a finite mixture of K trajectory classes. Each
   class has its own Markov dynamics, pooled across patients (so the model
   borrows strength across patients within a class, unlike the per-person S0
   smoother). Class membership is modeled conditional on the observed-data
   pattern, the defining feature of a pattern-mixture model (Little, 1993): the
   prior class weights depend on each patient's anchor-observation stratum.

Inference is classical throughout. Class dynamics and mixing weights are fit by
expectation maximization on the marginal likelihood of the observed anchors.
Each anchor is a loading-scaled average of the latent over its recall window
plus Gaussian noise, so the anchor vector of a patient is multivariate Gaussian
with a covariance the class dynamics imply in closed form. Given the fitted
classes, the per-patient latent trajectory is the responsibility-weighted
posterior mean of a batch linear-Gaussian smoother that conditions on the
retained anchors and, in simulation, on the same-day daily severity proxies.
Uncertainty is the model-based Gaussian interval from the smoother posterior
variance combined across classes by the law of total variance. There is no
ensemble, no Laplace layer, and no conformal calibration. Those are BALL
constructions. The contrast is deliberate.

The comparator estimates the composite latent severity L only. It does not
attempt to recover the sparse RDoC loading vector alpha or the proxy-explained
slow component. Identifying which proxy dimensions drive recovery is a
BALL-specific capability, so the RDoC-specific diagnostics (active-set F1,
rho_RDoC, denoised C recovery) are reported as not applicable for this method.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from ..model_utils import (
    AnchorSpec,
    MethodResult,
    SimulationData,
    make_rng,
    marginal_anchor_sd,
)


@dataclass(frozen=True)
class MarkovMixtureHyperparameters:
    """Hyperparameters for the Markov pattern-mixture comparator."""

    n_classes: int = 3
    n_strata: int = 3
    em_iters: int = 8
    em_tol: float = 1e-3
    param_opt_maxiter: int = 20
    fit_split: str | None = "train"
    max_fit_persons: int = 300
    use_daily: bool = True
    daily_obs_sd: float = 0.6
    level_prior_sd: float = 5.0
    min_phi: float = -0.95
    max_phi: float = 0.95
    jitter: float = 1e-6
    interval_scale: float = 1.96
    random_state: int = 0
    # Bounds for the per-class innovation and anchor-noise scales (z units).
    min_scale: float = 0.02
    max_scale: float = 6.0


# ---------------------------------------------------------------------------
# Per-person observed-data containers, assembled once and reused across EM.
# ---------------------------------------------------------------------------
@dataclass
class _PersonObs:
    person_id: int
    t_index: np.ndarray            # day positions 0..T-1 actually present
    n_days: int
    # Anchor design: each retained anchor is loading/|window| spread over its
    # window day positions. window_rows is a list of (positions, coeff, value).
    anchor_positions: list[np.ndarray]
    anchor_coeffs: list[float]
    anchor_values: np.ndarray      # standardized anchor values (length W)
    anchor_design: np.ndarray      # (W, n_days) windowed-mean design with loading
    # Daily same-day severity observations of L(t): value and day position.
    daily_positions: np.ndarray
    daily_values: np.ndarray
    stratum: int


def _build_person_obs(
    comp_i: pd.DataFrame,
    daily_i: pd.DataFrame | None,
    anchors_i: pd.DataFrame,
    use_daily: bool,
    daily_readout: tuple[float, np.ndarray, np.ndarray] | None = None,
) -> _PersonObs:
    comp_i = comp_i.sort_values("t").reset_index(drop=True)
    days = comp_i["t"].to_numpy(dtype=int)
    n_days = len(days)
    day_to_pos = {int(d): i for i, d in enumerate(days)}

    anchor_positions: list[np.ndarray] = []
    anchor_coeffs: list[float] = []
    anchor_values: list[float] = []
    rows: list[np.ndarray] = []
    for row in anchors_i.itertuples(index=False):
        if not bool(getattr(row, "observed", False)):
            continue
        value = float(getattr(row, "value"))
        if not np.isfinite(value):
            continue
        start = int(getattr(row, "window_start"))
        end = int(getattr(row, "window_end"))
        positions = np.array(
            [day_to_pos[d] for d in range(start, end + 1) if d in day_to_pos],
            dtype=int,
        )
        if positions.size == 0:
            # No overlap with present days: attach to nearest present day.
            nearest = int(np.argmin(np.abs(days - end)))
            positions = np.array([nearest], dtype=int)
        loading = float(getattr(row, "loading"))
        if not np.isfinite(loading) or loading == 0.0:
            loading = 1.0
        coeff = loading / positions.size
        design_row = np.zeros(n_days, dtype=float)
        design_row[positions] = coeff
        anchor_positions.append(positions)
        anchor_coeffs.append(coeff)
        anchor_values.append(value)
        rows.append(design_row)

    anchor_design = np.vstack(rows) if rows else np.zeros((0, n_days), dtype=float)

    daily_positions = np.array([], dtype=int)
    daily_values = np.array([], dtype=float)
    if use_daily and daily_readout is not None and daily_i is not None and not daily_i.empty:
        # Daily severity observation via a LEARNED readout (no oracle group hints):
        # the per-day reading of L(t) is the train-fit linear prediction of L from
        # ALL daily covariates, b + sum_j beta_j * X_j(t), with missing covariates
        # imputed to their train means. Permuting the covariate columns and refitting
        # leaves this prediction unchanged.
        b0, beta, means = daily_readout
        p_daily = len(beta)
        x_cols = [f"X{j}" for j in range(p_daily)]
        obs_cols = [f"obs_X{j}" for j in range(p_daily)]
        di = daily_i.sort_values("t").reset_index(drop=True)
        if set(x_cols).issubset(di.columns):
            xv = di[x_cols].to_numpy(dtype=float)
            xo = di[obs_cols].to_numpy(dtype=float) if set(obs_cols).issubset(di.columns) else np.isfinite(xv).astype(float)
            ximp = np.where(xo > 0.5, xv, means[None, :])
            pred_l = b0 + ximp @ beta
            d_days = di["t"].to_numpy(dtype=int)
            any_obs = (xo > 0.5).any(axis=1)
            pos_list: list[int] = []
            val_list: list[float] = []
            for r in range(len(di)):
                d = int(d_days[r])
                if d in day_to_pos and bool(any_obs[r]):
                    pos_list.append(day_to_pos[d])
                    val_list.append(float(pred_l[r]))
            daily_positions = np.array(pos_list, dtype=int)
            daily_values = np.array(val_list, dtype=float)

    return _PersonObs(
        person_id=int(comp_i["id"].iloc[0]),
        t_index=days,
        n_days=n_days,
        anchor_positions=anchor_positions,
        anchor_coeffs=anchor_coeffs,
        anchor_values=np.asarray(anchor_values, dtype=float),
        anchor_design=anchor_design,
        daily_positions=daily_positions,
        daily_values=daily_values,
        stratum=0,
    )


# ---------------------------------------------------------------------------
# Class dynamics: latent covariance and anchor marginal likelihood.
# ---------------------------------------------------------------------------
@dataclass
class _ClassParams:
    s_lev: float       # random-walk innovation SD (slow level)
    phi: float         # AR(1) coefficient of the fast residual
    s_fast: float      # AR(1) innovation SD of the fast residual
    anchor_sd: float   # pooled anchor measurement SD (z units)


def _latent_cov(params: _ClassParams, n_days: int, level_prior_var: float,
                min_grid: np.ndarray, lag_grid: np.ndarray) -> np.ndarray:
    """Closed-form Cov(L_s, L_t) for the level (random walk) + fast (AR1) model."""

    fast_marg = params.s_fast**2 / max(1.0 - params.phi**2, 1e-6)
    cov_level = level_prior_var + params.s_lev**2 * min_grid
    cov_fast = fast_marg * np.power(abs(params.phi), lag_grid)
    return cov_level + cov_fast


def _anchor_loglik(person: _PersonObs, cov_L: np.ndarray, anchor_sd: float,
                   jitter: float) -> float:
    """Marginal Gaussian log-likelihood of a person's anchors under a class."""

    w = person.anchor_design.shape[0]
    if w == 0:
        return 0.0
    M = person.anchor_design                       # (W, n_days)
    cov_anchor = M @ cov_L @ M.T                    # (W, W)
    cov_anchor = cov_anchor + (anchor_sd**2 + jitter) * np.eye(w)
    y = person.anchor_values
    try:
        c, low = cho_factor(cov_anchor, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        cov_anchor = cov_anchor + 1e-3 * np.eye(w)
        c, low = cho_factor(cov_anchor, lower=True, check_finite=False)
    alpha = cho_solve((c, low), y, check_finite=False)
    logdet = 2.0 * np.sum(np.log(np.diag(c)))
    quad = float(y @ alpha)
    return -0.5 * (quad + logdet + w * math.log(2.0 * math.pi))


# ---------------------------------------------------------------------------
# Batch linear-Gaussian smoother for one person under one class.
# ---------------------------------------------------------------------------
def _smooth_person(person: _PersonObs, params: _ClassParams,
                   hyper: MarkovMixtureHyperparameters) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (L_hat, level_hat, var_L) for one person under one class.

    Solves the weighted Gaussian system over the stacked state [level, fast]
    via the normal equations, then reads the latent posterior variance from the
    inverse information matrix.
    """

    n = person.n_days
    level0 = 0
    fast0 = n
    n_vars = 2 * n

    rows: list[np.ndarray] = []
    targets: list[float] = []

    def add(coeffs: list[tuple[int, float]], target: float, sd: float) -> None:
        weight = 1.0 / max(sd, 1e-8)
        row = np.zeros(n_vars, dtype=float)
        for col, value in coeffs:
            row[col] = value * weight
        rows.append(row)
        targets.append(target * weight)

    # State priors at the first day.
    add([(level0 + 0, 1.0)], 0.0, hyper.level_prior_sd)
    fast_marg_sd = params.s_fast / math.sqrt(max(1.0 - params.phi**2, 1e-6))
    add([(fast0 + 0, 1.0)], 0.0, max(fast_marg_sd, hyper.min_scale))

    # Markov transitions.
    for t in range(1, n):
        add([(level0 + t, 1.0), (level0 + t - 1, -1.0)], 0.0, params.s_lev)
        add([(fast0 + t, 1.0), (fast0 + t - 1, -params.phi)], 0.0, params.s_fast)

    # Anchor observations: loading-scaled window mean of L = value.
    for design_row, value in zip(person.anchor_design, person.anchor_values):
        positions = np.nonzero(design_row)[0]
        coeffs: list[tuple[int, float]] = []
        for p in positions:
            coeffs.append((level0 + p, design_row[p]))
            coeffs.append((fast0 + p, design_row[p]))
        add(coeffs, float(value), params.anchor_sd)

    # Same-day daily severity observations of L(t).
    if hyper.use_daily and person.daily_positions.size:
        for p, value in zip(person.daily_positions, person.daily_values):
            add([(level0 + int(p), 1.0), (fast0 + int(p), 1.0)], float(value), hyper.daily_obs_sd)

    A = np.vstack(rows)
    b = np.asarray(targets, dtype=float)
    info = A.T @ A + hyper.jitter * np.eye(n_vars)
    rhs = A.T @ b
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(info)
    x = cov @ rhs

    level_hat = x[level0:fast0]
    fast_hat = x[fast0:]
    l_hat = level_hat + fast_hat
    # Var(L_t) = Var(level_t) + Var(fast_t) + 2 Cov(level_t, fast_t).
    diag_level = np.diag(cov)[level0:fast0]
    diag_fast = np.diag(cov)[fast0:]
    cross = cov[np.arange(level0, fast0), np.arange(fast0, n_vars)]
    var_L = np.maximum(diag_level + diag_fast + 2.0 * cross, 1e-10)
    return l_hat, level_hat, var_L


# ---------------------------------------------------------------------------
# Expectation maximization for the class dynamics and mixing weights.
# ---------------------------------------------------------------------------
def _init_classes(hyper: MarkovMixtureHyperparameters, anchor_values: np.ndarray) -> list[_ClassParams]:
    base_sd = float(np.std(anchor_values)) if anchor_values.size else 1.0
    base_sd = min(max(base_sd, 0.2), 3.0)
    classes: list[_ClassParams] = []
    # Spread the initial classes across slow/fast regimes so EM can separate a
    # smooth-trajectory class from a volatile one.
    phis = np.linspace(0.2, 0.8, hyper.n_classes)
    lev_scales = np.linspace(0.15, 0.6, hyper.n_classes)
    for k in range(hyper.n_classes):
        classes.append(
            _ClassParams(
                s_lev=float(lev_scales[k]) * base_sd,
                phi=float(phis[k]),
                s_fast=0.5 * base_sd,
                anchor_sd=0.6 * base_sd,
            )
        )
    return classes


def _theta_to_params(theta: np.ndarray, hyper: MarkovMixtureHyperparameters) -> _ClassParams:
    s_lev = float(np.clip(math.exp(theta[0]), hyper.min_scale, hyper.max_scale))
    phi = float(np.clip(np.tanh(theta[1]), hyper.min_phi, hyper.max_phi))
    s_fast = float(np.clip(math.exp(theta[2]), hyper.min_scale, hyper.max_scale))
    anchor_sd = float(np.clip(math.exp(theta[3]), hyper.min_scale, hyper.max_scale))
    return _ClassParams(s_lev=s_lev, phi=phi, s_fast=s_fast, anchor_sd=anchor_sd)


def _params_to_theta(params: _ClassParams) -> np.ndarray:
    return np.array(
        [
            math.log(max(params.s_lev, 1e-3)),
            math.atanh(np.clip(params.phi, -0.95, 0.95)),
            math.log(max(params.s_fast, 1e-3)),
            math.log(max(params.anchor_sd, 1e-3)),
        ],
        dtype=float,
    )


def _fit_markov_mixture_params(
    fit_people: list[_PersonObs],
    hyper: MarkovMixtureHyperparameters,
    min_grid: np.ndarray,
    lag_grid: np.ndarray,
    level_prior_var: float,
    n_days: int,
) -> tuple[list[_ClassParams], np.ndarray]:
    """EM over class dynamics and stratum-conditional mixing weights.

    Returns the fitted class parameters and the (n_strata x n_classes) matrix of
    pattern-conditional mixing weights.
    """

    from scipy.optimize import minimize

    n_people = len(fit_people)
    if n_people == 0:
        classes = _init_classes(hyper, np.array([]))
        weights = np.full((hyper.n_strata, hyper.n_classes), 1.0 / hyper.n_classes)
        return classes, weights

    anchor_values = np.concatenate(
        [p.anchor_values for p in fit_people if p.anchor_values.size]
    ) if any(p.anchor_values.size for p in fit_people) else np.array([])
    classes = _init_classes(hyper, anchor_values)
    strata = np.array([p.stratum for p in fit_people], dtype=int)
    weights = np.full((hyper.n_strata, hyper.n_classes), 1.0 / hyper.n_classes)

    def class_loglik_vector(params: _ClassParams) -> np.ndarray:
        cov_L = _latent_cov(params, n_days, level_prior_var, min_grid, lag_grid)
        return np.array(
            [_anchor_loglik(p, cov_L, params.anchor_sd, hyper.jitter) for p in fit_people],
            dtype=float,
        )

    prev_total = -np.inf
    for _ in range(max(1, hyper.em_iters)):
        # E-step: per-person, per-class anchor log-likelihood and responsibilities.
        loglik = np.column_stack([class_loglik_vector(c) for c in classes])  # (N, K)
        log_weights = np.log(np.clip(weights[strata], 1e-8, None))            # (N, K)
        joint = loglik + log_weights
        row_max = joint.max(axis=1, keepdims=True)
        resp = np.exp(joint - row_max)
        resp_sum = resp.sum(axis=1, keepdims=True)
        resp = resp / np.where(resp_sum > 0, resp_sum, 1.0)                   # (N, K)
        total = float(np.sum(row_max.ravel() + np.log(np.clip(resp_sum.ravel(), 1e-12, None))))

        # M-step (weights): stratum-conditional mixing.
        new_weights = np.full_like(weights, 1.0 / hyper.n_classes)
        for s in range(hyper.n_strata):
            mask = strata == s
            if mask.any():
                col = resp[mask].sum(axis=0)
                if col.sum() > 0:
                    new_weights[s] = col / col.sum()
        weights = np.clip(new_weights, 1e-6, None)
        weights = weights / weights.sum(axis=1, keepdims=True)

        # M-step (dynamics): maximize responsibility-weighted anchor likelihood.
        new_classes: list[_ClassParams] = []
        for k, c in enumerate(classes):
            r_k = resp[:, k]
            if r_k.sum() < 1e-6:
                new_classes.append(c)
                continue

            def neg_obj(theta: np.ndarray) -> float:
                params = _theta_to_params(theta, hyper)
                cov_L = _latent_cov(params, n_days, level_prior_var, min_grid, lag_grid)
                ll = np.array(
                    [_anchor_loglik(p, cov_L, params.anchor_sd, hyper.jitter) for p in fit_people],
                    dtype=float,
                )
                return -float(np.sum(r_k * ll))

            res = minimize(
                neg_obj,
                _params_to_theta(c),
                method="L-BFGS-B",
                options={"maxiter": hyper.param_opt_maxiter},
            )
            new_classes.append(_theta_to_params(res.x, hyper))
        classes = new_classes

        if abs(total - prev_total) < hyper.em_tol * (abs(prev_total) + 1.0):
            break
        prev_total = total

    return classes, weights


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
def _estimate_markov_daily_readout(data, train_ids, p_daily, ridge=1.0):
    """Learn a single linear predictor of composite severity L from ALL daily
    covariates, on the TRAIN split only.

    The prelim L target is a crude per-patient anchor interpolation; the 24
    covariates are then ridge-regressed onto it, with missing covariates imputed
    to their train means. No oracle group/channel hints, so permuting the
    covariate columns and refitting leaves the predictor unchanged. Returns
    (intercept, beta[p_daily], train_means[p_daily]) or None.
    """

    x_cols = [f"X{j}" for j in range(p_daily)]
    obs_cols = [f"obs_X{j}" for j in range(p_daily)]
    comp_by = dict(tuple(data.components.groupby("id")))
    daily_by = dict(tuple(data.daily.groupby("id")))
    anc_by = dict(tuple(data.anchors.groupby("id")))
    X_all, O_all, L_all = [], [], []
    for pid in train_ids:
        comp_i = comp_by.get(pid); daily_i = daily_by.get(pid); anchors_i = anc_by.get(pid)
        if comp_i is None or daily_i is None or anchors_i is None:
            continue
        days = comp_i.sort_values("t")["t"].to_numpy(dtype=float)
        px, py = [], []
        for r in anchors_i.itertuples(index=False):
            if not bool(getattr(r, "observed", False)):
                continue
            v = float(getattr(r, "value"))
            if not np.isfinite(v):
                continue
            load = float(getattr(r, "loading")) or 1.0
            px.append(float(getattr(r, "window_end"))); py.append(v / load)
        if len(px) < 2:
            continue
        order = np.argsort(px)
        l_hat = np.interp(days, np.array(px)[order], np.array(py)[order])
        di = daily_i.sort_values("t").reset_index(drop=True)
        if not set(x_cols).issubset(di.columns):
            continue
        xv = di[x_cols].to_numpy(dtype=float)
        xo = di[obs_cols].to_numpy(dtype=float) if set(obs_cols).issubset(di.columns) else np.isfinite(xv).astype(float)
        n = min(len(days), len(xv))
        X_all.append(xv[:n]); O_all.append(xo[:n]); L_all.append(l_hat[:n])
    if not X_all:
        return None
    X = np.concatenate(X_all); O = np.concatenate(O_all); L = np.concatenate(L_all)
    means = np.array([X[O[:, j] > 0.5, j].mean() if (O[:, j] > 0.5).any() else 0.0 for j in range(p_daily)])
    ximp = np.where(O > 0.5, X, means[None, :])
    design = np.concatenate([np.ones((len(ximp), 1)), ximp], axis=1)
    reg = ridge * np.eye(p_daily + 1); reg[0, 0] = 0.0
    theta = np.linalg.solve(design.T @ design + reg, design.T @ L)
    return (float(theta[0]), theta[1:].astype(float), means.astype(float))


def fit_markov_pattern_mixture(
    data: SimulationData,
    hyperparameters: MarkovMixtureHyperparameters | None = None,
) -> MethodResult:
    """Fit the Markov pattern-mixture comparator on a simulated dataset.

    The returned intervals are model-based Gaussian intervals from the smoother
    posterior variance combined across mixture classes by the law of total
    variance. They are not split-conformal intervals. They are the classical
    counterpart to the BALL uncertainty stack, reported as a comparator.
    """

    hyper = hyperparameters or MarkovMixtureHyperparameters()
    config = data.config
    n_days = config.t

    # Per-day covariance grids, shared across all classes for the fixed horizon.
    idx = np.arange(n_days)
    min_grid = np.minimum.outer(idx, idx).astype(float)
    lag_grid = np.abs(np.subtract.outer(idx, idx)).astype(float)
    level_prior_var = hyper.level_prior_sd**2

    daily_groups = dict(tuple(data.daily.groupby("id", sort=True))) if data.daily is not None else {}
    anchor_groups = dict(tuple(data.anchors.groupby("id", sort=True)))

    # Learn the daily readout on the train split only (no oracle group hints).
    markov_train_ids = (
        set(int(i) for i in data.individuals.loc[data.individuals["split"] == "train", "id"])
        if "split" in data.individuals.columns else None
    )
    daily_readout = (
        _estimate_markov_daily_readout(data, markov_train_ids, data.config.p_daily)
        if (hyper.use_daily and markov_train_ids) else None
    )

    people: list[_PersonObs] = []
    for person_id, comp_i in data.components.groupby("id", sort=True):
        daily_i = daily_groups.get(person_id)
        anchors_i = anchor_groups.get(person_id)
        if anchors_i is None:
            anchors_i = pd.DataFrame(columns=data.anchors.columns)
        people.append(_build_person_obs(comp_i, daily_i, anchors_i, hyper.use_daily, daily_readout))

    # Pattern strata from the observed anchor count. This is what makes the
    # mixing weights pattern-conditional (pattern-mixture structure).
    anchor_counts = np.array([p.anchor_values.size for p in people], dtype=float)
    if hyper.n_strata > 1 and np.unique(anchor_counts).size >= hyper.n_strata:
        quantiles = np.quantile(anchor_counts, np.linspace(0, 1, hyper.n_strata + 1)[1:-1])
        strata = np.digitize(anchor_counts, quantiles).astype(int)
    else:
        strata = np.zeros(len(people), dtype=int)
    for p, s in zip(people, strata):
        p.stratum = int(min(s, hyper.n_strata - 1))

    # Parameter fitting is restricted to the configured split (default train) and
    # capped for runtime; the fitted classes are then applied to every patient.
    split_ids: set[int] | None = None
    if hyper.fit_split is not None and "split" in data.individuals.columns:
        split_ids = set(
            data.individuals.loc[data.individuals["split"] == hyper.fit_split, "id"].astype(int)
        )
    fit_people = [p for p in people if (split_ids is None or p.person_id in split_ids)]
    fit_people = [p for p in fit_people if p.anchor_values.size >= 2]
    if len(fit_people) > hyper.max_fit_persons:
        rng = make_rng(config.seed, "markov_mixture_fit", hyper.random_state)
        keep = rng.choice(len(fit_people), size=hyper.max_fit_persons, replace=False)
        fit_people = [fit_people[i] for i in sorted(keep)]

    classes, weights = _fit_markov_mixture_params(
        fit_people, hyper, min_grid, lag_grid, level_prior_var, n_days
    )

    # Precompute each class's latent covariance for the responsibility step.
    class_cov = [_latent_cov(c, n_days, level_prior_var, min_grid, lag_grid) for c in classes]

    pred_frames: list[pd.DataFrame] = []
    interval_frames: list[pd.DataFrame] = []
    n_active_classes: list[float] = []

    for person in people:
        days = person.t_index
        w = person.anchor_values.size
        # Responsibilities from the anchor marginal likelihood and the patient's
        # pattern-conditional mixing weights.
        if w > 0:
            ll = np.array(
                [_anchor_loglik(person, cov, c.anchor_sd, hyper.jitter) for c, cov in zip(classes, class_cov)],
                dtype=float,
            )
        else:
            ll = np.zeros(len(classes), dtype=float)
        log_w = np.log(np.clip(weights[person.stratum], 1e-8, None))
        joint = ll + log_w
        joint -= joint.max()
        resp = np.exp(joint)
        resp = resp / resp.sum()
        n_active_classes.append(float(np.sum(resp > 0.05)))

        # Smooth under each class, then mix the trajectories by responsibility.
        l_by_class = np.zeros((len(classes), person.n_days), dtype=float)
        var_by_class = np.zeros((len(classes), person.n_days), dtype=float)
        for k, c in enumerate(classes):
            l_hat, _level_hat, var_L = _smooth_person(person, c, hyper)
            l_by_class[k] = l_hat
            var_by_class[k] = var_L

        l_mix = np.einsum("k,kt->t", resp, l_by_class)
        # Law of total variance across mixture classes: within-class posterior
        # variance plus between-class disagreement of the posterior means.
        second_moment = np.einsum("k,kt->t", resp, var_by_class + l_by_class**2)
        var_total = np.maximum(second_moment - l_mix**2, 1e-10)
        raw_sd = np.sqrt(var_total)

        pred = pd.DataFrame({"id": person.person_id, "t": days, "L_hat": l_mix})
        # The classical comparator targets the composite severity, so both
        # disorder channels inherit the composite estimate. It does NOT report a
        # slow/fast decomposition: separating the RDoC-explained slow component
        # from the residual requires the proxy and sparsity identification that
        # is specific to BALL. Reporting a level/fast split here would not align
        # with the data-generating slow/delta convention, so the decomposition
        # and RDoC diagnostics are left not applicable for this comparator.
        pred["z_d_hat"] = l_mix
        pred["z_p_hat"] = l_mix
        pred_frames.append(pred)

        scale = hyper.interval_scale
        interval = pd.DataFrame(
            {
                "id": person.person_id,
                "t": days,
                "lower": l_mix - scale * raw_sd,
                "upper": l_mix + scale * raw_sd,
                "raw_sd": raw_sd,
                "total_sd": raw_sd,
                "z_d_lower": l_mix - scale * raw_sd,
                "z_d_upper": l_mix + scale * raw_sd,
                "z_p_lower": l_mix - scale * raw_sd,
                "z_p_upper": l_mix + scale * raw_sd,
                "z_d_raw_sd": raw_sd,
                "z_p_raw_sd": raw_sd,
                "interval_scale": scale,
            }
        )
        interval_frames.append(interval)

    predictions = pd.concat(pred_frames, ignore_index=True)
    intervals = pd.concat(interval_frames, ignore_index=True)

    metadata = {
        "status": "markov_pattern_mixture_comparator",
        "not_ball": True,
        "comparator_role": "old_school_statistics_vs_transformer_ml",
        "model": "finite mixture of linear-Gaussian Markov state-space trajectories",
        "latent_dynamics": "L(t) = level(t) [random walk] + fast(t) [AR(1)]",
        "pattern_mixture": "class prior conditioned on observed anchor-count stratum",
        "n_classes": hyper.n_classes,
        "n_strata": hyper.n_strata,
        "estimation": "EM on the anchor marginal likelihood; batch Gaussian smoother for the latent",
        "interval_type": "model_based_gaussian_law_of_total_variance",
        "uses_rdoc_proxy": False,
        "fit_split": hyper.fit_split,
        "n_fit_persons": len(fit_people),
        "mean_active_classes": float(np.mean(n_active_classes)) if n_active_classes else float("nan"),
        "class_params": [
            {"s_lev": c.s_lev, "phi": c.phi, "s_fast": c.s_fast, "anchor_sd": c.anchor_sd}
            for c in classes
        ],
        "stratum_mixing_weights": weights.tolist(),
    }
    return MethodResult("markov_pattern_mixture", predictions, intervals, metadata=metadata)
'''

# --- simulations.src.methods.ball_ssm (simulations/src/methods/ball_ssm.py) ---
SRC_SIMULATIONS_SRC_METHODS_BALL_SSM = r'''
"""BALL as written in the appendix: a deep state-space variational model.

This is the literal implementation of Sec. 3-4 (no structural-smoother
stand-in, no smoothness-penalty stand-in for the KL):

* Generative model p_theta(u) with explicit neural drift functions
  - alpha_{n+1} ~ N(alpha_n + f_theta(alpha_n, a_n), dt * Sigma_alpha)   (eq. 7)
  - alpha_{n,d} carries the adaptive-Lasso prior (eq. 8), a per-(n,d) Laplace
  - delta^.{n+1} ~ N(delta^.n + g_theta(delta^.n, x_EHR_n, a_n), dt * Sigma_delta)
  - z^d_n = alpha_n . RDoC_n + delta^d_n ,  z^p_n = alpha_n . RDoC_n + delta^p_n
  - anchors are recall-windowed means of z, observed via a Gaussian on the total
    score or a graded-response IRT model.

* Teacher q_psi(u | full record): a bidirectional transformer returning a
  Gaussian over u_n at every session, trained by maximizing the ELBO
      L = E_q[log p(y|u)] - KL(q_psi(u) || p_theta(u))
  with the reparameterization trick. The KL is the actual KL to the dynamics
  prior, evaluated by a single-sample Monte Carlo estimate of
  log q(u) - log p(u) under the transition densities above.

* Student q(u | past): the same backbone with a causal mask, trained by
  distilling the teacher's per-session Gaussian posterior (a Gaussian KL whose
  mean term is the spec's Mahalanobis distance to the teacher mean) plus direct
  re-anchoring (eta times the anchor log-likelihood), per the spec student
  objective L2 = ||mu - u_teacher||^2_{Sigma^-1} + eta * [-log p(y|u)].

Sec. 5 is implemented as a deep ensemble with a diagonal last-layer Laplace
approximation on the transformer's final mean head, followed by split conformal
calibration in the pipeline wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from ..model_utils import MethodResult, SimulationData, make_rng, marginal_anchor_sd

LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class SSMConfig:
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.10
    action_embed_dim: int = 8
    drift_hidden: int = 32
    teacher_epochs: int = 350
    student_epochs: int = 250
    batch_size: int = 64
    learning_rate: float = 1e-3
    grad_clip_norm: float = 5.0
    kl_warmup_epochs: int = 80          # anneal the KL weight 0 -> 1 (beta-VAE warmup)
    anchor_warmup_epochs: int = 120     # phase-1: anchor-only warmup sets latent scale/sign
    # The calibrated (sparse) anchors are up-weighted in the ELBO to correct for
    # the dense daily features being highly redundant (6 correlated groups, not
    # 24 independent observations); without this the dense likelihood swamps the
    # anchors and the optimizer drifts a channel's sign into a flipped basin.
    anchor_weight: float = 40.0
    ensemble_size: int = 5              # deep ensemble K (spec Sec. 5)
    delta_phi: float = 1.0              # delta AR coefficient; 1.0 = spec random walk, <1 = mean-reverting
    # Decomposition-pressure ablation knobs (validation step B). All default to the
    # current model behavior, so the canonical fit is unchanged unless an arm sets them.
    delta_drift_use_ehr: bool = True         # False drops EHR from g_theta (the delta-drift escape hatch)
    delta_persistence_penalty: float = 0.0   # >0 penalizes low-frequency (persistent) power in delta
    delta_persistence_window: int = 14       # window (days) for the persistence/anchor low-pass
    alpha_proxy_anchor_weight: float = 0.0   # >0 anchors the slow component to the low-pass composite
    use_alpha_slow: bool = True              # False: direct latent-drift model, no alpha'RDoC severity term
    # Direct RDoC drift head for parameter-recovery scenarios. When enabled, a
    # sparse linear B_t beta term is added to both delta-channel transition means.
    rdoc_drift_head: bool = False
    rdoc_drift_l1: float = 0.0
    rdoc_drift_adaptive: bool = False         # adaptive-Lasso: penalize beta with fixed pilot-derived weights
    rdoc_drift_adaptive_gamma: float = 1.0    # weight exponent gamma in w_j = 1/(|beta_pilot_j|+eps)^gamma
    rdoc_drift_adaptive_eps: float = 1e-3     # stabilizer eps in the pilot weight formula
    rdoc_drift_weights: tuple = ()            # FIXED per-coordinate weights from the pilot fit (empty = unit L1)
    reanchor_weight: float = 1.0        # eta in the student objective
    # Adaptive-Lasso alpha rate for the neural ELBO. This is not numerically
    # interchangeable with the structural posterior's xi because the VI loss
    # sums the Laplace prior across every time point and latent dimension.
    alpha_lasso_xi: float = 2.0
    laplace_prior_precision: float = 1.0
    laplace_scale: float = 1.0
    max_individuals: int = 1000
    seed: int = 1729


# ---------------------------------------------------------------------------
# Data tensors (built directly from the simulation so f_theta / g_theta get the
# action index and the structured-EHR signals they condition on).
# ---------------------------------------------------------------------------
@dataclass
class SSMBatch:
    ids: list[int]
    t_values: np.ndarray
    rdoc: torch.Tensor            # (B, T, q)   observed RDoC proxy (model input + z = alpha'RDoC)
    ehr: torch.Tensor            # (B, T, p)   structured-EHR daily features (g_theta input, imputed)
    x_raw: torch.Tensor          # (B, T, p)   raw daily observations (likelihood targets; 0 where missing)
    x_obs: torch.Tensor          # (B, T, p)   daily observation mask
    enc_features: torch.Tensor   # (B, T, F)   full encoder input
    action: torch.Tensor         # (B, T) long action index (0 = null action)
    dt: torch.Tensor             # (B, T) days to next session (here 1.0)
    anchor_value: torch.Tensor   # (B, T, 2)
    anchor_obs: torch.Tensor     # (B, T, 2)
    anchor_total: torch.Tensor   # (B, T, 2) raw integer graded-response total (IRT); 0 if Gaussian
    anchor_items: torch.Tensor   # (B, T, 2, J) integer per-item graded responses (IRT); 0 if Gaussian
    win_start: torch.Tensor      # (B, T, 2) long
    win_end: torch.Tensor        # (B, T, 2) long
    z_truth: torch.Tensor        # (B, T, 2) z_d, z_p (eval only)


def _impute(frame: pd.DataFrame, cols: list[str], *, causal: bool = False) -> np.ndarray:
    arr = frame[cols].astype(float)
    if causal:
        # Forward-only students cannot borrow future RDoC/EHR observations to
        # fill earlier missing sessions. Initial gaps have no past observation,
        # so they fall back to zero rather than a full-record mean.
        arr = arr.ffill().fillna(0.0)
    else:
        arr = arr.interpolate(limit_direction="both").fillna(arr.mean()).fillna(0.0)
    return arr.to_numpy(dtype=np.float32)


def build_ssm_batch(
    data: SimulationData,
    split: str | None,
    device: torch.device,
    max_individuals: int,
    feature_stats: tuple[np.ndarray, np.ndarray] | None = None,
    causal_impute: bool = False,
) -> tuple[SSMBatch, tuple[np.ndarray, np.ndarray]]:
    q = data.config.q
    p = data.config.p_daily
    ids_all = data.individuals
    if split is not None:
        ids_all = ids_all[ids_all["split"] == split]
    ids = sorted(int(i) for i in ids_all["id"].tolist())[:max_individuals]
    comp = data.components[data.components["id"].isin(ids)].sort_values(["id", "t"])
    daily = data.daily[data.daily["id"].isin(ids)].sort_values(["id", "t"])
    anchors = data.anchors[data.anchors["id"].isin(ids)]
    t_values = np.sort(comp["t"].unique())
    T = len(t_values)
    B = len(ids)
    t_index = {int(t): k for k, t in enumerate(t_values)}

    rdoc = np.zeros((B, T, q), dtype=np.float32)
    ehr = np.zeros((B, T, p), dtype=np.float32)
    x_raw = np.zeros((B, T, p), dtype=np.float32)
    x_obs = np.zeros((B, T, p), dtype=np.float32)
    x_input = np.zeros((B, T, p), dtype=np.float32)
    dt = np.ones((B, T), dtype=np.float32)
    action = np.zeros((B, T), dtype=np.int64)
    z_truth = np.zeros((B, T, 2), dtype=np.float32)
    anchor_value = np.zeros((B, T, 2), dtype=np.float32)
    anchor_obs = np.zeros((B, T, 2), dtype=np.float32)
    anchor_total = np.zeros((B, T, 2), dtype=np.float32)
    use_irt = str(getattr(data.config, "anchor_observation", "gaussian")).lower() == "irt"
    n_items = int(getattr(data.config, "irt_n_items", 9))
    anchor_items = np.zeros((B, T, 2, n_items), dtype=np.float32)
    win_start = np.zeros((B, T, 2), dtype=np.int64)
    win_end = np.zeros((B, T, 2), dtype=np.int64)

    b_cols = [f"B{d}" for d in range(q)]
    x_cols = [f"X{j}" for j in range(p)]
    comp_by_id = dict(tuple(comp.groupby("id")))
    daily_by_id = dict(tuple(daily.groupby("id")))
    anc_by_id = dict(tuple(anchors.groupby("id")))

    for row, pid in enumerate(ids):
        ci = comp_by_id[pid].sort_values("t")
        di = daily_by_id[pid].sort_values("t")
        rdoc[row] = _impute(ci, b_cols, causal=causal_impute)
        ehr[row] = _impute(di, x_cols, causal=causal_impute)
        obs_cols = [f"obs_X{j}" for j in range(p)]
        raw = di[x_cols].to_numpy(dtype=np.float32)
        x_obs[row] = di[obs_cols].to_numpy(dtype=np.float32) if obs_cols[0] in di.columns else np.isfinite(raw).astype(np.float32)
        input_cols = [f"input_X{j}" for j in range(p)]
        x_input[row] = (
            di[input_cols].to_numpy(dtype=np.float32)
            if input_cols[0] in di.columns else x_obs[row]
        )
        x_raw[row] = np.nan_to_num(raw, nan=0.0)
        if "dt" in ci.columns:
            dt[row] = pd.to_numeric(ci["dt"], errors="coerce").fillna(1.0).clip(lower=1e-3).to_numpy(dtype=np.float32)
        action[row] = (ci["a"].to_numpy(dtype=int) + 1)  # -1 (none) -> 0
        z_truth[row, :, 0] = ci["z_d"].to_numpy(dtype=np.float32)
        z_truth[row, :, 1] = ci["z_p"].to_numpy(dtype=np.float32)
        ai = anc_by_id.get(pid)
        if ai is not None:
            for arow in ai.itertuples(index=False):
                if not bool(arow.observed):
                    continue
                ch = 0 if arow.anchor == "Y1" else 1
                k = t_index.get(int(arow.t))
                if k is None:
                    continue
                if use_irt:
                    items = getattr(arow, "irt_items", None)
                    total = getattr(arow, "irt_total", float("nan"))
                    if items is None or not np.isfinite(total):
                        continue
                    anchor_items[row, k, ch, :] = np.asarray(items, dtype=np.float32)
                    anchor_total[row, k, ch] = float(total)
                    anchor_value[row, k, ch] = float(total)   # finite encoder feature; likelihood uses items
                else:
                    if not np.isfinite(arow.value):
                        continue
                    anchor_value[row, k, ch] = float(arow.value)
                    total = getattr(arow, "irt_total", float("nan"))
                    if total is not None and np.isfinite(total):
                        anchor_total[row, k, ch] = float(total)
                anchor_obs[row, k, ch] = 1.0
                win_start[row, k, ch] = max(0, t_index.get(int(arow.window_start), 0))
                win_end[row, k, ch] = t_index.get(int(arow.window_end), k)

    # Encoder features: standardized [EHR | EHR availability masks | RDoC |
    # anchor value*obs | obs flags] +
    # the action is added as a learned embedding inside the encoder.
    raw = np.concatenate([ehr, x_input, rdoc, anchor_value * anchor_obs, anchor_obs], axis=-1)
    no_encoder_scaling = data.metadata.get("encoder_feature_scaling") == "none"
    if no_encoder_scaling:
        feature_stats = (
            np.zeros(raw.shape[-1], dtype=np.float32),
            np.ones(raw.shape[-1], dtype=np.float32),
        )
    elif feature_stats is None:
        mean = raw.reshape(-1, raw.shape[-1]).mean(0)
        std = raw.reshape(-1, raw.shape[-1]).std(0)
        std = np.where(std > 1e-6, std, 1.0)
        feature_stats = (mean.astype(np.float32), std.astype(np.float32))
    mean, std = feature_stats
    enc = raw.astype(np.float32) if no_encoder_scaling else ((raw - mean) / std).astype(np.float32)
    # Some empirical covariates are deliberately represented as raw continuous
    # counts. Preserve those declared daily columns exactly in the encoder rather
    # than silently standardizing them with a cohort-derived denominator.
    unscaled_daily = data.metadata.get("encoder_unscaled_daily_indices", [])
    for col_idx in unscaled_daily:
        col_idx = int(col_idx)
        if col_idx < 0 or col_idx >= p:
            raise ValueError(f"Invalid unscaled daily feature index: {col_idx}")
        enc[:, :, col_idx] = raw[:, :, col_idx].astype(np.float32)
    if data.metadata.get("encoder_unscaled_daily_masks", False):
        enc[:, :, p:2 * p] = raw[:, :, p:2 * p].astype(np.float32)

    to = lambda a: torch.as_tensor(a, device=device)
    batch = SSMBatch(
        ids=ids,
        t_values=t_values,
        rdoc=to(rdoc),
        ehr=to(ehr),
        x_raw=to(x_raw),
        x_obs=to(x_obs),
        enc_features=to(enc),
        action=to(action),
        dt=to(dt),
        anchor_value=to(anchor_value),
        anchor_obs=to(anchor_obs),
        anchor_total=to(anchor_total),
        anchor_items=to(anchor_items),
        win_start=to(win_start),
        win_end=to(win_end),
        z_truth=to(z_truth),
    )
    return batch, feature_stats


# ---------------------------------------------------------------------------
# Generative model parameters: f_theta, g_theta, diffusion scales, Lasso rate,
# observation noise.  (theta in the appendix.)
# ---------------------------------------------------------------------------
class GenerativeModel(nn.Module):
    def __init__(self, q: int, ehr_dim: int, config: SSMConfig, n_actions: int,
                 obs_sd: tuple[float, float] = (0.95, 0.85),
                 loadings: tuple[float, float] = (1.0, 0.75),
                 anchor_observation: str = "gaussian",
                 irt_n_items: int = 9, irt_n_categories: int = 4,
                 irt_loc: float = 0.0, irt_scale: float = 1.0,
                 irt_item_disc=(), irt_item_thresholds=()) -> None:
        super().__init__()
        self.q = q
        self.anchor_observation = str(anchor_observation).lower()
        self.irt_n_items = int(irt_n_items)
        self.irt_n_categories = max(int(irt_n_categories), 2)
        self.action_embed = nn.Embedding(n_actions, config.action_embed_dim)
        # f_theta(alpha_n, a_n) -> drift on the q-dim loading
        self.f_theta = nn.Sequential(
            nn.Linear(q + config.action_embed_dim, config.drift_hidden),
            nn.Tanh(),
            nn.Linear(config.drift_hidden, q),
        )
        # g_theta(delta, [x_EHR], a) -> scalar drift, applied per channel (shared).
        # delta_drift_use_ehr=False drops the EHR input: the EHR-conditioned drift is
        # the path by which a flexible delta re-creates persistence from the covariates
        # and absorbs the slow signal (validation step B arm).
        self.delta_drift_use_ehr = bool(getattr(config, "delta_drift_use_ehr", True))
        g_in = 1 + config.action_embed_dim + (ehr_dim if self.delta_drift_use_ehr else 0)
        self.g_theta = nn.Sequential(
            nn.Linear(g_in, config.drift_hidden),
            nn.Tanh(),
            nn.Linear(config.drift_hidden, 1),
        )
        # Daily EHR observation head: the dense daily features X are noisy
        # correlates of the latent (notes, vitals, fills correlated with the
        # PHQ/PCL latent), modeled as a learnable linear-Gaussian readout of the
        # severity components. The model learns which features load on the latent
        # and which are uninformative (near-zero loading, absorbed as noise).
        # Readout from the independent latent coordinates [slow, delta_d, delta_p]
        # (z_d, z_p are linear combos of these; feeding them too would be
        # rank-deficient and induces channel sign-mixing). The delta coordinates
        # are ALSO provided within-person centered: the DGP fast-residual
        # proxies (X12-X15) track the zero-level dynamic residual while the
        # per-patient level lives inside delta, and a global readout bias cannot
        # absorb per-person levels. Without the centered coordinates the
        # level-free targets pull delta toward zero level and push the person
        # level into the slow component (the same level mis-assignment fixed in
        # the structural posterior on 2026-06-10). Centered and uncentered
        # deltas differ by a per-person constant, so across a batch they are not
        # collinear and the readout learns which convention fits each feature.
        self.daily_readout = nn.Linear(5, ehr_dim)
        self.log_sigma_x = nn.Parameter(torch.full((ehr_dim,), math.log(0.6)))
        # Per-day diffusion log-sds.
        self.log_sigma_alpha = nn.Parameter(torch.full((q,), math.log(0.05)))
        self.log_sigma_delta = nn.Parameter(torch.full((2,), math.log(0.35)))
        self.delta_phi = float(getattr(config, "delta_phi", 1.0))  # delta AR coef (1=random walk)
        self.rdoc_drift_head = bool(getattr(config, "rdoc_drift_head", False))
        self.rdoc_drift_l1 = float(getattr(config, "rdoc_drift_l1", 0.0))
        self.rdoc_drift_adaptive = bool(getattr(config, "rdoc_drift_adaptive", False))
        self.rdoc_drift_adaptive_gamma = float(getattr(config, "rdoc_drift_adaptive_gamma", 1.0))
        self.rdoc_drift_adaptive_eps = float(getattr(config, "rdoc_drift_adaptive_eps", 1e-3))
        self.rdoc_drift_beta = nn.Parameter(torch.zeros(q))
        # FIXED adaptive-Lasso weights (a buffer, not a parameter: never updated by
        # the optimizer). Empty config -> unit weights, which reduce the adaptive
        # penalty to plain L1. Populated by the two-stage pilot in the benchmark.
        _drift_w = tuple(getattr(config, "rdoc_drift_weights", ()) or ())
        if len(_drift_w) == q:
            self.register_buffer("rdoc_drift_weights", torch.tensor(_drift_w, dtype=torch.float32))
        else:
            self.register_buffer("rdoc_drift_weights", torch.ones(q))
        # Adaptive-Lasso rate xi is a hyperparameter (eq. 8), fixed: making it
        # learnable lets the model inflate log p via the log(xi/2) normalizer and
        # drive the KL negative.
        self.register_buffer("log_xi", torch.tensor(math.log(max(float(config.alpha_lasso_xi), 1e-8))))
        # Observation noise is a known property of the instrument (PHQ/PCL), so it
        # is fixed rather than learned (a learnable obs sd collapses by inflating
        # to explain the anchors away as noise).
        self.register_buffer("log_sigma_obs", torch.log(torch.tensor(obs_sd, dtype=torch.float32)))
        # Anchor loadings beta_k (Y1, Y2): the anchor measures beta_k * windowed
        # mean of the latent, so the likelihood compares against beta_k * zbar.
        self.register_buffer("anchor_loadings", torch.tensor(loadings, dtype=torch.float32))
        # Graded-response IRT observation (calibrated-instrument sensitivity). The
        # per-item discriminations and ordered thresholds and the affine trait map are
        # FIXED buffers from the frozen calibration (validation/irt_calibration.json),
        # not learned. Fixing the instrument is what identifies the latent scale, and
        # the same constants are shared with the S0 and Markov comparators. The integer
        # responses enter through the EXACT per-item graded-response likelihood.
        if self.anchor_observation == "irt":
            if not len(irt_item_disc) or not len(irt_item_thresholds):
                raise ValueError("anchor_observation='irt' requires a frozen calibration (irt_item_disc/thresholds)")
            self.register_buffer("irt_item_disc", torch.tensor(irt_item_disc, dtype=torch.float32))            # (J,)
            self.register_buffer("irt_item_thresholds", torch.tensor(irt_item_thresholds, dtype=torch.float32))  # (J, K-1)
            self.register_buffer("irt_loc", torch.tensor(float(irt_loc), dtype=torch.float32))
            self.register_buffer("irt_scale", torch.tensor(float(irt_scale) or 1.0, dtype=torch.float32))
        self.log_sigma0 = nn.Parameter(torch.tensor(math.log(1.5)))       # prior sd at n=1

    def drift_alpha(self, alpha: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        return self.f_theta(torch.cat([alpha, a_emb], dim=-1))

    def drift_delta(self, delta_channel: torch.Tensor, ehr: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        # delta_channel: (B,T); ehr: (B,T,p); a_emb: (B,T,e) -> (B,T)
        parts = [delta_channel.unsqueeze(-1)]
        if self.delta_drift_use_ehr:
            parts.append(ehr)
        parts.append(a_emb)
        return self.g_theta(torch.cat(parts, dim=-1)).squeeze(-1)

    def direct_rdoc_drift(self, rdoc: torch.Tensor) -> torch.Tensor:
        if not self.rdoc_drift_head:
            return torch.zeros(rdoc.shape[:-1], device=rdoc.device, dtype=rdoc.dtype)
        return torch.einsum("btq,q->bt", rdoc, self.rdoc_drift_beta)

    def log_prior(self, alpha, delta_d, delta_p, batch: SSMBatch) -> torch.Tensor:
        """log p_theta(u) for sampled trajectories. Returns (B,) summed over n,dims."""

        a_emb = self.action_embed(batch.action)           # (B,T,e)
        dt = batch.dt.clamp(min=1e-3)
        # Floor the diffusion sds so the learned prior cannot collapse to a
        # near-deterministic fit of the posterior (which would drive the KL to
        # zero/negative and remove all regularization).
        floor = math.log(0.03)
        sig_a2 = (self.log_sigma_alpha.clamp(min=floor).exp() ** 2).view(1, 1, -1)   # (1,1,q)
        sig_d2 = (self.log_sigma_delta.clamp(min=floor).exp() ** 2).view(1, 1, 2)
        sigma0_2 = self.log_sigma0.exp() ** 2

        # n = 1 prior: N(0, sigma0^2) for alpha and delta.
        def gauss_lp(x, mean, var):
            return -0.5 * (LOG_2PI + torch.log(var) + (x - mean) ** 2 / var)

        lp = gauss_lp(alpha[:, 0, :], torch.zeros_like(alpha[:, 0, :]), sigma0_2).sum(-1)
        lp = lp + gauss_lp(delta_d[:, 0], torch.zeros_like(delta_d[:, 0]), sigma0_2)
        lp = lp + gauss_lp(delta_p[:, 0], torch.zeros_like(delta_p[:, 0]), sigma0_2)

        # transitions n>=2 (eq. 7 for alpha, eq. analog for delta)
        f = self.drift_alpha(alpha[:, :-1, :], a_emb[:, :-1, :])             # (B,T-1,q)
        mean_alpha = alpha[:, :-1, :] + f
        var_alpha = dt[:, :-1].unsqueeze(-1) * sig_a2
        lp = lp + gauss_lp(alpha[:, 1:, :], mean_alpha, var_alpha).sum(dim=(1, 2))

        gd = self.drift_delta(delta_d[:, :-1], batch.ehr[:, :-1, :], a_emb[:, :-1, :])
        gp = self.drift_delta(delta_p[:, :-1], batch.ehr[:, :-1, :], a_emb[:, :-1, :])
        bd = self.direct_rdoc_drift(batch.rdoc[:, :-1, :])
        var_d = dt[:, :-1] * sig_d2[..., 0]
        var_p = dt[:, :-1] * sig_d2[..., 1]
        lp = lp + gauss_lp(delta_d[:, 1:], self.delta_phi * delta_d[:, :-1] + gd + bd, var_d).sum(1)
        lp = lp + gauss_lp(delta_p[:, 1:], self.delta_phi * delta_p[:, :-1] + gp + bd, var_p).sum(1)

        # Adaptive-Lasso prior on each alpha_{n,d} (eq. 8): marginalizing tau gives
        # a Laplace(0, 1/xi); log p = log(xi/2) - xi|alpha|.
        xi = self.log_xi.exp()
        lp = lp + (torch.log(xi / 2.0) - xi * alpha.abs()).sum(dim=(1, 2))
        return lp

    def anchor_log_lik(self, z_d, z_p, batch: SSMBatch) -> torch.Tensor:
        """Anchor observation log-likelihood, summed per patient (B,).

        Gaussian: N(value | loading * windowed-latent, sigma^2). IRT: the exact
        per-item graded-response likelihood through the fixed calibrated instrument
        (see _irt_channel_log_lik), with the latent mapped to the calibrated trait scale.
        """

        sig2 = (self.log_sigma_obs.exp() ** 2)   # (2,)
        z = [z_d, z_p]
        B, T = z_d.shape
        device = z_d.device
        arange = torch.arange(T, device=device).view(1, 1, T)
        total = torch.zeros(B, device=device)
        for ch in range(2):
            obs = batch.anchor_obs[:, :, ch] > 0.5         # (B,T)
            if not torch.any(obs):
                continue
            starts = batch.win_start[:, :, ch].unsqueeze(-1)   # (B,T,1)
            ends = batch.win_end[:, :, ch].unsqueeze(-1)
            win = ((arange >= starts) & (arange <= ends)).float()   # (B,T,T)
            win = win / win.sum(-1, keepdim=True).clamp(min=1.0)
            zbar = torch.einsum("bts,bs->bt", win, z[ch])          # (B,T) windowed mean
            if self.anchor_observation == "irt":
                lp = self._irt_channel_log_lik(zbar, batch.anchor_items[:, :, ch, :])
            else:
                resid2 = (batch.anchor_value[:, :, ch] - self.anchor_loadings[ch] * zbar) ** 2
                lp = -0.5 * (LOG_2PI + torch.log(sig2[ch]) + resid2 / sig2[ch])
            total = total + (lp * obs.float()).sum(1)
        return total

    def _irt_channel_log_lik(self, zbar: torch.Tensor, items_obs: torch.Tensor) -> torch.Tensor:
        """Exact per-item graded-response log-likelihood for one anchor channel.

        The windowed latent zbar (B,T) is mapped to the calibrated trait scale,
        trait = (zbar - loc) / scale, and each item's category probability is
        P(item_j = c | trait) = sigma(a_j (trait - b_{j,c})) - sigma(a_j (trait - b_{j,c+1})),
        with the fixed calibrated discriminations a_j and ordered thresholds b_{j,.}.
        The observed integer responses are scored exactly; the result is summed over
        the J items (B,T). items_obs is (B,T,J) and is masked by anchor_obs upstream.
        """

        trait = (zbar - self.irt_loc) / self.irt_scale                  # (B,T)
        a = self.irt_item_disc.view(1, 1, -1, 1)                        # (1,1,J,1)
        b = self.irt_item_thresholds.unsqueeze(0).unsqueeze(0)          # (1,1,J,K-1)
        pstar = torch.sigmoid(a * (trait[..., None, None] - b))         # (B,T,J,K-1) = P(>=k)
        ones = torch.ones_like(pstar[..., :1])
        zeros = torch.zeros_like(pstar[..., :1])
        pad = torch.cat([ones, pstar, zeros], dim=-1)                  # (B,T,J,K+1)
        probs = (pad[..., :-1] - pad[..., 1:]).clamp(min=1e-9)         # (B,T,J,K) = P(=c)
        idx = items_obs.long().clamp(0, probs.shape[-1] - 1).unsqueeze(-1)   # (B,T,J,1)
        p_resp = probs.gather(-1, idx).squeeze(-1)                     # (B,T,J)
        return torch.log(p_resp).sum(-1)                              # (B,T)

    def daily_log_lik(self, slow, delta_d, delta_p, batch: SSMBatch) -> torch.Tensor:
        """Dense daily-EHR observation log-likelihood, masked by missingness (B,).

        The daily features are noisy correlates of the latent; this is what pins
        the daily trajectory between the sparse anchors. Modeled as a linear
        readout of the independent latent coordinates [slow, delta_d, delta_p].
        """

        dd_c = delta_d - delta_d.mean(dim=1, keepdim=True)
        dp_c = delta_p - delta_p.mean(dim=1, keepdim=True)
        feat = torch.stack([slow, delta_d, delta_p, dd_c, dp_c], dim=-1)  # (B,T,5)
        pred = self.daily_readout(feat)                                   # (B,T,p)
        # Floor the daily observation noise so the dense features cannot be
        # over-trusted and swamp the calibrated (but sparse) anchors.
        sig2 = (self.log_sigma_x.clamp(min=math.log(0.4)).exp() ** 2).view(1, 1, -1)
        resid2 = (batch.x_raw - pred) ** 2
        lp = -0.5 * (LOG_2PI + torch.log(sig2) + resid2 / sig2)
        return (lp * batch.x_obs).sum(dim=(1, 2))


# ---------------------------------------------------------------------------
# Inference network q_psi (teacher: bidirectional; student: causal).
# ---------------------------------------------------------------------------
class InferenceTransformer(nn.Module):
    def __init__(self, feature_dim: int, q: int, max_t: int, config: SSMConfig,
                 action_embed: nn.Embedding, causal: bool) -> None:
        super().__init__()
        self.q = q
        self.causal = causal
        self.action_embed = action_embed
        in_dim = feature_dim + action_embed.embedding_dim
        self.input_proj = nn.Linear(in_dim, config.d_model)
        self.position = nn.Embedding(max_t, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.n_heads,
            dim_feedforward=4 * config.d_model, dropout=config.dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.mu_head = nn.Linear(config.d_model, q + 2)
        self.logvar_head = nn.Linear(config.d_model, q + 2)

    def _mask(self, length: int, device: torch.device):
        if not self.causal:
            return None
        return torch.triu(torch.full((length, length), float("-inf"), device=device), diagonal=1)

    def encode(self, batch: SSMBatch) -> torch.Tensor:
        B, T, _ = batch.enc_features.shape
        a_emb = self.action_embed(batch.action)
        x = torch.cat([batch.enc_features, a_emb], dim=-1)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.input_proj(x) + self.position(pos)
        return self.encoder(h, mask=self._mask(T, x.device))

    def forward(self, batch: SSMBatch) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(batch)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-8.0, 4.0)
        return mu, logvar


def _split_u(sample: torch.Tensor, q: int):
    return sample[..., :q], sample[..., q], sample[..., q + 1]


def _slow_from_alpha(alpha: torch.Tensor, rdoc: torch.Tensor, config: SSMConfig) -> torch.Tensor:
    if not bool(getattr(config, "use_alpha_slow", True)):
        return alpha.new_zeros(alpha.shape[:-1])
    return (alpha * rdoc).sum(-1)


def _gaussian_entropy(logvar: torch.Tensor) -> torch.Tensor:
    # H(q) summed over dims, per patient (B,)
    return 0.5 * (LOG_2PI + 1.0 + logvar).sum(dim=(1, 2))


def _causal_rolling_mean(x: torch.Tensor, window: int) -> torch.Tensor:
    """Trailing moving average over the last `window` steps, per (B, T) sequence."""
    B, T = x.shape
    csum = torch.cat([x.new_zeros(B, 1), x.cumsum(dim=1)], dim=1)
    idx = torch.arange(T, device=x.device)
    left = (idx - window + 1).clamp(min=0)
    sums = csum[:, idx + 1] - csum[:, left]
    counts = (idx + 1 - left).to(x.dtype)
    return sums / counts


def _decomposition_pressure(slow, dd, dp, config: SSMConfig) -> torch.Tensor:
    """Validation step B: optional penalties that push persistent signal out of the
    fast residual delta and into the proxy-explained slow component alpha'RDoC.

    Returns a scalar penalty, zero unless an arm enables a knob.
    """
    w = int(getattr(config, "delta_persistence_window", 14))
    pressure = slow.new_zeros(())
    pw = float(getattr(config, "delta_persistence_penalty", 0.0))
    if pw > 0.0:
        # A fast residual should carry little content slower than the recall window.
        pressure = pressure + pw * (
            _causal_rolling_mean(dd, w) ** 2 + _causal_rolling_mean(dp, w) ** 2
        ).mean()
    aw = float(getattr(config, "alpha_proxy_anchor_weight", 0.0))
    if aw > 0.0:
        # Anchor the slow component to the low-pass of the composite latent, so the
        # persistent part of severity is carried by alpha'RDoC rather than delta.
        z_avg = slow + 0.5 * (dd + dp)
        pressure = pressure + aw * ((_causal_rolling_mean(z_avg, w) - slow) ** 2).mean()
    return pressure


def _rdoc_drift_penalty(gen: GenerativeModel) -> torch.Tensor:
    if not getattr(gen, "rdoc_drift_head", False):
        return gen.rdoc_drift_beta.new_zeros(())
    weight = float(getattr(gen, "rdoc_drift_l1", 0.0))
    if weight <= 0.0:
        return gen.rdoc_drift_beta.new_zeros(())
    beta = gen.rdoc_drift_beta
    if getattr(gen, "rdoc_drift_adaptive", False):
        # Adaptive Lasso (Zou 2006): FIXED per-coordinate weights derived from an
        # unpenalized pilot fit (set on the model as rdoc_drift_weights), so the
        # penalized objective is not path-dependent. Plain L1 is the unit-weight case.
        return weight * (gen.rdoc_drift_weights * beta.abs()).sum()
    return weight * beta.abs().sum()


# ---------------------------------------------------------------------------
# Production fit / predict / deep ensemble for the literal BALL SSM.
# ---------------------------------------------------------------------------
import dataclasses as _dataclasses

_B_FIELDS = [
    "rdoc", "ehr", "x_raw", "x_obs", "enc_features", "action", "dt",
    "anchor_value", "anchor_obs", "anchor_total", "anchor_items", "win_start", "win_end", "z_truth",
]


def _subbatch(batch: SSMBatch, idx: torch.Tensor) -> SSMBatch:
    kw = {f: getattr(batch, f)[idx] for f in _B_FIELDS}
    kw["ids"] = [batch.ids[i] for i in idx.tolist()]
    kw["t_values"] = batch.t_values
    return _dataclasses.replace(batch, **kw)


def _make_models(data: SimulationData, train_batch: SSMBatch, config: SSMConfig,
                 device: torch.device, causal: bool):
    q = data.config.q
    gen = GenerativeModel(
        q, data.config.p_daily, config, n_actions=data.config.n_treatment_types + 1,
        # Marginal anchor error SD: the DGP anchor errors are AR(1), so the
        # instrument's marginal noise is error_sd / sqrt(1 - rho^2).
        obs_sd=(
            marginal_anchor_sd(data.config.y1, data.config.rho_serial_y1),
            marginal_anchor_sd(data.config.y2, data.config.rho_serial_y2),
        ),
        loadings=(data.config.y1.loading, data.config.y2.loading),
        anchor_observation=getattr(data.config, "anchor_observation", "gaussian"),
        irt_n_items=getattr(data.config, "irt_n_items", 9),
        irt_n_categories=getattr(data.config, "irt_n_categories", 4),
        irt_loc=getattr(data.config, "irt_loc", 0.0),
        irt_scale=getattr(data.config, "irt_scale", 1.0),
        irt_item_disc=getattr(data.config, "irt_item_discriminations", ()),
        irt_item_thresholds=getattr(data.config, "irt_item_thresholds", ()),
    ).to(device)
    net = InferenceTransformer(
        train_batch.enc_features.shape[-1], q, len(train_batch.t_values), config,
        gen.action_embed, causal=causal,
    ).to(device)
    return gen, net


def _train_one_ssm(data: SimulationData, train_batch: SSMBatch, config: SSMConfig,
                   device: torch.device, causal: bool, seed: int):
    """Train a single ELBO model with the anchor-warmup + anchor-upweighted
    curriculum validated to recover both channels. Used for the bidirectional
    teacher; the deployable causal student is produced by
    `_train_student_distilled`, not by re-running this with a causal mask."""

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    q = data.config.q
    gen, net = _make_models(data, train_batch, config, device, causal)
    params = list(dict.fromkeys(list(gen.parameters()) + list(net.parameters())))
    history: list[dict[str, float | int | str]] = []

    # Phase 1: daily-free anchor ELBO so the calibrated sparse anchors set the
    # latent scale and sign before the dense daily likelihood can lock a flipped
    # basin. It is a reparameterized sample with the KL term (not the posterior
    # mean), so the variational variance head is trained from the start rather
    # than entering Phase 2 untrained.
    opt1 = torch.optim.Adam(params, lr=2e-3)
    net.train()
    for ep in range(config.anchor_warmup_epochs):
        opt1.zero_grad()
        mu, logvar = net(train_batch)
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        u = mu + std * eps
        a, dd, dp = _split_u(u, q)
        slow = _slow_from_alpha(a, train_batch.rdoc, config)
        anchor_ll = gen.anchor_log_lik(slow + dd, slow + dp, train_batch)
        log_p = gen.log_prior(a, dd, dp, train_batch)
        log_q = (-0.5 * (LOG_2PI + logvar + eps ** 2)).sum(dim=(1, 2))
        kl = log_q - log_p
        klw = min(1.0, ep / max(config.kl_warmup_epochs, 1))
        loss = -(config.anchor_weight * anchor_ll - klw * kl).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, config.grad_clip_norm)
        opt1.step()
        history.append(
            {
                "phase": "anchor_warmup",
                "epoch": ep + 1,
                "loss": float(loss.detach().cpu()),
                "anchor_ll": float(anchor_ll.mean().detach().cpu()),
                "daily_ll": float("nan"),
                "kl": float(kl.mean().detach().cpu()),
                "kl_weight": float(klw),
            }
        )

    # Phase 2: full ELBO (anchor up-weighted) minibatched with KL annealing.
    # The KL weight deliberately re-anneals from zero here (a second warm
    # restart): the dense daily likelihood switches on at this phase boundary
    # and re-shapes the posterior, so the dynamics prior is reintroduced
    # gradually against the new reconstruction term rather than fighting it at
    # full strength from the first step.
    opt2 = torch.optim.Adam(params, lr=config.learning_rate)
    B = train_batch.z_truth.shape[0]
    bs = config.batch_size
    for ep in range(config.teacher_epochs):
        klw = min(1.0, ep / max(config.kl_warmup_epochs, 1))
        perm = torch.randperm(B, device=device)
        epoch_n = 0
        epoch_loss = 0.0
        epoch_anchor_ll = 0.0
        epoch_daily_ll = 0.0
        epoch_kl = 0.0
        for s in range(0, B, bs):
            idx = perm[s:s + bs]
            sb = _subbatch(train_batch, idx)
            opt2.zero_grad()
            mu, logvar = net(sb)
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            u = mu + std * eps
            a, dd, dp = _split_u(u, q)
            slow = _slow_from_alpha(a, sb.rdoc, config)
            anchor_ll = gen.anchor_log_lik(slow + dd, slow + dp, sb)
            daily_ll = gen.daily_log_lik(slow, dd, dp, sb)
            log_p = gen.log_prior(a, dd, dp, sb)
            log_q = (-0.5 * (LOG_2PI + logvar + eps ** 2)).sum(dim=(1, 2))
            kl = log_q - log_p
            loss = -(config.anchor_weight * anchor_ll + daily_ll - klw * kl).mean()
            loss = loss + _decomposition_pressure(slow, dd, dp, config)
            loss = loss + _rdoc_drift_penalty(gen)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, config.grad_clip_norm)
            opt2.step()
            n_batch = int(len(idx))
            epoch_n += n_batch
            epoch_loss += float(loss.detach().cpu()) * n_batch
            epoch_anchor_ll += float(anchor_ll.mean().detach().cpu()) * n_batch
            epoch_daily_ll += float(daily_ll.mean().detach().cpu()) * n_batch
            epoch_kl += float(kl.mean().detach().cpu()) * n_batch
        denom = max(epoch_n, 1)
        history.append(
            {
                "phase": "teacher_elbo",
                "epoch": ep + 1,
                "loss": epoch_loss / denom,
                "anchor_ll": epoch_anchor_ll / denom,
                "daily_ll": epoch_daily_ll / denom,
                "kl": epoch_kl / denom,
                "kl_weight": float(klw),
            }
        )
    return gen, net, history


def _train_student_distilled(data: SimulationData, teacher_train_batch: SSMBatch,
                             student_train_batch: SSMBatch,
                             config: SSMConfig, device: torch.device,
                             teacher_gen: GenerativeModel,
                             teacher_net: InferenceTransformer, seed: int):
    """Distill a causal student from a trained teacher (spec student objective).

    The loss is the Gaussian KL from the student's per-session posterior to the
    teacher's, KL(q_student || q_teacher), whose mean term is exactly the spec's
    Mahalanobis distance ||mu_s - mu_t||^2_{Sigma_t^-1} and whose variance terms
    train the student's variance head toward the teacher's, plus eta
    (reanchor_weight) times the direct anchor negative log-likelihood evaluated
    at the student mean. The generative parameters theta are shared with the
    teacher and frozen here; only the causal inference network is trained.
    """

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    q = data.config.q
    student_action_embed = nn.Embedding(
        teacher_gen.action_embed.num_embeddings,
        teacher_gen.action_embed.embedding_dim,
    ).to(device)
    student_action_embed.load_state_dict(teacher_gen.action_embed.state_dict())
    student = InferenceTransformer(
        student_train_batch.enc_features.shape[-1], q, len(student_train_batch.t_values), config,
        student_action_embed, causal=True,
    ).to(device)
    teacher_net.eval()
    with torch.no_grad():
        mu_t_full, logvar_t_full = teacher_net(teacher_train_batch)
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable_params, lr=config.learning_rate)
    B = student_train_batch.z_truth.shape[0]
    bs = config.batch_size
    student.train()
    history: list[dict[str, float | int | str]] = []
    for ep in range(config.student_epochs):
        perm = torch.randperm(B, device=device)
        epoch_n = 0
        epoch_loss = 0.0
        epoch_kl = 0.0
        epoch_anchor_ll = 0.0
        for s in range(0, B, bs):
            idx = perm[s:s + bs]
            sb = _subbatch(student_train_batch, idx)
            mu_t = mu_t_full[idx]
            var_t = logvar_t_full[idx].exp().clamp(min=1e-6)
            opt.zero_grad()
            mu_s, logvar_s = student(sb)
            var_s = logvar_s.exp()
            # KL(N(mu_s, var_s) || N(mu_t, var_t)) per element, summed per patient.
            kl = 0.5 * (
                logvar_t_full[idx] - logvar_s
                + (var_s + (mu_s - mu_t) ** 2) / var_t
                - 1.0
            ).sum(dim=(1, 2))
            a, dd, dp = _split_u(mu_s, q)
            slow = _slow_from_alpha(a, sb.rdoc, config)
            anchor_ll = teacher_gen.anchor_log_lik(slow + dd, slow + dp, sb)
            loss = (kl - config.reanchor_weight * anchor_ll).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, config.grad_clip_norm)
            opt.step()
            n_batch = int(len(idx))
            epoch_n += n_batch
            epoch_loss += float(loss.detach().cpu()) * n_batch
            epoch_kl += float(kl.mean().detach().cpu()) * n_batch
            epoch_anchor_ll += float(anchor_ll.mean().detach().cpu()) * n_batch
        denom = max(epoch_n, 1)
        history.append(
            {
                "phase": "student_distill",
                "epoch": ep + 1,
                "loss": epoch_loss / denom,
                "anchor_ll": epoch_anchor_ll / denom,
                "daily_ll": float("nan"),
                "kl": epoch_kl / denom,
                "kl_weight": float("nan"),
            }
        )
    return student, history


def _last_layer_laplace_variance(
    net: InferenceTransformer,
    train_batch: SSMBatch,
    eval_batch: SSMBatch,
    config: SSMConfig,
) -> torch.Tensor:
    """Diagonal last-layer Laplace variance for the final mean head.

    This is the lightweight Sec. 5 uncertainty layer: freeze the transformer
    feature extractor, approximate the final linear head posterior with a
    diagonal Gaussian precision prior + feature Gram diagonal, and propagate the
    resulting parameter variance to every latent output coordinate.
    """

    with torch.no_grad():
        h_train = net.encode(train_batch).reshape(-1, net.mu_head.in_features)
        h_eval = net.encode(eval_batch)
        precision = float(config.laplace_prior_precision) + h_train.pow(2).sum(dim=0)
        weight_cov_diag = float(config.laplace_scale) / precision.clamp(min=1e-8)
        bias_var = float(config.laplace_scale) / max(
            float(config.laplace_prior_precision) + float(h_train.shape[0]), 1e-8
        )
        output_var = torch.einsum("btd,d->bt", h_eval.pow(2), weight_cov_diag) + bias_var
        return output_var.unsqueeze(-1).expand(-1, -1, net.q + 2)


def _member_predict(
    gen: GenerativeModel,
    net: InferenceTransformer,
    batch: SSMBatch,
    q: int,
    laplace_train_batch: SSMBatch | None = None,
    config: SSMConfig | None = None,
):
    """Per-session posterior means and variances for one ensemble member.

    Returns dict of (B,T) arrays for z_d, z_p, slow, delta_d, delta_p and their
    posterior variances (var_zd, var_zp, var_L), plus per-dim alpha means.
    """

    net.eval()
    with torch.no_grad():
        mu, logvar = net(batch)
        a, dd, dp = _split_u(mu, q)
        var = logvar.exp()
        if laplace_train_batch is not None and config is not None and config.laplace_scale > 0:
            laplace_var = _last_layer_laplace_variance(net, laplace_train_batch, batch, config)
            var = var + laplace_var
        else:
            laplace_var = torch.zeros_like(var)
        cfg = config or SSMConfig()
        rdoc = batch.rdoc
        slow = _slow_from_alpha(a, rdoc, cfg)
        # When the slow alpha'RDoC term is disabled (direct-RDoC mode) it adds
        # nothing to severity, so its posterior variance must be zero too;
        # otherwise the raw interval width/coverage inflate spuriously.
        if getattr(cfg, "use_alpha_slow", True):
            var_slow = (rdoc ** 2 * var[..., :q]).sum(-1)        # Var(alpha'RDoC), diag
            lap_slow = (rdoc ** 2 * laplace_var[..., :q]).sum(-1)
        else:
            var_slow = torch.zeros_like(var[..., q])
            lap_slow = torch.zeros_like(laplace_var[..., q])
        var_dd = var[..., q]
        var_dp = var[..., q + 1]
        lap_dd = laplace_var[..., q]
        lap_dp = laplace_var[..., q + 1]
        z_d = slow + dd
        z_p = slow + dp
        var_zd = var_slow + var_dd
        var_zp = var_slow + var_dp
        lap_var_zd = lap_slow + lap_dd
        lap_var_zp = lap_slow + lap_dp
        # Cov(z_d, z_p) = Var(slow) (shared slow); Var(L) for L = 0.5(z_d+z_p).
        var_L = 0.25 * (var_zd + var_zp + 2.0 * var_slow)
        lap_var_L = 0.25 * (lap_var_zd + lap_var_zp + 2.0 * lap_slow)
    f = lambda x: x.cpu().numpy()
    return {
        "z_d": f(z_d), "z_p": f(z_p), "slow": f(slow), "dd": f(dd), "dp": f(dp),
        "alpha": f(a), "var_zd": f(var_zd), "var_zp": f(var_zp), "var_L": f(var_L),
        "lap_var_zd": f(lap_var_zd), "lap_var_zp": f(lap_var_zp), "lap_var_L": f(lap_var_L),
        "rdoc_drift_beta": f(gen.rdoc_drift_beta.detach()),
    }


def _assemble_ssm_ensemble(members, output_batch, q, method_name, metadata):
    """Combine ensemble member predictions into a MethodResult.

    Deep-ensemble law of total variance: total var = mean_k(member var) +
    var_k(member mean) [aleatoric + epistemic]. Shared by the causal student
    ensemble and the bidirectional teacher ensemble.
    """

    def _combine(key_mean, key_var):
        means = np.stack([m[key_mean] for m in members], axis=0)   # (K,B,T)
        variances = np.stack([m[key_var] for m in members], axis=0)
        mean = means.mean(0)
        total_var = variances.mean(0) + means.var(0)
        return mean, np.sqrt(np.clip(total_var, 1e-10, None))

    z_d_mean, z_d_sd = _combine("z_d", "var_zd")
    z_p_mean, z_p_sd = _combine("z_p", "var_zp")
    L_mean = 0.5 * (z_d_mean + z_p_mean)
    # L total var from member L variances + between-member L-mean disagreement.
    L_means = np.stack([0.5 * (m["z_d"] + m["z_p"]) for m in members], axis=0)
    L_vars = np.stack([m["var_L"] for m in members], axis=0)
    L_sd = np.sqrt(np.clip(L_vars.mean(0) + L_means.var(0), 1e-10, None))
    L_laplace_sd = np.sqrt(np.clip(np.stack([m["lap_var_L"] for m in members], axis=0).mean(0), 0.0, None))
    zd_laplace_sd = np.sqrt(np.clip(np.stack([m["lap_var_zd"] for m in members], axis=0).mean(0), 0.0, None))
    zp_laplace_sd = np.sqrt(np.clip(np.stack([m["lap_var_zp"] for m in members], axis=0).mean(0), 0.0, None))
    slow_mean = np.stack([m["slow"] for m in members], axis=0).mean(0)
    dd_mean = np.stack([m["dd"] for m in members], axis=0).mean(0)
    dp_mean = np.stack([m["dp"] for m in members], axis=0).mean(0)
    alpha_mean = np.stack([m["alpha"] for m in members], axis=0).mean(0)   # (B,T,q)
    beta_members = np.stack([m["rdoc_drift_beta"] for m in members], axis=0)
    metadata = dict(metadata)
    metadata["rdoc_drift_beta_hat"] = beta_members.mean(axis=0).tolist()
    metadata["rdoc_drift_beta_members"] = beta_members.tolist()

    ids = output_batch.ids
    t_values = output_batch.t_values
    rows_id = np.repeat(ids, len(t_values))
    rows_t = np.tile(t_values, len(ids))
    _flat = lambda arr: arr.reshape(-1)

    pred = pd.DataFrame({
        "id": rows_id, "t": rows_t,
        "L_hat": _flat(L_mean),
        "z_d_hat": _flat(z_d_mean), "z_p_hat": _flat(z_p_mean),
        "slow_hat": _flat(slow_mean), "proxy_slow_hat": _flat(slow_mean),
        "delta_hat": _flat(0.5 * (dd_mean + dp_mean)),
        "delta_d_hat": _flat(dd_mean), "delta_p_hat": _flat(dp_mean),
    })
    for d in range(q):
        pred[f"alpha_hat{d}"] = _flat(alpha_mean[:, :, d])

    L_sd_f = _flat(L_sd); zd_sd_f = _flat(z_d_sd); zp_sd_f = _flat(z_p_sd)
    intervals = pd.DataFrame({
        "id": rows_id, "t": rows_t,
        "lower": _flat(L_mean) - 1.96 * L_sd_f, "upper": _flat(L_mean) + 1.96 * L_sd_f,
        "z_d_lower": _flat(z_d_mean) - 1.96 * zd_sd_f, "z_d_upper": _flat(z_d_mean) + 1.96 * zd_sd_f,
        "z_p_lower": _flat(z_p_mean) - 1.96 * zp_sd_f, "z_p_upper": _flat(z_p_mean) + 1.96 * zp_sd_f,
        "raw_sd": L_sd_f, "total_sd": L_sd_f,
        "z_d_raw_sd": zd_sd_f, "z_d_total_sd": zd_sd_f,
        "z_p_raw_sd": zp_sd_f, "z_p_total_sd": zp_sd_f,
        "laplace_sd": _flat(L_laplace_sd),
        "z_d_laplace_sd": _flat(zd_laplace_sd),
        "z_p_laplace_sd": _flat(zp_laplace_sd),
        "interval_scale": 1.96,
        "interval_source": "ssm_deep_ensemble_laplace_raw",
    })
    return MethodResult(method_name, pred, intervals, metadata=metadata)


def fit_ball_ssm(
    data: SimulationData,
    config: SSMConfig | None = None,
    device: str | torch.device | None = None,
    causal: bool = False,
    prediction_split: str = "test",
    return_teacher: bool = False,
) -> "MethodResult | tuple[MethodResult, MethodResult]":
    """Fit the literal BALL deep state-space VI as a deep ensemble and return
    predictions + raw posterior intervals (law-of-total-variance combined).

    causal=False is the bidirectional ELBO teacher; causal=True trains a
    bidirectional teacher per member and then distills a forward-only student
    from it (Gaussian-KL distillation, whose mean term is the spec Mahalanobis,
    plus eta times direct re-anchoring), so the student is the spec's distilled
    deployable model rather than an independently trained causal ELBO fit.

    When causal=True and return_teacher=True, the bidirectional teacher ensemble
    the students distil from is ALSO evaluated and returned as a second
    MethodResult (the primary smoother), at no extra training cost since those
    teachers are already trained for distillation.
    """

    config = config or SSMConfig()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    q = data.config.q

    train_batch, stats = build_ssm_batch(data, "train", device, config.max_individuals)
    eval_batch, _ = build_ssm_batch(data, prediction_split, device, config.max_individuals, feature_stats=stats)
    if causal:
        student_train_batch, student_stats = build_ssm_batch(
            data, "train", device, config.max_individuals, causal_impute=True
        )
        student_eval_batch, _ = build_ssm_batch(
            data,
            prediction_split,
            device,
            config.max_individuals,
            feature_stats=student_stats,
            causal_impute=True,
        )
    else:
        student_train_batch = train_batch
        student_eval_batch = eval_batch

    members = []
    teacher_members = []
    loss_history: list[dict[str, float | int | str]] = []
    for k in range(max(1, config.ensemble_size)):
        member_seed = config.seed + 1000 * k
        gen, teacher_net, teacher_history = _train_one_ssm(
            data, train_batch, config, device, causal=False, seed=member_seed
        )
        for row in teacher_history:
            rec = dict(row)
            rec.update({"member": k, "member_seed": member_seed})
            loss_history.append(rec)
        if causal:
            if return_teacher:
                # Evaluate the bidirectional teacher (no extra training): it is the
                # primary smoother the student distils from.
                teacher_members.append(
                    _member_predict(gen, teacher_net, eval_batch, q, laplace_train_batch=train_batch, config=config)
                )
            net, student_history = _train_student_distilled(
                data,
                train_batch,
                student_train_batch,
                config,
                device,
                gen,
                teacher_net,
                seed=config.seed + 1000 * k + 500,
            )
            for row in student_history:
                rec = dict(row)
                rec.update({"member": k, "member_seed": config.seed + 1000 * k + 500})
                loss_history.append(rec)
        else:
            net = teacher_net
        members.append(
            _member_predict(
                gen,
                net,
                student_eval_batch if causal else eval_batch,
                q,
                laplace_train_batch=student_train_batch if causal else train_batch,
                config=config,
            )
        )
        # Release this member's GPU tensors before the next member trains, so
        # the deep ensemble does not accumulate device memory across members
        # (the empirical full cohort is marginal on an 8 GB device).
        del gen, teacher_net, net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    irt_prov = None
    if str(getattr(data.config, "anchor_observation", "gaussian")).lower() == "irt":
        irt_prov = {
            "loc": float(data.config.irt_loc),
            "scale": float(data.config.irt_scale),
            "n_items": int(data.config.irt_n_items),
            "n_categories": int(data.config.irt_n_categories),
            "discriminations": [float(a) for a in data.config.irt_item_discriminations],
            "thresholds": [[float(x) for x in b] for b in data.config.irt_item_thresholds],
        }

    student_metadata = {
        "status": "literal_deep_ssm_vi",
        "anchor_observation": getattr(data.config, "anchor_observation", "gaussian"),
        "irt_calibration": irt_prov,
        "ensemble_size": int(config.ensemble_size),
        "alpha_lasso_xi": float(config.alpha_lasso_xi),
        "use_alpha_slow": bool(config.use_alpha_slow),
        "rdoc_drift_head": bool(config.rdoc_drift_head),
        "rdoc_drift_l1": float(config.rdoc_drift_l1),
        "rdoc_drift_adaptive": bool(config.rdoc_drift_adaptive),
        "rdoc_drift_adaptive_gamma": float(config.rdoc_drift_adaptive_gamma) if config.rdoc_drift_adaptive else None,
        "rdoc_drift_adaptive_eps": float(getattr(config, "rdoc_drift_adaptive_eps", 1e-3)) if config.rdoc_drift_adaptive else None,
        "rdoc_drift_weights": list(config.rdoc_drift_weights) if config.rdoc_drift_adaptive else None,
        "last_layer_laplace": "diagonal_final_mean_head",
        "laplace_prior_precision": float(config.laplace_prior_precision),
        "laplace_scale": float(config.laplace_scale),
        "causal_student": bool(causal),
        "student_objective": (
            "gaussian_kl_distillation_to_teacher_plus_reanchoring" if causal else None
        ),
        "student_input_imputation": "forward_fill_no_future" if causal else None,
        "teacher_input_imputation": "bidirectional_interpolation",
        "student_epochs": int(config.student_epochs) if causal else None,
        "reanchor_weight": float(config.reanchor_weight) if causal else None,
        "anchor_warmup_epochs": int(config.anchor_warmup_epochs),
        "anchor_weight": float(config.anchor_weight),
        "anchor_weight_note": (
            "anchors are up-weighted in the ELBO, so the objective is a "
            "tempered likelihood, not an exact ELBO"
        ),
        "anchor_obs_sd": "marginal AR(1) sd: error_sd / sqrt(1 - rho_serial^2)",
        "teacher_epochs": int(config.teacher_epochs),
        "prediction_split": prediction_split,
        "loss_history": loss_history,
    }
    student_result = _assemble_ssm_ensemble(
        members,
        student_eval_batch if causal else eval_batch,
        q,
        "BALL-SSM-teacher" if not causal else "BALL-SSM-student",
        student_metadata,
    )
    if causal and return_teacher:
        teacher_metadata = {
            "status": "literal_deep_ssm_vi_teacher",
            "anchor_observation": getattr(data.config, "anchor_observation", "gaussian"),
            "irt_calibration": irt_prov,
            "ensemble_size": int(config.ensemble_size),
            "alpha_lasso_xi": float(config.alpha_lasso_xi),
            "use_alpha_slow": bool(config.use_alpha_slow),
            "rdoc_drift_head": bool(config.rdoc_drift_head),
            "rdoc_drift_l1": float(config.rdoc_drift_l1),
            "rdoc_drift_adaptive": bool(config.rdoc_drift_adaptive),
            "rdoc_drift_adaptive_gamma": float(config.rdoc_drift_adaptive_gamma) if config.rdoc_drift_adaptive else None,
            "rdoc_drift_weights": list(config.rdoc_drift_weights) if config.rdoc_drift_adaptive else None,
            "role": "bidirectional_teacher_smoother",
            "causal_student": False,
            "teacher_input_imputation": "bidirectional_interpolation",
            "teacher_epochs": int(config.teacher_epochs),
            "anchor_warmup_epochs": int(config.anchor_warmup_epochs),
            "anchor_weight": float(config.anchor_weight),
            "prediction_split": prediction_split,
            "loss_history": loss_history,
        }
        teacher_result = _assemble_ssm_ensemble(teacher_members, eval_batch, q, "BALL-SSM-teacher", teacher_metadata)
        return student_result, teacher_result
    return student_result
'''

# --- simulations.run_ball_pipeline (simulations/run_ball_pipeline.py) ---
SRC_SIMULATIONS_RUN_BALL_PIPELINE = r'''
"""Run the end-to-end BALL latent-recovery pipeline.

This runner is the canonical math-spec pipeline wrapper for the measurement
half of the appendix (Sec. 3 latent model, Sec. 4 two-stage inference, Sec. 5
uncertainty). It produces the latent-recovery artifacts in one place:

1. calibrated session-level severity intervals for z_d and z_p;
2. per-patient time-varying alpha loading vectors;
3. rho_RDoC, the share of recovery variance explained by RDoC features.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from simulations.src.dgp import generate_dataset
from simulations.src.diagnostics import assert_diagnostics_pass, diagnostics_frame, run_diagnostics
from simulations.src.metrics import (
    active_set_metrics,
    channel_interval_metrics,
    component_rmse,
    coverage,
    double_anchor_latent_metrics,
    identifiability_diagnostics,
    interval_score,
    latent_rmse,
    mean_width,
    proxy_state_metrics,
    rdoc_recovery_fraction,
    summarize_mcse,
)
from simulations.src.methods.ball_ssm import SSMConfig, fit_ball_ssm
from simulations.src.methods.ball_structural import (
    BallStructuralHyperparameters,
    conformal_calibrate_structural_posterior,
    fit_ball_structural_posterior,
    oracle_latent_conformal_intervals,
)
from simulations.src.methods.s0 import fit_s0_structural_prior_ssm
from simulations.src.methods.markov_pattern_mixture import fit_markov_pattern_mixture
from simulations.src.model_utils import (
    MethodResult,
    SimulationConfig,
    make_run_id,
    publish_latest_artifact,
    publish_latest_text,
)

OUTPUT_ROOT = ROOT / "outputs" / "ball_pipeline"
RUN_ROOT = ROOT / "runs"
EVAL_SPLIT = "test"

METRIC_COLS = [
    "latent_rmse",
    "slow_rmse",
    "delta_rmse",
    "z_d_rmse",
    "z_p_rmse",
    "z_joint_rmse",
    "delta_d_rmse",
    "delta_p_rmse",
    "coverage",
    "mean_width",
    "interval_score",
    "z_d_coverage",
    "z_p_coverage",
    "z_d_mean_width",
    "z_p_mean_width",
    "active_f1",
    "active_topk_f1",
    "c_proxy_rmse",
    "c_proxy_corr",
    "rho_rdoc",
    "slow_recovery_corr",
    "fast_recovery_corr",
    "slow_to_fast_leakage",
    "fast_to_slow_leakage",
    "slow_var_recovery",
    "fast_var_recovery",
    "loading_step_mean_abs",
    "elapsed_seconds",
]


# Minimum per-replicate scale for a run to feed the manuscript. Smoke/quick runs
# fall below these and are published under a separate SMOKE prefix so they can
# never overwrite the manuscript-grade latest_* artifacts or the Table 1 source.
MANUSCRIPT_GRADE_MIN = {
    "n": 1000, "t": 84, "teacher_epochs": 100, "student_epochs": 100,
    "d_model": 96, "n_layers": 3, "members": 5,
}


def _is_manuscript_grade(args) -> bool:
    """A run is manuscript-grade only if every scale knob meets the minimum."""

    return (
        int(args.n) >= MANUSCRIPT_GRADE_MIN["n"]
        and int(args.t) >= MANUSCRIPT_GRADE_MIN["t"]
        and int(args.teacher_epochs) >= MANUSCRIPT_GRADE_MIN["teacher_epochs"]
        and int(args.student_epochs) >= MANUSCRIPT_GRADE_MIN["student_epochs"]
        and int(args.d_model) >= MANUSCRIPT_GRADE_MIN["d_model"]
        and int(args.n_layers) >= MANUSCRIPT_GRADE_MIN["n_layers"]
        and int(args.members) >= MANUSCRIPT_GRADE_MIN["members"]
    )


def _torch_resource_summary() -> dict[str, object]:
    out: dict[str, object] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        out["cuda_device_name"] = torch.cuda.get_device_name(0)
        out["cuda_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full BALL latent-learning pipeline.")
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--t", type=int, default=42)
    parser.add_argument("--base-seed", type=int, default=21000)
    # Moderate manuscript scale (3 layers, width 96, 150/150 teacher/student
    # epochs). Appendix H states exactly these. Smoke runs must pass smaller
    # values explicitly, which the manuscript-grade gate then rejects.
    parser.add_argument("--teacher-epochs", type=int, default=300)
    parser.add_argument("--student-epochs", type=int, default=300)
    # The anchor-only warmup sets the latent scale/sign and is load-bearing for
    # SSM recovery; the known-good archived run used 120. Keep it at 120 so the
    # teacher trains properly rather than being starved of the warmup.
    parser.add_argument("--anchor-warmup-epochs", type=int, default=120,
                        help="Phase-1 anchor-only warmup epochs (sets latent scale/sign).")
    parser.add_argument("--kl-warmup-epochs", type=int, default=80,
                        help="KL-weight annealing epochs in phase 2.")
    parser.add_argument(
        "--label-source",
        type=str,
        default="ssm-teacher",
        choices=["ssm-teacher", "structural-teacher", "student-ensemble"],
        help="Posterior source for severity, alpha, and rho artifacts (default: the primary transformer teacher).",
    )
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--alpha-lasso-xi", type=float, default=2.0,
                        help="Adaptive-Lasso alpha rate for the BALL SSM prior.")
    parser.add_argument("--conformal-alpha", type=float, default=0.05)
    parser.add_argument("--skip-s0", action="store_true")
    parser.add_argument("--skip-markov", action="store_true",
                        help="Skip the Markov pattern-mixture comparator. Do not use for manuscript Table 1 runs.")
    parser.add_argument("--per-member-metrics", action="store_true", help="Also score each ensemble member (diagnostic; off by default for speed).")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def _metrics(
    replicate: int,
    data_seed: int,
    role: str,
    result: MethodResult,
    data,
    elapsed_seconds: float,
    member: int | None = None,
    member_seed: int | None = None,
) -> dict[str, float | str | int | None]:
    out: dict[str, float | str | int | None] = {
        "replicate": replicate,
        "data_seed": data_seed,
        "member": member,
        "member_seed": member_seed,
        "role": role,
        "method": result.method,
        "evaluation_split": EVAL_SPLIT,
        "latent_rmse": latent_rmse(result.predictions, data.components, data.individuals, EVAL_SPLIT),
        "slow_rmse": component_rmse(
            result.predictions,
            data.components,
            "slow_hat",
            "slow",
            data.individuals,
            EVAL_SPLIT,
        ),
        "delta_rmse": component_rmse(
            result.predictions,
            data.components,
            "delta_hat",
            "delta",
            data.individuals,
            EVAL_SPLIT,
        ),
        "coverage": coverage(result.intervals, data.components, data.individuals, EVAL_SPLIT)
        if result.intervals is not None
        else np.nan,
        "mean_width": mean_width(result.intervals, data.individuals, EVAL_SPLIT)
        if result.intervals is not None
        else np.nan,
        "interval_score": interval_score(result.intervals, data.components, 0.05, data.individuals, EVAL_SPLIT)
        if result.intervals is not None
        else np.nan,
        "rho_rdoc": rdoc_recovery_fraction(result.predictions, data, EVAL_SPLIT),
        "elapsed_seconds": elapsed_seconds,
    }
    out.update(identifiability_diagnostics(result.predictions, data, EVAL_SPLIT))
    out.update(active_set_metrics(result.predictions, data.components, data.config.q, individuals=data.individuals, split=EVAL_SPLIT))
    out.update(proxy_state_metrics(result.predictions, data.components, data.config.q, individuals=data.individuals, split=EVAL_SPLIT))
    out.update(double_anchor_latent_metrics(result.predictions, data.components, data.individuals, EVAL_SPLIT))
    out.update(
        channel_interval_metrics(result.intervals, data.components, data.individuals, EVAL_SPLIT)
        if result.intervals is not None
        else {
            "z_d_coverage": np.nan,
            "z_p_coverage": np.nan,
            "z_d_mean_width": np.nan,
            "z_p_mean_width": np.nan,
        }
    )
    return out


def _aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in metrics.groupby(["role", "method", "evaluation_split"], sort=True):
        role, method, eval_split = keys
        for metric in METRIC_COLS:
            rows.append(
                {
                    "role": role,
                    "method": method,
                    "evaluation_split": eval_split,
                    "metric": metric,
                    **summarize_mcse(group[metric]),
                }
            )
    return pd.DataFrame(rows)


def _artifact_frames(result: MethodResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha_cols = ["id", "t"] + [col for col in result.predictions.columns if col.startswith("alpha_hat")]
    severity_cols = [
        "id",
        "t",
        "L_hat",
        "z_d_hat",
        "z_p_hat",
        "slow_hat",
        "delta_hat",
        "delta_d_hat",
        "delta_p_hat",
    ]
    severity = result.predictions[[col for col in severity_cols if col in result.predictions.columns]].merge(
        result.intervals if result.intervals is not None else result.predictions[["id", "t"]],
        on=["id", "t"],
        how="left",
    )
    alpha = result.predictions[alpha_cols].copy()
    return severity, alpha


def main() -> None:
    args = parse_args()
    run_id = make_run_id("ball_pipeline")
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[pd.DataFrame] = []
    config_records: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    started = perf_counter()
    structural_hyper = BallStructuralHyperparameters()

    for replicate in range(args.replicates):
        data_seed = args.base_seed + replicate
        data = generate_dataset(SimulationConfig(n=args.n, t=args.t, seed=data_seed, double_anchor_latents=True))
        diagnostics = diagnostics_frame(run_diagnostics(data))
        assert_diagnostics_pass(run_diagnostics(data))
        diagnostics.insert(0, "replicate", replicate)
        diagnostics.insert(1, "data_seed", data_seed)
        diagnostic_rows.append(diagnostics)

        if not args.skip_s0:
            s0_start = perf_counter()
            s0 = fit_s0_structural_prior_ssm(data)
            metric_rows.append(_metrics(replicate, data_seed, "s0", s0, data, perf_counter() - s0_start))

        if not args.skip_markov:
            markov_start = perf_counter()
            markov = fit_markov_pattern_mixture(data)
            metric_rows.append(
                _metrics(replicate, data_seed, "markov_pattern_mixture", markov, data, perf_counter() - markov_start)
            )

        structural_start = perf_counter()
        structural_raw = fit_ball_structural_posterior(data, structural_hyper)
        structural_teacher = conformal_calibrate_structural_posterior(
            structural_raw,
            data,
            alpha=args.conformal_alpha,
        )
        structural_elapsed = perf_counter() - structural_start
        # Spec requires three latent-coverage rows: raw posterior (what the model
        # believes), anchor-residual split conformal (deployable, appendix Sec. 5
        # form f_hat +/- q), and oracle latent-conformal (simulation-only
        # ceiling). Reporting all three exposes the latent-calibration gap rather
        # than a single conflated number.
        metric_rows.append(
            _metrics(replicate, data_seed, "structural_raw", structural_raw, data, structural_elapsed)
        )
        metric_rows.append(
            _metrics(
                replicate,
                data_seed,
                "structural_teacher",
                structural_teacher,
                data,
                structural_elapsed,
            )
        )
        structural_oracle = oracle_latent_conformal_intervals(
            structural_raw, data, alpha=args.conformal_alpha
        )
        metric_rows.append(
            _metrics(
                replicate, data_seed, "structural_oracle_latent", structural_oracle, data, structural_elapsed
            )
        )

        rep_start = perf_counter()
        ssm_config = SSMConfig(
            d_model=args.d_model,
            n_layers=args.n_layers,
            teacher_epochs=args.teacher_epochs,
            student_epochs=args.student_epochs,
            anchor_warmup_epochs=args.anchor_warmup_epochs,
            kl_warmup_epochs=args.kl_warmup_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            alpha_lasso_xi=args.alpha_lasso_xi,
            ensemble_size=args.members,
            max_individuals=args.n,
            seed=args.base_seed + 50_000 + replicate,
        )
        config_records.append(
            {
                "replicate": replicate,
                "data_seed": data_seed,
                "student_architecture": "transformer-causal",
                "primary_model": "BALL deep state-space teacher/student",
                "config": asdict(ssm_config),
            }
        )
        print(
            f"rep={replicate} data_seed={data_seed} training BALL SSM teacher+student ensemble K={args.members}",
            flush=True,
        )
        # The transformer teacher (bidirectional smoother) is the PRIMARY model;
        # the causal student is the deployable approximation distilled from it.
        # return_teacher evaluates the same teachers used for distillation, so the
        # teacher number costs only prediction, not extra training.
        student_raw, teacher_raw = fit_ball_ssm(
            data,
            ssm_config,
            device=args.device,
            causal=True,
            prediction_split=None,
            return_teacher=True,
        )
        for row in student_raw.metadata.get("loss_history", []):
            rec = dict(row)
            rec.update({"replicate": replicate, "data_seed": data_seed})
            history_rows.append(rec)
        ensemble_elapsed = perf_counter() - rep_start
        teacher_conformal = conformal_calibrate_structural_posterior(
            teacher_raw,
            data,
            alpha=args.conformal_alpha,
        )
        student_ensemble = conformal_calibrate_structural_posterior(
            student_raw,
            data,
            alpha=args.conformal_alpha,
        )
        metric_rows.append(_metrics(replicate, data_seed, "ssm_teacher_raw", teacher_raw, data, ensemble_elapsed))
        metric_rows.append(_metrics(replicate, data_seed, "ssm_teacher", teacher_conformal, data, ensemble_elapsed))
        metric_rows.append(_metrics(replicate, data_seed, "student_raw", student_raw, data, ensemble_elapsed))
        student_metric = _metrics(replicate, data_seed, "student_ensemble", student_ensemble, data, ensemble_elapsed)
        metric_rows.append(student_metric)

        if args.label_source == "ssm-teacher":
            label_result = teacher_conformal
        elif args.label_source == "structural-teacher":
            label_result = structural_teacher
        else:
            label_result = student_ensemble
        label_metric = _metrics(replicate, data_seed, "label_source", label_result, data, 0.0)
        severity, alpha = _artifact_frames(label_result)
        rho = pd.DataFrame(
            [
                {
                    "replicate": replicate,
                    "data_seed": data_seed,
                    "method": label_result.method,
                    "split": split,
                    "rho_rdoc": rdoc_recovery_fraction(label_result.predictions, data, split),
                    "label_source": args.label_source,
                }
                for split in ["train", "validation", "conformal", "test", None]
            ]
        )
        rho["split"] = rho["split"].fillna("all")

        structural_teacher.predictions.to_csv(
            run_dir / f"structural_teacher_predictions_rep_{replicate:03d}.csv.gz",
            index=False,
            compression="gzip",
        )
        structural_teacher.intervals.to_csv(
            run_dir / f"structural_teacher_intervals_rep_{replicate:03d}.csv.gz",
            index=False,
            compression="gzip",
        )
        student_raw.predictions.to_csv(
            run_dir / f"student_raw_predictions_rep_{replicate:03d}.csv.gz",
            index=False,
            compression="gzip",
        )
        student_raw.intervals.to_csv(
            run_dir / f"student_raw_intervals_rep_{replicate:03d}.csv.gz",
            index=False,
            compression="gzip",
        )
        student_ensemble.predictions.to_csv(
            run_dir / f"student_ensemble_predictions_rep_{replicate:03d}.csv.gz",
            index=False,
            compression="gzip",
        )
        student_ensemble.intervals.to_csv(
            run_dir / f"student_ensemble_intervals_rep_{replicate:03d}.csv.gz",
            index=False,
            compression="gzip",
        )
        severity.to_csv(run_dir / f"severity_intervals_rep_{replicate:03d}.csv.gz", index=False, compression="gzip")
        alpha.to_csv(run_dir / f"alpha_loadings_rep_{replicate:03d}.csv.gz", index=False, compression="gzip")
        rho.to_csv(run_dir / f"rho_rdoc_rep_{replicate:03d}.csv", index=False)

        print(
            f"rep={replicate} label={args.label_source} "
            f"label_latent={label_metric['latent_rmse']:.4f} "
            f"student_latent={student_metric['latent_rmse']:.4f} "
            f"z_d_cov={label_metric['z_d_coverage']:.4f} "
            f"z_p_cov={label_metric['z_p_coverage']:.4f} "
            f"rho_test={label_metric['rho_rdoc']:.4f}",
            flush=True,
        )

    metrics = pd.DataFrame(metric_rows)
    diagnostics_all = pd.concat(diagnostic_rows, ignore_index=True)
    history_all = pd.DataFrame(history_rows)
    aggregate = _aggregate(metrics)

    metrics.to_csv(run_dir / "pipeline_replicate_metrics.csv", index=False)
    diagnostics_all.to_csv(run_dir / "pipeline_diagnostics.csv", index=False)
    history_all.to_csv(run_dir / "pipeline_loss_history.csv", index=False)
    aggregate.to_csv(run_dir / "pipeline_aggregate_mcse.csv", index=False)
    (run_dir / "pipeline_torch_configs.json").write_text(json.dumps(config_records, indent=2) + "\n", encoding="utf-8")

    # Smoke isolation: only a manuscript-grade run may write the latest_* files
    # the manuscript build reads. Sub-grade (smoke/quick) runs publish under a
    # SMOKE prefix so they can never clobber the Table 1 / figure source.
    grade = _is_manuscript_grade(args)
    latest_prefix = "latest_ball_pipeline_" if grade else "latest_ball_pipeline_SMOKE_"
    latest_publish = {}
    for filename in [
        "pipeline_replicate_metrics.csv",
        "pipeline_diagnostics.csv",
        "pipeline_loss_history.csv",
        "pipeline_aggregate_mcse.csv",
        "pipeline_torch_configs.json",
    ]:
        latest_publish[filename] = publish_latest_artifact(
            run_dir / filename,
            OUTPUT_ROOT / f"{latest_prefix}{filename}",
        )

    manifest = {
        "run_id": run_id,
        "status": "passed",
        "scope": "BALL latent-recovery pipeline wrapper.",
        "elapsed_seconds": perf_counter() - started,
        "args": vars(args),
        "manuscript_grade": grade,
        "manuscript_grade_min": MANUSCRIPT_GRADE_MIN,
        "label_source": args.label_source,
        "student_architecture": "transformer-causal",
        "torch_resources": _torch_resource_summary(),
        "math_spec_artifacts": {
            "structural_teacher_predictions": "structural_teacher_predictions_rep_*.csv.gz",
            "structural_teacher_intervals": "structural_teacher_intervals_rep_*.csv.gz",
            "student_raw_predictions": "student_raw_predictions_rep_*.csv.gz",
            "student_raw_intervals": "student_raw_intervals_rep_*.csv.gz",
            "student_ensemble_predictions": "student_ensemble_predictions_rep_*.csv.gz",
            "student_ensemble_intervals": "student_ensemble_intervals_rep_*.csv.gz",
            "severity_intervals": "severity_intervals_rep_*.csv.gz",
            "alpha_loadings": "alpha_loadings_rep_*.csv.gz",
            "rho_rdoc": "rho_rdoc_rep_*.csv",
        },
        "student_uncertainty_stack": [
            "K-member BALL SSM causal transformer student ensemble",
            "diagonal last-layer Laplace approximation on the final mean head",
            "teacher Gaussian-KL distillation plus direct anchor re-anchoring",
            "final split-conformal calibration on held-out anchor residuals",
        ],
        "structural_hyperparameters": asdict(structural_hyper),
        "guardrails": [
            "BALL's primary neural model is the transformer teacher/student system.",
            "The structural posterior is an interpretable statistical comparator.",
            "The default severity, alpha, and rho artifacts use the configured label_source.",
            "The transformer student is trained from the bidirectional BALL SSM teacher, not from the structural comparator.",
            "S0 and any Markov/pattern-mixture model are comparator-only.",
            "This project is restricted to anchored latent learning.",
        ],
        "latest_publish": latest_publish,
        "run_directory": str(run_dir),
        "latest_output_directory": str(OUTPUT_ROOT),
    }
    manifest_json = json.dumps(manifest, indent=2) + "\n"
    (run_dir / "ball_pipeline_manifest.json").write_text(manifest_json, encoding="utf-8")
    latest_publish["ball_pipeline_manifest.json"] = publish_latest_text(
        manifest_json,
        OUTPUT_ROOT / f"{latest_prefix}manifest.json",
    )

    key_metrics = aggregate[
        aggregate["metric"].isin(
            [
                "latent_rmse",
                "z_d_rmse",
                "z_p_rmse",
                "coverage",
                "z_d_coverage",
                "z_p_coverage",
                "mean_width",
                "rho_rdoc",
            ]
        )
    ].sort_values(["metric", "role"])
    md = [
        "# BALL Latent-Recovery Pipeline Report",
        "",
        f"Run ID: `{run_id}`",
        "",
        "## Scope",
        "",
        f"Latent-recovery pipeline: BALL SSM transformer teacher/student ensemble, structural posterior audit, final split-conformal calibration, severity intervals, alpha loadings, and rho_RDoC. Artifact label source: `{args.label_source}`.",
        "",
        "## Key Metrics",
        "",
        "```text",
        key_metrics[["role", "method", "metric", "mean", "mcse", "sd", "n", "min", "max"]].to_string(index=False),
        "```",
        "",
        "## Artifacts",
        "",
        f"- Run directory: `{run_dir}`",
        "- Structural comparator: `structural_teacher_predictions_rep_*.csv.gz`, `structural_teacher_intervals_rep_*.csv.gz`",
        "- BALL student raw ensemble: `student_raw_predictions_rep_*.csv.gz`, `student_raw_intervals_rep_*.csv.gz`",
        "- BALL student conformal ensemble: `student_ensemble_predictions_rep_*.csv.gz`, `student_ensemble_intervals_rep_*.csv.gz`",
        "- Severity intervals: `severity_intervals_rep_*.csv.gz`",
        "- Alpha loadings: `alpha_loadings_rep_*.csv.gz`",
        "- RDoC recovery fraction: `rho_rdoc_rep_*.csv`",
        "",
    ]
    md_text = "\n".join(md)
    (run_dir / "ball_pipeline_agent_report.md").write_text(md_text, encoding="utf-8")
    latest_publish["ball_pipeline_agent_report.md"] = publish_latest_text(
        md_text,
        OUTPUT_ROOT / f"{latest_prefix}agent_report.md",
    )

    print(key_metrics[["role", "method", "metric", "mean", "mcse", "sd", "n", "min", "max"]].to_string(index=False))
    print(f"BALL full pipeline complete: {run_dir}")


if __name__ == "__main__":
    main()
'''

# --- simulations.run_battle_tests (simulations/run_battle_tests.py) ---
SRC_SIMULATIONS_RUN_BATTLE_TESTS = r'''
"""Repeatable scaffold hardening checks for the simulation code.

This is not a replacement for the Scenario B pilot. It is a fast validation
gate that checks the generated data, diagnostics, method result schemas, seed
determinism, and the first-class r_ts knob before heavier model work begins.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from simulations.src.dgp import generate_dataset
from simulations.src.diagnostics import assert_diagnostics_pass, diagnostics_frame, run_diagnostics
from simulations.src.metrics import component_rmse, coverage, latent_rmse, mean_width
from simulations.src.methods.ball_structural import (
    conformal_calibrate_structural_posterior,
    fit_ball_structural_posterior,
)
from simulations.src.methods.ball_ssm import (
    InferenceTransformer,
    SSMConfig,
    build_ssm_batch,
    _impute as _ssm_impute,
)
from simulations.src.methods.baselines import linear_interpolation, locf, anchor_only_windowed_mean
from simulations.src.methods.s0 import fit_s0_structural_prior_ssm
from simulations.src.model_utils import MethodResult, SimulationConfig, load_config, make_run_id


OUTPUT_ROOT = ROOT / "outputs" / "diagnostics"
RUN_ROOT = ROOT / "runs"


def _frame_fingerprint(df: pd.DataFrame, columns: list[str]) -> int:
    stable = df[columns].copy()
    return int(pd.util.hash_pandas_object(stable, index=False).sum())


def _check_method_result(result: MethodResult, data_rows: int) -> dict[str, float | str]:
    pred = result.predictions
    if len(pred) != data_rows:
        raise AssertionError(f"{result.method} predictions have {len(pred)} rows, expected {data_rows}.")
    if pred.duplicated(["id", "t"]).any():
        raise AssertionError(f"{result.method} predictions contain duplicate id,t keys.")
    if not np.isfinite(pred["L_hat"].to_numpy(dtype=float)).all():
        raise AssertionError(f"{result.method} predictions contain non-finite values.")
    out: dict[str, float | str] = {"method": result.method}
    if result.intervals is not None:
        intervals = result.intervals
        if len(intervals) != data_rows:
            raise AssertionError(f"{result.method} intervals have {len(intervals)} rows, expected {data_rows}.")
        if (intervals["lower"] > intervals["upper"]).any():
            raise AssertionError(f"{result.method} intervals contain lower > upper.")
    return out


def _run_case(name: str, config: SimulationConfig, run_methods: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = generate_dataset(config)
    diagnostics = run_diagnostics(data)
    diag_df = diagnostics_frame(diagnostics)
    diag_df.insert(0, "case", name)
    diag_df.insert(1, "seed", config.seed)
    diag_df.insert(2, "n", config.n)
    diag_df.insert(3, "t", config.t)
    try:
        assert_diagnostics_pass(diagnostics)
    except AssertionError:
        print(f"\n{name} diagnostics:")
        print(diag_df.to_string(index=False))
        raise

    rows = [
        {
            "case": name,
            "method": "DGP",
            "rmse": np.nan,
            "coverage": np.nan,
            "mean_width": np.nan,
            "n": config.n,
            "t": config.t,
            "seed": config.seed,
        }
    ]
    if run_methods:
        methods = [
            linear_interpolation(data),
            locf(data),
            anchor_only_windowed_mean(data),
            fit_s0_structural_prior_ssm(data),
        ]
        for result in methods:
            _check_method_result(result, len(data.components))
            row = {
                "case": name,
                "method": result.method,
                "rmse": latent_rmse(result.predictions, data.components),
                "coverage": np.nan,
                "mean_width": np.nan,
                "n": config.n,
                "t": config.t,
                "seed": config.seed,
            }
            if result.intervals is not None:
                row["coverage"] = coverage(result.intervals, data.components)
                row["mean_width"] = mean_width(result.intervals)
            rows.append(row)
    return pd.DataFrame(rows), diag_df


def _check_determinism() -> None:
    config = SimulationConfig(n=60, t=35, seed=991)
    first = generate_dataset(config)
    second = generate_dataset(config)
    checks = [
        (
            "components",
            _frame_fingerprint(first.components, ["id", "t", "L", "slow", "delta"]),
            _frame_fingerprint(second.components, ["id", "t", "L", "slow", "delta"]),
        ),
        (
            "anchors",
            _frame_fingerprint(first.anchors.fillna(-9999.0), ["id", "anchor", "t", "value", "observed"]),
            _frame_fingerprint(second.anchors.fillna(-9999.0), ["id", "anchor", "t", "value", "observed"]),
        ),
        (
            "daily",
            _frame_fingerprint(first.daily.fillna(-9999.0), ["id", "t", "X0", "X8", "X12", "obs_X0"]),
            _frame_fingerprint(second.daily.fillna(-9999.0), ["id", "t", "X0", "X8", "X12", "obs_X0"]),
        ),
    ]
    for name, a, b in checks:
        if a != b:
            raise AssertionError(f"Seed determinism failed for {name}: {a} != {b}.")


def _check_config_load() -> None:
    config = load_config(ROOT / "config" / "default.yml")
    if config.n != 1000 or config.t != 84 or config.q != 6:
        raise AssertionError("default.yml did not load the expected core dimensions.")
    if config.y1.error_sd != 0.95:
        raise AssertionError("default.yml did not load the calibrated Y1 anchor noise.")


def _alpha_step_variance(data) -> float:
    """Scale-invariant loading drift: variance of CONTINUING-ACTIVE alpha steps
    divided by the variance of active alpha levels.

    Two DGP behaviors make the naive absolute step variance useless as a knob
    check. First, the per-person slow-variance normalization rescales alpha so
    that absolute step variance is nearly invariant to the innovation knobs.
    Second, support toggling (inactive -> fresh loading) creates large jumps
    governed by support_switch_prob, not by loading drift, and those jumps
    dominate the naive measure. Restricting to steps where the dimension is
    active on both days and normalizing by the active level variance restores a
    measure that is monotone in the loading-innovation scale.
    """

    step_values: list[float] = []
    level_values: list[float] = []
    sorted_components = data.components.sort_values(["id", "t"])
    grouped = sorted_components.groupby("id")
    for d in range(data.config.q):
        alpha_col = f"alpha{d}"
        active_col = f"active{d}"
        diffs = grouped[alpha_col].diff()
        active = sorted_components[active_col].astype(bool)
        prev_active = grouped[active_col].shift(1, fill_value=False).astype(bool)
        continuing = active & prev_active
        selected = diffs[continuing].dropna()
        if len(selected):
            step_values.extend(selected.to_list())
        level_values.extend(sorted_components.loc[active, alpha_col].to_list())
    if not step_values or not level_values:
        return float("nan")
    return float(np.var(step_values) / max(np.var(level_values), 1e-12))


def _check_r_ts_monotonicity() -> pd.DataFrame:
    rows = []
    for r_ts in [0.003, 0.03, 0.30]:
        config = SimulationConfig(n=180, t=84, seed=4421, r_ts=r_ts)
        data = generate_dataset(config)
        assert_diagnostics_pass(run_diagnostics(data))
        rows.append(
            {
                "r_ts": r_ts,
                "loading_drift_sd": data.metadata["loading_drift_sd"],
                "alpha_innovation_sd": data.metadata["alpha_innovation_sd"],
                "alpha_step_variance": _alpha_step_variance(data),
            }
        )
    out = pd.DataFrame(rows)
    diffs = np.diff(out["alpha_step_variance"].to_numpy())
    if not (diffs > 0).all():
        raise AssertionError(f"r_ts sweep is not monotone in alpha step variance:\n{out}")
    return out


def _check_loading_drift_sd_knob() -> pd.DataFrame:
    rows = []
    for loading_drift_sd in [0.005, 0.02, 0.08]:
        config = SimulationConfig(n=120, t=56, seed=4422, loading_drift_sd=loading_drift_sd)
        data = generate_dataset(config)
        rows.append(
            {
                "loading_drift_sd": loading_drift_sd,
                "alpha_innovation_sd": data.metadata["alpha_innovation_sd"],
                "alpha_step_variance": _alpha_step_variance(data),
            }
        )
    out = pd.DataFrame(rows)
    diffs = np.diff(out["alpha_step_variance"].to_numpy())
    if not (diffs > 0).all():
        raise AssertionError(f"loading_drift_sd sweep is not monotone in alpha step variance:\n{out}")
    return out


def _assert_joint_decomposition(result: MethodResult, label: str) -> None:
    pred = result.predictions
    # Spec decomposition has no intercept: L = slow + delta (level carried by delta).
    required = {"L_hat", "slow_hat", "delta_hat"}
    missing = required.difference(pred.columns)
    if missing:
        raise AssertionError(f"{label} is missing decomposition columns: {sorted(missing)}")
    reconstructed = pred["slow_hat"] + pred["delta_hat"]
    max_abs = float(np.max(np.abs(pred["L_hat"] - reconstructed)))
    if max_abs > 1e-8:
        raise AssertionError(f"{label} joint decomposition is inconsistent: max abs {max_abs:.6g}")
    if "proxy_slow_hat" not in pred.columns:
        raise AssertionError(f"{label} no longer exposes proxy_slow_hat as a diagnostic column.")


def _check_slow_decomposition_consistency() -> None:
    config = SimulationConfig(n=30, t=28, seed=18410, double_anchor_latents=True)
    data = generate_dataset(config)
    s0 = fit_s0_structural_prior_ssm(data)
    structural = fit_ball_structural_posterior(data)
    _assert_joint_decomposition(s0, "S0")
    _assert_joint_decomposition(structural, "BALL structural posterior")
    pred = structural.predictions
    required = {"z_d_hat", "z_p_hat", "slow_d_hat", "slow_p_hat", "delta_d_hat", "delta_p_hat"}
    missing = required.difference(pred.columns)
    if missing:
        raise AssertionError(f"Structural posterior missing channel decomposition columns: {sorted(missing)}")
    zd_reconstructed = pred["slow_d_hat"] + pred["delta_d_hat"]
    zp_reconstructed = pred["slow_p_hat"] + pred["delta_p_hat"]
    max_zd = float(np.max(np.abs(pred["z_d_hat"] - zd_reconstructed)))
    max_zp = float(np.max(np.abs(pred["z_p_hat"] - zp_reconstructed)))
    if max(max_zd, max_zp) > 1e-8:
        raise AssertionError(
            "Structural posterior channel decomposition is inconsistent: "
            f"max z_d {max_zd:.6g}, max z_p {max_zp:.6g}"
        )


def _check_causal_anchor_token_no_future_leak() -> None:
    data_config = SimulationConfig(n=30, t=28, seed=6060, double_anchor_latents=True)
    data = generate_dataset(data_config)
    torch_config = SSMConfig(
        d_model=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
        max_individuals=12,
        seed=6060,
    )
    batch, _ = build_ssm_batch(
        data,
        split="train",
        device=torch.device("cpu"),
        max_individuals=torch_config.max_individuals,
        causal_impute=True,
    )
    torch.manual_seed(torch_config.seed)
    action_embed = torch.nn.Embedding(data.config.n_treatment_types + 1, torch_config.action_embed_dim)
    model = InferenceTransformer(
        batch.enc_features.shape[-1],
        data.config.q,
        len(batch.t_values),
        torch_config,
        action_embed,
        causal=True,
    )
    model.eval()

    future_day = len(batch.t_values) - 1
    modified_features = batch.enc_features.clone()
    modified_features[:, future_day, :] += 25.0
    modified = replace(batch, enc_features=modified_features)

    with torch.no_grad():
        original_mu, original_logvar = model(batch)
        modified_mu, modified_logvar = model(modified)

    prefix = slice(0, future_day)
    for key, before_tensor, after_tensor in [
        ("mu", original_mu, modified_mu),
        ("logvar", original_logvar, modified_logvar),
    ]:
        before = before_tensor[:, prefix]
        after = after_tensor[:, prefix]
        max_abs = float(torch.max(torch.abs(before - after)).detach().cpu())
        if max_abs > 1e-5:
            raise AssertionError(f"Causal SSM student leaked future token into {key}: max abs diff {max_abs:.6g}.")


def _check_causal_proxy_imputation_no_future_fill() -> None:
    frame = pd.DataFrame(
        {
            "B0": [np.nan, np.nan, 5.0],
            "B1": [1.0, np.nan, 3.0],
        }
    )
    full_record = _ssm_impute(frame, ["B0", "B1"], causal=False)
    causal = _ssm_impute(frame, ["B0", "B1"], causal=True)

    if full_record[0, 0] != 5.0:
        raise AssertionError("Full-record SSM imputation no longer backfills as expected.")
    if causal[0, 0] != 0.0 or causal[1, 0] != 0.0:
        raise AssertionError("Causal SSM imputation used a future value for an initial missing proxy.")
    if causal[1, 1] != 1.0:
        raise AssertionError("Causal SSM imputation failed to carry forward an observed past proxy.")


def _check_structural_posterior_beats_s0() -> pd.DataFrame:
    """Protect the appendix-aligned severity/calibration smoke result."""

    config = SimulationConfig(n=120, t=42, seed=18400, double_anchor_latents=True)
    data = generate_dataset(config)
    assert_diagnostics_pass(run_diagnostics(data))

    s0 = fit_s0_structural_prior_ssm(data)
    raw = fit_ball_structural_posterior(data)
    ball = conformal_calibrate_structural_posterior(raw, data, alpha=0.05)
    _check_method_result(s0, len(data.components))
    _check_method_result(ball, len(data.components))

    split = "test"
    rows = []
    for role, result in [
        ("s0", s0),
        ("ball_structural_raw", raw),
        ("ball_structural_conformal", ball),
    ]:
        rows.append(
            {
                "role": role,
                "method": result.method,
                "seed": config.seed,
                "n": config.n,
                "t": config.t,
                "evaluation_split": split,
                "latent_rmse": latent_rmse(result.predictions, data.components, data.individuals, split),
                "slow_rmse": component_rmse(
                    result.predictions, data.components, "slow_hat", "slow", data.individuals, split
                ),
                "delta_rmse": component_rmse(
                    result.predictions, data.components, "delta_hat", "delta", data.individuals, split
                ),
                "coverage": coverage(result.intervals, data.components, data.individuals, split),
                "mean_width": mean_width(result.intervals, data.individuals, split),
                "conformal_q": result.metadata.get("conformal_q"),
            }
        )
    gate = pd.DataFrame(rows)
    pivot = gate.set_index("role")
    s0_row = pivot.loc["s0"]
    raw_row = pivot.loc["ball_structural_raw"]
    ball_row = pivot.loc["ball_structural_conformal"]

    # Like-for-like arms: S0's intervals are raw Gaussian, so the coverage and
    # width comparisons gate the RAW marginal-posterior arm. The anchor-conformal
    # arm is the deployable guarantee; it overcovers on the latent BY DESIGN
    # (its conformity scores include anchor measurement noise; see the
    # masterclass conformal pitfall box), so it gets sanity checks rather than
    # an S0 width comparison.
    failures = []
    if not ball_row["latent_rmse"] < s0_row["latent_rmse"] - 0.02:
        failures.append("latent RMSE did not improve over S0 by at least 0.02")
    if not np.isfinite(ball_row["slow_rmse"]):
        failures.append("corrected slow RMSE is not finite")
    if not ball_row["slow_rmse"] <= s0_row["slow_rmse"] * 1.10:
        failures.append("corrected slow RMSE was more than 10% worse than S0")
    if not ball_row["delta_rmse"] < s0_row["delta_rmse"] - 0.005:
        failures.append("delta RMSE did not improve over S0 by at least 0.005")
    if not 0.90 <= raw_row["coverage"] <= 0.99:
        failures.append("raw posterior coverage was outside the smoke acceptance band [0.90, 0.99]")
    if not abs(raw_row["coverage"] - 0.95) < abs(s0_row["coverage"] - 0.95):
        failures.append("raw posterior coverage was not closer to 0.95 than S0 coverage")
    if not raw_row["mean_width"] <= s0_row["mean_width"] * 1.05:
        failures.append("raw posterior intervals were more than 5% wider than S0 intervals")
    if not (np.isfinite(ball_row["conformal_q"]) and ball_row["conformal_q"] > 0):
        failures.append("anchor-conformal quantile was not a positive finite value")
    if not ball_row["coverage"] >= raw_row["coverage"] - 0.01:
        failures.append("anchor-conformal latent coverage fell below raw posterior coverage")
    if not ball_row["mean_width"] <= 6.0 * raw_row["mean_width"]:
        failures.append(
            "anchor-conformal intervals were more than 6x the raw posterior width "
            "(expected roughly 2x-5x from anchor noise; a larger blowup indicates a "
            "scale bug like the 2026-05-31 sqrt-window propagation regression)"
        )
    if failures:
        raise AssertionError("BALL structural posterior gate failed:\n" + "\n".join(failures) + f"\n\n{gate}")
    return gate


def _check_irt_anchor_mode() -> None:
    """Gate the calibrated graded-response IRT anchor path at the DGP level.

    The fixed calibrated instrument must emit per-item responses on the calibrated
    trait scale, leave unobserved anchors missing, stay de-saturated, and track the
    windowed latent. The neural exact-GRM properties (fixed buffers, anchor batching,
    exact likelihood, teacher/student execution, comparator symmetry) are covered by
    validation/test_irt_calibration.py.
    """

    n_items, kcut = 9, 3
    thresholds = tuple(tuple(float(x) for x in np.linspace(-2.0, 2.0, kcut)) for _ in range(n_items))
    disc = tuple([1.0] * n_items)
    config = SimulationConfig(
        n=200, t=42, seed=18450, anchor_observation="irt",
        irt_n_items=n_items, irt_n_categories=kcut + 1, irt_loc=0.0, irt_scale=1.0,
        irt_item_thresholds=thresholds, irt_item_discriminations=disc,
    )
    data = generate_dataset(config)
    assert_diagnostics_pass(run_diagnostics(data))
    anc = data.anchors
    obs = anc[anc["observed"]]
    if not obs["irt_items"].apply(lambda x: x is not None and len(x) == n_items).all():
        raise AssertionError("IRT anchors must carry per-item graded responses")
    unobs = anc[~anc["observed"]]
    if len(unobs) and not unobs["irt_total"].isna().all():
        raise AssertionError("unobserved IRT totals and items must be missing")
    tot = obs["irt_total"].to_numpy(dtype=float)
    maxt = n_items * kcut
    saturation = float(np.mean((tot <= 0.5) | (tot >= maxt - 0.5)))
    if saturation > 0.25:
        raise AssertionError(f"IRT instrument too saturated: floor+ceiling {saturation:.2f}")
    corr = float(obs["irt_total"].corr(obs["window_mean_L"]))
    if not (np.isfinite(corr) and corr >= 0.5):
        raise AssertionError(f"IRT total/window correlation too low: {corr:.3f}")


def _check_assignment_confounding() -> pd.DataFrame:
    """Gate that the treatment-assignment knob induces severity dependence.

    With gamma = 0 (random) treatment timing is independent of the latent; with
    strongly_adaptive it must depend on the within-person severity state. This
    protects the confounding-by-indication scenario from silently regressing to
    uniform assignment.
    """

    rows = []
    for assignment in ["random", "strongly_adaptive"]:
        config = SimulationConfig(n=200, t=84, seed=18460, assignment=assignment)
        data = generate_dataset(config)
        assert_diagnostics_pass(run_diagnostics(data))
        comp = data.components.sort_values(["id", "t"]).copy()
        comp["treated"] = (comp["a"] >= 0).astype(float)
        comp["L_prev"] = comp.groupby("id")["L"].shift(1)
        sub = comp.dropna(subset=["L_prev"])
        l_centered = sub["L_prev"] - sub.groupby("id")["L_prev"].transform("mean")
        t_centered = sub["treated"] - sub.groupby("id")["treated"].transform("mean")
        corr = float(np.corrcoef(l_centered, t_centered)[0, 1])
        counts = data.treatments.groupby("id").size()
        rows.append({"assignment": assignment, "timing_corr": corr,
                     "count_min": int(counts.min()), "count_max": int(counts.max())})
    gate = pd.DataFrame(rows)
    random_corr = float(gate.loc[gate["assignment"] == "random", "timing_corr"].iloc[0])
    adaptive_corr = float(gate.loc[gate["assignment"] == "strongly_adaptive", "timing_corr"].iloc[0])
    failures = []
    if not abs(random_corr) < 0.03:
        failures.append(f"random assignment timing correlation {random_corr:.4f} not near zero")
    if not adaptive_corr > random_corr + 0.02:
        failures.append(
            f"strongly_adaptive timing correlation {adaptive_corr:.4f} did not exceed random {random_corr:.4f}"
        )
    if failures:
        raise AssertionError("Assignment confounding gate failed:\n" + "\n".join(failures) + f"\n\n{gate}")
    return gate


def _write_outputs(
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    r_ts: pd.DataFrame,
    structural_gate: pd.DataFrame,
    run_id: str,
) -> dict[str, str]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = OUTPUT_ROOT / "latest_battle_test_summary.csv"
    diagnostics_path = OUTPUT_ROOT / "latest_battle_test_diagnostics.csv"
    r_ts_path = OUTPUT_ROOT / "latest_r_ts_monotonicity.csv"
    structural_gate_path = OUTPUT_ROOT / "latest_structural_posterior_gate.csv"
    report_json_path = OUTPUT_ROOT / "latest_agent_report.json"
    report_md_path = OUTPUT_ROOT / "latest_agent_report.md"

    timestamped_summary_path = run_dir / "battle_test_summary.csv"
    timestamped_diagnostics_path = run_dir / "battle_test_diagnostics.csv"
    timestamped_r_ts_path = run_dir / "r_ts_monotonicity.csv"
    timestamped_structural_gate_path = run_dir / "structural_posterior_gate.csv"
    timestamped_report_json_path = run_dir / "agent_report.json"
    timestamped_report_md_path = run_dir / "agent_report.md"

    for path in [summary_path, timestamped_summary_path]:
        summary.to_csv(path, index=False)
    for path in [diagnostics_path, timestamped_diagnostics_path]:
        diagnostics.to_csv(path, index=False)
    for path in [r_ts_path, timestamped_r_ts_path]:
        r_ts.to_csv(path, index=False)
    for path in [structural_gate_path, timestamped_structural_gate_path]:
        structural_gate.to_csv(path, index=False)

    s0_rows = summary[summary["method"] == "S0_structural_prior_lgssm"]
    report = {
        "run_id": run_id,
        "status": "passed",
        "artifacts": {
            "summary_csv": str(summary_path),
            "diagnostics_csv": str(diagnostics_path),
            "r_ts_csv": str(r_ts_path),
            "structural_posterior_gate_csv": str(structural_gate_path),
            "agent_report_json": str(report_json_path),
            "agent_report_md": str(report_md_path),
            "run_directory": str(run_dir),
        },
        "model_status": {
            "s0": "prototype_structural_prior_lgssm",
            "s0_is_scaffold_placeholder": False,
            "s0_is_ball": False,
            "s0_parameterization": "static sparse per-person alpha with structured L and delta dynamics",
            "intervals": "raw diagonal-precision Gaussian intervals, not conformal",
            "next_step": "Use this S0 as the prototype comparator, then add Scenario B pilot orchestration and later BALL teacher/student training.",
            "structural_posterior_gate": "BALL appendix-aligned two-channel structural posterior must beat S0 on test latent, slow, and delta RMSE. Coverage and width gate the RAW marginal-posterior arm against S0's raw Gaussian intervals (like-for-like). The anchor-conformal arm overcovers the latent by design and gets sanity checks only (positive finite q, width at most 6x raw).",
        },
        "s0_metrics": s0_rows.to_dict(orient="records"),
        "structural_posterior_gate": structural_gate.to_dict(orient="records"),
        "known_caveats": [
            "S0 is a real structural-prior smoother but not the full BALL teacher/student model.",
            "The structural posterior gate validates the appendix-aligned severity/calibration target; it is a direct smoother, not the amortized neural student.",
            "S0 uses static sparse alpha per person for speed; dynamic alpha(t) remains a later sensitivity.",
            "S0 intervals are raw diagonal-precision Gaussian intervals and are not calibrated uncertainty claims.",
            "Battle tests validate code plumbing and DGP behavior, not manuscript-ready Monte Carlo evidence.",
        ],
    }

    json_text = json.dumps(report, indent=2)
    for path in [report_json_path, timestamped_report_json_path]:
        path.write_text(json_text + "\n", encoding="utf-8")

    md = [
        "# BALL Simulation Battle-Test Report",
        "",
        f"Run ID: `{run_id}`",
        "",
        "## Status",
        "",
        "Passed scaffold battle tests.",
        "",
        "## Important Caveats",
        "",
    ]
    md.extend(f"- {item}" for item in report["known_caveats"])
    md.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Summary CSV: `{summary_path}`",
            f"- Diagnostics CSV: `{diagnostics_path}`",
            f"- r_ts CSV: `{r_ts_path}`",
            f"- Structural posterior gate CSV: `{structural_gate_path}`",
            f"- Timestamped run directory: `{run_dir}`",
            "",
            "## S0 Rows",
            "",
            "```text",
            s0_rows.to_string(index=False),
            "```",
            "",
            "## BALL Structural Posterior Gate",
            "",
            "```text",
            structural_gate.to_string(index=False),
            "```",
            "",
        ]
    )
    md_text = "\n".join(md)
    for path in [report_md_path, timestamped_report_md_path]:
        path.write_text(md_text, encoding="utf-8")

    return {
        "summary": str(summary_path),
        "diagnostics": str(diagnostics_path),
        "r_ts": str(r_ts_path),
        "structural_gate": str(structural_gate_path),
        "report_json": str(report_json_path),
        "report_md": str(report_md_path),
        "run_dir": str(run_dir),
    }


def main() -> None:
    run_id = make_run_id("battle_test")
    _check_config_load()
    _check_determinism()
    _check_causal_anchor_token_no_future_leak()
    _check_causal_proxy_imputation_no_future_fill()
    _check_slow_decomposition_consistency()
    loading_drift = _check_loading_drift_sd_knob()
    structural_gate = _check_structural_posterior_beats_s0()
    _check_irt_anchor_mode()
    assignment_gate = _check_assignment_confounding()

    cases = [
        ("smoke", SimulationConfig(n=50, t=42, seed=1729), True),
        ("default_size", SimulationConfig(n=1000, t=84, seed=1729), True),
        ("seed_2027", SimulationConfig(n=250, t=84, seed=2027), True),
        ("seed_2028_mnar", SimulationConfig(n=250, t=84, seed=2028, missingness_mechanism="mnar", mnar_gamma_l=0.5), True),
        ("random_proxy_timing", SimulationConfig(n=250, t=84, seed=2029, endogenous_proxy_observation=False), True),
    ]
    results = [_run_case(name, config, run_methods) for name, config, run_methods in cases]
    summary = pd.concat([result[0] for result in results], ignore_index=True)
    diagnostics = pd.concat([result[1] for result in results], ignore_index=True)
    r_ts = _check_r_ts_monotonicity()
    print("\nloading_drift_sd monotonicity check:")
    print(loading_drift.to_string(index=False))
    artifact_paths = _write_outputs(summary, diagnostics, r_ts, structural_gate, run_id)

    print("Battle-test method summary:")
    print(summary.to_string(index=False))
    print("\nBALL structural posterior gate:")
    print(structural_gate.to_string(index=False))
    print("\nassignment confounding gate:")
    print(assignment_gate.to_string(index=False))
    print("\nr_ts monotonicity check:")
    print(r_ts.to_string(index=False))
    print("\nAgent-review artifacts:")
    for name, path in artifact_paths.items():
        print(f"{name}: {path}")
    print("\nImportant caveat:")
    print("S0_structural_prior_lgssm is a real prototype structural-prior smoother, not BALL.")
    print("It uses static sparse per-person alpha and raw diagonal-precision intervals.")


if __name__ == "__main__":
    main()
'''

# --- simulations.run_pilot_batch (simulations/run_pilot_batch.py) ---
SRC_SIMULATIONS_RUN_PILOT_BATCH = r'''
"""Batch runner for the canonical BALL latent-recovery pilot.

Runs the replicates as a small pool of concurrent single-replicate pipeline
subprocesses (replicate-level parallelism: each replicate is independent, so
this is embarrassingly parallel and avoids fine-grained per-patient overhead),
then concatenates their per-replicate metrics and recomputes the aggregate Monte
Carlo standard errors.

Intended to run outside the interactive window. Progress is printed per
replicate. A ``DONE`` marker is written only when every requested replicate
finishes successfully; partial runs write ``PARTIAL`` and exit nonzero unless
``--allow-partial`` is supplied.

Example:
    python simulations/run_pilot_batch.py --replicates 10 --concurrency 3 \
        --members 5 --n 1000 --t 84 --student-epochs 120
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))

from simulations.run_ball_pipeline import _aggregate  # noqa: E402

RUN_DIR_RE = re.compile(r"pipeline complete:\s*(.+?)\s*$")


def _run_one_replicate(seed: int, args: argparse.Namespace) -> tuple[int, "pd.DataFrame | None", str]:
    """Run a single-replicate pipeline subprocess and return its metrics frame."""

    cmd = [
        sys.executable,
        str(REPO / "BALL.py"),
        "pipeline",
        "--replicates", "1",
        "--members", str(args.members),
        "--n", str(args.n),
        "--t", str(args.t),
        "--student-epochs", str(args.student_epochs),
        "--base-seed", str(seed),
    ]
    if args.skip_s0:
        cmd.append("--skip-s0")
    # Cap each worker's native (BLAS/OpenMP) thread pools so that
    # concurrency * threads stays near the core count; without this, many
    # concurrent workers each spawn one thread per core and thrash, turning a
    # ~2 min structural solve into a >30 min stall.
    env = dict(os.environ)
    env.update(
        {
            "OMP_NUM_THREADS": str(args.threads_per_worker),
            "MKL_NUM_THREADS": str(args.threads_per_worker),
            "OPENBLAS_NUM_THREADS": str(args.threads_per_worker),
            "NUMEXPR_NUM_THREADS": str(args.threads_per_worker),
        }
    )
    try:
        proc = subprocess.run(
            cmd, cwd=str(REPO), capture_output=True, text=True, timeout=args.per_timeout, env=env
        )
    except subprocess.TimeoutExpired:
        return seed, None, f"timeout after {args.per_timeout}s"
    if proc.returncode != 0:
        tail = (proc.stdout[-400:] + " | " + proc.stderr[-400:]).replace("\n", " ")
        return seed, None, f"exit {proc.returncode}: {tail}"
    run_dir = None
    for line in proc.stdout.splitlines():
        match = RUN_DIR_RE.search(line)
        if match:
            run_dir = match.group(1).strip()
    if run_dir is None:
        return seed, None, "no run directory found in stdout"
    csv_path = Path(run_dir) / "pipeline_replicate_metrics.csv"
    if not csv_path.exists():
        return seed, None, f"metrics csv missing at {run_dir}"
    frame = pd.read_csv(csv_path)
    frame["batch_seed"] = seed
    return seed, frame, "ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replicate-parallel BALL pilot batch.")
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=3, help="Concurrent single-replicate workers.")
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--t", type=int, default=84)
    parser.add_argument("--student-epochs", type=int, default=150)
    parser.add_argument("--base-seed", type=int, default=21000)
    parser.add_argument("--per-timeout", type=int, default=5400, help="Per-replicate wall-clock cap (s).")
    parser.add_argument("--threads-per-worker", type=int, default=3, help="BLAS/OpenMP threads per worker (concurrency*threads should be near core count).")
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "runs" / "pilot_batch"))
    parser.add_argument("--skip-s0", action="store_true",
                        help="Skip the S0 comparator inside each replicate. Do not use for manuscript Table 1 runs.")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Exit zero and write DONE even if some requested replicates fail.")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Explicit comma-separated seed list (overrides --replicates/--base-seed). "
                             "Use to backfill specific failed replicates.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    (out / "STARTED").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    # Distinct, well-separated seeds so the replicates are independent draws.
    # An explicit --seeds list overrides the generated range (used for backfill).
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        args.replicates = len(seeds)
    else:
        seeds = [args.base_seed + 1000 * r for r in range(args.replicates)]
    print(f"[batch] launching {len(seeds)} replicates, concurrency={args.concurrency}, "
          f"N={args.n}, T={args.t}, members={args.members}, epochs={args.student_epochs}, "
          "student=transformer-causal", flush=True)

    frames: list[pd.DataFrame] = []
    status: list[tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(_run_one_replicate, seed, args): seed for seed in seeds}
        for future in as_completed(futures):
            seed, frame, message = future.result()
            status.append((seed, message))
            print(f"[batch] seed={seed} -> {message}", flush=True)
            if frame is not None:
                frames.append(frame)

    if frames:
        replicate_metrics = pd.concat(frames, ignore_index=True)
        replicate_metrics.to_csv(out / "pilot_batch_replicate_metrics.csv", index=False)
        aggregate = _aggregate(replicate_metrics)
        aggregate.to_csv(out / "pilot_batch_aggregate_mcse.csv", index=False)
        key = aggregate[
            aggregate["metric"].isin(
                ["latent_rmse", "z_d_rmse", "z_p_rmse", "coverage", "z_d_coverage",
                 "z_p_coverage", "mean_width", "active_f1", "rho_rdoc"]
            )
            & aggregate["role"].isin(["structural_teacher", "student_ensemble"])
        ].sort_values(["metric", "role"])
        print(key[["role", "metric", "mean", "mcse", "sd", "n"]].to_string(index=False), flush=True)

    n_ok = len(frames)
    summary = "\n".join(f"seed={seed}: {message}" for seed, message in sorted(status))
    summary += f"\n\nreplicates_ok={n_ok}/{args.replicates}  elapsed={time.time() - started:.0f}s"
    (out / "pilot_batch_summary.txt").write_text(summary, encoding="utf-8")
    complete = n_ok == args.replicates
    marker_name = "DONE" if complete or args.allow_partial else ("PARTIAL" if n_ok else "FAILED")
    marker_text = time.strftime("%Y-%m-%d %H:%M:%S") + f"  ok={n_ok}/{args.replicates}"
    for stale in ["DONE", "PARTIAL", "FAILED"]:
        stale_path = out / stale
        if stale_path.exists() and stale != marker_name:
            stale_path.unlink()
    (out / marker_name).write_text(marker_text, encoding="utf-8")
    if complete:
        print(f"[batch] DONE ok={n_ok}/{args.replicates} elapsed={time.time() - started:.0f}s", flush=True)
    elif args.allow_partial:
        print(f"[batch] DONE partial ok={n_ok}/{args.replicates} elapsed={time.time() - started:.0f}s", flush=True)
    else:
        print(f"[batch] {marker_name} ok={n_ok}/{args.replicates} elapsed={time.time() - started:.0f}s", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
'''

# --- simulations.paper.plotting (simulations/paper/plotting.py) ---
SRC_SIMULATIONS_PAPER_PLOTTING = r'''
"""Matplotlib figure builders for the paper simulation figures."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "figure.dpi": 200,
    "savefig.dpi": 200,
})


def heatmap(ax, matrix, row_labels, col_labels, title, cbar_label, fmt="{:.2f}", cmap="viridis"):
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            v = matrix[r, c]
            if np.isfinite(v):
                ax.text(c, r, fmt.format(v), ha="center", va="center",
                        color="white" if im.norm(v) < 0.6 else "black", fontsize=7)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)
    return im


def figure2_recoverability(grid, anchor_levels, daily_levels, out_path, markov_grid=None):
    """grid[(anchor, daily)] = dict(latent_rmse=, coverage=).

    When markov_grid[(anchor, daily)] = dict(latent_rmse=) is supplied, a third
    panel shows the Markov pattern-mixture comparator's latent RMSE on the same
    grid so the classical and BALL recoverability maps can be read together on a
    shared color scale.
    """

    rmse = np.array([[grid[(a, d)]["latent_rmse"] for d in daily_levels] for a in anchor_levels])
    cov = np.array([[grid[(a, d)]["coverage"] for d in daily_levels] for a in anchor_levels])
    if markov_grid is not None:
        mrmse = np.array([[markov_grid[(a, d)]["latent_rmse"] for d in daily_levels] for a in anchor_levels])
        vmax = float(np.nanmax([np.nanmax(rmse), np.nanmax(mrmse)]))
        vmin = float(np.nanmin([np.nanmin(rmse), np.nanmin(mrmse)]))
        fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3))
        h0 = heatmap(axes[0], rmse, anchor_levels, daily_levels,
                     "BALL latent RMSE", "RMSE", cmap="viridis")
        h0.set_clim(vmin, vmax)
        h1 = heatmap(axes[1], mrmse, anchor_levels, daily_levels,
                     "Markov pattern-mixture latent RMSE", "RMSE", cmap="viridis")
        h1.set_clim(vmin, vmax)
        heatmap(axes[2], cov, anchor_levels, daily_levels,
                "BALL anchor-conformal coverage", "Coverage", cmap="cividis")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3))
        heatmap(axes[0], rmse, anchor_levels, daily_levels,
                "Test latent RMSE", "RMSE", cmap="viridis")
        heatmap(axes[1], cov, anchor_levels, daily_levels,
                "Anchor-conformal latent coverage", "Coverage", cmap="cividis")
    for ax in axes:
        ax.set_xlabel("Daily-covariate informativeness")
    axes[0].set_ylabel("Anchor quality")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def figure3_calibration(arm_summary, dist_strata, miss_strata, out_path):
    """arm_summary: dict arm -> dict(coverage, rel_width).
    dist_strata, miss_strata: DataFrames with columns bin, coverage, rel_width.
    """

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.2))
    arms = list(arm_summary.keys())
    cov = [arm_summary[a]["coverage"] for a in arms]
    wid = [arm_summary[a]["rel_width"] for a in arms]
    # A fixed palette keyed by arm name so the Markov comparator is colored
    # consistently when present; unknown arms fall back to a neutral gray.
    arm_colors = {
        "raw": "#888", "anchor conformal": "#3b7", "oracle": "#37b",
        "Markov model-based": "#e73",
    }
    colors = [arm_colors.get(a, "#aaa") for a in arms]

    axes[0, 0].bar(range(len(arms)), cov, color=colors)
    axes[0, 0].axhline(0.95, ls="--", c="k", lw=0.8)
    axes[0, 0].set_xticks(range(len(arms)))
    axes[0, 0].set_xticklabels(arms, rotation=20, ha="right")
    axes[0, 0].set_ylabel("Latent coverage")
    axes[0, 0].set_title("A. Coverage by interval arm")
    axes[0, 0].set_ylim(0.8, 1.02)

    axes[0, 1].bar(range(len(arms)), wid, color=colors)
    axes[0, 1].set_xticks(range(len(arms)))
    axes[0, 1].set_xticklabels(arms, rotation=20, ha="right")
    axes[0, 1].set_ylabel("Relative width (latent SD)")
    axes[0, 1].set_title("B. Width by interval arm")

    def strat_panel(ax, df, xlabel, title):
        x = range(len(df))
        ax.plot(x, df["coverage"], "o-", c="#37b", label="coverage")
        ax.axhline(0.95, ls="--", c="k", lw=0.8)
        ax.set_ylim(0.8, 1.02)
        ax.set_xticks(list(x))
        ax.set_xticklabels(df["bin"], rotation=20, ha="right")
        ax.set_ylabel("Coverage")
        ax.set_xlabel(xlabel)
        ax2 = ax.twinx()
        ax2.plot(x, df["rel_width"], "s--", c="#e73", label="width")
        ax2.set_ylabel("Relative width")
        ax.set_title(title)

    strat_panel(axes[1, 0], dist_strata, "Days to nearest anchor",
                "C. Raw posterior arm by anchor distance")
    strat_panel(axes[1, 1], miss_strata, "Daily missingness fraction",
                "D. Raw posterior arm by daily missingness")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def figure4_identifiability(leakage, info_limit, neg_controls, out_path):
    """leakage: 2x2 matrix [[slow->slow, slow->fast],[fast->slow, fast->fast]].
    info_limit: DataFrame with columns scenario, latent_rmse, rel_width.
    neg_controls: DataFrame with columns scenario, slow_leakage, latent_rmse.
    """

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    im = axes[0].imshow(leakage, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["true slow", "true fast"])
    axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["est slow", "est fast"])
    axes[0].set_title("A. Recovery and leakage correlations")
    for r in range(2):
        for c in range(2):
            axes[0].text(c, r, f"{leakage[r, c]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    x = np.arange(len(info_limit))
    has_markov = "markov_rmse" in info_limit.columns
    if has_markov:
        # Grouped bars compare BALL and the Markov pattern-mixture comparator
        # latent error under the same information-limit scenarios.
        w = 0.4
        axes[1].bar(x - w / 2, info_limit["latent_rmse"], width=w, color="#a33", alpha=0.8, label="BALL RMSE")
        axes[1].bar(x + w / 2, info_limit["markov_rmse"], width=w, color="#777", alpha=0.8, label="Markov RMSE")
        axes[1].legend(loc="upper left", fontsize=7)
    else:
        axes[1].bar(x, info_limit["latent_rmse"], color="#a33", alpha=0.7, label="latent RMSE")
    ax2 = axes[1].twinx()
    ax2.plot(x, info_limit["rel_width"], "s-", c="#37b", label="BALL rel width")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(info_limit["scenario"], rotation=25, ha="right")
    axes[1].set_ylabel("Latent RMSE")
    ax2.set_ylabel("Relative width")
    axes[1].set_title("B. Information limits")

    x = range(len(neg_controls))
    axes[2].bar(x, neg_controls["slow_leakage"].abs(), color="#777")
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(neg_controls["scenario"], rotation=25, ha="right")
    axes[2].set_ylabel("Absolute slow leakage correlation")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("C. Negative controls")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
'''

# --- simulations.paper.make_figure1 (simulations/paper/make_figure1.py) ---
SRC_SIMULATIONS_PAPER_MAKE_FIGURE1 = r'''
"""Figure 1. BALL model and inference schematic (four panels)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8.5})

BLUE, GREEN, ORANGE, GREY = "#cfe0f3", "#cfeede", "#fbe3cf", "#e8e8e8"


def box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                linewidth=1, edgecolor="#555", facecolor=color))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
                                 linewidth=1, color="#555", shrinkA=2, shrinkB=2))


def panelA(ax):
    ax.set_title("A. Latent decomposition", loc="left", fontsize=9)
    box(ax, 0.05, 0.62, 0.38, 0.22, "slow proxy B(t)", BLUE)
    box(ax, 0.05, 0.30, 0.38, 0.22, "sparse drifting\nloading alpha(t)", BLUE)
    box(ax, 0.57, 0.46, 0.38, 0.22, "slow component\nalpha(t)' B(t)", GREEN)
    box(ax, 0.57, 0.10, 0.38, 0.22, "fast residual\ndelta(t)", ORANGE)
    box(ax, 0.30, 0.78, 0.40, 0.18, "latent severity L(t)", GREY)
    arrow(ax, 0.43, 0.73, 0.57, 0.62); arrow(ax, 0.43, 0.41, 0.57, 0.55)
    arrow(ax, 0.76, 0.68, 0.62, 0.78); arrow(ax, 0.76, 0.32, 0.55, 0.78)


def panelB(ax):
    ax.set_title("B. Recall-windowed anchor", loc="left", fontsize=9)
    ax.plot([0.08, 0.92], [0.5, 0.5], color="#999", lw=1)
    for d in [0.2, 0.35, 0.5, 0.65, 0.8]:
        ax.plot([d, d], [0.47, 0.53], color="#999", lw=1)
    ax.add_patch(FancyBboxPatch((0.5, 0.42), 0.3, 0.16, boxstyle="round,pad=0.01",
                                linewidth=1, edgecolor=ORANGE, facecolor=ORANGE, alpha=0.5))
    ax.text(0.65, 0.5, "recall window", ha="center", va="center", fontsize=7.5)
    box(ax, 0.55, 0.68, 0.34, 0.18, "anchor Y(w)", GREEN)
    arrow(ax, 0.65, 0.58, 0.70, 0.68)
    ax.text(0.5, 0.22, "Y(w) = loading x mean of L\nover the window + noise", ha="center", fontsize=7.5)


def panelC(ax):
    ax.set_title("C. Teacher and student", loc="left", fontsize=9)
    box(ax, 0.05, 0.55, 0.42, 0.30, "bidirectional\nteacher smoother\n(full record, ELBO)", BLUE)
    box(ax, 0.53, 0.55, 0.42, 0.30, "causal student\nfilter (past only)", GREEN)
    arrow(ax, 0.47, 0.70, 0.53, 0.70)
    ax.text(0.5, 0.76, "distill", ha="center", fontsize=7)
    box(ax, 0.25, 0.12, 0.50, 0.22, "re-anchor to\nobserved scales", ORANGE)
    arrow(ax, 0.74, 0.34, 0.74, 0.55)


def panelD(ax):
    ax.set_title("D. Uncertainty stack", loc="left", fontsize=9)
    box(ax, 0.06, 0.60, 0.88, 0.20, "deep ensemble (K = 5)", BLUE)
    box(ax, 0.06, 0.36, 0.88, 0.20, "last-layer Laplace", GREEN)
    box(ax, 0.06, 0.12, 0.88, 0.20, "split conformal on\nheld-out anchor residuals", ORANGE)
    arrow(ax, 0.5, 0.60, 0.5, 0.56); arrow(ax, 0.5, 0.36, 0.5, 0.32)


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0))
    for ax in axes.ravel():
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    panelA(axes[0, 0]); panelB(axes[0, 1]); panelC(axes[1, 0]); panelD(axes[1, 1])
    fig.tight_layout()
    fig.savefig(OUT / "figure1_schematic.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "figure1_schematic.png")


if __name__ == "__main__":
    main()
'''

# --- simulations.paper.scenarios (simulations/paper/scenarios.py) ---
SRC_SIMULATIONS_PAPER_SCENARIOS = r'''
"""Scenario builders and the fit-and-evaluate harness for the paper figures.

This layer sits on top of the simulation core. It builds the paper's scenario
grid by combining config knobs that already exist (anchor error, missingness,
slow fraction, proxy error, assignment) with a few post-hoc data corruptions
that do not require changing the data-generating process. The corruptions are
applied to a generated SimulationData and are clearly named.

The fit-and-evaluate harness uses the BALL SSM causal transformer student as
the primary method, with the Markov pattern-mixture model as the classical
statistics comparator. The structural posterior can still be requested as an
audit arm, but it is not the figure headline.
"""

from __future__ import annotations

from dataclasses import replace
import copy

import numpy as np
import pandas as pd

from simulations.src.model_utils import SimulationConfig, SimulationData, AnchorSpec, make_rng
from simulations.src.dgp import generate_dataset
from simulations.src.methods.baselines import linear_interpolation, locf, anchor_only_windowed_mean
from simulations.src.methods.s0 import fit_s0_structural_prior_ssm
from simulations.src.methods.markov_pattern_mixture import fit_markov_pattern_mixture
from simulations.src.methods.ball_structural import (
    fit_ball_structural_posterior,
    conformal_calibrate_structural_posterior,
    oracle_latent_conformal_intervals,
)
from simulations.src.methods.ball_ssm import SSMConfig, fit_ball_ssm
from simulations.src import metrics as M


# ---------------------------------------------------------------------------
# Config knobs for the two recoverability axes and the stress conditions.
# ---------------------------------------------------------------------------
def anchor_quality_config(base: SimulationConfig, level: str) -> SimulationConfig:
    """Scale anchor error and missingness to set anchor quality."""

    factor = {"strong": 0.6, "moderate": 1.0, "weak": 1.8}[level]
    miss = {"strong": 0.05, "moderate": None, "weak": 0.35}[level]
    y1 = replace(base.y1, error_sd=base.y1.error_sd * factor,
                 missing_probability=base.y1.missing_probability if miss is None else miss)
    y2 = replace(base.y2, error_sd=base.y2.error_sd * factor,
                 missing_probability=base.y2.missing_probability if miss is None else max(miss, 0.10))
    return replace(base, y1=y1, y2=y2)


# Daily-covariate informativeness is set post-hoc by inflating daily noise.
DAILY_NOISE_SCALE = {"strong": 1.0, "moderate": 1.8, "noise": 6.0}


def degrade_daily(data: SimulationData, scale: float, seed_key: str = "degrade_daily") -> SimulationData:
    """Add Gaussian noise to the observed daily covariates to weaken their signal.

    scale = 1.0 leaves the data unchanged. Larger scales add proportional noise
    to every observed X value, lowering its information about the latent state
    without touching the latent state or the anchors.
    """

    if scale <= 1.0:
        return data
    rng = make_rng(data.config.seed, seed_key, int(round(scale * 100)))
    daily = data.daily.copy()
    x_cols = [c for c in daily.columns if c.startswith("X") and not c.startswith("obs_")]
    base_sd = daily[x_cols].std().to_numpy(dtype=float)
    extra = (scale - 1.0)
    for j, col in enumerate(x_cols):
        observed = daily[col].notna().to_numpy()
        noise = rng.normal(0.0, extra * (base_sd[j] if np.isfinite(base_sd[j]) else 1.0), size=len(daily))
        vals = daily[col].to_numpy(dtype=float)
        vals[observed] = vals[observed] + noise[observed]
        daily[col] = vals
    return replace(data, daily=daily)


def null_proxy(data: SimulationData, seed_key: str = "null_proxy") -> SimulationData:
    """Replace the observed proxy B with pure noise (negative control).

    The proxy columns B0..B{q-1} in the components table are overwritten with
    Gaussian noise on the observed positions. The true slow process C is left
    intact so that latent recovery from anchors and daily covariates still
    works, but the proxy carries no information about C.
    """

    rng = make_rng(data.config.seed, seed_key)
    comp = data.components.copy()
    b_cols = [f"B{d}" for d in range(data.config.q)]
    for col in b_cols:
        observed = comp[col].notna().to_numpy()
        noise = rng.normal(0.0, 1.0, size=len(comp))
        vals = comp[col].to_numpy(dtype=float)
        vals[observed] = noise[observed]
        comp[col] = vals
    return replace(data, components=comp)


def permute_proxy(data: SimulationData, seed_key: str = "permute_proxy") -> SimulationData:
    """Permute proxy columns across patients so B no longer matches C (control).

    Each patient's proxy block is replaced by a different patient's proxy block
    of the same length. The proxy keeps realistic marginal structure but is
    decoupled from this patient's true slow process.
    """

    rng = make_rng(data.config.seed, seed_key)
    comp = data.components.sort_values(["id", "t"]).copy()
    b_cols = [f"B{d}" for d in range(data.config.q)]
    ids = list(comp["id"].unique())
    perm = list(ids)
    rng.shuffle(perm)
    mapping = dict(zip(ids, perm))
    blocks = {pid: comp.loc[comp["id"] == pid, b_cols].to_numpy() for pid in ids}
    out = comp.copy()
    for pid in ids:
        src = blocks[mapping[pid]]
        dst_idx = out.index[out["id"] == pid]
        n = len(dst_idx)
        src_use = src[:n] if len(src) >= n else np.vstack([src, src[: n - len(src)]])
        out.loc[dst_idx, b_cols] = src_use
    return replace(data, components=out)


# ---------------------------------------------------------------------------
# Fit and evaluate.
# ---------------------------------------------------------------------------
def fit_and_evaluate(data: SimulationData, split: str = "test",
                     include_baselines: bool = True,
                     include_arms: bool = True,
                     include_markov: bool = True,
                     include_structural_audit: bool = False,
                     ssm_config: SSMConfig | None = None,
                     device: str | None = None) -> pd.DataFrame:
    """Fit the method set on one dataset and return a tidy metrics frame.

    The Markov pattern-mixture comparator is the classical statistics arm of the
    old-school-versus-machine-learning comparison. It is controlled separately by
    include_markov so the figures can request it alongside the structural
    posterior without also re-fitting the cheap baselines.
    """

    truth = data.components
    ind = data.individuals
    rows: list[dict] = []

    def point_row(method: str, result) -> dict:
        return {
            "method": method,
            "latent_rmse": M.latent_rmse(result.predictions, truth, ind, split),
            "slow_rmse": M.component_rmse(result.predictions, truth, "slow_hat", "slow", ind, split),
            "delta_rmse": M.component_rmse(result.predictions, truth, "delta_hat", "delta", ind, split),
        }

    if include_baselines:
        for name, fn in [("interpolation", linear_interpolation), ("locf", locf),
                         ("anchor_only", anchor_only_windowed_mean)]:
            res = fn(data)
            rows.append({"method": name,
                         "latent_rmse": M.latent_rmse(res.predictions, truth, ind, split),
                         "slow_rmse": float("nan"), "delta_rmse": float("nan")})
        s0 = fit_s0_structural_prior_ssm(data)
        r = point_row("s0", s0)
        r.update({"coverage": M.coverage(s0.intervals, truth, ind, split),
                  "rel_width": M.latent_relative_width(s0.intervals, truth, ind, split),
                  "interval_score": M.interval_score(s0.intervals, truth, individuals=ind, split=split)})
        rows.append(r)

    if include_markov:
        markov = fit_markov_pattern_mixture(data)
        mrow = point_row("markov_pattern_mixture", markov)
        mrow.update({"coverage": M.coverage(markov.intervals, truth, ind, split),
                     "rel_width": M.latent_relative_width(markov.intervals, truth, ind, split),
                     "interval_score": M.interval_score(markov.intervals, truth, individuals=ind, split=split)})
        rows.append(mrow)

    if ssm_config is None:
        ssm_config = SSMConfig(
            d_model=64,
            n_layers=2,
            n_heads=4,
            teacher_epochs=40,
            student_epochs=150,
            batch_size=32,
            ensemble_size=5,
            max_individuals=data.config.n,
            seed=data.config.seed + 91_000,
        )
    student_raw = fit_ball_ssm(
        data,
        ssm_config,
        device=device,
        causal=True,
        prediction_split=None,
    )
    diag = M.identifiability_diagnostics(student_raw.predictions, data, split)
    aset = M.active_set_metrics(student_raw.predictions, truth, data.config.q, individuals=ind, split=split)

    raw_row = point_row("student_raw", student_raw)
    raw_row.update({
        "coverage": M.coverage(student_raw.intervals, truth, ind, split),
        "rel_width": M.latent_relative_width(student_raw.intervals, truth, ind, split),
        "interval_score": M.interval_score(student_raw.intervals, truth, individuals=ind, split=split),
        **diag, **aset,
    })
    rows.append(raw_row)

    if include_arms:
        conf = conformal_calibrate_structural_posterior(student_raw, data)
        crow = point_row("student_ensemble", conf)
        crow.update({
            "coverage": M.coverage(conf.intervals, truth, ind, split),
            "rel_width": M.latent_relative_width(conf.intervals, truth, ind, split),
            "interval_score": M.interval_score(conf.intervals, truth, individuals=ind, split=split),
            **diag, **aset,
        })
        rows.append(crow)

        oracle = oracle_latent_conformal_intervals(student_raw, data)
        orow = point_row("student_oracle", oracle)
        orow.update({
            "coverage": M.coverage(oracle.intervals, truth, ind, split),
            "rel_width": M.latent_relative_width(oracle.intervals, truth, ind, split),
            "interval_score": M.interval_score(oracle.intervals, truth, individuals=ind, split=split),
        })
        rows.append(orow)

    if include_structural_audit:
        structural = fit_ball_structural_posterior(data)
        sdiag = M.identifiability_diagnostics(structural.predictions, data, split)
        saset = M.active_set_metrics(structural.predictions, truth, data.config.q, individuals=ind, split=split)
        srow = point_row("structural_raw", structural)
        srow.update({
            "coverage": M.coverage(structural.intervals, truth, ind, split),
            "rel_width": M.latent_relative_width(structural.intervals, truth, ind, split),
            "interval_score": M.interval_score(structural.intervals, truth, individuals=ind, split=split),
            **sdiag, **saset,
        })
        rows.append(srow)

    return pd.DataFrame(rows)


def structural_intervals(data: SimulationData):
    """Return (raw, conformal, oracle) MethodResults for the stratified figure."""

    structural = fit_ball_structural_posterior(data)
    conf = conformal_calibrate_structural_posterior(structural, data)
    oracle = oracle_latent_conformal_intervals(structural, data)
    return structural, conf, oracle


def primary_intervals(
    data: SimulationData,
    ssm_config: SSMConfig | None = None,
    device: str | None = None,
):
    """Return (raw, conformal, oracle) MethodResults for the BALL student."""

    if ssm_config is None:
        ssm_config = SSMConfig(
            d_model=64,
            n_layers=2,
            n_heads=4,
            teacher_epochs=40,
            student_epochs=150,
            batch_size=32,
            ensemble_size=5,
            max_individuals=data.config.n,
            seed=data.config.seed + 91_000,
        )
    raw = fit_ball_ssm(data, ssm_config, device=device, causal=True, prediction_split=None)
    conf = conformal_calibrate_structural_posterior(raw, data)
    oracle = oracle_latent_conformal_intervals(raw, data)
    return raw, conf, oracle
'''

# --- simulations.paper.run_fig2_recoverability (simulations/paper/run_fig2_recoverability.py) ---
SRC_SIMULATIONS_PAPER_RUN_FIG2_RECOVERABILITY = r'''
"""Figure 2. Recoverability map over anchor quality x daily-covariate signal.

Sweeps a grid of anchor quality and daily informativeness, fits the BALL SSM
causal transformer student per cell across replicates, and records test latent
RMSE and anchor-conformal latent coverage. Writes the cell table and heatmap.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simulations.src.model_utils import SimulationConfig
from simulations.src.dgp import generate_dataset
from simulations.src.methods.ball_ssm import SSMConfig
from simulations.paper.scenarios import (
    anchor_quality_config, degrade_daily, DAILY_NOISE_SCALE,
    fit_and_evaluate,
)
from simulations.paper import plotting

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

ANCHOR_LEVELS = ["strong", "moderate", "weak"]
DAILY_LEVELS = ["strong", "moderate", "noise"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--replicates", type=int, default=10)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--t", type=int, default=84)
    p.add_argument("--base-seed", type=int, default=50000)
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--teacher-epochs", type=int, default=40)
    p.add_argument("--student-epochs", type=int, default=150)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    rows = []
    for a in ANCHOR_LEVELS:
        for d in DAILY_LEVELS:
            for r in range(args.replicates):
                seed = args.base_seed + 1000 * r
                cfg = anchor_quality_config(
                    SimulationConfig(n=args.n, t=args.t, seed=seed), a)
                data = generate_dataset(cfg)
                data = degrade_daily(data, DAILY_NOISE_SCALE[d])
                ssm_config = SSMConfig(
                    d_model=args.d_model,
                    n_layers=args.n_layers,
                    teacher_epochs=args.teacher_epochs,
                    student_epochs=args.student_epochs,
                    batch_size=args.batch_size,
                    ensemble_size=args.members,
                    max_individuals=args.n,
                    seed=seed + 91_000,
                )
                metrics = fit_and_evaluate(
                    data,
                    include_baselines=False,
                    include_arms=True,
                    include_markov=True,
                    ssm_config=ssm_config,
                    device=args.device,
                )
                conf = metrics[metrics["method"] == "student_ensemble"].iloc[0]
                raw = metrics[metrics["method"] == "student_raw"].iloc[0]
                markov = metrics[metrics["method"] == "markov_pattern_mixture"]
                markov_rmse = float(markov["latent_rmse"].iloc[0]) if len(markov) else float("nan")
                rows.append({"anchor": a, "daily": d, "replicate": r,
                             "latent_rmse": raw["latent_rmse"],
                             "coverage": conf["coverage"],
                             "markov_rmse": markov_rmse})
            print(f"cell anchor={a} daily={d} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "fig2_recoverability_cells.csv", index=False)
    agg = df.groupby(["anchor", "daily"]).agg(
        latent_rmse=("latent_rmse", "mean"),
        latent_rmse_mcse=("latent_rmse", lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0),
        coverage=("coverage", "mean"),
        markov_rmse=("markov_rmse", "mean"),
    ).reset_index()
    agg.to_csv(OUT / "fig2_recoverability_aggregate.csv", index=False)

    grid = {(row["anchor"], row["daily"]): {"latent_rmse": row["latent_rmse"], "coverage": row["coverage"]}
            for _, row in agg.iterrows()}
    markov_grid = {(row["anchor"], row["daily"]): {"latent_rmse": row["markov_rmse"]}
                   for _, row in agg.iterrows()}
    plotting.figure2_recoverability(grid, ANCHOR_LEVELS, DAILY_LEVELS,
                                    OUT / "figure2_recoverability.png", markov_grid=markov_grid)
    print("wrote", OUT / "figure2_recoverability.png")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
'''

# --- simulations.paper.run_fig3_calibration (simulations/paper/run_fig3_calibration.py) ---
SRC_SIMULATIONS_PAPER_RUN_FIG3_CALIBRATION = r'''
"""Figure 3. Interval calibration and the latent-calibration gap.

Fits the BALL SSM causal transformer student on the default scenario across
replicates and reports, for the three interval arms (raw, anchor conformal,
oracle), latent coverage and relative width. It also reports the raw arm's
coverage and width stratified by days to the nearest observed anchor and by
daily-covariate missingness.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simulations.src.model_utils import SimulationConfig
from simulations.src.dgp import generate_dataset
from simulations.src import metrics as M
from simulations.src.methods.ball_ssm import SSMConfig
from simulations.src.methods.markov_pattern_mixture import fit_markov_pattern_mixture
from simulations.paper.scenarios import primary_intervals
from simulations.paper import plotting

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
SPLIT = "test"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--replicates", type=int, default=10)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--t", type=int, default=84)
    p.add_argument("--base-seed", type=int, default=60000)
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--teacher-epochs", type=int, default=40)
    p.add_argument("--student-epochs", type=int, default=150)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    arm_rows, dist_rows, miss_rows = [], [], []
    for r in range(args.replicates):
        seed = args.base_seed + 1000 * r
        data = generate_dataset(SimulationConfig(n=args.n, t=args.t, seed=seed))
        truth, ind = data.components, data.individuals
        ssm_config = SSMConfig(
            d_model=args.d_model,
            n_layers=args.n_layers,
            teacher_epochs=args.teacher_epochs,
            student_epochs=args.student_epochs,
            batch_size=args.batch_size,
            ensemble_size=args.members,
            max_individuals=args.n,
            seed=seed + 91_000,
        )
        raw, conf, oracle = primary_intervals(data, ssm_config, device=args.device)
        # The Markov pattern-mixture comparator reports model-based Gaussian
        # intervals, the classical counterpart to the BALL conformal and oracle
        # arms. It is shown as a fourth arm so the calibration of model-based
        # intervals can be read against the conformal construction.
        markov = fit_markov_pattern_mixture(data)
        for arm, res in [("student raw", raw), ("student anchor conformal", conf), ("student oracle", oracle),
                         ("Markov model-based", markov)]:
            arm_rows.append({"replicate": r, "arm": arm,
                             "coverage": M.coverage(res.intervals, truth, ind, SPLIT),
                             "rel_width": M.latent_relative_width(res.intervals, truth, ind, SPLIT)})
        # Stratify the RAW posterior arm, not the conformal arm. The conformal
        # arm applies one constant half-width per channel, so it cannot reveal
        # whether the model widens its own intervals where information is weak.
        # The raw arm has day-varying width and is the right test of the
        # overconfidence-under-weak-information risk.
        d = M.coverage_by_anchor_distance(raw.intervals, truth, data.anchors, ind, SPLIT)
        d["replicate"] = r
        dist_rows.append(d)
        m = M.coverage_by_daily_missingness(raw.intervals, truth, data.daily, ind, SPLIT)
        m["replicate"] = r
        miss_rows.append(m)
        print(f"replicate {r} done", flush=True)

    arms = pd.DataFrame(arm_rows)
    arms.to_csv(OUT / "fig3_arms.csv", index=False)
    dist = pd.concat(dist_rows, ignore_index=True)
    dist.to_csv(OUT / "fig3_by_anchor_distance.csv", index=False)
    miss = pd.concat(miss_rows, ignore_index=True)
    miss.to_csv(OUT / "fig3_by_daily_missingness.csv", index=False)

    arm_summary = {a: {"coverage": float(g["coverage"].mean()), "rel_width": float(g["rel_width"].mean())}
                   for a, g in arms.groupby("arm")}
    # keep arm display order
    arm_summary = {
        a: arm_summary[a]
        for a in ["student raw", "student anchor conformal", "student oracle", "Markov model-based"]
        if a in arm_summary
    }
    dist_strata = dist.groupby("bin", sort=False).agg(
        coverage=("coverage", "mean"), rel_width=("rel_width", "mean")).reset_index()
    miss_strata = miss.groupby("bin", sort=False).agg(
        coverage=("coverage", "mean"), rel_width=("rel_width", "mean")).reset_index()

    plotting.figure3_calibration(arm_summary, dist_strata, miss_strata, OUT / "figure3_calibration.png")
    print("wrote", OUT / "figure3_calibration.png")
    print("arms:", arm_summary)


if __name__ == "__main__":
    main()
'''

# --- simulations.paper.run_fig4_identifiability (simulations/paper/run_fig4_identifiability.py) ---
SRC_SIMULATIONS_PAPER_RUN_FIG4_IDENTIFIABILITY = r'''
"""Figure 4. Identifiability diagnostics, information limits, negative controls.

Panel A. Recovery and leakage correlations between the recovered and true slow
and fast components on the default scenario.
Panel B. Latent error and relative interval width in information-limit
scenarios in which anchors and daily covariates are jointly weakened.
Panel C. Absolute slow-component leakage correlation in negative controls with
a null proxy, a permuted proxy, and noise-only daily covariates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simulations.src.model_utils import SimulationConfig
from simulations.src.dgp import generate_dataset
from simulations.src import metrics as M
from simulations.src.methods.ball_ssm import SSMConfig
from simulations.paper.scenarios import (
    anchor_quality_config, degrade_daily, null_proxy, permute_proxy,
    fit_and_evaluate, DAILY_NOISE_SCALE,
)
from simulations.paper import plotting

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
SPLIT = "test"


def _default(seed, n, t):
    return generate_dataset(SimulationConfig(n=n, t=t, seed=seed))


def _info_limit_builders():
    return {
        "weak anchors\n+ weak daily": lambda s, n, t: degrade_daily(
            generate_dataset(anchor_quality_config(SimulationConfig(n=n, t=t, seed=s), "weak")),
            DAILY_NOISE_SCALE["noise"]),
        "anchor-only\n(noise daily)": lambda s, n, t: degrade_daily(
            generate_dataset(SimulationConfig(n=n, t=t, seed=s)), DAILY_NOISE_SCALE["noise"]),
        "weak anchors\nonly": lambda s, n, t: generate_dataset(
            anchor_quality_config(SimulationConfig(n=n, t=t, seed=s), "weak")),
    }


def _neg_control_builders():
    return {
        "null proxy": lambda s, n, t: null_proxy(generate_dataset(SimulationConfig(n=n, t=t, seed=s))),
        "permuted proxy": lambda s, n, t: permute_proxy(generate_dataset(SimulationConfig(n=n, t=t, seed=s))),
        "noise daily": lambda s, n, t: degrade_daily(
            generate_dataset(SimulationConfig(n=n, t=t, seed=s)), DAILY_NOISE_SCALE["noise"]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--replicates", type=int, default=5)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--t", type=int, default=84)
    p.add_argument("--base-seed", type=int, default=70000)
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--teacher-epochs", type=int, default=40)
    p.add_argument("--student-epochs", type=int, default=150)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    def ssm_config(seed: int) -> SSMConfig:
        return SSMConfig(
            d_model=args.d_model,
            n_layers=args.n_layers,
            teacher_epochs=args.teacher_epochs,
            student_epochs=args.student_epochs,
            batch_size=args.batch_size,
            ensemble_size=args.members,
            max_individuals=args.n,
            seed=seed + 91_000,
        )

    # Panel A. Default-scenario recovery and leakage correlations.
    leak = {"slow_recovery_corr": [], "fast_recovery_corr": [],
            "slow_to_fast_leakage": [], "fast_to_slow_leakage": []}
    for r in range(args.replicates):
        data = _default(args.base_seed + 1000 * r, args.n, args.t)
        m = fit_and_evaluate(
            data,
            include_baselines=False,
            include_arms=False,
            include_markov=False,
            ssm_config=ssm_config(data.config.seed),
            device=args.device,
        )
        raw = m[m["method"] == "student_raw"].iloc[0]
        for k in leak:
            leak[k].append(raw[k])
    leak_mat = np.array([
        [np.nanmean(leak["slow_recovery_corr"]), np.nanmean(leak["slow_to_fast_leakage"])],
        [np.nanmean(leak["fast_to_slow_leakage"]), np.nanmean(leak["fast_recovery_corr"])],
    ])
    print("panel A done", flush=True)

    # Panel B. Information-limit scenarios.
    info_rows = []
    for name, builder in _info_limit_builders().items():
        rmses, widths, markov_rmses = [], [], []
        for r in range(args.replicates):
            data = builder(args.base_seed + 1000 * r, args.n, args.t)
            m = fit_and_evaluate(
                data,
                include_baselines=False,
                include_arms=True,
                include_markov=True,
                ssm_config=ssm_config(data.config.seed),
                device=args.device,
            )
            raw = m[m["method"] == "student_raw"].iloc[0]
            conf = m[m["method"] == "student_ensemble"].iloc[0]
            markov = m[m["method"] == "markov_pattern_mixture"]
            rmses.append(raw["latent_rmse"]); widths.append(conf["rel_width"])
            if len(markov):
                markov_rmses.append(float(markov["latent_rmse"].iloc[0]))
        info_rows.append({"scenario": name, "latent_rmse": np.nanmean(rmses),
                          "rel_width": np.nanmean(widths),
                          "markov_rmse": np.nanmean(markov_rmses) if markov_rmses else float("nan")})
        print(f"info-limit {name!r} done", flush=True)
    info_df = pd.DataFrame(info_rows)

    # Panel C. Negative controls.
    neg_rows = []
    for name, builder in _neg_control_builders().items():
        leaks, rmses = [], []
        for r in range(args.replicates):
            data = builder(args.base_seed + 1000 * r, args.n, args.t)
            m = fit_and_evaluate(
                data,
                include_baselines=False,
                include_arms=False,
                include_markov=False,
                ssm_config=ssm_config(data.config.seed),
                device=args.device,
            )
            raw = m[m["method"] == "student_raw"].iloc[0]
            leaks.append(raw["slow_to_fast_leakage"]); rmses.append(raw["latent_rmse"])
        neg_rows.append({"scenario": name, "slow_leakage": np.nanmean(leaks),
                         "latent_rmse": np.nanmean(rmses)})
        print(f"neg-control {name!r} done", flush=True)
    neg_df = pd.DataFrame(neg_rows)

    pd.DataFrame({"cell": ["slow->slow", "slow->fast", "fast->slow", "fast->fast"],
                  "corr": leak_mat.ravel()}).to_csv(OUT / "fig4_leakage.csv", index=False)
    info_df.to_csv(OUT / "fig4_info_limit.csv", index=False)
    neg_df.to_csv(OUT / "fig4_negative_controls.csv", index=False)
    plotting.figure4_identifiability(leak_mat, info_df, neg_df, OUT / "figure4_identifiability.png")
    print("wrote", OUT / "figure4_identifiability.png")
    print(info_df.to_string(index=False))
    print(neg_df.to_string(index=False))


if __name__ == "__main__":
    main()
'''

# --- simulations.paper.run_rts_identification (simulations/paper/run_rts_identification.py) ---
SRC_SIMULATIONS_PAPER_RUN_RTS_IDENTIFICATION = r'''
"""Time-scale-ratio identification sweep (Proposition A2 test).

Sweeps the loading-to-residual innovation-variance ratio r_ts and reports, for
the BALL SSM causal transformer student, the within-person recovery correlations of the slow
and fast components and the cross leakage correlations. Proposition A2 predicts
that as r_ts increases toward 1 the two components move on similar time scales,
component recovery degrades, and leakage rises. Outputs a table with Monte
Carlo standard errors and a figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulations.src.model_utils import SimulationConfig
from simulations.src.dgp import generate_dataset
from simulations.src import metrics as M
from simulations.src.methods.ball_ssm import SSMConfig, fit_ball_ssm

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
# Sweep into the regime where the loading innovation overtakes the residual
# innovation (around r_ts = 9), which is where Proposition A2 predicts the
# time-scale separation collapses and leakage rises.
R_TS = [0.03, 0.3, 1.0, 3.0, 10.0, 30.0]
SPLIT = "test"


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--replicates", type=int, default=5)
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--t", type=int, default=84)
    p.add_argument("--base-seed", type=int, default=80000)
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--teacher-epochs", type=int, default=40)
    p.add_argument("--student-epochs", type=int, default=150)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    rows = []
    for rts in R_TS:
        for r in range(args.replicates):
            cfg = SimulationConfig(n=args.n, t=args.t, seed=args.base_seed + 1000 * r, r_ts=rts)
            data = generate_dataset(cfg)
            ssm_config = SSMConfig(
                d_model=args.d_model,
                n_layers=args.n_layers,
                teacher_epochs=args.teacher_epochs,
                student_epochs=args.student_epochs,
                batch_size=args.batch_size,
                ensemble_size=args.members,
                max_individuals=args.n,
                seed=cfg.seed + 91_000,
            )
            res = fit_ball_ssm(data, ssm_config, device=args.device, causal=True, prediction_split=None)
            diag = M.identifiability_diagnostics(res.predictions, data, SPLIT)
            rows.append({"r_ts": rts, "replicate": r,
                         "slow_recovery": diag["slow_recovery_corr"],
                         "fast_recovery": diag["fast_recovery_corr"],
                         "slow_to_fast_leakage": diag["slow_to_fast_leakage"],
                         "fast_to_slow_leakage": diag["fast_to_slow_leakage"],
                         "latent_rmse": M.latent_rmse(res.predictions, data.components, data.individuals, SPLIT)})
        print(f"r_ts={rts} done", flush=True)

    df = pd.DataFrame(rows)
    def agg(col):
        g = df.groupby("r_ts")[col]
        return g.mean(), g.std(ddof=1) / np.sqrt(g.count())
    summary = pd.DataFrame({"r_ts": R_TS})
    for col in ["slow_recovery", "fast_recovery", "slow_to_fast_leakage", "fast_to_slow_leakage", "latent_rmse"]:
        m, se = agg(col)
        summary[col] = [m[r] for r in R_TS]
        summary[col + "_mcse"] = [se[r] for r in R_TS]
    summary.to_csv(OUT / "rts_identification.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.errorbar(R_TS, summary["slow_recovery"], yerr=summary["slow_recovery_mcse"], marker="o", label="slow recovery", color="#37b")
    ax.errorbar(R_TS, summary["fast_recovery"], yerr=summary["fast_recovery_mcse"], marker="o", label="fast recovery", color="#3b7")
    ax.errorbar(R_TS, summary["slow_to_fast_leakage"].abs(), yerr=summary["slow_to_fast_leakage_mcse"], marker="s", ls="--", label="slow->fast leakage", color="#e73")
    ax.errorbar(R_TS, summary["fast_to_slow_leakage"].abs(), yerr=summary["fast_to_slow_leakage_mcse"], marker="s", ls="--", label="fast->slow leakage", color="#a33")
    ax.set_xscale("log"); ax.set_xlabel("Time-scale ratio r_ts (log scale)")
    ax.set_ylabel("Within-person correlation"); ax.set_ylim(-0.05, 1.0)
    ax.set_title("Component recovery and leakage versus time-scale ratio")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "figure_rts_identification.png", bbox_inches="tight")
    plt.close(fig)
    pd.set_option("display.width", 160)
    print(summary.round(3).to_string(index=False))
    print("wrote", OUT / "rts_identification.csv", "and figure_rts_identification.png")


if __name__ == "__main__":
    main()
'''

# --- simulations.paper.run_stress_panels (simulations/paper/run_stress_panels.py) ---
SRC_SIMULATIONS_PAPER_RUN_STRESS_PANELS = r'''
"""Generate Supplementary Table 2 stress-panel aggregates.

Each panel is a simulation scenario fit with the BALL SSM causal transformer
student, Markov pattern-mixture comparator, S0, and cheap point baselines. The
aggregate files are intentionally named to match the table builder's expected
post-2026-06-12 stress-panel source pattern.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simulations.src.model_utils import SimulationConfig
from simulations.src.dgp import generate_dataset
from simulations.src import metrics as M
from simulations.src.methods.ball_ssm import SSMConfig
from simulations.paper.scenarios import anchor_quality_config, degrade_daily, fit_and_evaluate

OUT = ROOT / "simulations" / "outputs" / "ball_pipeline"
OUT.mkdir(parents=True, exist_ok=True)


def _panel_config(panel: str, seed: int, n: int, t: int) -> SimulationConfig:
    base = SimulationConfig(n=n, t=t, seed=seed)
    if panel == "moderate":
        return base
    if panel == "mnar":
        return replace(base, missingness_mechanism="mnar", mnar_gamma_l=0.5)
    if panel == "random_proxy":
        return replace(base, endogenous_proxy_observation=False)
    if panel == "low_anchor":
        return anchor_quality_config(base, "weak")
    if panel == "confounding":
        return replace(base, assignment="strongly_adaptive")
    if panel == "site_confound":
        return replace(base, site_bias_sd=0.45, note_density_error_multiplier=2.0)
    raise ValueError(f"Unknown stress panel: {panel}")


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    metric_names = [
        "latent_rmse",
        "slow_rmse",
        "delta_rmse",
        "coverage",
        "rel_width",
        "interval_score",
        "active_f1",
        "active_topk_f1",
        "rho_rdoc",
    ]
    rows = []
    for role, group in frame.groupby("role", sort=True):
        method = str(group["method_label"].iloc[0])
        for metric in metric_names:
            if metric not in group.columns:
                values = pd.Series([], dtype=float)
            else:
                values = pd.to_numeric(group[metric], errors="coerce")
            rows.append({
                "role": role,
                "method": method,
                "evaluation_split": "test",
                "metric": metric,
                **M.summarize_mcse(values),
            })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--replicates", type=int, default=10)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--t", type=int, default=84)
    p.add_argument("--base-seed", type=int, default=90000)
    p.add_argument("--panels", type=str, default="moderate,mnar,random_proxy,low_anchor,confounding,site_confound")
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--teacher-epochs", type=int, default=40)
    p.add_argument("--student-epochs", type=int, default=150)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    panels = [panel.strip() for panel in args.panels.split(",") if panel.strip()]
    for panel_idx, panel in enumerate(panels):
        rows = []
        for r in range(args.replicates):
            seed = args.base_seed + 10_000 * panel_idx + 1000 * r
            cfg = _panel_config(panel, seed, args.n, args.t)
            data = generate_dataset(cfg)
            if panel == "low_anchor":
                # Low-anchor is already weak on the anchor axis; mildly weaken
                # daily covariates too so the panel is an actual information
                # stress test rather than just a missing-anchor variant.
                data = degrade_daily(data, 1.8)
            ssm_config = SSMConfig(
                d_model=args.d_model,
                n_layers=args.n_layers,
                teacher_epochs=args.teacher_epochs,
                student_epochs=args.student_epochs,
                batch_size=args.batch_size,
                ensemble_size=args.members,
                max_individuals=args.n,
                seed=seed + 91_000,
            )
            metrics = fit_and_evaluate(
                data,
                include_baselines=True,
                include_arms=True,
                include_markov=True,
                include_structural_audit=True,
                ssm_config=ssm_config,
                device=args.device,
            )
            metrics.insert(0, "panel", panel)
            metrics.insert(1, "replicate", r)
            metrics.insert(2, "data_seed", seed)
            metrics["role"] = metrics["method"]
            metrics["method_label"] = metrics["method"]
            rows.append(metrics)
            print(f"panel={panel} replicate={r} done", flush=True)
        replicate_frame = pd.concat(rows, ignore_index=True)
        aggregate = _aggregate(replicate_frame)
        replicate_frame.to_csv(OUT / f"post20260612_{panel}_replicate_metrics.csv", index=False)
        aggregate.to_csv(OUT / f"post20260612_{panel}_aggregate_mcse.csv", index=False)
        print(f"wrote {OUT / f'post20260612_{panel}_aggregate_mcse.csv'}", flush=True)


if __name__ == "__main__":
    main()
'''

# --- simulations.paper.make_tables (simulations/paper/make_tables.py) ---
SRC_SIMULATIONS_PAPER_MAKE_TABLES = r'''
"""Generate Table 1 and Supplementary Table 2 from run aggregates.

Table 1 reads the primary-comparison aggregate (the pilot, or the 100-replicate
main grid once it finishes) and emits a tidy CSV and a Markdown table with the
method rows the manuscript uses. Supplementary Table 2 reads the post-2026-06-12
stress-panel aggregates.

The main grid path is checked first. If its aggregate is absent or stale, the
script falls back to the corrected pilot aggregate and labels the source.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

MAIN_GRID = ROOT / "simulations" / "runs" / "main_grid" / "pilot_batch_aggregate_mcse.csv"
PILOT = ROOT / "simulations" / "runs" / "pilot_batch" / "pilot_batch_aggregate_mcse.csv"
LATEST_PIPELINE = ROOT / "simulations" / "outputs" / "ball_pipeline" / "latest_ball_pipeline_pipeline_aggregate_mcse.csv"
STRESS_DIR = ROOT / "simulations" / "outputs" / "ball_pipeline"
DONE_RE = re.compile(r"ok=(\d+)/(\d+)")

# Role -> manuscript label for Table 1, ordered with the PRIMARY deployable
# transformer student first. The bidirectional teacher is a transformer smoother
# diagnostic, not the lead result. Classical comparators follow.
TABLE1_ROLES = {
    "student_ensemble": "BALL transformer student (deployable), anchor conformal",
    "ssm_teacher": "BALL transformer teacher (smoother diagnostic), anchor conformal",
    "markov_pattern_mixture": "Markov pattern-mixture model",
    "s0": "S0 structural-prior smoother",
    "structural_raw": "BALL structural posterior (diagnostic), raw interval",
}
REQUIRED_TABLE1_ROLES = {
    "ssm_teacher",
    "student_ensemble",
    "structural_raw",
    "markov_pattern_mixture",
    "s0",
}


def _completion_counts(run_dir: Path) -> tuple[int, int] | None:
    done = run_dir / "DONE"
    if not done.exists():
        return None
    match = DONE_RE.search(done.read_text(encoding="utf-8", errors="replace"))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _required_roles_present(df: pd.DataFrame, required_roles: set[str]) -> bool:
    roles = set(df["role"].dropna().astype(str).unique())
    return required_roles.issubset(roles)


def _min_metric_n(df: pd.DataFrame, roles: set[str], metric: str = "latent_rmse") -> float:
    rows = df[(df["role"].isin(roles)) & (df["metric"] == metric)]
    if rows.empty:
        return 0.0
    return float(rows["n"].min())


def _is_fresh_main_grid(expected_replicates: int) -> bool:
    """Use the main grid only if its current run has actually finished.

    The directory can hold stale aggregates from an earlier (pre-fix) run while
    a new run is still in progress. We require a DONE marker that postdates the
    STARTED marker, reports exact completion, and an aggregate file written at
    or after STARTED. The aggregate also must contain every manuscript Table 1
    role with at least the expected number of latent-RMSE replicates.
    """

    if not MAIN_GRID.exists():
        return False
    started = MAIN_GRID.parent / "STARTED"
    done = MAIN_GRID.parent / "DONE"
    try:
        if started.exists():
            started_mtime = started.stat().st_mtime
            if not done.exists() or done.stat().st_mtime < started_mtime:
                return False
            if MAIN_GRID.stat().st_mtime < started_mtime:
                return False
        counts = _completion_counts(MAIN_GRID.parent)
        if counts is None:
            return False
        ok, total = counts
        if ok != total or ok < expected_replicates:
            return False
        d = pd.read_csv(MAIN_GRID)
        if not _required_roles_present(d, REQUIRED_TABLE1_ROLES):
            return False
        return _min_metric_n(d, REQUIRED_TABLE1_ROLES) >= expected_replicates
    except Exception:
        return False


def _pivot(df: pd.DataFrame, role: str, metrics: list[str]) -> dict:
    out = {}
    sub = df[df["role"] == role]
    for m in metrics:
        row = sub[sub["metric"] == m]
        out[m] = (float(row["mean"].iloc[0]), float(row["mcse"].iloc[0])) if len(row) else (float("nan"), float("nan"))
    return out


def _source_label(source: Path, df: pd.DataFrame, expected_main_replicates: int) -> str:
    counts = _completion_counts(source.parent)
    if counts is not None:
        ok, total = counts
        count_label = f"{ok}-of-{total}"
    else:
        n = _min_metric_n(df, set(df["role"].unique()))
        count_label = f"{int(n)}"
    if source == MAIN_GRID and counts and counts[0] >= expected_main_replicates:
        return f"{expected_main_replicates}-replicate main grid"
    if source == MAIN_GRID:
        return f"{count_label}-replicate main grid"
    if source == LATEST_PIPELINE:
        return f"{count_label}-replicate latest BALL pipeline"
    return f"{count_label}-replicate corrected pilot"


def _latest_pipeline_is_grade() -> bool:
    """The latest BALL pipeline aggregate may feed the manuscript only if its run
    was manuscript-grade. Smoke/quick runs publish under a SMOKE prefix and set
    manuscript_grade=False, so a stale sub-grade latest can never become Table 1.
    """

    import json
    manifest = LATEST_PIPELINE.parent / "latest_ball_pipeline_manifest.json"
    try:
        return bool(json.loads(manifest.read_text(encoding="utf-8")).get("manuscript_grade", False))
    except Exception:
        return False


def _select_table1_source(expected_main_replicates: int) -> Path:
    if _is_fresh_main_grid(expected_main_replicates):
        return MAIN_GRID
    if PILOT.exists():
        df = pd.read_csv(PILOT)
        if _required_roles_present(df, REQUIRED_TABLE1_ROLES):
            return PILOT
    if LATEST_PIPELINE.exists():
        if not _latest_pipeline_is_grade():
            raise RuntimeError(
                f"Refusing to use non-manuscript-grade latest pipeline for Table 1: {LATEST_PIPELINE}. "
                "Run BALL.py pipeline with manuscript-grade settings or provide a completed pilot/main grid."
            )
        candidate = LATEST_PIPELINE
        df = pd.read_csv(candidate)
        if _required_roles_present(df, REQUIRED_TABLE1_ROLES):
            return candidate
    raise RuntimeError(
        "No valid Table 1 source found. Run BALL.py pipeline with manuscript-grade settings, "
        "or complete the pilot/main grid with all required roles."
    )


def make_table1(expected_main_replicates: int) -> None:
    source = _select_table1_source(expected_main_replicates)
    if not source.exists():
        raise FileNotFoundError(f"Table 1 source is missing: {source}")
    df = pd.read_csv(source)
    missing_roles = sorted(REQUIRED_TABLE1_ROLES.difference(set(df["role"].dropna().astype(str).unique())))
    if missing_roles:
        raise RuntimeError(
            f"Table 1 source {source} is missing required role(s): {', '.join(missing_roles)}. "
            "Rerun BALL.py pipeline or BALL.py pilot-batch without --skip-s0/--skip-markov."
        )
    label = _source_label(source, df, expected_main_replicates)
    metrics = ["latent_rmse", "slow_rmse", "delta_rmse", "coverage", "mean_width", "interval_score"]
    rows = []
    # Roles that intentionally estimate only the composite severity, so their
    # slow and fast component columns are not applicable rather than pending.
    composite_only_roles = {"markov_pattern_mixture"}
    for role, name in TABLE1_ROLES.items():
        vals = _pivot(df, role, metrics)
        lr, lr_mcse = vals["latent_rmse"]
        component_missing = "n/a" if role in composite_only_roles else "[pending]"
        rows.append({
            "method": name,
            "latent_rmse": f"{lr:.3f} ({lr_mcse:.3f})" if np.isfinite(lr) else "[pending]",
            "slow_rmse": f"{vals['slow_rmse'][0]:.3f}" if np.isfinite(vals["slow_rmse"][0]) else component_missing,
            "delta_rmse": f"{vals['delta_rmse'][0]:.3f}" if np.isfinite(vals["delta_rmse"][0]) else component_missing,
            "coverage": f"{vals['coverage'][0]:.3f}" if np.isfinite(vals["coverage"][0]) else "[pending]",
            "mean_width": f"{vals['mean_width'][0]:.3f}" if np.isfinite(vals["mean_width"][0]) else "[pending]",
            "interval_score": f"{vals['interval_score'][0]:.3f}" if np.isfinite(vals["interval_score"][0]) else "[pending]",
        })
    # Baselines are point methods. Their latent RMSE comes from battle/pilot
    # baseline rows where available; otherwise left as the documented pilot values.
    baseline_rmse = {"Linear interpolation": 1.30, "Last observation carried forward": 1.51,
                     "Anchor-only windowed mean": 0.97}
    for name, rmse in baseline_rmse.items():
        rows.append({"method": name, "latent_rmse": f"{rmse:.2f}", "slow_rmse": "n/a",
                     "delta_rmse": "n/a", "coverage": "n/a", "mean_width": "n/a", "interval_score": "n/a"})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "table1_primary_comparison.csv", index=False)
    md = ["| Method | Latent RMSE (MCSE) | Slow RMSE | Fast RMSE | Coverage | Mean width | Interval score |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in table.iterrows():
        md.append("| " + " | ".join(str(r[c]) for c in ["method", "latent_rmse", "slow_rmse",
                  "delta_rmse", "coverage", "mean_width", "interval_score"]) + " |")
    md.append("")
    md.append(f"Source: {label} ({source.relative_to(ROOT)}).")
    (OUT / "table1_primary_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Table 1 written from {label}")
    print(table.to_string(index=False))


def make_supp_table2() -> None:
    panels = ["moderate", "mnar", "random_proxy", "low_anchor", "confounding", "site_confound"]
    keep = {
        ("s0",): "S0",
        ("markov_pattern_mixture",): "Markov",
        ("student_ensemble", "student_conformal"): "Student",
    }
    rows = []
    missing_sources = []
    for panel in panels:
        f = STRESS_DIR / f"post20260612_{panel}_aggregate_mcse.csv"
        if not f.exists():
            missing_sources.append(f.name)
            continue
        df = pd.read_csv(f)
        row = {"panel": panel}
        for roles, short in keep.items():
            for metric, col in [("latent_rmse", "latent"), ("coverage", "cov"), ("active_f1", "f1")]:
                sel = df[(df["role"].isin(roles)) & (df["metric"] == metric)]
                row[f"{short}_{col}"] = round(float(sel["mean"].iloc[0]), 3) if len(sel) else float("nan")
        rows.append(row)
    if missing_sources:
        raise FileNotFoundError(
            "Supplementary Table 2 is missing stress-grid aggregate source(s) under "
            f"{STRESS_DIR}: {', '.join(missing_sources)}"
        )
    if not rows:
        raise FileNotFoundError(f"Supplementary Table 2 found no stress-grid aggregate sources under {STRESS_DIR}.")
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "supp_table2_stress.csv", index=False)
    print("\nSupplementary Table 2 written")
    print(table.to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--which", choices=["table1", "supp2", "all"], default="all")
    p.add_argument("--expected-main-replicates", type=int, default=100)
    args = p.parse_args()
    if args.which in ("table1", "all"):
        make_table1(args.expected_main_replicates)
    if args.which in ("supp2", "all"):
        make_supp_table2()


if __name__ == "__main__":
    main()
'''

# --- simulations.paper.run_all (simulations/paper/run_all.py) ---
SRC_SIMULATIONS_PAPER_RUN_ALL = r'''
"""Orchestrate all simulation tables and figures for the paper.

Runs Table 1 and Supplementary Table 2 (cheap, read from aggregates), then the
three simulation figures (Figure 2 recoverability, Figure 3 calibration, Figure
4 identifiability). Figures fit the structural posterior per cell, so they are
the expensive step. Use --quick for a fast small-scale draft, or pass full
arguments for the manuscript-scale run.

Manuscript-scale invocation (heavy):
    python simulations/paper/run_all.py --replicates 10 --n 1000 --t 84

Quick draft:
    python simulations/paper/run_all.py --quick
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def run(script: str, args: list[str]) -> None:
    command_map = {
        "make_tables.py": "paper-tables",
        "run_ball_pipeline.py": "pipeline",
        "run_fig2_recoverability.py": "paper-fig2",
        "run_fig3_calibration.py": "paper-fig3",
        "run_fig4_identifiability.py": "paper-fig4",
        "run_stress_panels.py": "paper-stress",
    }
    cmd = [sys.executable, str(ROOT / "BALL.py"), command_map[script]] + args
    print(f"\n=== {script} {' '.join(args)} ===", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--replicates", type=int, default=10)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--t", type=int, default=84)
    p.add_argument("--quick", action="store_true", help="Small-scale draft run.")
    p.add_argument("--tables-only", action="store_true")
    args = p.parse_args()

    if args.quick:
        reps, n, t = "2", "150", "42"
        model_args = [
            "--members", "1",
            "--teacher-epochs", "1",
            "--student-epochs", "1",
            "--d-model", "16",
            "--n-layers", "1",
            "--batch-size", "16",
        ]
    else:
        reps, n, t = str(args.replicates), str(args.n), str(args.t)
        model_args = []

    run("run_ball_pipeline.py", ["--replicates", reps, "--n", n, "--t", t, "--base-seed", "21000"] + model_args)
    run("run_stress_panels.py", ["--replicates", reps, "--n", n, "--t", t, "--base-seed", "90000"] + model_args)
    run("make_tables.py", ["--which", "all"])
    if args.tables_only:
        return
    run("run_fig2_recoverability.py", ["--replicates", reps, "--n", n, "--t", t, "--base-seed", "50000"] + model_args)
    run("run_fig3_calibration.py", ["--replicates", reps, "--n", n, "--t", t, "--base-seed", "60000"] + model_args)
    run("run_fig4_identifiability.py", ["--replicates", reps, "--n", n, "--t", t, "--base-seed", "70000"] + model_args)
    print("\nAll simulation tables and figures written to", HERE / "outputs")


if __name__ == "__main__":
    main()
'''

# --- empirical.assemble_rdoc_llm (empirical/assemble_rdoc_llm.py) ---
SRC_EMPIRICAL_ASSEMBLE_RDOC_LLM = r'''
"""Assemble empirical RDoC scores from raw/RDoC_LLM_scorer.csv.

The empirical example uses the LLM-constructed RDoC feature file as the sole
RDoC source. The raw file is a long note-field table with two independent LLM
score sets per domain. This command keeps rows for patients in the empirical
rTMS session cohort, averages the two model scores for each domain, standardizes
the six domains across the retained raw rows, and writes rdoc_scores_llm.csv in
the column convention consumed by empirical-build-rdoc-proxy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATA = ROOT / "data"
OUT = ROOT / "derived"
RAW_SCORES = REPO / "raw" / "RDoC_LLM_scorer.csv"
SESSION_DATA = DATA / "rtms_paper_analytic_sessions.csv"

DOMAIN_SOURCES = {
    "rdoc_negative_valence": (
        "negative_valence_systems_gemma-4-31B-it",
        "negative_valence_systems_Qwen3-Next-80B-A3B-Instruct",
    ),
    "rdoc_positive_valence": (
        "positive_valence_systems_gemma-4-31B-it",
        "positive_valence_systems_Qwen3-Next-80B-A3B-Instruct",
    ),
    "rdoc_cognitive_systems": (
        "cognitive_systems_gemma-4-31B-it",
        "cognitive_systems_Qwen3-Next-80B-A3B-Instruct",
    ),
    "rdoc_social_processes": (
        "social_processes_gemma-4-31B-it",
        "social_processes_Qwen3-Next-80B-A3B-Instruct",
    ),
    "rdoc_arousal_regulatory": (
        "arousal_and_regulatory_systems_gemma-4-31B-it",
        "arousal_and_regulatory_systems_Qwen3-Next-80B-A3B-Instruct",
    ),
    "rdoc_sensorimotor": (
        "sensorimotor_systems_gemma-4-31B-it",
        "sensorimotor_systems_Qwen3-Next-80B-A3B-Instruct",
    ),
}
COLS = list(DOMAIN_SOURCES.keys())
EXCLUDED_FIELD_NAMES = {"dx", "dx_diagnosis"}


def _to_datetime(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce")


def _excluded_field_mask(field_names: pd.Series) -> pd.Series:
    norm = field_names.fillna("").astype(str).str.strip().str.lower()
    norm = norm.str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    return norm.isin(EXCLUDED_FIELD_NAMES) | norm.str.contains("diagnosis", na=False)


def _session_first_dates() -> pd.Series:
    sess = pd.read_csv(SESSION_DATA, usecols=["PatientFID", "ServiceDate"], low_memory=False)
    sess["ServiceDate"] = _to_datetime(sess["ServiceDate"])
    sess = sess.dropna(subset=["PatientFID", "ServiceDate"]).copy()
    sess["PatientFID"] = pd.to_numeric(sess["PatientFID"], errors="coerce")
    sess = sess.dropna(subset=["PatientFID"])
    sess["PatientFID"] = sess["PatientFID"].astype(int)
    return sess.groupby("PatientFID")["ServiceDate"].min()


def main() -> None:
    if not RAW_SCORES.exists():
        raise FileNotFoundError(f"{RAW_SCORES} not found.")
    OUT.mkdir(parents=True, exist_ok=True)

    first_session = _session_first_dates()
    empirical_patients = set(first_session.index.astype(int))
    source_cols = sorted({c for pair in DOMAIN_SOURCES.values() for c in pair})
    usecols = [
        "FieldValue_UID", "FieldFID", "FieldName", "PatientFID", "AppointmentFID",
        "ServiceDate", "note_length", "num_words", "ApptTypes",
    ] + source_cols
    raw = pd.read_csv(RAW_SCORES, usecols=usecols, low_memory=False)
    raw["PatientFID"] = pd.to_numeric(raw["PatientFID"], errors="coerce")
    raw = raw[raw["PatientFID"].isin(empirical_patients)].copy()
    raw["PatientFID"] = raw["PatientFID"].astype(int)
    rows_after_patient_filter = len(raw)
    excluded_fields = raw.loc[_excluded_field_mask(raw["FieldName"]), "FieldName"].value_counts(dropna=False).to_dict()
    raw = raw.loc[~_excluded_field_mask(raw["FieldName"])].copy()
    rows_after_field_filter = len(raw)
    duplicate_fieldvalue_rows = int(raw.duplicated("FieldValue_UID").sum())
    raw = raw.drop_duplicates("FieldValue_UID")
    rows_after_dedup = len(raw)
    raw["ServiceDate"] = _to_datetime(raw["ServiceDate"])
    invalid_service_date_rows = int(raw["ServiceDate"].isna().sum())
    raw = raw.dropna(subset=["ServiceDate"]).copy()
    rows_after_date_filter = len(raw)
    raw["first_session_date"] = raw["PatientFID"].map(first_session)
    raw["day"] = (raw["ServiceDate"] - raw["first_session_date"]).dt.days.astype(int)

    for out_col, model_cols in DOMAIN_SOURCES.items():
        model_scores = raw[list(model_cols)].apply(pd.to_numeric, errors="coerce").clip(0.0, 1.0)
        raw[out_col] = model_scores.mean(axis=1)
    raw = raw.dropna(subset=COLS).copy()

    means = raw[COLS].mean()
    sds = raw[COLS].std().replace(0.0, 1.0)
    std = (raw[COLS] - means) / sds
    out = pd.DataFrame({
        "id": raw["PatientFID"].astype(int),
        "AppointmentFID": pd.to_numeric(raw["AppointmentFID"], errors="coerce").astype("Int64"),
        "ServiceDate": raw["ServiceDate"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "day": raw["day"].astype(int),
        "note_tier": "raw_llm_field",
        "note_char_count": pd.to_numeric(raw["note_length"], errors="coerce").fillna(0).astype(int),
        "FieldValue_UID": raw["FieldValue_UID"],
        "FieldName": raw["FieldName"],
        "ApptTypes": raw["ApptTypes"],
    })
    out = pd.concat([out, std], axis=1)
    out.to_csv(OUT / "rdoc_scores_llm.csv", index=False)
    print(f"wrote {OUT/'rdoc_scores_llm.csv'} ({len(out)} rows)")

    manifest = {
        "source": str(RAW_SCORES.relative_to(REPO)),
        "empirical_session_source": str(SESSION_DATA.relative_to(REPO)),
        "sole_rdoc_source_for_empirical_example": True,
        "same_day_records_excluded": True,
        "excluded_field_names": "normalized FieldName is dx/dx_diagnosis or contains diagnosis",
        "excluded_field_name_counts": {str(k): int(v) for k, v in excluded_fields.items()},
        "raw_rows_read_for_empirical_patients": int(rows_after_patient_filter),
        "rows_after_field_exclusion": int(rows_after_field_filter),
        "duplicate_fieldvalue_uid_rows_dropped": duplicate_fieldvalue_rows,
        "rows_after_dedup": int(rows_after_dedup),
        "invalid_service_date_rows_dropped": invalid_service_date_rows,
        "rows_after_date_filter": int(rows_after_date_filter),
        "rows_written": int(len(out)),
        "patients_written": int(out["id"].nunique()),
        "patient_day_rows": int(out[["id", "day"]].drop_duplicates().shape[0]),
        "service_date_range": [str(raw["ServiceDate"].min()), str(raw["ServiceDate"].max())],
        "day_range_relative_to_first_session": [int(raw["day"].min()), int(raw["day"].max())],
        "raw_score_scale": "0, 0.5, 1 per model-domain field",
        "model_ensemble": "mean of Gemma and Qwen score columns for each domain",
        "standardization": "per-domain z-score across retained empirical-patient raw rows",
        "domains": COLS,
        "source_columns": {domain: list(cols) for domain, cols in DOMAIN_SOURCES.items()},
    }
    (OUT / "rdoc_scores_llm_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"patients with raw LLM RDoC scores: {manifest['patients_written']}")
    print(f"patient-days with raw LLM RDoC scores: {manifest['patient_day_rows']}")
    print(f"wrote {OUT/'rdoc_scores_llm_manifest.json'}")


if __name__ == "__main__":
    main()
'''

# --- empirical.build_rdoc_proxy (empirical/build_rdoc_proxy.py) ---
SRC_EMPIRICAL_BUILD_RDOC_PROXY = r'''
"""Carry RDoC note scores forward to a per-session six-dimensional proxy B(t).

The RDoC scores live on the patient day-axis at the days notes were written.
The proxy B(t) at a session day is the most recent note's RDoC profile strictly
before that day. This is a last-observation-carried-forward on the day-axis.
Session days before a patient's first scored note receive the corpus mean
(zero on the standardized scale) and are flagged as having no prior note.

Output is empirical/derived/sessions_with_rdoc.csv with columns id, session,
day, B0..B5, rdoc_days_stale (days since the note that supplied the proxy), and
rdoc_observed (whether any prior note existed). B0..B5 map to the six RDoC
domains in the order of RDOC_TERMS in rdoc_score.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "derived"
SESSION_DATA = DATA / "rtms_paper_analytic_sessions.csv"

RDOC_COLS = [
    "rdoc_negative_valence", "rdoc_positive_valence", "rdoc_cognitive_systems",
    "rdoc_social_processes", "rdoc_arousal_regulatory", "rdoc_sensorimotor",
]


def _session_axis() -> pd.DataFrame:
    sess = pd.read_csv(SESSION_DATA, low_memory=False)
    sess["ServiceDate"] = pd.to_datetime(sess["ServiceDate"], errors="coerce")
    sess = sess.dropna(subset=["ServiceDate"]).sort_values(["PatientFID", "ServiceDate"]).reset_index(drop=True)
    sess["session"] = sess.groupby("PatientFID").cumcount()
    first = sess.groupby("PatientFID")["ServiceDate"].transform("min")
    sess["day"] = (sess["ServiceDate"] - first).dt.days.astype(int)
    sess["PatientFID"] = sess["PatientFID"].astype(int)
    return sess[["PatientFID", "session", "day"]].rename(columns={"PatientFID": "id"})


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="rdoc_scores_llm.csv",
                    help="Raw LLM RDoC score file in derived/. Must be generated from raw/RDoC_LLM_scorer.csv.")
    a = ap.parse_args()
    if a.scores != "rdoc_scores_llm.csv":
        raise ValueError("The empirical RDoC proxy must use rdoc_scores_llm.csv generated from raw/RDoC_LLM_scorer.csv.")
    print(f"using scores: {a.scores}")
    score_path = OUT / a.scores
    if not score_path.exists():
        raise FileNotFoundError(f"{score_path} not found. Run empirical-assemble-rdoc-llm first.")
    scores = pd.read_csv(score_path)
    scores = scores.dropna(subset=["day", "id"]).copy()
    scores["day"] = scores["day"].astype(int)
    scores["id"] = scores["id"].astype(int)
    # Average scores of notes written on the same patient-day.
    notes = scores.groupby(["id", "day"])[RDOC_COLS].mean().reset_index()
    notes["day"] = notes["day"].astype(int)
    notes["note_day"] = notes["day"]
    # merge_asof requires both frames globally sorted by the on key.
    notes = notes.sort_values("day").reset_index(drop=True)

    sess = _session_axis().sort_values("day").reset_index(drop=True)

    # Per-patient backward as-of merge: each session day takes the most recent
    # note strictly before it. Same-day order is unavailable in the extract.
    merged = pd.merge_asof(
        sess, notes, on="day", by="id", direction="backward", suffixes=("", "_note"),
        allow_exact_matches=False,
    )
    merged["rdoc_observed"] = merged["note_day"].notna()
    merged["rdoc_days_stale"] = (merged["day"] - merged["note_day"]).astype("float")
    for j, col in enumerate(RDOC_COLS):
        merged[f"B{j}"] = merged[col].fillna(0.0)

    keep = ["id", "session", "day", "rdoc_observed", "rdoc_days_stale"] + [f"B{j}" for j in range(6)]
    out = merged[keep].copy()
    out.to_csv(OUT / "sessions_with_rdoc.csv", index=False)

    # Within-patient variability of the proxy, to quantify how slowly it varies.
    b_cols = [f"B{j}" for j in range(6)]
    within_sd = out[out["rdoc_observed"]].groupby("id")[b_cols].std().mean().mean()
    between_sd = out[out["rdoc_observed"]].groupby("id")[b_cols].mean().std().mean()
    manifest = {
        "n_sessions": int(len(out)),
        "session_days_with_proxy": int(out["rdoc_observed"].sum()),
        "proxy_coverage": float(out["rdoc_observed"].mean()),
        "patients_with_any_proxy": int(out.loc[out["rdoc_observed"], "id"].nunique()),
        "median_days_stale": float(out.loc[out["rdoc_observed"], "rdoc_days_stale"].median()),
        "within_patient_proxy_sd_mean": float(within_sd),
        "between_patient_proxy_sd_mean": float(between_sd),
        "B_columns": {f"B{j}": RDOC_COLS[j] for j in range(6)},
        "score_source": "empirical/derived/rdoc_scores_llm.csv",
        "score_source_manifest": "empirical/derived/rdoc_scores_llm_manifest.json",
        "raw_score_source": "raw/RDoC_LLM_scorer.csv",
        "empirical_session_source": "empirical/data/rtms_paper_analytic_sessions.csv",
        "sole_rdoc_source_for_empirical_example": True,
        "carry_forward": "most recent note RDoC profile strictly before each session day; corpus mean (0) before first note",
        "finding": "RDoC proxy uses only the LLM-constructed raw RDoC scorer output; empirical direct-transition coefficients remain descriptive and should be reported with the permutation control rather than interpreted as validated recovery of a true beta.",
    }
    (OUT / "rdoc_proxy_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"sessions: {len(out)}  proxy coverage: {manifest['proxy_coverage']:.3f}  "
          f"patients with proxy: {manifest['patients_with_any_proxy']}")
    print(f"median days stale: {manifest['median_days_stale']:.0f}")
    print(f"within-patient proxy SD {within_sd:.3f} vs between-patient SD {between_sd:.3f}")
    print(f"wrote {OUT/'sessions_with_rdoc.csv'}, {OUT/'rdoc_proxy_manifest.json'}")


if __name__ == "__main__":
    main()
'''

# --- empirical.build_sparse_anchors (empirical/build_sparse_anchors.py) ---
SRC_EMPIRICAL_BUILD_SPARSE_ANCHORS = r'''
"""Build the BALL empirical illustration dataset from the rTMS cohort.

Reads ONLY the isolated working copy in empirical/data/ (never the DBH AMD
source). Produces sparse-anchor datasets mapped to the BALL general
simulation objects:

  Y1/Y2/Y3 sparse anchors <- PHQ-9, GAD-7, BDI. Each scale is genuinely measured
                            at a high per-session rate in this cohort (PHQ-9
                            ~85%, BDI ~72%, GAD-7 ~42%), so each is artificially
                            sparsified to a target cadence. The genuine
                            measurements NOT chosen as anchors are retained as a
                            held-out recovery-validation target -- a quasi-latent
                            the empirical study can actually score recovery
                            against, which a truly unobservable latent cannot.
  a_im treatment events   <- rTMS protocol action_id per session.
  X(t) daily state        <- session-level covariates.
  B(t) RDoC proxy         <- notes-derived McCoy/Perlis LLM RDoC profile
                            (6-dim), used in the empirical direct-transition
                            coefficient analysis.

Two modes:
  (default)   build one anchor set at --cadence-days (weekly by default).
  --sweep     build anchor sets at several cadences for the recovery-vs-sparsity
              sensitivity analysis; each cadence held-out eval set lets us trace
              latent-recovery quality as anchors get sparser.

Outputs to empirical/derived/:
  anchors_sparse[_<cadence>d].csv   id, anchor, session, date, value,
                                    window_start_day, window_end_day,
                                    role in {anchor, heldout_eval}, cadence_days
  sessions.csv                      id, session, date, action_id, covariates
  build_manifest.json               counts per cadence, provenance, rules
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "derived"
OUT.mkdir(parents=True, exist_ok=True)
SESSION_DATA = DATA / "rtms_paper_analytic_sessions.csv"

# Recall windows (days) per scale, mirroring the simulation anchor defaults.
RECALL_WINDOW_DAYS = {"PHQ9": 14, "GAD7": 7, "BDI": 7}
SCALES = ("PHQ9", "GAD7", "BDI")
VALUE_RANGE = {"PHQ9": [0, 27], "GAD7": [0, 21], "BDI": [0, 63]}
DEFAULT_CADENCE = 7
SWEEP_CADENCES = (7, 14, 21, 28)  # weekly -> 4-weekly anchor spacing


def _load() -> pd.DataFrame:
    df = pd.read_csv(SESSION_DATA, low_memory=False)
    df["ServiceDate"] = pd.to_datetime(df["ServiceDate"], errors="coerce")
    df = df.dropna(subset=["ServiceDate"]).copy()
    df = df.sort_values(["PatientFID", "ServiceDate"]).reset_index(drop=True)
    df["session"] = df.groupby("PatientFID").cumcount()
    first = df.groupby("PatientFID")["ServiceDate"].transform("min")
    df["day"] = (df["ServiceDate"] - first).dt.days.astype(int)
    return df


def _genuine_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Per-session masks where each scale was genuinely (not imputed) measured."""

    return {
        "PHQ9": (df.get("phq9_is_imputed", 0) == 0)
        & (df.get("phq9_is_interpolated", 0) == 0)
        & df["phq9"].notna(),
        "GAD7": df.get("gad_measured", 0) == 1,
        "BDI": df.get("bdi_measured", 0) == 1,
    }


def _value(df: pd.DataFrame, scale: str) -> pd.Series:
    return {"PHQ9": df["phq9"], "GAD7": df["gad_filled"], "BDI": df["bdi_filled"]}[scale]


def _sparsify(df: pd.DataFrame, genuine: pd.Series, cadence_days: int) -> pd.Series:
    """First genuine measurement in each within-patient cadence-day bin = anchor.

    Returns a role Series over the genuine rows ('anchor' / 'heldout_eval');
    non-genuine rows are NaN. The held-out genuine measurements at finer
    resolution are the empirical recovery-validation target.
    """

    role = pd.Series(np.nan, index=df.index, dtype=object)
    g = df[genuine].copy()
    g["bin"] = g["day"] // int(cadence_days)
    first_in_bin = g.groupby(["PatientFID", "bin"]).head(1).index
    role.loc[g.index] = "heldout_eval"
    role.loc[first_in_bin] = "anchor"
    return role


def _build_anchor_set(df: pd.DataFrame, masks: dict[str, pd.Series], cadence_days: int) -> pd.DataFrame:
    rows: list[dict] = []
    for scale in SCALES:
        genuine = masks[scale]
        role = _sparsify(df, genuine, cadence_days)
        vals = _value(df, scale)
        win = RECALL_WINDOW_DAYS[scale]
        for idx in df.index[genuine]:
            v = vals.loc[idx]
            if not np.isfinite(v):
                continue
            row = df.loc[idx]
            rows.append(
                {
                    "id": int(row["PatientFID"]),
                    "anchor": scale,
                    "session": int(row["session"]),
                    "date": row["ServiceDate"].date().isoformat(),
                    "day": int(row["day"]),
                    "value": float(v),
                    "window_start_day": int(row["day"]) - win + 1,
                    "window_end_day": int(row["day"]),
                    "role": role.loc[idx],
                    "cadence_days": int(cadence_days),
                }
            )
    return pd.DataFrame(rows).sort_values(["id", "day", "anchor"]).reset_index(drop=True)


def _anchor_counts(anchors: pd.DataFrame, n_patients: int) -> dict:
    out = {}
    for scale in SCALES:
        s = anchors[anchors["anchor"] == scale]
        n_anchor = int((s["role"] == "anchor").sum())
        out[scale] = {
            "anchor": n_anchor,
            "heldout_eval": int((s["role"] == "heldout_eval").sum()),
            "mean_anchors_per_patient": round(n_anchor / max(n_patients, 1), 2),
        }
    return out


def _write_sessions(df: pd.DataFrame) -> None:
    cov_cols = [
        c for c in [
            "PatientFID", "session", "day", "ServiceDate", "action_id", "prev_action_id",
            "facility_id", "phq9_momentum", "phq9_change_from_baseline", "phq9_baseline",
            "global_session", "max_session", "gad_measured", "bdi_measured",
        ] if c in df.columns
    ]
    sessions = df[cov_cols].rename(columns={"PatientFID": "id", "ServiceDate": "date"}).copy()
    sessions["date"] = sessions["date"].dt.date.astype(str)
    sessions.to_csv(OUT / "sessions.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BALL empirical sparse anchors.")
    parser.add_argument("--cadence-days", type=int, default=DEFAULT_CADENCE)
    parser.add_argument("--sweep", action="store_true", help="Build all SWEEP_CADENCES for the sparsity sensitivity.")
    args = parser.parse_args()

    df = _load()
    masks = _genuine_masks(df)
    n_patients = int(df["PatientFID"].nunique())
    _write_sessions(df)

    cadences = SWEEP_CADENCES if args.sweep else (args.cadence_days,)
    manifest = {
        "source": "isolated copy of DBH AMD/processed/analytic_dataset_with_recommendations.xlsx sessions sheet",
        "n_patients": n_patients,
        "n_sessions": int(len(df)),
        "date_range": [str(df["ServiceDate"].min()), str(df["ServiceDate"].max())],
        "sessions_per_patient_median": float(df.groupby("PatientFID").size().median()),
        "sparsification_rule": "first genuine measurement per within-patient cadence-day bin = anchor; rest = heldout_eval",
        "recall_windows_days": RECALL_WINDOW_DAYS,
        "value_ranges": VALUE_RANGE,
        "all_anchors_sparsified": True,
        "notes_rdoc_proxy": "McCoy/Perlis-style LLM RDoC 6-dim profile from clinical notes -> direct-transition proxy B(t)",
        "cadences": {},
    }
    for cadence in cadences:
        anchors = _build_anchor_set(df, masks, cadence)
        suffix = "" if (not args.sweep and cadence == DEFAULT_CADENCE) else f"_{cadence}d"
        anchors.to_csv(OUT / f"anchors_sparse{suffix}.csv", index=False)
        manifest["cadences"][f"{cadence}d"] = _anchor_counts(anchors, n_patients)
        print(f"cadence={cadence}d -> " + ", ".join(
            f"{s}: {manifest['cadences'][f'{cadence}d'][s]['anchor']} anchors / "
            f"{manifest['cadences'][f'{cadence}d'][s]['heldout_eval']} held-out" for s in SCALES))

    (OUT / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"patients={n_patients} sessions={len(df)} -> wrote empirical/derived/")


if __name__ == "__main__":
    main()
'''

# --- empirical.build_comorbidities (empirical/build_comorbidities.py) ---
SRC_EMPIRICAL_BUILD_COMORBIDITIES = r'''
"""Build strictly prior, session-level psychiatric diagnosis features.

The input diagnosis text and the patient-level output are protected data. This
command therefore refuses to write inside a Git working tree. It emits only
numeric indicators and counts. It never writes diagnosis strings, ICD codes,
calendar dates, appointment identifiers, or raw EHR field values.

For a session on calendar date d, a diagnosis record contributes only when its
recorded ServiceDate is strictly earlier than d. Same-day records are excluded
because the available extracts do not contain a reliable within-day timestamp.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DIAGNOSIS_FIELDS = ("DX_Diagnosis", "MH_diagnosis")
CATEGORY_NAMES = (
    "depression",
    "anxiety",
    "ptsd",
    "bipolar",
    "ocd",
    "adhd",
    "psychotic_spectrum",
    "substance_use",
    "eating_disorder",
    "personality_disorder",
    "autism_spectrum",
    "sleep_disorder",
)
FEATURE_COLUMNS = (
    "dx_history_available",
    "dx_comorbidity_count",
    *(f"dx_{name}" for name in CATEGORY_NAMES),
)

_CODE_RE = re.compile(r"(?i)(?<![A-Z0-9])([A-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?)(?![A-Z0-9])")
_TEXT_PATTERNS = {
    "depression": re.compile(r"(?i)\b(depress(?:ion|ive)?|major depressive|dysthymi\w*)\b"),
    "anxiety": re.compile(r"(?i)\b(anxiety|anxious|generalized anxiety|panic disorder|social phobia)\b"),
    "ptsd": re.compile(r"(?i)\b(ptsd|post[- ]traumatic stress)\b"),
    "bipolar": re.compile(r"(?i)\b(bipolar|manic depression|mania)\b"),
    "ocd": re.compile(r"(?i)\b(ocd|obsessive[- ]compulsive)\b"),
    "adhd": re.compile(r"(?i)\b(adhd|attention[- ]deficit|attention deficit)\b"),
    "psychotic_spectrum": re.compile(
        r"(?i)\b(schizophren\w*|schizoaffective|psychotic disorder|unspecified psychosis)\b"
    ),
    "substance_use": re.compile(
        r"(?i)\b(substance use|alcohol use disorder|opioid use disorder|cocaine use disorder|"
        r"cannabis use disorder|stimulant use disorder|addiction)\b"
    ),
    "eating_disorder": re.compile(r"(?i)\b(eating disorder|anorexia|bulimia|binge eating)\b"),
    "personality_disorder": re.compile(
        r"(?i)\b(personality disorder|borderline personality|obsessive[- ]compulsive personality)\b"
    ),
    "autism_spectrum": re.compile(r"(?i)\b(autism|autistic|asperger)\b"),
    "sleep_disorder": re.compile(r"(?i)\b(insomnia|sleep disorder|sleep apnea)\b"),
}


def _git_ancestor(path: Path) -> Path | None:
    """Return the nearest Git working tree containing path, including worktrees."""

    resolved = path.expanduser().resolve()
    start = resolved if resolved.is_dir() else resolved.parent
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _assert_protected_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    git_root = _git_ancestor(resolved)
    if git_root is not None:
        raise ValueError(
            "Refusing to write patient-level diagnosis features inside a Git working tree. "
            "Choose a protected output directory outside the repository."
        )
    if resolved.suffix.lower() != ".csv":
        raise ValueError("Protected comorbidity output must be a .csv file.")
    return resolved


def _session_axis(path: Path) -> pd.DataFrame:
    sessions = pd.read_csv(path, low_memory=False)
    required = {"PatientFID", "ServiceDate"}
    missing = required.difference(sessions.columns)
    if missing:
        raise ValueError(f"Session file is missing required columns: {sorted(missing)}")
    sessions = sessions[["PatientFID", "ServiceDate"]].copy()
    sessions["PatientFID"] = pd.to_numeric(sessions["PatientFID"], errors="coerce")
    sessions["ServiceDate"] = pd.to_datetime(sessions["ServiceDate"], errors="coerce").dt.normalize()
    sessions = sessions.dropna(subset=["PatientFID", "ServiceDate"]).copy()
    sessions["PatientFID"] = sessions["PatientFID"].astype(int)
    sessions["_source_order"] = np.arange(len(sessions), dtype=int)
    sessions = sessions.sort_values(
        ["PatientFID", "ServiceDate", "_source_order"], kind="mergesort"
    ).reset_index(drop=True)
    sessions["session"] = sessions.groupby("PatientFID", sort=False).cumcount().astype(int)
    first = sessions.groupby("PatientFID", sort=False)["ServiceDate"].transform("min")
    sessions["day"] = (sessions["ServiceDate"] - first).dt.days.astype(int)
    return sessions[["PatientFID", "session", "day", "ServiceDate"]]


def _icd_categories(code: str) -> set[str]:
    compact = code.upper().replace(".", "")
    stem3 = compact[:3]
    out: set[str] = set()
    if stem3 in {"F32", "F33"} or compact.startswith(("F341", "F53", "F0631")):
        out.add("depression")
    if stem3 in {"F40", "F41"} or compact.startswith("F064"):
        out.add("anxiety")
    if compact.startswith("F431"):
        out.add("ptsd")
    if stem3 in {"F30", "F31"} or compact.startswith("F340"):
        out.add("bipolar")
    if stem3 == "F42":
        out.add("ocd")
    if stem3 == "F90":
        out.add("adhd")
    if len(stem3) == 3 and stem3[0] == "F" and stem3[1:].isdigit() and 20 <= int(stem3[1:]) <= 29:
        out.add("psychotic_spectrum")
    if len(stem3) == 3 and stem3[0] == "F" and stem3[1:].isdigit() and 10 <= int(stem3[1:]) <= 19:
        out.add("substance_use")
    if stem3 == "F50":
        out.add("eating_disorder")
    if stem3 in {"F60", "F61"}:
        out.add("personality_disorder")
    if stem3 == "F84":
        out.add("autism_spectrum")
    if stem3 in {"F51", "G47"}:
        out.add("sleep_disorder")
    return out


def classify_diagnosis(value: object) -> set[str]:
    """Map one protected diagnosis string to broad, nonexclusive categories."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    # Split exports that concatenate adjacent codes, such as F33.2F41.1.
    code_text = re.sub(r"(?<=[0-9])(?=[A-Z][0-9]{2})", " ", text.upper())
    out: set[str] = set()
    for code in _CODE_RE.findall(code_text):
        out.update(_icd_categories(code))
    for name, pattern in _TEXT_PATTERNS.items():
        if pattern.search(text):
            out.add(name)
    return out


def _read_diagnosis_events(raw_dir: Path, patient_ids: set[int]) -> pd.DataFrame:
    files = sorted(raw_dir.glob("amd*.sas7bdat"))
    if not files:
        raise FileNotFoundError("No amd*.sas7bdat files were found in the protected raw directory.")
    frames: list[pd.DataFrame] = []
    keep = ["FieldValue_UID", "FieldName", "PatientFID", "ServiceDate", "Value"]
    for path in files:
        raw = pd.read_sas(path, format="sas7bdat", encoding="latin1")
        missing = set(keep).difference(raw.columns)
        if missing:
            raise ValueError(f"A raw SAS part is missing required columns: {sorted(missing)}")
        part = raw.loc[raw["FieldName"].isin(DIAGNOSIS_FIELDS), keep].copy()
        if part.empty:
            continue
        part["PatientFID"] = pd.to_numeric(part["PatientFID"], errors="coerce")
        part = part[part["PatientFID"].isin(patient_ids)].copy()
        if part.empty:
            continue
        part["PatientFID"] = part["PatientFID"].astype(int)
        part["ServiceDate"] = pd.to_datetime(part["ServiceDate"], errors="coerce").dt.normalize()
        part = part.dropna(subset=["ServiceDate", "Value"])
        frames.append(part)
    if not frames:
        return pd.DataFrame(columns=keep)
    events = pd.concat(frames, ignore_index=True)
    # The raw export can repeat records across SAS parts. Prefer the stable field
    # value identifier, then remove exact duplicate fallbacks.
    uid_present = events["FieldValue_UID"].notna()
    with_uid = events[uid_present].drop_duplicates("FieldValue_UID", keep="first")
    without_uid = events[~uid_present].drop_duplicates(
        ["PatientFID", "ServiceDate", "FieldName", "Value"], keep="first"
    )
    events = pd.concat([with_uid, without_uid], ignore_index=True)
    events["categories"] = events["Value"].map(classify_diagnosis)
    return events.sort_values(["PatientFID", "ServiceDate"], kind="mergesort").reset_index(drop=True)


def construct_session_features(sessions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Construct cumulative features using only records strictly before a session."""

    event_groups = {
        int(pid): grp.sort_values("ServiceDate", kind="mergesort").reset_index(drop=True)
        for pid, grp in events.groupby("PatientFID", sort=False)
    }
    rows: list[dict] = []
    for pid, sess_group in sessions.groupby("PatientFID", sort=False):
        pid = int(pid)
        diagnoses = event_groups.get(pid, pd.DataFrame())
        event_pos = 0
        active_categories: set[str] = set()
        record_count = 0
        for session in sess_group.sort_values("session", kind="mergesort").itertuples(index=False):
            # Strict inequality is intentional. Same-day records could have been
            # entered after treatment and are therefore not available as features.
            while event_pos < len(diagnoses) and diagnoses.iloc[event_pos]["ServiceDate"] < session.ServiceDate:
                event = diagnoses.iloc[event_pos]
                active_categories.update(event["categories"])
                record_count += 1
                event_pos += 1
            observed = record_count > 0
            comorbidity_count = len(active_categories.difference({"depression"}))
            row = {
                "id": pid,
                "session": int(session.session),
                "day": int(session.day),
                "dx_history_available": int(observed),
                "dx_comorbidity_count": int(comorbidity_count),
            }
            for category in CATEGORY_NAMES:
                row[f"dx_{category}"] = int(category in active_categories)
            rows.append(row)
    out = pd.DataFrame(rows)
    expected = ["id", "session", "day", *FEATURE_COLUMNS]
    out = out[expected].sort_values(["id", "session"], kind="mergesort").reset_index(drop=True)
    forbidden = {"Value", "ServiceDate", "FieldValue_UID", "PatientFID", "AppointmentFID"}
    if forbidden.intersection(out.columns):
        raise RuntimeError("Protected raw fields reached the numeric feature output.")
    return out


def _aggregate_manifest(features: pd.DataFrame, raw_parts: int) -> dict:
    observed = features["dx_history_available"] == 1
    category_prevalence = {
        category: {
            "sessions": int(features[f"dx_{category}"].sum()),
            "patients": int(features.loc[features[f"dx_{category}"] == 1, "id"].nunique()),
        }
        for category in CATEGORY_NAMES
    }
    return {
        "rule": "diagnosis ServiceDate strictly earlier than session ServiceDate",
        "same_day_records_excluded": True,
        "carry_forward": "cumulative after first strictly prior diagnosis record",
        "raw_fields_used": list(DIAGNOSIS_FIELDS),
        "raw_sas_parts_read": int(raw_parts),
        "n_sessions": int(len(features)),
        "n_patients": int(features["id"].nunique()),
        "sessions_with_prior_diagnosis_list": int(observed.sum()),
        "patients_with_prior_diagnosis_list": int(features.loc[observed, "id"].nunique()),
        "feature_columns": list(FEATURE_COLUMNS),
        "category_prevalence": category_prevalence,
        "contains_raw_text": False,
        "contains_calendar_dates": False,
        "contains_appointment_identifiers": False,
        "patient_level_output_is_protected": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strictly prior BALL comorbidity features.")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Protected directory containing amd*.sas7bdat.")
    parser.add_argument("--sessions", type=Path, required=True, help="Protected rTMS session cohort CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Protected output CSV outside every Git working tree.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional protected aggregate JSON manifest.")
    args = parser.parse_args()

    output = _assert_protected_output(args.output)
    manifest_path = args.manifest.expanduser().resolve() if args.manifest else None
    if manifest_path is not None:
        if _git_ancestor(manifest_path) is not None:
            raise ValueError("Refusing to write the comorbidity manifest inside a Git working tree.")
        if manifest_path.suffix.lower() != ".json":
            raise ValueError("Aggregate manifest must be a .json file.")
    sessions = _session_axis(args.sessions.expanduser().resolve())
    patient_ids = set(sessions["PatientFID"].astype(int))
    events = _read_diagnosis_events(args.raw_dir.expanduser().resolve(), patient_ids)
    features = construct_session_features(sessions, events)

    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)
    if args.manifest:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = _aggregate_manifest(features, len(list(args.raw_dir.glob("amd*.sas7bdat"))))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"built {len(features)} protected session feature rows for "
        f"{features['id'].nunique()} patients; no raw diagnosis text was written"
    )


if __name__ == "__main__":
    main()
'''

# --- empirical.fit_empirical (empirical/fit_empirical.py) ---
SRC_EMPIRICAL_FIT_EMPIRICAL = r'''
"""Empirical BALL held-out recovery analysis on the rTMS cohort.

Implements the deployable empirical estimator. For each patient it recovers a
recall-windowed latent severity trajectory from the retained sparse anchors and
a slow smoothing prior, then predicts the held-out genuine measurements. The
prediction of a held-out anchor is the mean of the recovered latent over that
anchor's recall window. Recovery is scored against the held-out measurements and
compared with linear interpolation, last observation carried forward, and the
anchor-only windowed mean on the identical held-out set. Errors are stratified
by days to the nearest retained anchor. Split conformal intervals are calibrated
on a held-out-anchor calibration fold and evaluated for coverage. The whole
analysis is repeated across the anchor-cadence sweep.

Two channels are modeled. The depression channel is anchored by PHQ-9 and BDI.
The anxiety channel is anchored by GAD-7. Scales are placed on a common
standardized severity scale using cohort statistics computed on retained
anchors only, so the held-out measurements never inform the scaling.

The same recovered trajectories are also used for a descriptive empirical
direct-RDoC transition analysis. That analysis regresses adjacent-session latent
change on the six note-derived RDoC proxy dimensions, adjusted for current
latent level, inter-session gap, and proxy staleness, and reports a permutation
control. Because the empirical data do not contain a true transition beta, this
is an observable association and predictive-increment check rather than a
parameter-recovery claim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.sparse.linalg import lsqr

from simulations.src.model_utils import AnchorSpec, SimulationConfig, SimulationData, marginal_anchor_sd
from simulations.src.methods.ball_ssm import SSMConfig, fit_ball_ssm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "derived"
# 7-day cadence is structurally all-embargoed: at 7d, every held-out recall
# window overlaps a retained anchor window, so _channel_eval returns no rows.
# It is excluded so the run does not train a model that produces zero output.
CADENCES = [14, 21, 28]
# Channel membership and anchor noise (standardized scale). PHQ-9 is the
# depression reference. GAD-7 is anxiety. BDI joins depression.
CHANNELS = {"depression": ["PHQ9", "BDI"], "anxiety": ["GAD7"]}
SMOOTH_SD = 0.7          # random-walk smoothing scale on the latent (z units)
ANCHOR_SD = 0.6          # anchor measurement noise (z units)
CONFORMAL_ALPHA = 0.05
GAP_BINS = [0, 1, 2, 4, 7, 1000]
GAP_LABELS = ["0-1", "1-2", "2-4", "4-7", "7+"]
RDOC_COLS = [f"B{j}" for j in range(6)]
RDOC_NUISANCE_COLS = ["L_current", "dt", "rdoc_days_stale"]
RDOC_RIDGE = 1.0
RDOC_FOLDS = 5
RDOC_PERMUTATIONS = 200
COMORBIDITY_CATEGORY_COLS = [
    "dx_depression",
    "dx_anxiety",
    "dx_ptsd",
    "dx_bipolar",
    "dx_ocd",
    "dx_adhd",
    "dx_psychotic_spectrum",
    "dx_substance_use",
    "dx_eating_disorder",
    "dx_personality_disorder",
    "dx_autism_spectrum",
    "dx_sleep_disorder",
]
COMORBIDITY_TABLE_COLS = [
    "dx_history_available",
    "dx_comorbidity_count",
    *COMORBIDITY_CATEGORY_COLS,
]
COMORBIDITY_FEATURE_COLS = ["dx_comorbidity_count", *COMORBIDITY_CATEGORY_COLS]
EMPIRICAL_FEATURE_NAMES = ["treatment_session_number", *COMORBIDITY_FEATURE_COLS]

# Markov pattern-mixture comparator (classical statistics arm). A finite mixture
# of linear-Gaussian Markov trajectory classes is pooled across patients within a
# channel, fit by EM on the anchor marginal likelihood, with class membership
# conditioned on each patient's retained-anchor count (the pattern-mixture
# structure). Each patient's latent is the responsibility-weighted posterior mean
# of a level (random walk) plus fast (AR1) state-space smoother. Held-out
# measurements are predicted by the recall-window mean of the recovered latent,
# the same prediction rule used for every other method.
MARKOV_K = 3
MARKOV_STRATA = 3
MARKOV_EM_ITERS = 6
MARKOV_OPT_MAXITER = 15
MARKOV_LEVEL_PRIOR_SD = 5.0
MARKOV_MIN_SCALE = 0.02
MARKOV_MAX_SCALE = 6.0
MARKOV_MIN_PHI = -0.95
MARKOV_MAX_PHI = 0.95
MARKOV_JITTER = 1e-6
MARKOV_MAX_FIT_PATIENTS = 200  # cap patients used for EM parameter fitting (applied to all)
EMPIRICAL_SEED = 41421
FIT_FINAL_FILES = [
    "empirical_recovery_overall.csv",
    "empirical_recovery_by_gap.csv",
    "empirical_calibration.csv",
    "empirical_bootstrap_diff.csv",
    "empirical_rdoc_transition_rows.csv",
    "empirical_rdoc_transition.csv",
    "empirical_rdoc_transition_coefficients.csv",
]


@dataclass
class _MClass:
    s_lev: float
    phi: float
    s_fast: float
    anchor_sd: float


def _m_theta_to_params(theta: np.ndarray) -> _MClass:
    return _MClass(
        s_lev=float(np.clip(math.exp(theta[0]), MARKOV_MIN_SCALE, MARKOV_MAX_SCALE)),
        phi=float(np.clip(np.tanh(theta[1]), MARKOV_MIN_PHI, MARKOV_MAX_PHI)),
        s_fast=float(np.clip(math.exp(theta[2]), MARKOV_MIN_SCALE, MARKOV_MAX_SCALE)),
        anchor_sd=float(np.clip(math.exp(theta[3]), MARKOV_MIN_SCALE, MARKOV_MAX_SCALE)),
    )


def _m_params_to_theta(p: _MClass) -> np.ndarray:
    return np.array([
        math.log(max(p.s_lev, 1e-3)),
        math.atanh(np.clip(p.phi, -0.95, 0.95)),
        math.log(max(p.s_fast, 1e-3)),
        math.log(max(p.anchor_sd, 1e-3)),
    ], dtype=float)


@dataclass
class _MPatient:
    design: np.ndarray      # (W, S) windowed-mean anchor design over session positions
    values: np.ndarray      # (W,) standardized retained anchor values
    n_sess: int
    min_grid: np.ndarray    # (S, S) min(s, t)
    lag_grid: np.ndarray    # (S, S) |s - t|
    stratum: int


def _m_build_patient(sess_days: np.ndarray, retained: pd.DataFrame, stats: dict) -> _MPatient | None:
    S = len(sess_days)
    if S == 0:
        return None
    day_to_pos = {int(d): i for i, d in enumerate(sess_days)}
    rows, vals = [], []
    for row in retained.itertuples(index=False):
        mu, sd = stats[row.anchor]
        z = (float(row.value) - mu) / sd
        win = [day_to_pos[int(d)] for d in sess_days if row.window_start_day <= d <= row.window_end_day]
        if not win:
            win = [int(np.argmin(np.abs(sess_days - row.day)))]
        coeff = 1.0 / len(win)
        r = np.zeros(S, dtype=float)
        for j in win:
            r[j] = coeff
        rows.append(r)
        vals.append(z)
    if not rows:
        return None
    idx = np.arange(S)
    return _MPatient(
        design=np.vstack(rows),
        values=np.asarray(vals, dtype=float),
        n_sess=S,
        min_grid=np.minimum.outer(idx, idx).astype(float),
        lag_grid=np.abs(np.subtract.outer(idx, idx)).astype(float),
        stratum=0,
    )


def _m_cov_L(p: _MClass, patient: _MPatient, level_prior_var: float) -> np.ndarray:
    fast_marg = p.s_fast**2 / max(1.0 - p.phi**2, 1e-6)
    return level_prior_var + p.s_lev**2 * patient.min_grid + fast_marg * np.power(abs(p.phi), patient.lag_grid)


def _m_anchor_loglik(patient: _MPatient, cov_L: np.ndarray, anchor_sd: float) -> float:
    w = patient.design.shape[0]
    if w == 0:
        return 0.0
    C = patient.design @ cov_L @ patient.design.T + (anchor_sd**2 + MARKOV_JITTER) * np.eye(w)
    try:
        c, low = cho_factor(C, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        c, low = cho_factor(C + 1e-3 * np.eye(w), lower=True, check_finite=False)
    a = cho_solve((c, low), patient.values, check_finite=False)
    logdet = 2.0 * np.sum(np.log(np.diag(c)))
    return -0.5 * (float(patient.values @ a) + logdet + w * math.log(2.0 * math.pi))


def _m_smooth(patient: _MPatient, p: _MClass) -> tuple[np.ndarray, np.ndarray]:
    """Batch level+fast Gaussian smoother for one patient under one class."""

    S = patient.n_sess
    level0, fast0 = 0, S
    n_vars = 2 * S
    rows, targets = [], []

    def add(coeffs, target, sd):
        w = 1.0 / max(sd, 1e-8)
        row = np.zeros(n_vars, dtype=float)
        for col, val in coeffs:
            row[col] = val * w
        rows.append(row)
        targets.append(target * w)

    add([(level0, 1.0)], 0.0, MARKOV_LEVEL_PRIOR_SD)
    fast_marg_sd = max(p.s_fast / math.sqrt(max(1.0 - p.phi**2, 1e-6)), MARKOV_MIN_SCALE)
    add([(fast0, 1.0)], 0.0, fast_marg_sd)
    for tpos in range(1, S):
        add([(level0 + tpos, 1.0), (level0 + tpos - 1, -1.0)], 0.0, p.s_lev)
        add([(fast0 + tpos, 1.0), (fast0 + tpos - 1, -p.phi)], 0.0, p.s_fast)
    for design_row, value in zip(patient.design, patient.values):
        positions = np.nonzero(design_row)[0]
        coeffs = []
        for j in positions:
            coeffs.append((level0 + int(j), design_row[j]))
            coeffs.append((fast0 + int(j), design_row[j]))
        add(coeffs, float(value), p.anchor_sd)

    A = np.vstack(rows)
    b = np.asarray(targets, dtype=float)
    info = A.T @ A + MARKOV_JITTER * np.eye(n_vars)
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(info)
    x = cov @ (A.T @ b)
    l_hat = x[level0:fast0] + x[fast0:]
    diag = np.diag(cov)
    cross = cov[np.arange(level0, fast0), np.arange(fast0, n_vars)]
    var_L = np.maximum(diag[level0:fast0] + diag[fast0:] + 2.0 * cross, 1e-10)
    return l_hat, var_L


def _fit_markov_channel(patients: list[_MPatient]) -> tuple[list[_MClass], np.ndarray]:
    """EM over class dynamics and stratum-conditional mixing for one channel.

    Parameter fitting is capped at MARKOV_MAX_FIT_PATIENTS deterministically
    sampled patients for runtime; the fitted classes and mixing weights are then
    applied to every patient in the channel.
    """

    level_prior_var = MARKOV_LEVEL_PRIOR_SD**2
    all_vals = np.concatenate([p.values for p in patients]) if patients else np.array([])
    base = float(np.std(all_vals)) if all_vals.size else 1.0
    base = min(max(base, 0.2), 3.0)
    phis = np.linspace(0.2, 0.8, MARKOV_K)
    lev_scales = np.linspace(0.15, 0.6, MARKOV_K)
    classes = [
        _MClass(s_lev=float(lev_scales[k]) * base, phi=float(phis[k]), s_fast=0.5 * base, anchor_sd=0.6 * base)
        for k in range(MARKOV_K)
    ]
    weights = np.full((MARKOV_STRATA, MARKOV_K), 1.0 / MARKOV_K)
    if not patients:
        return classes, weights
    if len(patients) > MARKOV_MAX_FIT_PATIENTS:
        sel = np.random.default_rng(7).choice(len(patients), size=MARKOV_MAX_FIT_PATIENTS, replace=False)
        patients = [patients[i] for i in sorted(sel)]
    strata = np.array([p.stratum for p in patients], dtype=int)

    def class_loglik(p: _MClass) -> np.ndarray:
        return np.array([_m_anchor_loglik(pt, _m_cov_L(p, pt, level_prior_var), p.anchor_sd) for pt in patients])

    prev = -np.inf
    for _ in range(MARKOV_EM_ITERS):
        loglik = np.column_stack([class_loglik(c) for c in classes])
        log_w = np.log(np.clip(weights[strata], 1e-8, None))
        joint = loglik + log_w
        rmax = joint.max(axis=1, keepdims=True)
        resp = np.exp(joint - rmax)
        rsum = resp.sum(axis=1, keepdims=True)
        resp = resp / np.where(rsum > 0, rsum, 1.0)
        total = float(np.sum(rmax.ravel() + np.log(np.clip(rsum.ravel(), 1e-12, None))))

        new_w = np.full_like(weights, 1.0 / MARKOV_K)
        for s in range(MARKOV_STRATA):
            mask = strata == s
            if mask.any():
                col = resp[mask].sum(axis=0)
                if col.sum() > 0:
                    new_w[s] = col / col.sum()
        weights = np.clip(new_w, 1e-6, None)
        weights = weights / weights.sum(axis=1, keepdims=True)

        new_classes = []
        for k, c in enumerate(classes):
            r_k = resp[:, k]
            if r_k.sum() < 1e-6:
                new_classes.append(c)
                continue

            def neg_obj(theta, r_k=r_k):
                p = _m_theta_to_params(theta)
                ll = np.array([_m_anchor_loglik(pt, _m_cov_L(p, pt, level_prior_var), p.anchor_sd) for pt in patients])
                return -float(np.sum(r_k * ll))

            res = minimize(neg_obj, _m_params_to_theta(c), method="L-BFGS-B",
                           options={"maxiter": MARKOV_OPT_MAXITER})
            new_classes.append(_m_theta_to_params(res.x))
        classes = new_classes
        if abs(total - prev) < 1e-3 * (abs(prev) + 1.0):
            break
        prev = total
    return classes, weights


def _markov_latent(patient: _MPatient, classes: list[_MClass], weights: np.ndarray) -> np.ndarray:
    """Responsibility-weighted mixed latent trajectory for one patient."""

    level_prior_var = MARKOV_LEVEL_PRIOR_SD**2
    ll = np.array([_m_anchor_loglik(patient, _m_cov_L(c, patient, level_prior_var), c.anchor_sd) for c in classes])
    joint = ll + np.log(np.clip(weights[patient.stratum], 1e-8, None))
    joint -= joint.max()
    resp = np.exp(joint)
    resp = resp / resp.sum()
    l_mix = np.zeros(patient.n_sess, dtype=float)
    for c, r in zip(classes, resp):
        l_hat, _ = _m_smooth(patient, c)
        l_mix += r * l_hat
    return l_mix


def _conformal_quantile(scores: np.ndarray, alpha: float, default: float = float("nan")) -> float:
    clean = np.asarray(scores, dtype=float)
    clean = clean[np.isfinite(clean)]
    n = len(clean)
    if n == 0:
        return float(default)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if rank > n:
        return float("inf")
    return float(np.sort(clean)[max(rank, 1) - 1])


def _standardizers(anchors: pd.DataFrame) -> dict:
    """Prespecified scale transforms that use no cohort or future values."""

    fixed = {
        "PHQ9": (13.5, 13.5),
        "GAD7": (10.5, 10.5),
        "BDI": (31.5, 31.5),
    }
    observed = set(anchors["anchor"].dropna().astype(str).unique())
    unknown = observed.difference(fixed)
    if unknown:
        raise ValueError(f"No prespecified scale transform for anchors: {sorted(unknown)}")
    return {scale: fixed[scale] for scale in observed}


def _coerce_sessions(sessions_full: pd.DataFrame) -> pd.DataFrame:
    sessions = sessions_full.copy()
    sessions["id"] = pd.to_numeric(sessions["id"], errors="coerce")
    sessions["session"] = pd.to_numeric(sessions.get("session"), errors="coerce")
    sessions["day"] = pd.to_numeric(sessions["day"], errors="coerce")
    sessions = sessions.dropna(subset=["id", "session", "day"]).copy()
    sessions["id"] = sessions["id"].astype(int)
    sessions["session"] = sessions["session"].astype(int)
    sessions["day"] = sessions["day"].astype(int)
    return sessions.sort_values(["id", "session", "day"]).reset_index(drop=True)


def _git_ancestor(path: Path) -> Path | None:
    resolved = path.expanduser().resolve()
    start = resolved if resolved.is_dir() else resolved.parent
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _load_comorbidity_sessions(path_value: str | Path) -> pd.DataFrame:
    """Load the protected numeric feature table without exposing its location."""

    path = Path(path_value).expanduser().resolve()
    if _git_ancestor(path) is not None:
        raise ValueError(
            "The comorbidity feature table must remain outside every Git working tree."
        )
    if not path.exists():
        raise FileNotFoundError("The protected comorbidity feature table was not found.")
    features = pd.read_csv(path, low_memory=False)
    required = {"id", "session", "day", *COMORBIDITY_TABLE_COLS}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Comorbidity feature table is missing columns: {sorted(missing)}")
    forbidden = {
        "PatientFID", "AppointmentFID", "ServiceDate", "DOB", "Value",
        "FieldValue_UID", "DX_Diagnosis", "MH_diagnosis", "note_text",
    }
    present_forbidden = forbidden.intersection(features.columns)
    if present_forbidden:
        raise ValueError(
            "Protected raw columns are forbidden in the model feature table: "
            f"{sorted(present_forbidden)}"
        )
    keep = ["id", "session", "day", *COMORBIDITY_TABLE_COLS]
    features = features[keep].copy()
    for col in keep:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    if features[keep].isna().any().any():
        raise ValueError("Comorbidity feature table contains missing or nonnumeric values.")
    features[["id", "session", "day"]] = features[["id", "session", "day"]].astype(int)
    binary_cols = ["dx_history_available", *COMORBIDITY_CATEGORY_COLS]
    invalid_binary = [c for c in binary_cols if not features[c].isin([0, 1]).all()]
    if invalid_binary:
        raise ValueError(f"Comorbidity indicator columns must contain only 0/1: {invalid_binary}")
    if features.duplicated(["id", "session"]).any():
        raise ValueError("Comorbidity feature table has duplicate patient-session rows.")
    return features.sort_values(["id", "session"], kind="mergesort").reset_index(drop=True)


def _empirical_neural_dataset(
    anchors: pd.DataFrame,
    sessions_full: pd.DataFrame,
    rdoc_sessions: pd.DataFrame | None,
    comorbidity_sessions: pd.DataFrame,
    stats: dict,
    *,
    max_patients: int | None = None,
) -> tuple[SimulationData, dict[int, np.ndarray]]:
    """Build the rectangular empirical tensor consumed by the BALL SSM.

    The retained sparse anchors are the only symptom-scale values supplied to
    the model. Held-out genuine measurements stay out of this object and are
    used only in the downstream recovery evaluation.
    """

    sessions = _coerce_sessions(sessions_full)
    retained = anchors[anchors["role"] == "anchor"].copy()
    ids = sorted(set(retained["id"].astype(int)).intersection(set(sessions["id"].astype(int))))
    if max_patients is not None:
        ids = ids[: int(max_patients)]
    if not ids:
        raise ValueError("No empirical patients with retained anchors and sessions.")

    sessions = sessions[sessions["id"].isin(ids)].copy()
    retained = retained[retained["id"].isin(ids)].copy()
    session_groups = {int(pid): g.sort_values(["session", "day"]).reset_index(drop=True)
                      for pid, g in sessions.groupby("id", sort=True)}
    session_days = {pid: g["day"].to_numpy(dtype=int) for pid, g in session_groups.items()}
    max_t = max(len(v) for v in session_days.values())

    if rdoc_sessions is not None and not rdoc_sessions.empty:
        rdoc_use = rdoc_sessions[rdoc_sessions["id"].isin(ids)].copy()
        rdoc_use["id"] = rdoc_use["id"].astype(int)
        rdoc_use["day"] = rdoc_use["day"].astype(int)
        rdoc_by = {(int(r.id), int(r.day)): r for r in rdoc_use.itertuples(index=False)}
    else:
        rdoc_by = {}

    action_values = []
    if "action_id" in sessions.columns:
        action_values = sorted(v for v in sessions["action_id"].dropna().unique().tolist())
    action_map = {v: i for i, v in enumerate(action_values)}
    n_actions = max(len(action_map), 1)

    comorbidity_use = comorbidity_sessions[comorbidity_sessions["id"].isin(ids)].copy()
    comorbidity_by = {
        (int(r.id), int(r.session)): r for r in comorbidity_use.itertuples(index=False)
    }
    expected_keys = {
        (int(r.id), int(r.session))
        for r in sessions[["id", "session"]].itertuples(index=False)
    }
    missing_keys = expected_keys.difference(comorbidity_by)
    if missing_keys:
        raise ValueError(
            f"Protected comorbidity table is missing {len(missing_keys)} cohort patient-session rows."
        )

    x_cols = [f"X{j}" for j in range(len(EMPIRICAL_FEATURE_NAMES))]
    component_rows, daily_rows, anchor_rows = [], [], []
    for pid in ids:
        g = session_groups[pid]
        days = session_days[pid]
        for tpos in range(max_t):
            real = tpos < len(g)
            row = g.iloc[tpos] if real else None
            day = int(row["day"]) if real else int(tpos)
            session_index = int(row["session"]) if real else int(tpos)
            session_number = (
                float(row.get("global_session"))
                if real and pd.notna(row.get("global_session"))
                else float(session_index + 1) if real else 0.0
            )
            dx_row = comorbidity_by.get((pid, session_index)) if real else None
            rdoc_row = rdoc_by.get((pid, day))
            b = np.zeros(len(RDOC_COLS), dtype=float)
            rdoc_observed = 0.0
            stale = 0.0
            if rdoc_row is not None:
                b = np.array([float(getattr(rdoc_row, c, 0.0) or 0.0) for c in RDOC_COLS], dtype=float)
                rdoc_observed = float(bool(getattr(rdoc_row, "rdoc_observed", False)))
                stale_val = getattr(rdoc_row, "rdoc_days_stale", 0.0)
                stale = float(stale_val) if stale_val is not None and np.isfinite(stale_val) else 0.0
            action_raw = row.get("action_id") if real and "action_id" in g.columns else None
            a = action_map.get(action_raw, -1) if real else -1
            next_gap = max(
                float(int(days[tpos + 1]) - day) if real and tpos + 1 < len(days) else 1.0,
                1e-3,
            )
            recent_tx = float(row.get("recent_treatment", 0.0) or 0.0) if real else 0.0
            burden = float(row.get("treatment_burden", row.get("side_effect_burden", 0.0)) or 0.0) if real else 0.0
            component = {
                "id": pid, "t": tpos, "a": int(a), "z_d": 0.0, "z_p": 0.0,
                "L": 0.0, "slow": 0.0, "delta": 0.0, "proxy_observed": bool(rdoc_observed),
                "dt": next_gap, "session_observed": bool(real),
                # Fields the fair direct comparators (S0/Markov) read; the empirical
                # tensor must carry the same covariate streams BALL gets so the
                # comparison is input-matched. recent_treatment/treatment_burden are
                # not available empirically and are set to zero (an honest absence).
                "subtype": 0, "recent_treatment": recent_tx, "treatment_burden": burden,
            }
            for j, col in enumerate(RDOC_COLS):
                component[col] = float(b[j])
                component[f"C{j}"] = float(b[j])
                component[f"alpha{j}"] = 0.0
                component[f"active{j}"] = False
            component_rows.append(component)

            dx_values = {
                col: float(getattr(dx_row, col)) if dx_row is not None else 0.0
                for col in COMORBIDITY_TABLE_COLS
            }
            features = np.array([
                session_number,
                dx_values["dx_comorbidity_count"],
                *[dx_values[col] for col in COMORBIDITY_CATEGORY_COLS],
            ], dtype=float)
            if len(features) != len(EMPIRICAL_FEATURE_NAMES):
                raise RuntimeError("Empirical feature vector does not match its declared schema.")
            daily = {"id": pid, "t": tpos}
            for j, val in enumerate(features):
                daily[x_cols[j]] = float(val)
                daily[f"input_{x_cols[j]}"] = bool(real) and (
                    j == 0 or bool(dx_values["dx_history_available"])
                )
                # These columns are conditioning covariates, not noisy outcomes
                # for the daily reconstruction likelihood.
                daily[f"obs_{x_cols[j]}"] = False
            daily_rows.append(daily)

        pid_anchors = retained[retained["id"].astype(int) == pid]
        for arow in pid_anchors.itertuples(index=False):
            scale = str(arow.anchor)
            if scale not in stats:
                continue
            mu, sd = stats[scale]
            value = (float(arow.value) - mu) / sd
            channel = "Y2" if scale == "GAD7" else "Y1"
            days_i = session_days[pid]
            win_pos = np.where((days_i >= int(arow.window_start_day)) & (days_i <= int(arow.window_end_day)))[0]
            if win_pos.size == 0:
                nearest = int(np.argmin(np.abs(days_i - int(arow.day))))
                win_pos = np.array([nearest], dtype=int)
            anchor_t = int(arow.session)
            if anchor_t < 0 or anchor_t >= len(days_i):
                anchor_t = int(np.argmin(np.abs(days_i - int(arow.day))))
            anchor_rows.append({
                "id": pid,
                "anchor": channel,
                "t": anchor_t,
                "value": float(value),
                "observed": True,
                "loading": 1.0,
                "window_start": int(win_pos.min()),
                "window_end": int(win_pos.max()),
            })

    cfg = SimulationConfig(
        seed=EMPIRICAL_SEED,
        n=len(ids),
        t=max_t,
        q=len(RDOC_COLS),
        p_daily=len(x_cols),
        n_treatment_types=n_actions,
        y1=AnchorSpec("Y1", 7, 14, 1.0, ANCHOR_SD, 0.0),
        y2=AnchorSpec("Y2", 7, 7, 1.0, ANCHOR_SD, 0.0),
        rho_serial_y1=0.0,
        rho_serial_y2=0.0,
    )
    individuals = pd.DataFrame({"id": ids, "subtype": 0, "split": "train"})
    data = SimulationData(
        config=cfg,
        individuals=individuals,
        daily=pd.DataFrame(daily_rows),
        anchors=pd.DataFrame(anchor_rows),
        treatments=pd.DataFrame(),
        components=pd.DataFrame(component_rows),
        metadata={
            "source": "empirical_rtms_sparse_anchor_tensor",
            "anchor_note": "Retained anchors only; held-out genuine measurements excluded from training.",
            "feature_names": EMPIRICAL_FEATURE_NAMES,
            "feature_timing": "All diagnosis records are strictly earlier than the modeled session date.",
            "time_encoding": "Raw treatment session number as context; raw inter-session days enter the transition interval dt.",
            "encoder_unscaled_daily_indices": list(range(len(EMPIRICAL_FEATURE_NAMES))),
            "empirical_feature_scaling": "none",
            "encoder_feature_scaling": "none",
            "encoder_unscaled_daily_masks": True,
            "raw_continuous_time_features": ["treatment_session_number", "transition_dt_days"],
            "diagnosis_history_availability_is_a_mask": True,
            "daily_features_are_conditioning_covariates": True,
            "future_course_denominators": False,
            "same_session_symptom_measurement_flags": False,
            "action_is_separate_from_ehr_features": True,
            "padded_time_steps": int(max_t),
            "n_patients": int(len(ids)),
        },
    )
    return data, session_days


def _fit_empirical_teacher_student(
    anchors: pd.DataFrame,
    sessions_full: pd.DataFrame,
    rdoc_sessions: pd.DataFrame | None,
    comorbidity_sessions: pd.DataFrame,
    stats: dict,
    args,
    cadence: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    data, session_days = _empirical_neural_dataset(
        anchors,
        sessions_full,
        rdoc_sessions,
        comorbidity_sessions,
        stats,
        max_patients=args.max_patients,
    )
    cfg = SSMConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        teacher_epochs=args.teacher_epochs,
        student_epochs=args.student_epochs,
        anchor_warmup_epochs=args.anchor_warmup,
        kl_warmup_epochs=args.kl_warmup,
        batch_size=args.batch_size,
        ensemble_size=args.members,
        max_individuals=data.config.n,
        seed=EMPIRICAL_SEED + int(args.seed_offset),
        rdoc_drift_head=False,
        use_alpha_slow=False,
        delta_phi=args.delta_phi,
    )
    student, teacher = fit_ball_ssm(data, cfg, device=args.device, causal=True, prediction_split=None, return_teacher=True)
    manifest = {
        "method": "BALL teacher/student transformer",
        "primary_empirical_estimator": "BALL causal student",
        "cadence_days": int(cadence),
        "teacher_epochs": int(args.teacher_epochs),
        "student_epochs": int(args.student_epochs),
        "anchor_warmup": int(args.anchor_warmup),
        "kl_warmup": int(args.kl_warmup),
        "d_model": int(args.d_model),
        "n_layers": int(args.n_layers),
        "n_heads": int(args.n_heads),
        "members": int(args.members),
        "n_patients": int(data.config.n),
        "t": int(data.config.t),
        "empirical_feature_names": list(EMPIRICAL_FEATURE_NAMES),
        "comorbidity_feature_rows": int(len(comorbidity_sessions)),
        "comorbidity_patients": int(comorbidity_sessions["id"].nunique()),
        "future_derived_features": False,
        "uses_rdoc_proxy": False,
        "max_patients": args.max_patients,
        "student_metadata": student.metadata,
        "teacher_metadata": teacher.metadata,
    }
    (OUT / f"empirical_teacher_student_manifest_{int(cadence)}d.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    # Fair direct comparators on the SAME empirical SimulationData (same inputs:
    # anchors + the same strictly prior daily covariates + RDoC proxy B), fixed linear basis so they
    # face the same mis-specification the transformer does. Returned as {id: L over
    # session index} dicts, windowed for held-out measurements like the BALL paths.
    dc_args = _dc_args("linear")
    s0_predictions = _dc_predictions(_dc_fit_direct_map(data, dc_args)[0])
    markov_predictions = _dc_predictions(_dc_fit_markov_direct(data, dc_args, basis="linear"))
    return student.predictions, teacher.predictions, s0_predictions, markov_predictions, session_days


def _prediction_lookup(predictions: pd.DataFrame, value_col: str) -> dict[int, np.ndarray]:
    out = {}
    for pid, grp in predictions.groupby("id", sort=False):
        g = grp.sort_values("t")
        out[int(pid)] = g[value_col].to_numpy(dtype=float)
    return out


def _predict_lookup_window(pred_by_id: dict[int, np.ndarray], pid: int, sess_days: np.ndarray, w0: int, w1: int, day: int) -> float:
    arr = pred_by_id.get(int(pid))
    if arr is None or len(arr) == 0:
        return float("nan")
    n = min(len(arr), len(sess_days))
    days = sess_days[:n]
    vals = arr[:n]
    win = vals[(days >= w0) & (days <= w1)]
    if win.size == 0:
        return float(vals[int(np.argmin(np.abs(days - day)))])
    return float(np.nanmean(win))


def _solve_patient(sess_days: np.ndarray, retained: pd.DataFrame, stats: dict):
    """Recover the latent severity over a patient's session days.

    Returns (L_hat, raw_sd) arrays aligned to sess_days, or (None, None) if the
    patient has fewer than two retained anchors.
    """

    S = len(sess_days)
    if S == 0 or len(retained) < 2:
        return None, None
    day_to_idx = {int(d): i for i, d in enumerate(sess_days)}
    rows, cols, vals, targets = [], [], [], []
    r = 0

    def add(coeffs, target, sd):
        nonlocal r
        w = 1.0 / max(sd, 1e-6)
        for c, v in coeffs:
            if v != 0.0:
                rows.append(r); cols.append(c); vals.append(v * w)
        targets.append(target * w)
        r += 1

    # Anchor rows: windowed mean of latent equals the standardized anchor value.
    for row in retained.itertuples(index=False):
        mu, sd = stats[row.anchor]
        z = (float(row.value) - mu) / sd
        win = [day_to_idx[int(d)] for d in sess_days
               if row.window_start_day <= d <= row.window_end_day]
        if not win:
            j = int(np.argmin(np.abs(sess_days - row.day)))
            win = [j]
        coeff = 1.0 / len(win)
        add([(j, coeff) for j in win], z, ANCHOR_SD)

    # Random-walk smoothing on the latent.
    for i in range(1, S):
        add([(i, 1.0), (i - 1, -1.0)], 0.0, SMOOTH_SD)

    A = sparse.coo_matrix((vals, (rows, cols)), shape=(r, S)).tocsr()
    b = np.asarray(targets, dtype=float)
    sol = lsqr(A, b, atol=1e-7, btol=1e-7, iter_lim=2000)
    L = sol[0]
    # Diagonal precision approximation for the raw interval.
    prec = np.asarray(A.power(2).sum(axis=0)).ravel()
    raw_sd = 1.0 / np.sqrt(np.maximum(prec, 1e-8))
    return L, raw_sd


def _predict_window(L: np.ndarray, sess_days: np.ndarray, w0: int, w1: int, day: int) -> float:
    win = L[(sess_days >= w0) & (sess_days <= w1)]
    if win.size == 0:
        return float(L[int(np.argmin(np.abs(sess_days - day)))])
    return float(win.mean())


def _overlaps_retained(w0, w1, retained_windows) -> bool:
    return any(rw0 <= w1 and w0 <= rw1 for rw0, rw1 in retained_windows)


# ---------------------------------------------------------------------------
# Fair direct-RDoC comparators for the EMPIRICAL recovery: S0 direct LGSSM and
# Markov direct transition. Faithful ports of the simulation benchmark
# comparators (validation/direct_rdoc_fair_comparator.py fit_direct_map and
# validation/direct_rdoc_benchmark.py fit_markov_direct) so the empirical S0/Markov
# see the SAME inputs BALL gets (anchors + daily covariates X0-9 + RDoC proxy B)
# and use the same direct B(t)*beta + nuisance transition design. They run on the
# empirical SimulationData from _empirical_neural_dataset and return a latent
# trajectory per (id, session index), windowed for held-out measurements exactly
# like the BALL predictions. A fixed linear basis matches the simulation's fair
# default, so the comparators face the same mis-specification the transformer does.
# ---------------------------------------------------------------------------

class _DCArgs:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _dc_args(basis: str = "linear") -> "_DCArgs":
    return _DCArgs(daily_ridge=1.0, s0_basis=basis, iters=5, beta_ridge=10.0,
                   prior_sd=5.0, transition_sd=0.75, markov_strata=3,
                   markov_iters=4, markov_ridge=10.0)


def _dc_impute_by_time(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    vals = frame[cols].astype(float).interpolate(limit_direction="both")
    vals = vals.fillna(vals.mean()).fillna(0.0)
    return vals.to_numpy(dtype=float)


def _dc_fit_daily_readout(data, train_ids, ridge: float = 1.0):
    x_cols = [f"X{j}" for j in range(data.config.p_daily)]
    obs_cols = [f"obs_X{j}" for j in range(data.config.p_daily)]
    daily_by = dict(tuple(data.daily.groupby("id", sort=True)))
    comp_by = dict(tuple(data.components.groupby("id", sort=True)))
    anc_by = dict(tuple(data.anchors.groupby("id", sort=True)))
    x_blocks, o_blocks, y_blocks = [], [], []
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
            value = float(getattr(row, "value"))
            loading = float(getattr(row, "loading"))
            if not np.isfinite(value) or not np.isfinite(loading) or abs(loading) < 1e-8:
                continue
            points_x.append(float(getattr(row, "window_end")))
            points_y.append(value / loading)
        if len(points_x) < 2:
            continue
        order = np.argsort(points_x)
        days = comp_i.sort_values("t")["t"].to_numpy(dtype=float)
        target = np.interp(days, np.asarray(points_x)[order], np.asarray(points_y)[order])
        di = daily_i.sort_values("t").reset_index(drop=True)
        if not set(x_cols).issubset(di.columns):
            continue
        x = di[x_cols].to_numpy(dtype=float)
        obs = di[obs_cols].to_numpy(dtype=float) if set(obs_cols).issubset(di.columns) else np.isfinite(x).astype(float)
        n = min(len(target), len(x))
        x_blocks.append(x[:n]); o_blocks.append(obs[:n]); y_blocks.append(target[:n])
    if not x_blocks:
        return 0.0, np.zeros(data.config.p_daily, dtype=float), np.zeros(data.config.p_daily, dtype=float), 1.0
    x = np.concatenate(x_blocks); obs = np.concatenate(o_blocks); y = np.concatenate(y_blocks)
    means = np.array([x[obs[:, j] > 0.5, j].mean() if (obs[:, j] > 0.5).any() else 0.0 for j in range(x.shape[1])])
    x_imp = np.where(obs > 0.5, x, means[None, :])
    design = np.column_stack([np.ones(len(x_imp)), x_imp])
    penalty = ridge * np.eye(design.shape[1]); penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    sigma = max(float(np.std(y - design @ coef)), 0.35)
    return float(coef[0]), coef[1:].astype(float), means.astype(float), sigma


def _dc_person_arrays(data, pid, daily_readout):
    comp_i = data.components[data.components["id"] == pid].sort_values("t").reset_index(drop=True)
    daily_i = data.daily[data.daily["id"] == pid].sort_values("t").reset_index(drop=True)
    anchors_i = data.anchors[data.anchors["id"] == pid].sort_values(["anchor", "t"]).reset_index(drop=True)
    q = data.config.q
    b = _dc_impute_by_time(comp_i, [f"B{j}" for j in range(q)])
    x_cols = [f"X{j}" for j in range(data.config.p_daily)]
    obs_cols = [f"obs_X{j}" for j in range(data.config.p_daily)]
    x = daily_i[x_cols].to_numpy(dtype=float)
    obs = daily_i[obs_cols].to_numpy(dtype=float) if set(obs_cols).issubset(daily_i.columns) else np.isfinite(x).astype(float)
    intercept, x_beta, x_means, daily_sd = daily_readout
    x_imp = np.where(obs > 0.5, x, x_means[None, :])
    daily_pred = intercept + x_imp @ x_beta
    any_daily = (obs > 0.5).any(axis=1)
    action = comp_i["a"].to_numpy(dtype=int)
    recent = comp_i["recent_treatment"].to_numpy(dtype=float)
    burden = comp_i["treatment_burden"].to_numpy(dtype=float)
    return comp_i, anchors_i, b, daily_pred, any_daily, daily_sd, action, recent, burden


def _dc_transition_features(b, action, recent, burden, n_actions, *, basis="linear", subtype=None, n_subtypes=3):
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
            feat.extend((b_k ** 2).tolist())
            feat.extend(np.tanh(b_k).tolist())
        if basis in {"heterogeneous", "full"}:
            subtype_i = int(subtype) if subtype is not None else 0
            for s in range(int(n_subtypes)):
                feat.extend((b_k * (1.0 if subtype_i == s else 0.0)).tolist())
        rows.append(feat)
    return np.asarray(rows, dtype=float)


def _dc_solve_person(data, pid, theta, daily_readout, args):
    comp_i, anchors_i, b, daily_pred, any_daily, daily_sd, action, recent, burden = _dc_person_arrays(data, pid, daily_readout)
    t_count = len(comp_i)
    basis = getattr(args, "s0_basis", "linear")
    subtype = int(comp_i["subtype"].iloc[0]) if "subtype" in comp_i.columns and len(comp_i) else 0
    feats = _dc_transition_features(b, action, recent, burden, data.config.n_treatment_types,
                                    basis=basis, subtype=subtype, n_subtypes=data.config.n_subtypes)
    trans_mean = feats @ theta
    rows, targets = [], []

    def add(coeffs, target, sd):
        row = np.zeros(t_count, dtype=float)
        for idx, value in coeffs:
            row[idx] = value / max(sd, 1e-8)
        rows.append(row)
        targets.append(float(target) / max(sd, 1e-8))

    add([(0, 1.0)], 0.0, args.prior_sd)
    for row in anchors_i.itertuples(index=False):
        if not bool(getattr(row, "observed", False)):
            continue
        value = float(getattr(row, "value")); loading = float(getattr(row, "loading"))
        if not np.isfinite(value) or not np.isfinite(loading) or abs(loading) < 1e-8:
            continue
        start = max(0, int(getattr(row, "window_start"))); end = min(t_count - 1, int(getattr(row, "window_end")))
        if end < start:
            continue
        window = list(range(start, end + 1))
        spec = data.config.y1 if getattr(row, "anchor") == "Y1" else data.config.y2
        rho = data.config.rho_serial_y1 if getattr(row, "anchor") == "Y1" else data.config.rho_serial_y2
        add([(k, loading / len(window)) for k in window], value, marginal_anchor_sd(spec, rho))
    for k, observed in enumerate(any_daily):
        if observed and np.isfinite(daily_pred[k]):
            add([(k, 1.0)], float(daily_pred[k]), daily_sd)
    for k in range(1, t_count):
        add([(k, 1.0), (k - 1, -1.0)], float(trans_mean[k - 1]), args.transition_sd)
    design = np.vstack(rows); target = np.asarray(targets, dtype=float)
    sol = np.linalg.lstsq(design, target, rcond=None)[0]
    pred = pd.DataFrame({"id": comp_i["id"].to_numpy(dtype=int), "t": comp_i["t"].to_numpy(dtype=int), "L_hat": sol})
    return pred, b, action, recent, burden


def _dc_fit_direct_map(data, args):
    train_ids = set(data.individuals.loc[data.individuals["split"] == "train", "id"].astype(int))
    daily_readout = _dc_fit_daily_readout(data, train_ids, ridge=args.daily_ridge)
    basis = getattr(args, "s0_basis", "linear")
    n_features = _dc_transition_features(np.zeros((2, data.config.q)), np.zeros(2, dtype=int), np.zeros(2), np.zeros(2),
                                         data.config.n_treatment_types, basis=basis, subtype=0,
                                         n_subtypes=data.config.n_subtypes).shape[1]
    theta = np.zeros(n_features, dtype=float)
    pred_by_pid, arrays_by_pid = {}, {}
    ids = sorted(data.individuals["id"].astype(int).tolist())
    for _ in range(args.iters):
        x_rows, y_rows = [], []
        pred_by_pid.clear(); arrays_by_pid.clear()
        for pid in ids:
            pred, b, action, recent, burden = _dc_solve_person(data, pid, theta, daily_readout, args)
            pred_by_pid[pid] = pred; arrays_by_pid[pid] = (b, action, recent, burden)
            if pid not in train_ids:
                continue
            l = pred["L_hat"].to_numpy(dtype=float)
            subtype = int(data.individuals.loc[data.individuals["id"] == pid, "subtype"].iloc[0])
            feats = _dc_transition_features(b, action, recent, burden, data.config.n_treatment_types,
                                            basis=basis, subtype=subtype, n_subtypes=data.config.n_subtypes)
            x_rows.append(feats); y_rows.append(np.diff(l))
        if not x_rows:
            break
        x = np.vstack(x_rows); y = np.concatenate(y_rows)
        penalty = args.beta_ridge * np.eye(n_features); penalty[0, 0] = 0.0
        theta = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    preds = pd.concat([pred_by_pid[pid] for pid in ids], ignore_index=True)
    return preds, theta


def _dc_anchor_pattern_strata(data, n_strata):
    n_strata = max(1, int(n_strata))
    ids = sorted(data.individuals["id"].astype(int).tolist())
    observed = data.anchors.loc[data.anchors["observed"].astype(bool)]
    counts = observed.groupby("id").size()
    vals = np.asarray([float(counts.get(pid, 0.0)) for pid in ids], dtype=float)
    if n_strata == 1 or len(np.unique(vals)) <= 1:
        return {pid: 0 for pid in ids}
    edges = np.quantile(vals, np.linspace(0.0, 1.0, n_strata + 1)[1:-1])
    return {pid: int(min(n_strata - 1, max(0, np.searchsorted(edges, vals[i], side="right")))) for i, pid in enumerate(ids)}


def _dc_markov_design(base_feats, l_hat, stratum, n_strata):
    n = min(len(base_feats), max(len(l_hat) - 1, 0))
    if n <= 0:
        return np.zeros((0, base_feats.shape[1] + 2 * n_strata), dtype=float), np.zeros(0, dtype=float)
    base = base_feats[:n]
    lag = np.asarray(l_hat[:n], dtype=float)
    onehot = np.zeros((n, n_strata), dtype=float); onehot[:, int(stratum)] = 1.0
    x = np.column_stack([base, onehot, onehot * lag[:, None]])
    y = np.diff(np.asarray(l_hat[: n + 1], dtype=float))
    return x, y


def _dc_solve_markov_person(data, pid, theta, daily_readout, args, *, basis, stratum, n_strata):
    comp_i, anchors_i, b, daily_pred, any_daily, daily_sd, action, recent, burden = _dc_person_arrays(data, pid, daily_readout)
    t_count = len(comp_i)
    subtype = int(comp_i["subtype"].iloc[0]) if "subtype" in comp_i.columns and len(comp_i) else 0
    feats = _dc_transition_features(b, action, recent, burden, data.config.n_treatment_types,
                                    basis=basis, subtype=subtype, n_subtypes=data.config.n_subtypes)
    n_base = feats.shape[1]
    base_theta = theta[:n_base]
    stratum_offsets = theta[n_base:n_base + n_strata]
    phis = np.clip(theta[n_base + n_strata:n_base + 2 * n_strata], -0.95, 0.95)
    phi = float(phis[int(stratum)]) if len(phis) else 0.0
    trans_mean = feats @ base_theta
    if len(stratum_offsets):
        trans_mean = trans_mean + float(stratum_offsets[int(stratum)])
    rows, targets = [], []

    def add(coeffs, target, sd):
        row = np.zeros(t_count, dtype=float)
        for idx, value in coeffs:
            row[idx] = value / max(sd, 1e-8)
        rows.append(row); targets.append(float(target) / max(sd, 1e-8))

    add([(0, 1.0)], 0.0, args.prior_sd)
    for row in anchors_i.itertuples(index=False):
        if not bool(getattr(row, "observed", False)):
            continue
        value = float(getattr(row, "value")); loading = float(getattr(row, "loading"))
        if not np.isfinite(value) or not np.isfinite(loading) or abs(loading) < 1e-8:
            continue
        start = max(0, int(getattr(row, "window_start"))); end = min(t_count - 1, int(getattr(row, "window_end")))
        if end < start:
            continue
        window = list(range(start, end + 1))
        spec = data.config.y1 if getattr(row, "anchor") == "Y1" else data.config.y2
        rho = data.config.rho_serial_y1 if getattr(row, "anchor") == "Y1" else data.config.rho_serial_y2
        add([(k, loading / len(window)) for k in window], value, marginal_anchor_sd(spec, rho))
    for k, observed in enumerate(any_daily):
        if observed and np.isfinite(daily_pred[k]):
            add([(k, 1.0)], float(daily_pred[k]), daily_sd)
    for k in range(1, t_count):
        add([(k, 1.0), (k - 1, -(1.0 + phi))], float(trans_mean[k - 1]), args.transition_sd)
    design = np.vstack(rows); target = np.asarray(targets, dtype=float)
    sol = np.linalg.lstsq(design, target, rcond=None)[0]
    return pd.DataFrame({"id": comp_i["id"].to_numpy(dtype=int), "t": comp_i["t"].to_numpy(dtype=int), "L_hat": sol})


def _dc_fit_markov_direct(data, args, *, basis):
    train_ids = set(data.individuals.loc[data.individuals["split"] == "train", "id"].astype(int))
    ids = sorted(data.individuals["id"].astype(int).tolist())
    daily_readout = _dc_fit_daily_readout(data, train_ids, ridge=args.daily_ridge)
    init_args = _DCArgs(**vars(args)); init_args.s0_basis = basis
    init_preds, base_theta = _dc_fit_direct_map(data, init_args)
    n_base = int(len(base_theta))
    n_strata = max(1, int(args.markov_strata))
    strata = _dc_anchor_pattern_strata(data, n_strata)
    theta = np.zeros(n_base + 2 * n_strata, dtype=float); theta[:n_base] = base_theta
    pred_by_pid = {int(pid): frame.sort_values("t").reset_index(drop=True) for pid, frame in init_preds.groupby("id", sort=True)}
    for _ in range(max(1, int(args.markov_iters))):
        x_rows, y_rows = [], []
        for pid in ids:
            if pid not in train_ids or pid not in pred_by_pid:
                continue
            comp_i, _, b, _, _, _, action, recent, burden = _dc_person_arrays(data, pid, daily_readout)
            subtype = int(comp_i["subtype"].iloc[0]) if "subtype" in comp_i.columns and len(comp_i) else 0
            base_feats = _dc_transition_features(b, action, recent, burden, data.config.n_treatment_types,
                                                 basis=basis, subtype=subtype, n_subtypes=data.config.n_subtypes)
            l_hat = pred_by_pid[pid]["L_hat"].to_numpy(dtype=float)
            x, y = _dc_markov_design(base_feats, l_hat, strata[pid], n_strata)
            if len(y):
                x_rows.append(x); y_rows.append(y)
        if not x_rows:
            break
        x = np.vstack(x_rows); y = np.concatenate(y_rows)
        penalty = float(args.markov_ridge) * np.eye(theta.size); penalty[0, 0] = 0.0
        theta = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        theta[n_base + n_strata:] = np.clip(theta[n_base + n_strata:], -0.95, 0.95)
        pred_by_pid = {pid: _dc_solve_markov_person(data, pid, theta, daily_readout, args, basis=basis,
                                                    stratum=strata[pid], n_strata=n_strata) for pid in ids}
    return pd.concat([pred_by_pid[pid] for pid in ids], ignore_index=True)


def _dc_predictions(preds: pd.DataFrame) -> dict[int, np.ndarray]:
    """Map a direct-comparator prediction frame to {id: L over session index}."""
    out = {}
    for pid, g in preds.groupby("id", sort=True):
        out[int(pid)] = g.sort_values("t")["L_hat"].to_numpy(dtype=float)
    return out


def _channel_eval(
    anchors_ch: pd.DataFrame,
    sessions: pd.DataFrame,
    stats: dict,
    neural_predictions: dict[str, dict[int, np.ndarray]] | None = None,
    session_days_by_id: dict[int, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Held-out recovery rows for one channel, all methods.

    BALL (teacher and student), the fair S0 direct LGSSM, and the fair Markov
    direct-transition comparator all receive the same inputs (anchors + daily
    covariates X0-9 + RDoC proxy B) via predictions computed once on the empirical
    SimulationData, and are evaluated on the identical embargo-clean held-out set.
    The anchor-only smoother supplies the raw-interval sd; interpolation, LOCF, and
    anchor-only are naive anchor-only baselines.
    """

    predictions = neural_predictions or {}
    has_neural = "student" in predictions
    patients: list[dict] = []
    for pid, ach in anchors_ch.groupby("id"):
        sess = sessions[sessions["id"] == pid]
        if sess.empty:
            continue
        # Per-session-row days in the SAME order the model produced its latents
        # (sort by session then day, duplicates kept). The prediction arrays are
        # indexed by session-row position, so the lookup must use this array, not
        # a unique-day reconstruction, or duplicate calendar days shift every
        # later position onto the wrong day.
        if session_days_by_id is not None and int(pid) in session_days_by_id:
            sess_days = np.asarray(session_days_by_id[int(pid)], dtype=int)
        else:
            sess_days = np.sort(sess["day"].unique()).astype(int)
        retained = ach[ach["role"] == "anchor"]
        heldout = ach[ach["role"] == "heldout_eval"]
        if len(retained) < 2 or heldout.empty:
            continue
        L, raw_sd = _solve_patient(sess_days, retained, stats)
        if L is None:
            continue
        ret_days = retained["day"].to_numpy(dtype=float)
        ret_z = np.array([(float(v) - stats[s][0]) / stats[s][1]
                          for v, s in zip(retained["value"], retained["anchor"])])
        order = np.argsort(ret_days)
        patients.append({
            "id": int(pid), "sess_days": sess_days, "heldout": heldout, "L": L, "raw_sd": raw_sd,
            "ret_days_s": ret_days[order], "ret_z_s": ret_z[order],
            "retained_windows": list(zip(retained["window_start_day"], retained["window_end_day"])),
        })

    if not patients:
        return pd.DataFrame()

    recs = []
    for p in patients:
        sess_days = p["sess_days"]
        L = p["L"]
        raw_sd = p["raw_sd"]
        ret_days_s, ret_z_s = p["ret_days_s"], p["ret_z_s"]
        retained_windows = p["retained_windows"]
        pid = p["id"]
        for h in p["heldout"].itertuples(index=False):
            w0, w1, day = int(h.window_start_day), int(h.window_end_day), int(h.day)
            if _overlaps_retained(w0, w1, retained_windows):
                continue  # recall-window embargo
            mu, sd = stats[h.anchor]
            z_true = (float(h.value) - mu) / sd
            gap = float(np.min(np.abs(ret_days_s - day)))
            if has_neural:
                ball = _predict_lookup_window(predictions["student"], pid, sess_days, w0, w1, day)
                ball_teacher = _predict_lookup_window(predictions["teacher"], pid, sess_days, w0, w1, day)
            else:
                ball = _predict_window(L, sess_days, w0, w1, day)
                ball_teacher = float("nan")
            s0 = (_predict_lookup_window(predictions["s0"], pid, sess_days, w0, w1, day)
                  if "s0" in predictions else _predict_window(L, sess_days, w0, w1, day))
            markov = (_predict_lookup_window(predictions["markov"], pid, sess_days, w0, w1, day)
                      if "markov" in predictions else float("nan"))
            interp = float(np.interp(day, ret_days_s, ret_z_s))
            locf = float(ret_z_s[ret_days_s <= day][-1]) if np.any(ret_days_s <= day) else float(ret_z_s[0])
            cover = ret_z_s[(ret_days_s >= w0) & (ret_days_s <= w1)]
            anchor_only = float(cover.mean()) if cover.size else interp
            si = int(np.argmin(np.abs(sess_days - day)))
            recs.append({
                "id": pid, "anchor": h.anchor, "day": day, "gap": gap, "z_true": z_true,
                "ball": ball, "ball_teacher": ball_teacher, "s0_direct_lgssm": s0,
                "markov_direct_transition": markov, "interpolation": interp, "locf": locf,
                "anchor_only": anchor_only, "ball_sd": float(raw_sd[si]),
            })
    return pd.DataFrame(recs)


def _load_rdoc_sessions() -> pd.DataFrame | None:
    path = OUT / "sessions_with_rdoc.csv"
    if not path.exists():
        return None
    manifest_path = OUT / "rdoc_proxy_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("rdoc_proxy_manifest.json is required to verify the empirical RDoC score source.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("raw_score_source") != "raw/RDoC_LLM_scorer.csv"
        or manifest.get("empirical_session_source") != "empirical/data/rtms_paper_analytic_sessions.csv"
        or manifest.get("sole_rdoc_source_for_empirical_example") is not True
        or manifest.get("same_day_records_excluded") is not True
    ):
        raise ValueError(
            "Empirical fit requires RDoC scores generated only from raw/RDoC_LLM_scorer.csv "
            "on empirical/data/rtms_paper_analytic_sessions.csv with same-day notes excluded."
        )
    rdoc = pd.read_csv(path)
    required = {"id", "day", "rdoc_observed", "rdoc_days_stale", *RDOC_COLS}
    if not required.issubset(rdoc.columns):
        missing = sorted(required.difference(rdoc.columns))
        raise ValueError(f"sessions_with_rdoc.csv is missing columns: {missing}")
    rdoc = rdoc.dropna(subset=["id", "day"]).copy()
    rdoc["id"] = rdoc["id"].astype(int)
    rdoc["day"] = rdoc["day"].astype(int)
    rdoc["rdoc_observed"] = rdoc["rdoc_observed"].astype(bool)
    rdoc["rdoc_days_stale"] = pd.to_numeric(rdoc["rdoc_days_stale"], errors="coerce").fillna(0.0)
    for col in RDOC_COLS:
        rdoc[col] = pd.to_numeric(rdoc[col], errors="coerce").fillna(0.0)
    keep = ["id", "session", "day", "rdoc_observed", "rdoc_days_stale"] + RDOC_COLS
    keep = [c for c in keep if c in rdoc.columns]
    return rdoc[keep].sort_values(["id", "day"]).reset_index(drop=True)


def _channel_transition_rows(
    anchors_ch: pd.DataFrame,
    rdoc_sessions: pd.DataFrame | None,
    stats: dict,
    cadence: int,
    channel: str,
    neural_predictions: dict[str, dict[int, np.ndarray]] | None = None,
    session_days_by_id: dict[int, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Build adjacent-session direct-RDoC transition rows from recovered latents."""

    if rdoc_sessions is None or rdoc_sessions.empty:
        return pd.DataFrame()
    use_neural = (
        neural_predictions is not None
        and "student" in neural_predictions
        and session_days_by_id is not None
    )
    recs = []
    for pid, ach in anchors_ch.groupby("id"):
        retained = ach[ach["role"] == "anchor"]
        if len(retained) < 2:
            continue
        rd = rdoc_sessions[rdoc_sessions["id"] == int(pid)]
        if rd.empty:
            continue
        if use_neural and int(pid) in session_days_by_id:
            # Student latents are indexed by session-row position in the model's
            # tensor order. Use the canonical per-row days and fetch each row's
            # RDoC proxy by (id, day) exactly as the tensor did (rdoc_by, last
            # row wins per duplicate day), so L[i] and b_mat[i] stay aligned.
            L_full = neural_predictions["student"].get(int(pid))
            if L_full is None:
                continue
            sess_days = np.asarray(session_days_by_id[int(pid)], dtype=int)
            n = min(len(L_full), len(sess_days))
            if n < 2:
                continue
            L = np.asarray(L_full[:n], dtype=float)
            sess_days = sess_days[:n]
            feat_by_day: dict[int, tuple] = {}
            for r in rd.itertuples(index=False):
                feat_by_day[int(r.day)] = (
                    np.array([float(getattr(r, c)) for c in RDOC_COLS], dtype=float),
                    bool(r.rdoc_observed),
                    float(r.rdoc_days_stale),
                )
            b_mat = np.zeros((n, len(RDOC_COLS)), dtype=float)
            observed = np.zeros(n, dtype=bool)
            stale = np.zeros(n, dtype=float)
            for i, d in enumerate(sess_days):
                feat = feat_by_day.get(int(d))
                if feat is not None:
                    b_mat[i], observed[i], stale[i] = feat
        else:
            sess = rd.sort_values(["day", "session"], na_position="last")
            sess = sess.drop_duplicates("day", keep="first").reset_index(drop=True)
            sess_days = sess["day"].to_numpy(dtype=int)
            if len(sess_days) < 2:
                continue
            L, _ = _solve_patient(sess_days, retained, stats)
            if L is None:
                continue
            b_mat = sess[RDOC_COLS].to_numpy(dtype=float)
            observed = sess["rdoc_observed"].to_numpy(dtype=bool)
            stale = sess["rdoc_days_stale"].to_numpy(dtype=float)
        for i in range(len(sess_days) - 1):
            dt = int(sess_days[i + 1] - sess_days[i])
            if dt <= 0 or not observed[i] or not np.all(np.isfinite(b_mat[i])):
                continue
            row = {
                "cadence": int(cadence),
                "channel": channel,
                "id": int(pid),
                "day": int(sess_days[i]),
                "next_day": int(sess_days[i + 1]),
                "dt": float(dt),
                "L_current": float(L[i]),
                "delta": float(L[i + 1] - L[i]),
                "delta_per_day": float((L[i + 1] - L[i]) / dt),
                "rdoc_days_stale": float(stale[i]) if np.isfinite(stale[i]) else 0.0,
            }
            for j, col in enumerate(RDOC_COLS):
                row[col] = float(b_mat[i, j])
            recs.append(row)
    return pd.DataFrame(recs)


def _design_matrix(frame: pd.DataFrame, features: list[str], norm: tuple[np.ndarray, np.ndarray] | None = None):
    X = frame[features].to_numpy(dtype=float)
    if norm is None:
        mu = np.nanmean(X, axis=0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        X_fill = np.where(np.isfinite(X), X, mu)
        sd = np.nanstd(X_fill, axis=0)
        sd = np.where((sd > 1e-8) & np.isfinite(sd), sd, 1.0)
    else:
        mu, sd = norm
        X_fill = np.where(np.isfinite(X), X, mu)
    Xs = (X_fill - mu) / sd
    return np.column_stack([np.ones(len(frame)), Xs]), (mu, sd)


def _ridge_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X_train, norm = _design_matrix(train, features)
    y_train = train["delta_per_day"].to_numpy(dtype=float)
    X_test, _ = _design_matrix(test, features, norm)
    pen = np.eye(X_train.shape[1])
    pen[0, 0] = 0.0
    lhs = X_train.T @ X_train + RDOC_RIDGE * pen
    rhs = X_train.T @ y_train
    try:
        coef = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(lhs) @ rhs
    return X_test @ coef, coef


def _patient_folds(frame: pd.DataFrame) -> list[np.ndarray]:
    ids = np.asarray(sorted(frame["id"].unique()), dtype=int)
    rng = np.random.default_rng(29)
    rng.shuffle(ids)
    k = min(RDOC_FOLDS, len(ids))
    if k < 2:
        return []
    return [fold for fold in np.array_split(ids, k) if len(fold)]


def _cv_error(frame: pd.DataFrame, features: list[str]) -> tuple[float, float]:
    y_all, pred_all = [], []
    for test_ids in _patient_folds(frame):
        test_mask = frame["id"].isin(test_ids)
        train = frame.loc[~test_mask]
        test = frame.loc[test_mask]
        if len(train) <= len(features) + 1 or test.empty:
            continue
        pred, _ = _ridge_predict(train, test, features)
        y_all.append(test["delta_per_day"].to_numpy(dtype=float))
        pred_all.append(pred)
    if not y_all:
        return float("nan"), float("nan")
    y = np.concatenate(y_all)
    pred = np.concatenate(pred_all)
    err = y - pred
    return float(np.sqrt(np.mean(err**2))), float(np.sum(err**2))


def _transition_one_group(frame: pd.DataFrame, cadence: int, channel: str) -> tuple[dict, list[dict]]:
    frame = frame.dropna(subset=["delta_per_day", *RDOC_NUISANCE_COLS, *RDOC_COLS]).copy()
    if len(frame) < 25 or frame["id"].nunique() < 5:
        return {
            "cadence": cadence, "channel": channel, "n": int(len(frame)),
            "n_patients": int(frame["id"].nunique()), "rmse_nuisance": float("nan"),
            "rmse_rdoc": float("nan"), "rmse_improvement": float("nan"),
            "incremental_r2": float("nan"), "beta_norm": float("nan"),
            "top_domain": "", "permutation_p": float("nan"),
        }, []

    base_features = list(RDOC_NUISANCE_COLS)
    full_features = list(RDOC_NUISANCE_COLS) + list(RDOC_COLS)
    base_rmse, base_sse = _cv_error(frame, base_features)
    full_rmse, full_sse = _cv_error(frame, full_features)
    improvement = base_rmse - full_rmse if np.isfinite(base_rmse) and np.isfinite(full_rmse) else float("nan")
    incr_r2 = 1.0 - full_sse / base_sse if np.isfinite(base_sse) and base_sse > 0 and np.isfinite(full_sse) else float("nan")

    _, coef = _ridge_predict(frame, frame, full_features)
    coef_map = dict(zip(["intercept"] + full_features, coef))
    beta = np.array([coef_map.get(col, 0.0) for col in RDOC_COLS], dtype=float)
    beta_norm = float(np.linalg.norm(beta))
    top_domain = RDOC_COLS[int(np.argmax(np.abs(beta)))] if beta.size else ""

    rng = np.random.default_rng(1100 + int(cadence) + (0 if channel == "depression" else 100))
    perm_better = 0
    perm_done = 0
    if np.isfinite(improvement):
        b_values = frame[RDOC_COLS].to_numpy(dtype=float)
        for _ in range(RDOC_PERMUTATIONS):
            perm = frame.copy()
            perm[RDOC_COLS] = b_values[rng.permutation(len(frame))]
            perm_rmse, _ = _cv_error(perm, full_features)
            if np.isfinite(perm_rmse):
                perm_done += 1
                if base_rmse - perm_rmse >= improvement:
                    perm_better += 1
    perm_p = (perm_better + 1.0) / (perm_done + 1.0) if perm_done else float("nan")

    summary = {
        "cadence": int(cadence),
        "channel": channel,
        "n": int(len(frame)),
        "n_patients": int(frame["id"].nunique()),
        "rmse_nuisance": base_rmse,
        "rmse_rdoc": full_rmse,
        "rmse_improvement": improvement,
        "incremental_r2": incr_r2,
        "beta_norm": beta_norm,
        "top_domain": top_domain,
        "permutation_p": perm_p,
    }
    coefs = [
        {"cadence": int(cadence), "channel": channel, "domain": col, "coefficient": float(coef_map.get(col, 0.0))}
        for col in RDOC_COLS
    ]
    return summary, coefs


def _summarize_rdoc_transitions(transitions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, coefs = [], []
    if transitions.empty:
        return pd.DataFrame(), pd.DataFrame()
    for (cadence, channel), group in transitions.groupby(["cadence", "channel"], sort=True):
        summary, coef_rows = _transition_one_group(group, int(cadence), str(channel))
        summaries.append(summary)
        coefs.extend(coef_rows)
    return pd.DataFrame(summaries), pd.DataFrame(coefs)


def _write_fit_tables(
    out_dir: Path,
    overall_rows: list[dict],
    strata_rows: list[dict],
    calib_rows: list[dict],
    boot_rows: list[dict],
    transition_frames: list[pd.DataFrame],
    *,
    suffix: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    overall = pd.DataFrame(overall_rows)
    strata = pd.DataFrame(strata_rows)
    calib = pd.DataFrame(calib_rows)
    boot = pd.DataFrame(boot_rows)
    transitions = pd.concat(transition_frames, ignore_index=True) if transition_frames else pd.DataFrame()
    transition_summary, transition_coef = _summarize_rdoc_transitions(transitions)
    overall.to_csv(out_dir / f"empirical_recovery_overall{suffix}.csv", index=False)
    strata.to_csv(out_dir / f"empirical_recovery_by_gap{suffix}.csv", index=False)
    calib.to_csv(out_dir / f"empirical_calibration{suffix}.csv", index=False)
    boot.to_csv(out_dir / f"empirical_bootstrap_diff{suffix}.csv", index=False)
    transitions.to_csv(out_dir / f"empirical_rdoc_transition_rows{suffix}.csv", index=False)
    transition_summary.to_csv(out_dir / f"empirical_rdoc_transition{suffix}.csv", index=False)
    transition_coef.to_csv(out_dir / f"empirical_rdoc_transition_coefficients{suffix}.csv", index=False)
    return overall, strata, calib, boot, transitions, transition_summary, transition_coef


def _archive_existing_final_outputs(run_dir: Path) -> list[str]:
    archive_dir = run_dir / "superseded_final_outputs"
    moved = []
    for name in FIT_FINAL_FILES:
        src = OUT / name
        if not src.exists():
            continue
        archive_dir.mkdir(parents=True, exist_ok=True)
        dest = archive_dir / name
        src.replace(dest)
        moved.append(name)
    return moved


def _write_fit_run_manifest(
    run_dir: Path,
    *,
    status: str,
    requested_cadences: list[int],
    completed_cadences: list[int],
    args,
    archived_outputs: list[str],
    sessions_full: pd.DataFrame | None = None,
    rdoc_sessions: pd.DataFrame | None = None,
    comorbidity_sessions: pd.DataFrame | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    fit_args = {
        k: ("[protected external file]" if k == "comorbidity_features" else
            int(v) if isinstance(v, np.integer) else v)
        for k, v in vars(args).items()
    }
    payload = {
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "requested_cadences": [int(c) for c in requested_cadences],
        "completed_cadences": [int(c) for c in completed_cadences],
        "fit_args": fit_args,
        "archived_final_outputs": archived_outputs,
        "empirical_session_source": "empirical/data/rtms_paper_analytic_sessions.csv",
        "final_outputs_written": status == "complete",
    }
    if sessions_full is not None and not sessions_full.empty:
        payload["n_sessions"] = int(len(sessions_full))
        payload["n_patients"] = int(sessions_full["id"].nunique()) if "id" in sessions_full.columns else None
    if rdoc_sessions is not None and not rdoc_sessions.empty:
        payload["rdoc_score_source"] = "raw/RDoC_LLM_scorer.csv"
        payload["rdoc_transition_analysis"] = True
        payload["rdoc_session_rows"] = int(len(rdoc_sessions))
        payload["rdoc_patients"] = int(rdoc_sessions["id"].nunique()) if "id" in rdoc_sessions.columns else None
        payload["rdoc_proxy_coverage"] = float(rdoc_sessions["rdoc_observed"].mean()) if "rdoc_observed" in rdoc_sessions.columns else None
    else:
        payload["rdoc_transition_analysis"] = False
    if comorbidity_sessions is not None and not comorbidity_sessions.empty:
        payload["comorbidity_feature_rows"] = int(len(comorbidity_sessions))
        payload["comorbidity_patients"] = int(comorbidity_sessions["id"].nunique())
        payload["comorbidity_prior_list_coverage"] = float(comorbidity_sessions["dx_history_available"].mean())
        payload["comorbidity_source"] = "protected external numeric feature table"
        payload["comorbidity_strictly_prior"] = True
    (run_dir / "fit_run_manifest.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Empirical BALL teacher/student held-out recovery analysis.")
    parser.add_argument("--teacher-epochs", type=int, default=120)
    parser.add_argument("--student-epochs", type=int, default=120)
    parser.add_argument("--anchor-warmup", type=int, default=40)
    parser.add_argument("--kl-warmup", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--members", type=int, default=5)  # deep ensemble K (spec Sec. 5)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--delta-phi", type=float, default=0.3)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cadences", type=int, nargs="+", default=CADENCES)
    parser.add_argument("--run-label", default=None)
    parser.add_argument(
        "--rdoc-transition-analysis",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the separate descriptive RDoC transition analysis. Never supplies RDoC to the recovery model.",
    )
    parser.add_argument(
        "--comorbidity-features",
        required=True,
        help="Protected session-level comorbidity CSV outside every Git working tree.",
    )
    parser.add_argument("--clean-final-outputs", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    requested_cadences = list(dict.fromkeys(int(c) for c in args.cadences))
    invalid_cadences = sorted(set(requested_cadences).difference(CADENCES))
    if invalid_cadences:
        raise ValueError(f"Unsupported empirical cadences: {invalid_cadences}; expected subset of {CADENCES}")
    OUT.mkdir(parents=True, exist_ok=True)
    sessions_full = _coerce_sessions(pd.read_csv(OUT / "sessions.csv"))
    sessions = sessions_full[["id", "day"]].dropna().drop_duplicates()
    sessions["id"] = sessions["id"].astype(int)
    sessions["day"] = sessions["day"].astype(int)
    rdoc_sessions = _load_rdoc_sessions() if args.rdoc_transition_analysis else None
    comorbidity_sessions = _load_comorbidity_sessions(args.comorbidity_features)
    methods = ["ball", "ball_teacher", "s0_direct_lgssm", "markov_direct_transition", "interpolation", "locf", "anchor_only"]
    all_strata, all_overall, calib_rows, boot_rows, transition_frames = [], [], [], [], []
    run_label = args.run_label or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUT / "empirical_fit_runs" / run_label
    archived_outputs = _archive_existing_final_outputs(run_dir) if args.clean_final_outputs else []
    completed_cadences: list[int] = []
    _write_fit_run_manifest(
        run_dir,
        status="started",
        requested_cadences=requested_cadences,
        completed_cadences=completed_cadences,
        args=args,
        archived_outputs=archived_outputs,
        sessions_full=sessions_full,
        rdoc_sessions=rdoc_sessions,
        comorbidity_sessions=comorbidity_sessions,
    )
    if archived_outputs:
        print(f"archived existing final empirical outputs under {run_dir / 'superseded_final_outputs'}", flush=True)
    print(f"clean empirical fit run directory: {run_dir}", flush=True)

    for cad in requested_cadences:
        cad_strata, cad_overall, cad_calib_rows, cad_boot_rows, cad_transition_frames = [], [], [], [], []
        anchors = pd.read_csv(OUT / f"anchors_sparse_{cad}d.csv")
        stats = _standardizers(anchors)
        print(f"cadence {cad}d: fitting BALL teacher/student transformer", flush=True)
        student_pred, teacher_pred, s0_pred, markov_pred, session_days = _fit_empirical_teacher_student(
            anchors, sessions_full, None, comorbidity_sessions, stats, args, cad
        )
        neural_predictions = {
            "student": _prediction_lookup(student_pred, "L_hat"),
            "teacher": _prediction_lookup(teacher_pred, "L_hat"),
            "s0": s0_pred,
            "markov": markov_pred,
        }
        ch_frames = []
        for ch, scales in CHANNELS.items():
            sub = anchors[anchors["anchor"].isin(scales)].copy()
            if args.max_patients is not None:
                sub = sub[sub["id"].astype(int).isin(neural_predictions["student"].keys())].copy()
            ev = _channel_eval(sub, sessions, stats, neural_predictions, session_days)
            if not ev.empty:
                ev["channel"] = ch
                ch_frames.append(ev)
            tr = _channel_transition_rows(sub, rdoc_sessions, stats, cad, ch, neural_predictions, session_days)
            if not tr.empty:
                transition_frames.append(tr)
                cad_transition_frames.append(tr)
        ev = pd.concat(ch_frames, ignore_index=True) if ch_frames else pd.DataFrame()
        if ev.empty:
            _write_fit_tables(run_dir, cad_overall, cad_strata, cad_calib_rows, cad_boot_rows, cad_transition_frames, suffix=f"_{cad}d")
            completed_cadences.append(int(cad))
            _write_fit_run_manifest(
                run_dir,
                status="in_progress",
                requested_cadences=requested_cadences,
                completed_cadences=completed_cadences,
                args=args,
                archived_outputs=archived_outputs,
                sessions_full=sessions_full,
                rdoc_sessions=rdoc_sessions,
                comorbidity_sessions=comorbidity_sessions,
            )
            continue
        ev["gap_bin"] = pd.cut(ev["gap"], bins=GAP_BINS, labels=GAP_LABELS, include_lowest=True, right=False)

        # Overall held-out RMSE per method (patient-clustered mean of squared error).
        per_pt_se = {}
        for m in methods:
            err2 = (ev[m] - ev["z_true"]) ** 2
            per_pt = err2.groupby(ev["id"]).mean()
            per_pt_se[m] = per_pt
            row = {"cadence": cad, "method": m,
                   "rmse": float(np.sqrt(per_pt.mean())),
                   "n": int(len(ev))}
            all_overall.append(row)
            cad_overall.append(row)
        # Patient-clustered bootstrap CIs for the BALL minus interpolation and
        # the BALL minus Markov RMSE differences (the old-school-versus-BALL
        # head-to-head). Resample patients, recompute pooled RMSE per method.
        pts = per_pt_se["ball"].index.to_numpy()
        rng = np.random.default_rng(101)
        diffs_interp, diffs_markov = [], []
        for _ in range(2000):
            samp = rng.choice(pts, size=len(pts), replace=True)
            rb = float(np.sqrt(per_pt_se["ball"].reindex(samp).mean()))
            ri = float(np.sqrt(per_pt_se["interpolation"].reindex(samp).mean()))
            rm = float(np.sqrt(per_pt_se["markov_direct_transition"].reindex(samp).mean()))
            diffs_interp.append(rb - ri)
            diffs_markov.append(rb - rm)
        diffs_interp = np.array(diffs_interp)
        diffs_markov = np.array(diffs_markov)
        boot_row = {"cadence": cad,
                    "ball_minus_interp_rmse": float(np.sqrt(per_pt_se["ball"].mean()) - np.sqrt(per_pt_se["interpolation"].mean())),
                    "ci_lo": float(np.quantile(diffs_interp, 0.025)),
                    "ci_hi": float(np.quantile(diffs_interp, 0.975)),
                    "ball_minus_markov_rmse": float(np.sqrt(per_pt_se["ball"].mean()) - np.sqrt(per_pt_se["markov_direct_transition"].mean())),
                    "markov_ci_lo": float(np.quantile(diffs_markov, 0.025)),
                    "markov_ci_hi": float(np.quantile(diffs_markov, 0.975)),
                    "n_patients": int(len(pts))}
        boot_rows.append(boot_row)
        cad_boot_rows.append(boot_row)
        # Gap-stratified RMSE per method.
        for m in methods:
            for lab in GAP_LABELS:
                mask = ev["gap_bin"] == lab
                if not mask.any():
                    continue
                err2 = (ev.loc[mask, m] - ev.loc[mask, "z_true"]) ** 2
                row = {"cadence": cad, "method": m, "gap_bin": lab,
                       "rmse": float(np.sqrt(err2.mean())), "n": int(mask.sum())}
                all_strata.append(row)
                cad_strata.append(row)

        # Split conformal on held-out anchors: half the patients calibrate, half test.
        ids = np.sort(ev["id"].unique())
        rng = np.random.default_rng(13)
        calib_ids = set(rng.choice(ids, size=len(ids) // 2, replace=False))
        cal = ev[ev["id"].isin(calib_ids)]
        tst = ev[~ev["id"].isin(calib_ids)]
        if len(cal) and len(tst):
            scores = np.abs(cal["ball"] - cal["z_true"]).to_numpy()
            q = _conformal_quantile(scores, CONFORMAL_ALPHA)
            covered = (np.abs(tst["ball"] - tst["z_true"]) <= q)
            per_pt = covered.groupby(tst["id"]).mean()
            z_sd = float(np.std(ev["z_true"]))
            calib_row = {"cadence": cad, "conformal_q": q,
                         "coverage": float(per_pt.mean()),
                         "rel_width": float(2 * q / z_sd) if z_sd else float("nan"),
                         "n_test": int(len(tst))}
            calib_rows.append(calib_row)
            cad_calib_rows.append(calib_row)
        print(f"cadence {cad}d: {len(ev)} held-out evaluations across {ev['id'].nunique()} patients", flush=True)
        _write_fit_tables(run_dir, cad_overall, cad_strata, cad_calib_rows, cad_boot_rows, cad_transition_frames, suffix=f"_{cad}d")
        _write_fit_tables(run_dir, all_overall, all_strata, calib_rows, boot_rows, transition_frames, suffix="_partial")
        completed_cadences.append(int(cad))
        _write_fit_run_manifest(
            run_dir,
            status="in_progress",
            requested_cadences=requested_cadences,
            completed_cadences=completed_cadences,
            args=args,
            archived_outputs=archived_outputs,
            sessions_full=sessions_full,
            rdoc_sessions=rdoc_sessions,
            comorbidity_sessions=comorbidity_sessions,
        )
        print(f"cadence {cad}d: wrote per-cadence outputs to {run_dir}", flush=True)

        # Release this cadence's GPU allocations before the next cadence builds
        # its tensors. The 8 GB device is marginal for the full cohort; the
        # caching allocator otherwise fragments across cadences (14d fits, 21d
        # OOMs). gc.collect() drops the now-unreferenced models from the
        # just-returned fit_ball_ssm call so empty_cache can actually free them.
        import gc as _gc
        import torch as _torch
        del student_pred, teacher_pred, s0_pred, markov_pred, neural_predictions, session_days
        _gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()

    overall, strata, calib, boot, transitions, transition_summary, transition_coef = _write_fit_tables(
        OUT, all_overall, all_strata, calib_rows, boot_rows, transition_frames
    )
    _write_fit_tables(run_dir, all_overall, all_strata, calib_rows, boot_rows, transition_frames, suffix="_complete")
    _write_fit_run_manifest(
        run_dir,
        status="complete",
        requested_cadences=requested_cadences,
        completed_cadences=completed_cadences,
        args=args,
        archived_outputs=archived_outputs,
        sessions_full=sessions_full,
        rdoc_sessions=rdoc_sessions,
        comorbidity_sessions=comorbidity_sessions,
    )
    print("\n=== BALL minus interpolation and BALL minus Markov RMSE, patient-clustered bootstrap 95% CI ===")
    print(boot.round(4).to_string(index=False))
    print("\n=== overall held-out RMSE (z units) by cadence and method ===")
    print(overall.pivot(index="cadence", columns="method", values="rmse").round(3).to_string())
    print("\n=== BALL conformal calibration by cadence ===")
    print(calib.round(3).to_string(index=False))
    if not transition_summary.empty:
        print("\n=== empirical direct-RDoC transition increment ===")
        print(transition_summary.round(4).to_string(index=False))
    print(f"\nwrote empirical_recovery_overall.csv, empirical_recovery_by_gap.csv, empirical_calibration.csv")
    print("wrote empirical_rdoc_transition.csv and empirical_rdoc_transition_coefficients.csv")


if __name__ == "__main__":
    main()
'''

# --- empirical.link_notes (empirical/link_notes.py) ---
SRC_EMPIRICAL_LINK_NOTES = r'''
"""Link the de-identified clinical notes to the rTMS session data.

The note export in raw/ is a long-format field/value table keyed by
PatientFID, AppointmentFID, and ServiceDate, the same keys as the session-level
analytic file. This script pivots the narrative note fields to one row per
appointment, attaches the BALL session and day index from the session data,
and writes a session-aligned note table. The combined note text per session is
the input the McCoy/Perlis RDoC scorer consumes to produce the six-dimensional
slow proxy B(t).

Privacy. The raw export carries demographic quasi-identifiers (DOB, gender,
city, ZIP). Those are NOT carried into derived outputs. Text-bearing scorer
inputs are written under empirical/derived_phi/ by default, while the
manuscript-facing empirical/derived/ directory receives only text-free
manifests and numeric scores.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT.parent / "raw"
DATA = ROOT / "data"
OUT = ROOT / "derived"
OUT.mkdir(parents=True, exist_ok=True)
PHI_OUT = ROOT / "derived_phi"
SESSION_DATA = DATA / "rtms_paper_analytic_sessions.csv"

# Narrative note fields, in the order they are concatenated into the combined
# note text. These are free-text clinical fields useful for RDoC scoring.
NARRATIVE_FIELDS = [
    "CC_ChiefComplaint",
    "HPI_Description",
    "HPI_Description_consult",
    "AP_Assessment",
    "AP_CarePlan",
    "pasthx",
    "pastmedicalhx",
    "socialhx",
    "develophx",
    "opiodhx",
    "CurrentMedsDetail",
    "Iop_Content",
    "Iop_Ipn",
    "TCR_additional",
    "TMS_Daily_Tx_PHQ_9_GAD_7_Notes",
]


def _load_notes() -> pd.DataFrame:
    files = sorted(glob.glob(str(RAW / "AMD_OutcomeData3_*.xlsx")))
    if not files:
        raise FileNotFoundError(f"No note export found in {RAW}")
    frames = [pd.read_excel(f) for f in files]
    notes = pd.concat(frames, ignore_index=True)
    notes["AppointmentFID"] = pd.to_numeric(notes["AppointmentFID"], errors="coerce")
    notes["PatientFID"] = pd.to_numeric(notes["PatientFID"], errors="coerce")
    notes["ServiceDate"] = pd.to_datetime(notes["ServiceDate"], errors="coerce")
    return notes


# Fields that carry a genuine clinical narrative (long free text).
RICH_FIELDS = [
    "HPI_Description", "HPI_Description_consult", "AP_Assessment", "AP_CarePlan",
    "pasthx", "pastmedicalhx", "socialhx", "develophx",
]
# Fields that carry a short structured signal (diagnosis, chief complaint).
DIAGNOSIS_FIELDS = ["CC_ChiefComplaint"]


def _note_tier(row) -> str:
    """Classify a note by the richness of its text for RDoC scoring."""

    def has(field, min_len):
        v = row.get(field)
        return isinstance(v, str) and v.strip().lower() != "nan" and len(v.strip()) >= min_len

    if any(has(f, 50) for f in RICH_FIELDS):
        return "rich_narrative"
    if any(has(f, 1) for f in DIAGNOSIS_FIELDS):
        return "diagnosis_only"
    return "daily_brief"


def _session_index() -> pd.DataFrame:
    """Recompute the BALL session and day index from the session data.

    Matches the convention in build_sparse_anchors so the note table aligns to
    the same per-patient session axis.
    """

    sess = _session_frame()
    return sess[["PatientFID", "AppointmentFID", "session", "day"]].rename(
        columns={"PatientFID": "id"}
    )


def _session_frame() -> pd.DataFrame:
    sess = pd.read_csv(SESSION_DATA, low_memory=False)
    sess["ServiceDate"] = pd.to_datetime(sess["ServiceDate"], errors="coerce")
    sess = sess.dropna(subset=["ServiceDate"]).copy()
    sess = sess.sort_values(["PatientFID", "ServiceDate"]).reset_index(drop=True)
    sess["session"] = sess.groupby("PatientFID").cumcount()
    first = sess.groupby("PatientFID")["ServiceDate"].transform("min")
    sess["day"] = (sess["ServiceDate"] - first).dt.days.astype(int)
    return sess


def _patient_first_dates() -> pd.Series:
    sess = _session_frame()
    return sess.groupby("PatientFID")["ServiceDate"].min()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Link clinical notes to the empirical rTMS session axis.")
    p.add_argument("--phi-out-dir", default=str(PHI_OUT),
                   help="Directory for text-bearing PHI scorer inputs. Keep this out of manuscript artifacts.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    phi_out = Path(args.phi_out_dir)
    phi_out.mkdir(parents=True, exist_ok=True)
    notes = _load_notes()
    narr = notes[notes["FieldName"].isin(NARRATIVE_FIELDS)].copy()
    narr = narr.dropna(subset=["AppointmentFID"])
    narr["Value"] = narr["Value"].astype(str)

    # One value string per (appointment, field). Multiple rows of the same field
    # for an appointment are joined with newlines.
    per_field = (
        narr.groupby(["PatientFID", "AppointmentFID", "ServiceDate", "FieldName"])["Value"]
        .apply(lambda s: "\n".join(v for v in s if v and v.lower() != "nan"))
        .reset_index()
    )
    wide = per_field.pivot_table(
        index=["PatientFID", "AppointmentFID", "ServiceDate"],
        columns="FieldName", values="Value", aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    # Combined note text per appointment, labeled by field, in NARRATIVE_FIELDS order.
    def combine(row) -> str:
        parts = []
        for field in NARRATIVE_FIELDS:
            val = row.get(field)
            if isinstance(val, str) and val.strip() and val.strip().lower() != "nan":
                parts.append(f"[{field}] {val.strip()}")
        return "\n\n".join(parts)

    wide["combined_note_text"] = wide.apply(combine, axis=1)
    wide["note_char_count"] = wide["combined_note_text"].str.len()

    # Attach the BALL session and day index by AppointmentFID.
    idx = _session_index()
    wide = wide.rename(columns={"PatientFID": "id"})
    wide["AppointmentFID"] = pd.to_numeric(wide["AppointmentFID"], errors="coerce")
    linked = wide.merge(idx, on=["id", "AppointmentFID"], how="left")
    in_session = linked["session"].notna()

    keep_cols = ["id", "AppointmentFID", "ServiceDate", "session", "day"] + \
                [c for c in NARRATIVE_FIELDS if c in linked.columns] + \
                ["combined_note_text", "note_char_count"]
    full = linked[keep_cols].copy()
    full.to_csv(phi_out / "session_notes_wide.csv", index=False)

    # Compact RDoC-scorer input. Only sessions that link to the rTMS session axis.
    scorer_input = linked.loc[in_session,
                              ["id", "AppointmentFID", "ServiceDate", "session", "day",
                               "combined_note_text", "note_char_count"]].copy()
    scorer_input["session"] = scorer_input["session"].astype(int)
    scorer_input["day"] = scorer_input["day"].astype(int)
    scorer_input = scorer_input[scorer_input["note_char_count"] > 0]
    scorer_input.to_csv(phi_out / "session_notes.csv", index=False)

    # ------------------------------------------------------------------
    # RDoC-scorer input on the patient day-axis. The substantive narratives
    # (HPI, assessment, care plan, histories) sit on periodic follow-up and MD
    # visits, not daily treatment sessions, so they often do not share an
    # AppointmentFID with a TMS session. We instead place every narrative note
    # on the patient's day-axis by ServiceDate (day = days since the patient's
    # first TMS session). The RDoC scorer scores each note, and the
    # slow proxy B(t) is the most recent profile carried forward to each session
    # day. This matches the slow, slowly varying nature of the proxy.
    # ------------------------------------------------------------------
    first_dates = _patient_first_dates()
    wide["ServiceDate"] = pd.to_datetime(wide["ServiceDate"], errors="coerce")
    rdoc = wide[wide["id"].isin(first_dates.index)].copy()
    rdoc["first_session_date"] = rdoc["id"].map(first_dates)
    rdoc["day"] = (rdoc["ServiceDate"] - rdoc["first_session_date"]).dt.days
    rdoc["note_tier"] = rdoc.apply(_note_tier, axis=1)
    rdoc = rdoc[rdoc["note_char_count"] > 0]
    rdoc_cols = ["id", "AppointmentFID", "ServiceDate", "day", "note_tier",
                 "combined_note_text", "note_char_count"]
    rdoc[rdoc_cols].sort_values(["id", "day"]).to_csv(phi_out / "rdoc_input.csv", index=False)
    tier_counts = rdoc["note_tier"].value_counts().to_dict()
    n_rich = int(tier_counts.get("rich_narrative", 0))
    rich_pts = int(rdoc.loc[rdoc["note_tier"] == "rich_narrative", "id"].nunique())

    manifest = {
        "note_files": [Path(f).name for f in sorted(glob.glob(str(RAW / "AMD_OutcomeData3_*.xlsx")))],
        "note_date_range": [str(notes["ServiceDate"].min()), str(notes["ServiceDate"].max())],
        "note_appointments_total": int(notes["AppointmentFID"].nunique()),
        "narrative_appointments": int(wide.shape[0]),
        "session_linked_daily_notes": int(in_session.sum()),
        "session_linked_patients": int(linked.loc[in_session, "id"].nunique()),
        "mean_chars_session_daily_note": float(scorer_input["note_char_count"].mean()) if len(scorer_input) else 0.0,
        "rdoc_input_rows": int(len(rdoc)),
        "rdoc_input_patients": int(rdoc["id"].nunique()),
        "note_tier_counts": {k: int(v) for k, v in tier_counts.items()},
        "rich_narrative_notes": n_rich,
        "rich_narrative_patients": rich_pts,
        "narrative_fields": NARRATIVE_FIELDS,
        "rich_fields": RICH_FIELDS,
        "diagnosis_fields": DIAGNOSIS_FIELDS,
        "privacy": "demographic quasi-identifiers (DOB, gender, city, ZIP) dropped; text-bearing scorer inputs are written outside manuscript-facing derived artifacts",
        "phi_output_dir": str(phi_out.relative_to(ROOT.parent) if phi_out.is_relative_to(ROOT.parent) else phi_out),
        "text_bearing_files": [
            str((phi_out / "session_notes_wide.csv").relative_to(ROOT.parent) if phi_out.is_relative_to(ROOT.parent) else phi_out / "session_notes_wide.csv"),
            str((phi_out / "session_notes.csv").relative_to(ROOT.parent) if phi_out.is_relative_to(ROOT.parent) else phi_out / "session_notes.csv"),
            str((phi_out / "rdoc_input.csv").relative_to(ROOT.parent) if phi_out.is_relative_to(ROOT.parent) else phi_out / "rdoc_input.csv"),
        ],
        "link_key": "AppointmentFID with PatientFID for daily session notes; PatientFID + ServiceDate day-offset for the RDoC timeline",
        "finding": "rich clinical narratives (HPI, assessment, care plan) are sparse and concentrated at periodic follow-up and MD visits. Most session-linked notes are short daily treatment notes or diagnosis stubs. The RDoC slow proxy B(t) will therefore be near-constant per patient. Treat it as a weak slow proxy and run the documentation-density and note-sparsity fragility checks from the simulation before any decomposition claim.",
        "next_step": "score rdoc_input rows (prioritize note_tier rich_narrative, then diagnosis_only) with the McCoy/Perlis RDoC scorer into 6-dim B(t), carry forward to session days on the day-axis, attach to empirical/derived/sessions.csv",
    }
    (OUT / "session_notes_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"narrative appointments: {wide.shape[0]}")
    print(f"session-linked daily notes: {int(in_session.sum())} appts, "
          f"{int(linked.loc[in_session, 'id'].nunique())} patients, "
          f"mean {manifest['mean_chars_session_daily_note']:.0f} chars")
    print(f"RDoC timeline input: {len(rdoc)} notes, {rdoc['id'].nunique()} patients")
    print(f"  note tiers: {tier_counts}")
    print(f"  rich narratives: {n_rich} notes across {rich_pts} patients")
    print(f"wrote text-bearing scorer inputs under {phi_out}")
    print(f"wrote text-free manifest {OUT/'session_notes_manifest.json'}")


if __name__ == "__main__":
    main()
'''

# --- empirical.make_figureS5 (empirical/make_figureS5.py) ---
SRC_EMPIRICAL_MAKE_FIGURES5 = r'''
"""Supplementary Figure 5. Empirical held-out recovery and calibration."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DER = ROOT / "derived"
FIGOUT = ROOT.parent / "simulations" / "paper" / "outputs"
FIGOUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 9, "savefig.dpi": 200})

METHODS = ["ball", "markov", "interpolation", "locf", "anchor_only"]
LABELS = {"ball": "BALL", "markov": "Markov", "interpolation": "interpolation",
          "locf": "LOCF", "anchor_only": "anchor-only"}
COLORS = {"ball": "#37b", "markov": "#a05", "interpolation": "#e73",
          "locf": "#999", "anchor_only": "#3b7"}


def main() -> None:
    overall = pd.read_csv(DER / "empirical_recovery_overall.csv")
    strata = pd.read_csv(DER / "empirical_recovery_by_gap.csv")
    calib = pd.read_csv(DER / "empirical_calibration.csv")
    cads = sorted(overall["cadence"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))

    # Panel A. Overall held-out RMSE by method across cadences.
    ax = axes[0]
    x = np.arange(len(cads))
    w = 0.16
    offset = (len(METHODS) - 1) / 2.0
    for k, m in enumerate(METHODS):
        vals = [overall[(overall.cadence == c) & (overall.method == m)]["rmse"].iloc[0] for c in cads]
        ax.bar(x + (k - offset) * w, vals, w, label=LABELS[m], color=COLORS[m])
    ax.set_xticks(x); ax.set_xticklabels([f"{c}d" for c in cads])
    ax.set_xlabel("Anchor cadence"); ax.set_ylabel("Held-out RMSE (latent SD units)")
    ax.set_title("A. Held-out recovery by method")
    ax.legend(fontsize=7, loc="upper left")

    # Panel B. Gap-stratified RMSE at the sparsest cadence.
    ax = axes[1]
    cad = max(cads)
    sub = strata[strata.cadence == cad]
    gaps = [g for g in ["0-1", "1-2", "2-4", "4-7", "7+"] if g in set(sub["gap_bin"])]
    for m in ["ball", "markov", "interpolation", "anchor_only"]:
        vals = [sub[(sub.gap_bin == g) & (sub.method == m)]["rmse"].iloc[0]
                if len(sub[(sub.gap_bin == g) & (sub.method == m)]) else np.nan for g in gaps]
        ax.plot(range(len(gaps)), vals, "o-", label=LABELS[m], color=COLORS[m])
    ax.set_xticks(range(len(gaps))); ax.set_xticklabels(gaps)
    ax.set_xlabel("Days to nearest retained anchor"); ax.set_ylabel("Held-out RMSE")
    ax.set_title(f"B. By anchor distance ({cad}d cadence)")
    ax.legend(fontsize=7)

    # Panel C. BALL conformal calibration by cadence.
    ax = axes[2]
    ax.plot(calib["cadence"], calib["coverage"], "o-", color="#37b", label="coverage")
    ax.axhline(0.95, ls="--", c="k", lw=0.8)
    ax.set_ylim(0.85, 1.0); ax.set_xlabel("Anchor cadence (days)"); ax.set_ylabel("Coverage")
    ax2 = ax.twinx()
    ax2.plot(calib["cadence"], calib["rel_width"], "s--", color="#e73", label="rel width")
    ax2.set_ylabel("Relative width (latent SD)")
    ax.set_title("C. Anchor-conformal calibration")

    fig.tight_layout()
    fig.savefig(FIGOUT / "figureS5_empirical.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGOUT / "figureS5_empirical.png")


if __name__ == "__main__":
    main()
'''

# --- empirical.rdoc_score (empirical/rdoc_score.py) ---
SRC_EMPIRICAL_RDOC_SCORE = r'''
"""Score linked clinical notes into a six-dimensional RDoC profile.

This emulates the structure of the McCoy and Perlis dictionary approach to
extracting Research Domain Criteria (RDoC) dimensions from clinical text. Each
note is scored against a term dictionary for each of the six RDoC domains. The
raw score for a domain is the count of matched dictionary terms divided by the
note token count, which normalizes for note length. Scores are then
standardized across the corpus so each domain has mean zero and unit variance.

The term dictionaries here are transparent RDoC seed lexicons. They are not the
proprietary embedding-expanded dictionaries from the source paper. The official
dictionaries should be dropped in when available by replacing RDOC_TERMS. The
scoring pipeline remains unchanged.

Privacy. Note text is read locally and only the six numeric scores per note are
written. No note text leaves this process.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "derived"

# Six RDoC domains (the 2024 matrix, including Sensorimotor). Seed term lists.
# Terms are matched as whole words, case-insensitive, after light stemming of
# trailing s/ed/ing.
RDOC_TERMS = {
    "negative_valence": [
        "fear", "anxious", "anxiety", "panic", "worry", "worried", "threat",
        "sad", "sadness", "depress", "hopeless", "guilt", "worthless", "loss",
        "grief", "frustrat", "anger", "irritable", "distress", "tearful",
        "suicid", "self harm", "dread",
    ],
    "positive_valence": [
        "reward", "motivat", "anhedonia", "pleasure", "interest", "enjoy",
        "hobby", "apath", "reinforce", "craving", "goal", "engaged",
        "looking forward", "joy",
    ],
    "cognitive_systems": [
        "attention", "concentrat", "memory", "forget", "cognit", "confus",
        "focus", "distract", "executive", "decision", "comprehension",
        "orientation", "disorient", "processing",
    ],
    "social_processes": [
        "social", "relationship", "family", "friend", "isolat", "withdrawn",
        "lonely", "attachment", "communicat", "interpersonal", "support",
        "conflict", "marital", "spouse", "partner",
    ],
    "arousal_regulatory": [
        "sleep", "insomnia", "arousal", "agitat", "restless", "energy",
        "fatigue", "tired", "circadian", "appetite", "hypervigilan",
        "startle", "tense", "hyperarousal",
    ],
    "sensorimotor": [
        "motor", "movement", "psychomotor", "retardation", "slowed",
        "tremor", "gait", "coordination", "akathisia", "restlessness",
        "posture", "reflex",
    ],
}


def _build_patterns():
    pats = {}
    for domain, terms in RDOC_TERMS.items():
        # Match the term as a word prefix so light stemming is captured.
        escaped = [re.escape(t) for t in terms]
        pats[domain] = re.compile(r"\b(" + "|".join(escaped) + r")", re.IGNORECASE)
    return pats


_TOKEN = re.compile(r"[A-Za-z]+")


def score_text(text: str, patterns) -> dict[str, float]:
    if not isinstance(text, str) or not text.strip():
        return {d: 0.0 for d in RDOC_TERMS}
    n_tokens = max(len(_TOKEN.findall(text)), 1)
    return {d: len(pat.findall(text)) / n_tokens for d, pat in patterns.items()}


def main() -> None:
    src = OUT / "rdoc_input.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found. Run empirical/link_notes.py first.")
    df = pd.read_csv(src)
    patterns = _build_patterns()
    raw = df["combined_note_text"].apply(lambda t: score_text(t, patterns)).apply(pd.Series)
    raw.columns = [f"rdoc_{c}" for c in raw.columns]

    # Standardize each domain across the corpus (mean zero, unit variance).
    cols = list(raw.columns)
    means = raw[cols].mean()
    sds = raw[cols].std().replace(0.0, 1.0)
    std = (raw[cols] - means) / sds

    out = pd.concat([df[["id", "AppointmentFID", "ServiceDate", "day", "note_tier", "note_char_count"]],
                     std], axis=1)
    out.to_csv(OUT / "rdoc_scores.csv", index=False)

    manifest = {
        "scorer": "dictionary-based RDoC emulation (McCoy/Perlis structure); seed lexicons, not the official expanded dictionaries",
        "domains": list(RDOC_TERMS.keys()),
        "n_notes_scored": int(len(out)),
        "raw_score": "matched RDoC terms divided by note token count",
        "standardization": "per-domain z-score across the scored corpus",
        "mean_nonzero_domains_per_note": float((raw[cols] > 0).sum(axis=1).mean()),
        "note_tier_counts": df["note_tier"].value_counts().to_dict() if "note_tier" in df else {},
        "caveat": "replace RDOC_TERMS with the official McCoy/Perlis dictionaries for the final analysis; scores on short diagnosis-only notes are coarse",
    }
    (OUT / "rdoc_scores_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"scored {len(out)} notes into 6 RDoC domains")
    print(f"mean nonzero domains per note: {manifest['mean_nonzero_domains_per_note']:.2f}")
    print("by tier, mean matched-term rate (any domain):")
    rate = (raw[cols].sum(axis=1)).groupby(df["note_tier"]).mean()
    print(rate.to_string())
    print(f"wrote {OUT/'rdoc_scores.csv'}, {OUT/'rdoc_scores_manifest.json'}")


if __name__ == "__main__":
    main()
'''

PACKAGE_SOURCES = {
    'simulations': ('simulations/__init__.py', SRC_SIMULATIONS),
    'simulations.src': ('simulations/src/__init__.py', SRC_SIMULATIONS_SRC),
    'simulations.src.methods': ('simulations/src/methods/__init__.py', SRC_SIMULATIONS_SRC_METHODS),
    'simulations.paper': ('simulations/paper/__init__.py', SRC_SIMULATIONS_PAPER),
    'empirical': ('empirical/__init__.py', SRC_EMPIRICAL),
}

MODULE_SOURCES = {
    'simulations.src.model_utils': ('simulations/src/model_utils.py', SRC_SIMULATIONS_SRC_MODEL_UTILS),
    'simulations.src.missingness': ('simulations/src/missingness.py', SRC_SIMULATIONS_SRC_MISSINGNESS),
    'simulations.src.anchors': ('simulations/src/anchors.py', SRC_SIMULATIONS_SRC_ANCHORS),
    'simulations.src.dgp': ('simulations/src/dgp.py', SRC_SIMULATIONS_SRC_DGP),
    'simulations.src.diagnostics': ('simulations/src/diagnostics.py', SRC_SIMULATIONS_SRC_DIAGNOSTICS),
    'simulations.src.metrics': ('simulations/src/metrics.py', SRC_SIMULATIONS_SRC_METRICS),
    'simulations.src.methods.baselines': ('simulations/src/methods/baselines.py', SRC_SIMULATIONS_SRC_METHODS_BASELINES),
    'simulations.src.methods.s0': ('simulations/src/methods/s0.py', SRC_SIMULATIONS_SRC_METHODS_S0),
    'simulations.src.methods.ball_structural': ('simulations/src/methods/ball_structural.py', SRC_SIMULATIONS_SRC_METHODS_BALL_STRUCTURAL),
    'simulations.src.methods.markov_pattern_mixture': ('simulations/src/methods/markov_pattern_mixture.py', SRC_SIMULATIONS_SRC_METHODS_MARKOV_PATTERN_MIXTURE),
    'simulations.src.methods.ball_ssm': ('simulations/src/methods/ball_ssm.py', SRC_SIMULATIONS_SRC_METHODS_BALL_SSM),
    'simulations.run_ball_pipeline': ('simulations/run_ball_pipeline.py', SRC_SIMULATIONS_RUN_BALL_PIPELINE),
    'simulations.run_battle_tests': ('simulations/run_battle_tests.py', SRC_SIMULATIONS_RUN_BATTLE_TESTS),
    'simulations.run_pilot_batch': ('simulations/run_pilot_batch.py', SRC_SIMULATIONS_RUN_PILOT_BATCH),
    'simulations.paper.plotting': ('simulations/paper/plotting.py', SRC_SIMULATIONS_PAPER_PLOTTING),
    'simulations.paper.make_figure1': ('simulations/paper/make_figure1.py', SRC_SIMULATIONS_PAPER_MAKE_FIGURE1),
    'simulations.paper.scenarios': ('simulations/paper/scenarios.py', SRC_SIMULATIONS_PAPER_SCENARIOS),
    'simulations.paper.run_fig2_recoverability': ('simulations/paper/run_fig2_recoverability.py', SRC_SIMULATIONS_PAPER_RUN_FIG2_RECOVERABILITY),
    'simulations.paper.run_fig3_calibration': ('simulations/paper/run_fig3_calibration.py', SRC_SIMULATIONS_PAPER_RUN_FIG3_CALIBRATION),
    'simulations.paper.run_fig4_identifiability': ('simulations/paper/run_fig4_identifiability.py', SRC_SIMULATIONS_PAPER_RUN_FIG4_IDENTIFIABILITY),
    'simulations.paper.run_rts_identification': ('simulations/paper/run_rts_identification.py', SRC_SIMULATIONS_PAPER_RUN_RTS_IDENTIFICATION),
    'simulations.paper.run_stress_panels': ('simulations/paper/run_stress_panels.py', SRC_SIMULATIONS_PAPER_RUN_STRESS_PANELS),
    'simulations.paper.make_tables': ('simulations/paper/make_tables.py', SRC_SIMULATIONS_PAPER_MAKE_TABLES),
    'simulations.paper.run_all': ('simulations/paper/run_all.py', SRC_SIMULATIONS_PAPER_RUN_ALL),
    'empirical.assemble_rdoc_llm': ('empirical/assemble_rdoc_llm.py', SRC_EMPIRICAL_ASSEMBLE_RDOC_LLM),
    'empirical.build_comorbidities': ('empirical/build_comorbidities.py', SRC_EMPIRICAL_BUILD_COMORBIDITIES),
    'empirical.build_rdoc_proxy': ('empirical/build_rdoc_proxy.py', SRC_EMPIRICAL_BUILD_RDOC_PROXY),
    'empirical.build_sparse_anchors': ('empirical/build_sparse_anchors.py', SRC_EMPIRICAL_BUILD_SPARSE_ANCHORS),
    'empirical.fit_empirical': ('empirical/fit_empirical.py', SRC_EMPIRICAL_FIT_EMPIRICAL),
    'empirical.link_notes': ('empirical/link_notes.py', SRC_EMPIRICAL_LINK_NOTES),
    'empirical.make_figureS5': ('empirical/make_figureS5.py', SRC_EMPIRICAL_MAKE_FIGURES5),
    'empirical.rdoc_score': ('empirical/rdoc_score.py', SRC_EMPIRICAL_RDOC_SCORE),
}

COMMANDS = {
    'pipeline': 'simulations.run_ball_pipeline',
    'battle-tests': 'simulations.run_battle_tests',
    'pilot-batch': 'simulations.run_pilot_batch',
    'paper-figure1': 'simulations.paper.make_figure1',
    'paper-fig2': 'simulations.paper.run_fig2_recoverability',
    'paper-fig3': 'simulations.paper.run_fig3_calibration',
    'paper-fig4': 'simulations.paper.run_fig4_identifiability',
    'paper-rts': 'simulations.paper.run_rts_identification',
    'paper-stress': 'simulations.paper.run_stress_panels',
    'paper-tables': 'simulations.paper.make_tables',
    'paper-all': 'simulations.paper.run_all',
    'empirical-assemble-rdoc-llm': 'empirical.assemble_rdoc_llm',
    'empirical-build-comorbidities': 'empirical.build_comorbidities',
    'empirical-build-rdoc-proxy': 'empirical.build_rdoc_proxy',
    'empirical-build-anchors': 'empirical.build_sparse_anchors',
    'empirical-fit': 'empirical.fit_empirical',
    'empirical-link-notes': 'empirical.link_notes',
    'empirical-figureS5': 'empirical.make_figureS5',
    'empirical-score-rdoc': 'empirical.rdoc_score',
}

_LOADED = False

def _bind_parent(name: str, module: types.ModuleType) -> None:
    parent_name, _, child = name.rpartition(".")
    if parent_name and parent_name in sys.modules:
        setattr(sys.modules[parent_name], child, module)


def _make_module(name: str, rel_path: str, *, is_package: bool) -> types.ModuleType:
    module = types.ModuleType(name)
    file_path = REPO_ROOT / rel_path
    module.__file__ = str(file_path)
    module.__loader__ = None
    if is_package:
        module.__package__ = name
        module.__path__ = [str(file_path.parent)]
        module.__spec__ = importlib.util.spec_from_loader(name, loader=None, is_package=True)
    else:
        module.__package__ = name.rpartition(".")[0]
        module.__spec__ = importlib.util.spec_from_loader(name, loader=None, is_package=False)
    return module


def _install_virtual_package() -> None:
    global _LOADED
    if _LOADED:
        return
    for name, (rel_path, source) in PACKAGE_SOURCES.items():
        module = _make_module(name, rel_path, is_package=True)
        sys.modules[name] = module
        _bind_parent(name, module)
        if source:
            exec(compile(source, str(REPO_ROOT / rel_path), "exec"), module.__dict__)
    for name, (rel_path, source) in MODULE_SOURCES.items():
        module = _make_module(name, rel_path, is_package=False)
        sys.modules[name] = module
        _bind_parent(name, module)
        exec(compile(source, str(REPO_ROOT / rel_path), "exec"), module.__dict__)
    _LOADED = True


def _run_module_main(module_name: str, argv: list[str]) -> None:
    _install_virtual_package()
    module = sys.modules[module_name]
    if not hasattr(module, "main"):
        raise SystemExit(f"Embedded module {module_name} has no main().")
    old_argv = sys.argv[:]
    sys.argv = [getattr(module, "__file__", module_name), *argv]
    try:
        module.main()
    finally:
        sys.argv = old_argv


def _print_commands() -> None:
    print("Available BALL.py commands:")
    for command in sorted(COMMANDS):
        print(f"  {command}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonical one-file BALL simulation, empirical, and paper-code runner.",
        add_help=False,
    )
    parser.add_argument("command", nargs="?")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parser.add_argument("-h", "--help", action="store_true")
    ns = parser.parse_args()
    if ns.help or ns.command is None or ns.command in {"help", "commands", "list"}:
        _print_commands()
        return
    if ns.command not in COMMANDS:
        _print_commands()
        raise SystemExit(f"Unknown command: {ns.command}")
    _run_module_main(COMMANDS[ns.command], ns.args)


if __name__ == "__main__":
    main()
