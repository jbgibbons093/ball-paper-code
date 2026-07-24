#!/usr/bin/env python
"""Build the frozen graded-response IRT calibration for the observation-model sensitivity.

The IRT sensitivity treats the questionnaire as a CALIBRATED INSTRUMENT: its trait
location/scale, item discriminations, and item-specific ordered thresholds are fixed
in advance from an INDEPENDENT calibration population (a dedicated seed disjoint from
every analysis seed) and then reused, byte-for-byte, across all seeds, shares,
methods, and negative controls. Fixing the instrument is what identifies the latent
scale: items-only IRT with jointly free item and latent parameters does not.

The model latent is expressed on the calibrated trait scale via a fixed affine map
    trait = (windowed_latent - loc) / scale,
so the calibration population's trait has mean 0 and SD 1 and item thresholds sit on
an approximately [-3, +3] standardized scale. Per-item thresholds overlap (each item
has its own ordered cut points) rather than sharing one grid.

This script writes validation/irt_calibration.json. It must be run once, before any
benchmark, and the resulting artifact is the single source of truth for the link.
Nothing here reads or depends on any analysis run.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ball_validation_harness as H  # noqa: E402

# --- Frozen calibration definition (do not change without re-deriving the artifact) ---
CALIBRATION_SEED = 90210                       # disjoint from analysis seeds (1729, 2027, ...)
CALIBRATION_CELLS_SHARES = [("linear", 0.10), ("linear", 0.25)]  # representative drift regime
CALIBRATION_N = 400                            # patients per (cell, share) block
CALIBRATION_T = 84
N_ITEMS = 9
N_CATEGORIES = 4                               # K categories -> K-1 ordered thresholds per item
DISCRIMINATION = 1.5                           # fixed per-item slope on the trait scale
ITEM_DIFFICULTY_RANGE = (-2.0, 2.0)            # item locations spread across the trait scale
WITHIN_ITEM_SPACING = (-1.0, 1.0)              # category cut offsets around each item location
THRESHOLD_JITTER_SD = 0.15                     # mild per-item overlap (deterministic from the seed)

OUT_PATH = Path(__file__).resolve().parent / "irt_calibration.json"


def _calibration_windowed_latent() -> np.ndarray:
    """Pool the windowed-latent the instrument reads over the independent calibration population."""
    blocks = []
    for k, (cell, share) in enumerate(CALIBRATION_CELLS_SHARES):
        cfg = dataclasses.replace(
            H.SimulationConfig(),
            seed=CALIBRATION_SEED + k,
            n=CALIBRATION_N,
            t=CALIBRATION_T,
            slow_fraction=0.75,
            delta_ar=tuple([0.3] * H.N_SUBTYPES),
            rdoc_drift_share=float(share),
            rdoc_drift_active_dims=3,
            rdoc_drift_beta_seed=31011,
            rdoc_drift_min_abs=0.35,
            anchor_observation="gaussian",   # only the windowed latent is needed, not IRT anchors
        )
        data = H.generate_dataset(cfg)
        anc = data.anchors
        for col in ("window_mean_zd", "window_mean_zp"):
            v = anc.loc[anc["observed"], col].to_numpy(dtype=float)
            blocks.append(v[np.isfinite(v)])
    return np.concatenate(blocks)


def _design_items(rng: np.random.Generator) -> tuple[list[float], list[list[float]]]:
    """Fixed per-item ordered thresholds (overlapping) and discriminations on the trait scale."""
    difficulties = np.linspace(*ITEM_DIFFICULTY_RANGE, N_ITEMS)
    spacing = np.linspace(*WITHIN_ITEM_SPACING, N_CATEGORIES - 1)
    thresholds = []
    for d in difficulties:
        jit = rng.normal(0.0, THRESHOLD_JITTER_SD, size=N_CATEGORIES - 1)
        thresholds.append(np.sort(d + spacing + jit).tolist())
    discriminations = [float(DISCRIMINATION)] * N_ITEMS
    return discriminations, thresholds


def _grm_item_category_probs(trait: np.ndarray, a: float, b: np.ndarray) -> np.ndarray:
    """Category probabilities P(item = c | trait) for one item, c = 0..K-1. Shape (N, K)."""
    p_ge = 1.0 / (1.0 + np.exp(-a * (trait[:, None] - b[None, :])))   # (N, K-1) = P(>=1..K-1)
    pad = np.concatenate([np.ones((trait.size, 1)), p_ge, np.zeros((trait.size, 1))], axis=1)
    return np.clip(pad[:, :-1] - pad[:, 1:], 0.0, 1.0)               # (N, K)


def _expected_stats(trait: np.ndarray, disc, thresholds):
    """Expected category frequencies and floor/ceiling rates over the calibration trait."""
    cat_freq = np.zeros(N_CATEGORIES)
    p0 = np.ones(trait.size)        # P(total at floor) = prod_j P(item_j = 0)
    pmax = np.ones(trait.size)      # P(total at ceiling) = prod_j P(item_j = K-1)
    for a, b in zip(disc, thresholds):
        probs = _grm_item_category_probs(trait, float(a), np.asarray(b))
        cat_freq += probs.mean(axis=0)
        p0 *= probs[:, 0]
        pmax *= probs[:, -1]
    cat_freq /= N_ITEMS
    return cat_freq.tolist(), float(p0.mean()), float(pmax.mean())


def main() -> None:
    wz = _calibration_windowed_latent()
    loc = float(wz.mean())
    scale = float(wz.std())
    rng = np.random.default_rng(np.random.SeedSequence([CALIBRATION_SEED, 7]))
    disc, thresholds = _design_items(rng)

    trait = (wz - loc) / scale
    cat_freq, floor_rate, ceil_rate = _expected_stats(trait, disc, thresholds)

    spec = {
        "calibration_seed": CALIBRATION_SEED,
        "sample_definition": {
            "cells_shares": CALIBRATION_CELLS_SHARES,
            "n_per_block": CALIBRATION_N,
            "t": CALIBRATION_T,
            "n_windowed_latent": int(wz.size),
            "note": "independent calibration population; disjoint seed; excluded from all analysis runs",
        },
        "affine": {"loc": loc, "scale": scale,
                   "definition": "trait = (windowed_latent - loc) / scale"},
        "n_items": N_ITEMS,
        "n_categories": N_CATEGORIES,
        "discriminations": disc,
        "thresholds": thresholds,          # (n_items, n_categories - 1), ordered per item
        "diagnostics": {
            "calibration_trait_mean": float(trait.mean()),
            "calibration_trait_sd": float(trait.std()),
            "expected_category_frequencies": cat_freq,
            "floor_rate": floor_rate,
            "ceiling_rate": ceil_rate,
        },
    }
    OUT_PATH.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    print(f"affine: loc={loc:.3f} scale={scale:.3f}  (trait mean {trait.mean():.3f} sd {trait.std():.3f})")
    print(f"expected category freqs: {[round(c,3) for c in cat_freq]}")
    print(f"floor rate={floor_rate:.4f}  ceiling rate={ceil_rate:.4f}")


if __name__ == "__main__":
    main()
