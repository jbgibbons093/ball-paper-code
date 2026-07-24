#!/usr/bin/env python
"""Anchor information-structure identifiability check (Codex's revised next step).

Validation A showed the residual time scale delta_phi is not recoverable from the
sparse windowed anchors: honest validation-anchor selection lands near a random
walk and the RDoC decomposition collapses, even though the matched (oracle) prior
recovers it. The open question is whether that is a true limitation or an artifact
of the simulation's anchor information structure, since the real PHQ-9 is a denser,
cleaner severity observation.

This check varies only the anchor observation model (config-only, no BALL.py edit)
on the reverting empirical-calibrated DGP, and asks whether a denser, cleaner, or
shorter-recall anchor makes delta_phi estimable from honest observables. It
isolates the three anchor knobs (noise, cadence, recall window) and a combined
near-direct variant, because a long recall window averages the fast residual away
before it is observed and is the suspected reason the time scale is unidentifiable.

Decision rule (Codex): if better anchors move the observable-selected delta_phi
toward the true reverting value and restore the decomposition, the earlier failure
is a simulation information-structure artifact and the DGP should change before
transformer step B. If selection still lands near a random walk and the
decomposition stays collapsed, the decomposition claim is prior-dependent and the
manuscript must say so.

Estimated delta_phi is always selected from deployable observables only. Matched
delta_phi rows are oracle references.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ball_validation_harness as H  # noqa: E402  (triggers BALL load + virtual package)

AnchorSpec = sys.modules["simulations.src.model_utils"].AnchorSpec
generate_dataset = H.generate_dataset

# AnchorSpec(name, cadence, recall_window, loading, error_sd, missing_probability).
# Defaults (BALL.py): Y1=(7, 14, 1.0, 0.95, 0.10), Y2=(7, 7, 0.75, 0.85, 0.20).
DEFAULT_Y1 = ("Y1", 7, 14, 1.0, 0.95, 0.10)
DEFAULT_Y2 = ("Y2", 7, 7, 0.75, 0.85, 0.20)


def _spec(name, cadence, recall_window, loading, error_sd, missing):
    return AnchorSpec(name, cadence, recall_window, loading, error_sd, missing)


# Each variant overrides (y1, y2). Knobs isolated: noise (error_sd, missing),
# cadence (days between anchors), recall_window (averaging length, the suspected
# identifiability knob). near-direct combines all three: dense, clean, same-day.
ANCHOR_VARIANTS = {
    "sparse-baseline": (_spec(*DEFAULT_Y1), _spec(*DEFAULT_Y2)),
    "cleaner": (_spec("Y1", 7, 14, 1.0, 0.30, 0.05), _spec("Y2", 7, 7, 0.75, 0.30, 0.05)),
    "denser-4day": (_spec("Y1", 4, 14, 1.0, 0.95, 0.10), _spec("Y2", 4, 7, 0.75, 0.85, 0.20)),
    "short-window": (_spec("Y1", 7, 3, 1.0, 0.95, 0.10), _spec("Y2", 7, 3, 0.75, 0.85, 0.20)),
    "near-direct": (_spec("Y1", 3, 1, 1.0, 0.30, 0.05), _spec("Y2", 3, 1, 1.0, 0.30, 0.05)),
}
VARIANT_ORDER = ["sparse-baseline", "cleaner", "denser-4day", "short-window", "near-direct"]

# empirical-cal is the decisive reverting cell; baseline is the random-walk control.
DEFAULT_CELLS = ["empirical-cal", "baseline"]
DEFAULT_SEEDS = [1729, 2027, 2028]
DEFAULT_N = 200
QUICK_SEEDS = [1729]
QUICK_N = 120


def make_config(cell_name, seed, n, t, y1, y2):
    knobs = H.CELLS[cell_name]
    return dataclasses.replace(
        H.SimulationConfig(),
        seed=seed,
        n=n,
        t=t,
        slow_fraction=knobs["slow_fraction"],
        delta_ar=tuple([float(knobs["delta_ar"])] * H.N_SUBTYPES),
        y1=y1,
        y2=y2,
    )


def make_cached_fit(data):
    cache = {}

    def f(phi):
        key = round(float(phi), 6)
        if key not in cache:
            cache[key] = H.fit_structural(data, key)
        return cache[key]

    return f


def n_observed_val_anchors(data) -> int:
    val_ids = set(data.individuals.loc[data.individuals["split"] == "validation", "id"])
    anc = data.anchors
    return int(((anc["observed"]) & (anc["id"].isin(val_ids))).sum())


def run_one(cell_name, variant, seed, n, t) -> dict:
    y1, y2 = ANCHOR_VARIANTS[variant]
    data = generate_dataset(make_config(cell_name, seed, n, t, y1, y2))
    cached_fit = make_cached_fit(data)
    matched_phi = float(H.CELLS[cell_name]["delta_ar"])

    matched = H.score(data, cached_fit(matched_phi))
    sel_phi, curve = H.select_delta_phi(data, cached_fit)
    estimated = H.score(data, cached_fit(sel_phi))

    return {
        "cell": cell_name,
        "variant": variant,
        "seed": seed,
        "n": n,
        "t": t,
        "n_val_anchors": n_observed_val_anchors(data),
        "phi_matched": matched_phi,
        "matched_slow_corr": matched["slow_recovery_corr"],
        "matched_rho_rdoc": matched["rho_rdoc"],
        "matched_s2f_leak": matched["slow_to_fast_leakage"],
        "matched_comp_rmse": matched["comp_rmse"],
        "phi_selected": sel_phi,
        "est_slow_corr": estimated["slow_recovery_corr"],
        "est_rho_rdoc": estimated["rho_rdoc"],
        "est_s2f_leak": estimated["slow_to_fast_leakage"],
        "est_comp_rmse": estimated["comp_rmse"],
        "phi_crit_curve": ";".join(f"{k:.2f}:{v:.4f}" for k, v in curve.items()),
    }


def aggregate(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    agg_keys = [
        "n_val_anchors",
        "phi_selected",
        "est_slow_corr",
        "est_rho_rdoc",
        "est_s2f_leak",
        "est_comp_rmse",
        "matched_rho_rdoc",
        "matched_slow_corr",
    ]
    out = []
    for cell in df["cell"].unique():
        for variant in VARIANT_ORDER:
            g = df[(df["cell"] == cell) & (df["variant"] == variant)]
            if g.empty:
                continue
            rec = {"cell": cell, "variant": variant, "n_seeds": int(len(g))}
            for key in agg_keys:
                vals = g[key].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                rec[f"{key}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
                rec[f"{key}_mcse"] = (
                    float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                )
            out.append(rec)
    return pd.DataFrame(out)


def print_summary(agg: pd.DataFrame) -> None:
    for cell in agg["cell"].unique():
        sub = agg[agg["cell"] == cell]
        true_phi = H.CELLS[cell]["delta_ar"]
        print(f"\n=== cell={cell} (true delta_ar={true_phi}) | estimated arm selects phi from observables ===")
        print(
            f"{'variant':<16}{'n_val':>7}{'phi_sel':>14}{'est_slow':>14}"
            f"{'est_rho':>14}{'est_leak':>14}{'matched_rho':>14}"
        )
        for variant in VARIANT_ORDER:
            r = sub[sub["variant"] == variant]
            if r.empty:
                continue
            r = r.iloc[0]

            def cs(key):
                return f"{r[f'{key}_mean']:.3f}+-{r[f'{key}_mcse']:.3f}"

            print(
                f"{variant:<16}{r['n_val_anchors_mean']:>7.0f}{cs('phi_selected'):>14}"
                f"{cs('est_slow_corr'):>14}{cs('est_rho_rdoc'):>14}{cs('est_s2f_leak'):>14}"
                f"{cs('matched_rho_rdoc'):>14}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Anchor information-structure identifiability check.")
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--t", type=int, default=H.DEFAULT_T)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--cells", type=str, nargs="+", default=DEFAULT_CELLS, choices=H.CELL_ORDER)
    parser.add_argument("--variants", type=str, nargs="+", default=VARIANT_ORDER, choices=VARIANT_ORDER)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--quick", action="store_true", help=f"fast: n={QUICK_N}, seeds={QUICK_SEEDS}")
    args = parser.parse_args()

    n = QUICK_N if args.quick else args.n
    seeds = QUICK_SEEDS if args.quick else args.seeds
    variants = [v for v in VARIANT_ORDER if v in set(args.variants)]
    cells = [c for c in H.CELL_ORDER if c in set(args.cells)]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else (H.REPO_ROOT / "validation" / "outputs" / f"anchor_check_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"anchor identifiability check: n={n}, seeds={seeds}, cells={cells}, variants={variants}")
    print(f"output -> {out_dir}")

    rows = []
    for cell in cells:
        for variant in variants:
            for seed in seeds:
                r = run_one(cell, variant, seed, n, args.t)
                rows.append(r)
                print(
                    f"  [{cell:<13} {variant:<14} seed={seed}] phi_sel={r['phi_selected']:.2f} "
                    f"est_rho={r['est_rho_rdoc']:.3f} matched_rho={r['matched_rho_rdoc']:.3f} "
                    f"n_val={r['n_val_anchors']}"
                )

    per_seed = pd.DataFrame(rows)
    agg = aggregate(rows)
    per_seed.to_csv(out_dir / "per_seed.csv", index=False)
    agg.to_csv(out_dir / "aggregate.csv", index=False)
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command_line": " ".join(sys.argv),
        "ball_py_sha256": H.file_sha256(H.BALL_PATH),
        "ball_py_mtime": datetime.fromtimestamp(H.BALL_PATH.stat().st_mtime).isoformat(timespec="seconds"),
        "seeds": list(seeds),
        "n_individuals": n,
        "t": args.t,
        "cells": cells,
        "phi_grid": list(H.PHI_GRID),
        "anchor_variants": {
            v: {"y1": list(ANCHOR_VARIANTS[v][0].__dict__.values()) if hasattr(ANCHOR_VARIANTS[v][0], "__dict__")
                else str(ANCHOR_VARIANTS[v][0]),
                "y2": list(ANCHOR_VARIANTS[v][1].__dict__.values()) if hasattr(ANCHOR_VARIANTS[v][1], "__dict__")
                else str(ANCHOR_VARIANTS[v][1])}
            for v in variants
        },
        "anchor_spec_fields": ["name", "cadence", "recall_window", "loading", "error_sd", "missing_probability"],
        "phi_estimator": "delta_phi selected over phi_grid by minimum held-out validation-anchor RMSE (no oracle)",
        "decision_rule": (
            "if better anchors move selected delta_phi toward true reverting value and restore rho_RDoC, "
            "the failure is an information-structure artifact (fix DGP before step B); else the decomposition "
            "claim is prior-dependent"
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print_summary(agg)
    print(f"\nwrote:\n  {out_dir / 'per_seed.csv'}\n  {out_dir / 'aggregate.csv'}\n  {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
