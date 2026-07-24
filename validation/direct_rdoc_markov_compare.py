#!/usr/bin/env python
"""Standalone native Markov direct-RDoC comparator.

This script intentionally uses a native transition beta target. The Markov arm
uses the same scored target as BALL and S0:

    L[t] - L[t-1] = B[t-1] beta + nuisance[t-1] gamma + pattern Markov terms.

Truth is used only for evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ball_validation_harness as H  # noqa: E402
from direct_rdoc_benchmark import (  # noqa: E402
    DEFAULT_CELLS,
    _matched_s0_basis,
    extract_s0_beta,
    fit_markov_direct,
    make_data,
)
from direct_rdoc_common import beta_metric_fields, mcse  # noqa: E402

METRICS = ["beta_cosine", "beta_abs_cosine", "beta_topk_f1", "beta_hat_norm", "latent_rmse", "val_anchor_rmse", "elapsed_seconds"]


def run_one(args, seed: int, share: float, cell: str) -> dict:
    data = make_data(args, seed, share, cell)
    basis = _matched_s0_basis(cell) if args.s0_basis == "matched" else args.s0_basis
    start = perf_counter()
    preds, theta, n_base = fit_markov_direct(data, args, basis=basis)
    elapsed = perf_counter() - start
    beta_true = np.asarray(data.metadata.get("rdoc_drift_beta_unit", []), dtype=float)
    beta_hat = extract_s0_beta(theta[:n_base], data.config.q, data.config.n_treatment_types, basis, data.config.n_subtypes)
    return {
        "cell": cell,
        "share": float(share),
        "seed": int(seed),
        "method": "markov_direct_transition",
        "latent_rmse": H.latent_rmse(preds, data.components, data.individuals, H.EVAL_SPLIT),
        "val_anchor_rmse": H.val_anchor_rmse(data, preds),
        "elapsed_seconds": float(elapsed),
        **beta_metric_fields(beta_hat, beta_true, args.rdoc_active_dims),
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone native Markov direct-RDoC comparator.")
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
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--transition-sd", type=float, default=0.75)
    parser.add_argument("--prior-sd", type=float, default=5.0)
    parser.add_argument("--daily-ridge", type=float, default=1.0)
    parser.add_argument("--beta-ridge", type=float, default=10.0)
    parser.add_argument(
        "--s0-basis",
        choices=["matched", "linear", "interaction", "nonlinear", "heterogeneous", "full"],
        default="matched",
    )
    parser.add_argument("--markov-strata", type=int, default=3)
    parser.add_argument("--markov-iters", type=int, default=4)
    parser.add_argument("--markov-ridge", type=float, default=10.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = Path(args.out) if args.out else H.REPO_ROOT / "validation" / "outputs" / f"direct_rdoc_markov_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cell in args.cells:
        for share in args.shares:
            for seed in args.seeds:
                print(f"cell={cell} share={share:.2f} seed={seed} markov_direct_transition")
                rows.append(run_one(args, seed, share, cell))

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
                "note": "Native Markov direct-RDoC transition comparator with in-model beta recovery.",
                "args": vars(args),
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
