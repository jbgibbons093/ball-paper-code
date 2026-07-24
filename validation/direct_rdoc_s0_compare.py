#!/usr/bin/env python
"""S0 direct comparator on direct-RDoC DGP cells.

For this validation target, S0 is defined as the classical direct linear-
Gaussian comparator to the transformer direct head. It uses the same observed
inputs and estimates the same explicit transition parameter:

    L[t] - L[t-1] = B[t-1] beta + nuisance[t-1] gamma + error

It does not use the legacy structural equation L = alpha'B + delta, and it does
does not get a secondary beta readout from a structurally advantaged trajectory. The
true simulated latent is used only for evaluation metrics.
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
import ball_validation_harness as H  # noqa: E402
from direct_rdoc_common import (  # noqa: E402
    aggregate_wide,
    beta_metric_fields,
)
from direct_rdoc_fair_comparator import fit_direct_map  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare direct S0 MAP smoother on direct-RDoC DGP cells.")
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
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = Path(args.out) if args.out else H.REPO_ROOT / "validation" / "outputs" / f"direct_rdoc_s0_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for share in args.shares:
        for seed in args.seeds:
            print(f"share={share:.2f} seed={seed} s0_direct_lgssm")
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
            rows.append(
                {
                    "share": float(share),
                    "seed": int(seed),
                    "latent_rmse": H.latent_rmse(preds, data.components, data.individuals, H.EVAL_SPLIT),
                    "val_anchor_rmse": H.val_anchor_rmse(data, preds),
                    **beta_metric_fields(beta_hat, beta_true, args.rdoc_active_dims),
                }
            )

    per = pd.DataFrame(rows)
    agg = aggregate_wide(
        per,
        "s0_direct_lgssm",
        ["beta_cosine", "beta_abs_cosine", "beta_topk_f1", "beta_hat_norm", "latent_rmse", "val_anchor_rmse"],
    )
    per.to_csv(out_dir / "per_run.csv", index=False)
    agg.to_csv(out_dir / "aggregate.csv", index=False)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "command_line": " ".join(sys.argv),
                "ball_py_sha256": H.file_sha256(H.BALL_PATH),
                "note": "S0 direct linear-Gaussian MAP comparator with explicit B_t beta transition; no legacy alpha structural equation and no secondary beta readout.",
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
