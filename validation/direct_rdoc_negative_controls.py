#!/usr/bin/env python
"""Negative controls for direct RDoC drift recovery.

The primary benchmark asks whether each method can recover a sparse RDoC drift
direction when the latent trajectory actually contains a direct C_t beta drift.
This runner tests the complementary safety property: beta recovery should
collapse when the direct signal is absent or the observed RDoC proxy is broken.
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
from direct_rdoc_benchmark import (  # noqa: E402
    aggregate,
    aggregate_deltas,
    irt_calibration_provenance,
    make_data,
    paired_deltas,
    run_ball,
    run_markov,
    run_s0,
)


CONTROL_LABELS = {
    "positive": "Positive direct-RDoC signal",
    "null": "No direct-RDoC signal",
    "permuted_proxy": "Permuted observed RDoC proxy",
    "noise_proxy": "Noise observed RDoC proxy",
}


def _break_proxy(data, seed: int, control: str):
    comp = data.components.copy()
    q = data.config.q
    cols = [f"B{j}" for j in range(q)]
    rng = np.random.default_rng(int(seed) + 918273)
    if control == "permuted_proxy":
        for col in cols:
            vals = comp[col].to_numpy(dtype=float).copy()
            rng.shuffle(vals)
            comp[col] = vals
    elif control == "noise_proxy":
        for col in cols:
            comp[col] = rng.normal(0.0, 1.0, size=len(comp))
    else:
        raise ValueError(f"Unknown proxy control: {control}")
    metadata = dict(data.metadata)
    metadata["direct_rdoc_negative_control"] = control
    metadata["direct_rdoc_negative_control_note"] = (
        "Latent trajectory and anchors retain the generated direct signal, but "
        "the observed RDoC proxy supplied to all fitted methods is broken."
    )
    return dataclasses.replace(data, components=comp, metadata=metadata)


def make_control_data(args, seed: int, control: str):
    if control == "null":
        data = make_data(args, seed, 0.0, "linear")
        metadata = dict(data.metadata)
        metadata["direct_rdoc_negative_control"] = control
        metadata["direct_rdoc_negative_control_note"] = "No direct C_t beta drift is injected into the latent trajectory."
        return dataclasses.replace(data, metadata=metadata)
    data = make_data(args, seed, args.share, "linear")
    if control in {"permuted_proxy", "noise_proxy"}:
        return _break_proxy(data, seed, control)
    metadata = dict(data.metadata)
    metadata["direct_rdoc_negative_control"] = control
    metadata["direct_rdoc_negative_control_note"] = "Positive-control linear direct-RDoC signal."
    return dataclasses.replace(data, metadata=metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct RDoC negative-control benchmark.")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--t", type=int, default=84)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1729, 2027, 2028])
    parser.add_argument("--controls", type=str, nargs="+", default=list(CONTROL_LABELS), choices=list(CONTROL_LABELS))
    parser.add_argument("--share", type=float, default=0.25)
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
    parser.add_argument("--s0-basis", choices=["matched", "linear", "interaction", "nonlinear", "heterogeneous", "full"], default="linear")
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
                        help="irt: frozen calibrated graded-response instrument shared by all methods")
    parser.add_argument("--irt-n-items", type=int, default=9)
    parser.add_argument("--irt-discrimination", type=float, default=1.5)
    parser.add_argument("--delta-phi", type=float, default=0.3)
    parser.add_argument("--ball-no-ehr-drift", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.rdoc_drift_adaptive and args.rdoc_drift_l1 <= 0.0:
        parser.error("--rdoc-drift-adaptive requires --rdoc-drift-l1 > 0 (adaptive weights only act through the L1 penalty)")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = Path(args.out) if args.out else H.REPO_ROOT / "validation" / "outputs" / f"direct_rdoc_negative_controls_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for control in args.controls:
        for seed in args.seeds:
            print(f"control={control} seed={seed} building data")
            data = make_control_data(args, seed, control)
            for runner in (run_s0, run_markov, run_ball):
                result = runner(data, args, control, seed, float(data.metadata.get("rdoc_drift_share", args.share)))
                result_rows = result if isinstance(result, list) else [result]
                for row in result_rows:
                    row["control"] = control
                    row["control_label"] = CONTROL_LABELS[control]
                    rows.append(row)

    per = pd.DataFrame(rows)
    agg = aggregate(per.rename(columns={"control": "_control"}))
    agg = agg.rename(columns={"cell": "control"})
    delta = paired_deltas(per.rename(columns={"control": "_control"}))
    delta_agg = aggregate_deltas(delta)
    delta = delta.rename(columns={"cell": "control"})
    if not delta_agg.empty:
        delta_agg = delta_agg.rename(columns={"cell": "control"})

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
                "controls": CONTROL_LABELS,
                "note": "Direct-RDoC negative controls. Truth is used only for scoring; all fitted methods see the same controlled observed data.",
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"\noutput -> {out_dir}")
    irt = str(getattr(args, "anchor_observation", "gaussian")).lower() == "irt"
    if irt:
        metrics = [
            "beta_cosine_mean",
            "beta_topk_f1_mean",
            "beta_hat_norm_trait_mean",
            "latent_rmse_trait_mean",
            "irt_item_nll_mean",
            "irt_expected_total_rmse_mean",
        ]
    else:
        metrics = [
            "beta_cosine_mean",
            "beta_topk_f1_mean",
            "latent_rmse_mean",
            "val_anchor_rmse_mean",
        ]
    cols = ["control", "share", "method", "n_seeds"] + metrics
    print(agg[[col for col in cols if col in agg.columns]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
