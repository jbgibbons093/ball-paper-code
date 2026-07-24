#!/usr/bin/env python
"""Transformer validation for direct RDoC drift-parameter recovery.

This is the direct-parameter counterpart to the slow/fast decomposition harness.
It generates DGP cells with an explicit sparse C_t beta contribution to latent
drift and tests whether BALL's constrained RDoC drift head recovers beta.

Primary diagnostics:
  - beta cosine against the true RDoC drift direction
  - top-k active-domain F1 for beta
  - latent/severity fit as context

The slow/fast decomposition metrics are intentionally not primary here.
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

_ssm = sys.modules["simulations.src.methods.ball_ssm"]
SSMConfig = _ssm.SSMConfig
fit_ball_ssm = _ssm.fit_ball_ssm


DEFAULT_SHARES = (0.0, 0.10, 0.25)
DEFAULT_ARMS = ("no_head", "head", "head_noehr", "direct_head", "direct_head_noehr")


def make_data(args, seed: int, share: float):
    return H.generate_dataset(
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


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 1e-12 else float("nan")


def topk_f1(beta_hat: np.ndarray, beta_true: np.ndarray, k: int) -> float:
    k = max(1, min(int(k), len(beta_true)))
    true = set(np.flatnonzero(np.abs(beta_true) > 1e-12).tolist())
    pred = set(np.argsort(-np.abs(beta_hat))[:k].tolist())
    if not true and not pred:
        return 1.0
    if not true or not pred:
        return 0.0
    tp = len(true & pred)
    precision = tp / len(pred)
    recall = tp / len(true)
    return float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def arm_flags(name: str) -> dict:
    if name == "no_head":
        return {"rdoc_drift_head": False}
    if name == "head":
        return {"rdoc_drift_head": True}
    if name == "head_noehr":
        return {"rdoc_drift_head": True, "delta_drift_use_ehr": False}
    if name == "direct_head":
        return {"rdoc_drift_head": True, "use_alpha_slow": False}
    if name == "direct_head_noehr":
        return {"rdoc_drift_head": True, "use_alpha_slow": False, "delta_drift_use_ehr": False}
    raise ValueError(f"unknown arm {name!r}")


def make_config(args, seed: int, arm: str):
    flags = arm_flags(arm)
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
        delta_phi=args.delta_phi,
        max_individuals=args.n,
        **flags,
    )


def score_role(data, result, *, role: str, arm: str, seed: int, share: float, args) -> dict:
    beta_true = np.asarray(data.metadata.get("rdoc_drift_beta_unit", []), dtype=float)
    beta_hat = np.asarray(result.metadata.get("rdoc_drift_beta_hat", []), dtype=float)
    if beta_hat.size != beta_true.size:
        beta_hat = np.zeros_like(beta_true)
    active_k = int(data.metadata.get("rdoc_drift_active_dims", args.rdoc_active_dims))
    latent_rmse = H.latent_rmse(result.predictions, data.components, data.individuals, H.EVAL_SPLIT)
    val_anchor = H.val_anchor_rmse(data, result.predictions)
    has_signal = float(share) > 0.0 and bool(result.metadata.get("rdoc_drift_head", False))
    beta_cos = cosine(beta_hat, beta_true) if has_signal else float("nan")
    return {
        "share": float(share),
        "arm": arm,
        "role": role,
        "seed": seed,
        "n": args.n,
        "t": args.t,
        "rdoc_drift_head": bool(result.metadata.get("rdoc_drift_head", False)),
        "use_alpha_slow": bool(result.metadata.get("use_alpha_slow", True)),
        "delta_drift_use_ehr": bool("noehr" not in arm),
        "beta_cosine": beta_cos,
        "beta_abs_cosine": abs(beta_cos) if np.isfinite(beta_cos) else float("nan"),
        "beta_topk_f1": topk_f1(beta_hat, beta_true, active_k) if has_signal else float("nan"),
        "beta_hat_norm": float(np.linalg.norm(beta_hat)),
        "latent_rmse": float(latent_rmse),
        "val_anchor_rmse": float(val_anchor),
        **{f"beta_hat_{j}": float(beta_hat[j]) for j in range(len(beta_hat))},
        **{f"beta_true_{j}": float(beta_true[j]) for j in range(len(beta_true))},
    }


def run_one(args, seed: int, share: float, arm: str) -> list[dict]:
    data = make_data(args, seed, share)
    cfg = make_config(args, seed, arm)
    if args.student:
        student, teacher = fit_ball_ssm(
            data,
            cfg,
            device=args.device,
            causal=True,
            prediction_split=None,
            return_teacher=True,
        )
        return [
            score_role(data, teacher, role="teacher", arm=arm, seed=seed, share=share, args=args),
            score_role(data, student, role="student", arm=arm, seed=seed, share=share, args=args),
        ]
    result = fit_ball_ssm(data, cfg, device=args.device, causal=False, prediction_split=None)
    return [score_role(data, result, role="teacher", arm=arm, seed=seed, share=share, args=args)]


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "beta_cosine",
        "beta_abs_cosine",
        "beta_topk_f1",
        "beta_hat_norm",
        "latent_rmse",
        "val_anchor_rmse",
    ]
    rows = []
    for keys, group in df.groupby(["share", "arm", "role"], sort=True):
        row = {"share": keys[0], "arm": keys[1], "role": keys[2], "n_seeds": int(group["seed"].nunique())}
        for metric in metric_cols:
            vals = group[metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{metric}_mcse"] = (
                float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct RDoC drift-head transformer ablation.")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--t", type=int, default=84)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1729])
    parser.add_argument("--shares", type=float, nargs="+", default=list(DEFAULT_SHARES))
    parser.add_argument("--arms", type=str, nargs="+", default=["no_head", "head"], choices=list(DEFAULT_ARMS))
    parser.add_argument("--slow-fraction", type=float, default=0.75)
    parser.add_argument("--delta-ar", type=float, default=0.3)
    parser.add_argument("--delta-phi", type=float, default=0.3)
    parser.add_argument("--rdoc-active-dims", type=int, default=3)
    parser.add_argument("--rdoc-beta-seed", type=int, default=31011)
    parser.add_argument("--rdoc-min-abs", type=float, default=0.35)
    parser.add_argument("--rdoc-drift-l1", type=float, default=0.0)
    parser.add_argument("--alpha-lasso-xi", type=float, default=2.0)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--anchor-warmup", type=int, default=30)
    parser.add_argument("--kl-warmup", type=int, default=20)
    parser.add_argument("--teacher-epochs", type=int, default=60)
    parser.add_argument("--student-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--student", action="store_true", help="also train/distill the causal student")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = Path(args.out) if args.out else H.REPO_ROOT / "validation" / "outputs" / f"direct_rdoc_transformer_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for share in args.shares:
        for seed in args.seeds:
            for arm in args.arms:
                print(f"share={share:.2f} seed={seed} arm={arm}")
                rows.extend(run_one(args, seed, share, arm))

    per = pd.DataFrame(rows)
    agg = aggregate(per)
    per.to_csv(out_dir / "per_run.csv", index=False)
    agg.to_csv(out_dir / "aggregate.csv", index=False)
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command_line": " ".join(sys.argv),
        "ball_py_sha256": H.file_sha256(H.BALL_PATH),
        "args": vars(args),
        "note": (
            "Direct RDoC parameter-recovery validation. The DGP injects C_t beta "
            "into latent drift when share > 0; the model arm 'head' adds an "
            "explicit constrained B_t beta transition term."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(f"\noutput -> {out_dir}")
    print("\naggregate:")
    cols = [
        "share",
        "arm",
        "role",
        "n_seeds",
        "beta_cosine_mean",
        "beta_topk_f1_mean",
        "beta_hat_norm_mean",
        "latent_rmse_mean",
        "val_anchor_rmse_mean",
    ]
    print(agg[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
