"""Shared diagnostics for direct RDoC drift validation."""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def ensure_channel_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    out = preds.copy()
    if "z_d_hat" not in out.columns and "L_hat" in out.columns:
        out["z_d_hat"] = out["L_hat"]
    if "z_p_hat" not in out.columns and "L_hat" in out.columns:
        out["z_p_hat"] = out["L_hat"]
    return out


def beta_metric_fields(beta_hat: np.ndarray, beta_true: np.ndarray, active_k: int) -> dict:
    beta_cos = cosine(beta_hat, beta_true)
    return {
        "beta_cosine": beta_cos,
        "beta_abs_cosine": abs(beta_cos) if np.isfinite(beta_cos) else float("nan"),
        "beta_topk_f1": topk_f1(beta_hat, beta_true, active_k),
        "beta_hat_norm": float(np.linalg.norm(beta_hat)),
        **{f"beta_hat_{j}": float(beta_hat[j]) for j in range(len(beta_hat))},
        **{f"beta_true_{j}": float(beta_true[j]) for j in range(len(beta_true))},
    }


def aggregate_wide(df: pd.DataFrame, method: str, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for share, group in df.groupby("share", sort=True):
        row = {"share": float(share), "method": method, "n_seeds": int(group["seed"].nunique())}
        for metric in metrics:
            vals = group[metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{metric}_mcse"] = mcse(vals)
        rows.append(row)
    return pd.DataFrame(rows)
