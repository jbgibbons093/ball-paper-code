'''Run the oracle-to-observed transition-coefficient diagnostic.'''

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import audit_classical_coefficient_recovery as audit
import direct_rdoc_benchmark as benchmark


VALIDATION_DIR = Path(__file__).resolve().parent
DEFAULT_CORE_GLOB = "direct_rdoc_gaussian_core_20260806_v6_seed_*/per_run.csv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _core_results(core_root: Path, pattern: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(core_root.glob(pattern)):
        frame = pd.read_csv(path)
        frame["source_file"] = str(path.resolve())
        frame["source_sha256"] = _sha256(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    records: list[dict] = []
    for keys, group in rows.groupby(
        ["cell", "share", "diagnostic_rung", "method"],
        sort=True,
    ):
        values = pd.to_numeric(group["beta_cosine"], errors="coerce").dropna()
        records.append(
            {
                "cell": keys[0],
                "share": float(keys[1]),
                "diagnostic_rung": keys[2],
                "method": keys[3],
                "n_seeds": int(group["seed"].nunique()),
                "beta_cosine_mean": float(values.mean()) if len(values) else np.nan,
                "beta_cosine_min": float(values.min()) if len(values) else np.nan,
                "beta_cosine_max": float(values.max()) if len(values) else np.nan,
                "positive_cosine_fraction": float((values > 0).mean()) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2028, 2029])
    parser.add_argument("--shares", nargs="+", type=float, default=[0.10, 0.25])
    parser.add_argument("--cells", nargs="+", default=["linear", "missingness"])
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--t", type=int, default=84)
    parser.add_argument(
        "--core-root",
        type=Path,
        default=VALIDATION_DIR / "outputs",
    )
    parser.add_argument("--core-glob", default=DEFAULT_CORE_GLOB)
    cli = parser.parse_args()
    cli.out.mkdir(parents=True, exist_ok=True)

    fit_args: SimpleNamespace = audit._fit_args(cli.n, cli.t)
    rows: list[dict] = []
    for cell in cli.cells:
        for share in cli.shares:
            for seed in cli.seeds:
                data = benchmark.make_data(fit_args, int(seed), float(share), str(cell))
                for profile_label, true_profile in (
                    ("true_latent_true_profile", True),
                    ("true_latent_observed_profile", False),
                ):
                    beta_hat, beta_true, cosine = audit._oracle_shared_beta(
                        data,
                        fit_args,
                        true_profile=true_profile,
                    )
                    rows.append(
                        {
                            "cell": str(cell),
                            "share": float(share),
                            "seed": int(seed),
                            "diagnostic_rung": profile_label,
                            "method": "oracle_transition_regression",
                            "beta_cosine": float(cosine),
                            **{
                                f"beta_hat_{index}": float(beta_hat[index])
                                for index in range(len(beta_hat))
                            },
                            **{
                                f"beta_true_{index}": float(beta_true[index])
                                for index in range(len(beta_true))
                            },
                            "source_file": "deterministically regenerated simulation data",
                            "source_sha256": "",
                        }
                    )
                    print(
                        f"cell={cell} share={share:.2f} seed={seed} "
                        f"rung={profile_label} cosine={cosine:.4f}",
                        flush=True,
                    )

    core = _core_results(cli.core_root, cli.core_glob)
    reconstructed_expected = 0
    reconstructed_available = 0
    if not core.empty:
        methods = {
            "s0_direct_lgssm": "lgssm_reconstructed_latent_observed_profile",
            "markov_direct_transition": "markov_reconstructed_latent_observed_profile",
        }
        reconstructed_expected = (
            len(cli.cells) * len(cli.shares) * len(cli.seeds) * len(methods)
        )
        selected = core.loc[
            core["cell"].astype(str).isin(set(cli.cells))
            & core["share"].astype(float).isin(set(float(value) for value in cli.shares))
            & core["seed"].astype(int).isin(set(int(value) for value in cli.seeds))
            & core["method"].astype(str).isin(set(methods))
        ].copy()
        reconstructed_available = len(selected)
        for row in selected.itertuples(index=False):
            rows.append(
                {
                    "cell": str(row.cell),
                    "share": float(row.share),
                    "seed": int(row.seed),
                    "diagnostic_rung": methods[str(row.method)],
                    "method": str(row.method),
                    "beta_cosine": float(row.beta_cosine),
                    **{
                        f"beta_hat_{index}": float(getattr(row, f"beta_hat_{index}"))
                        for index in range(6)
                    },
                    **{
                        f"beta_true_{index}": float(getattr(row, f"beta_true_{index}"))
                        for index in range(6)
                    },
                    "source_file": str(row.source_file),
                    "source_sha256": str(row.source_sha256),
                }
            )

    results = pd.DataFrame(rows).sort_values(
        ["cell", "share", "seed", "diagnostic_rung", "method"],
        kind="stable",
    )
    summary = _summary(results)
    results_path = cli.out / "oracle_coefficient_ladder.csv"
    summary_path = cli.out / "oracle_coefficient_ladder_summary.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)

    manifest = {
        "status": (
            "complete"
            if reconstructed_available == reconstructed_expected
            else "oracle_complete_reconstructed_rungs_pending"
        ),
        "completed": datetime.now().isoformat(timespec="seconds"),
        "seeds": cli.seeds,
        "shares": cli.shares,
        "cells": cli.cells,
        "n": cli.n,
        "t": cli.t,
        "oracle_rows": int(
            results["method"].eq("oracle_transition_regression").sum()
        ),
        "reconstructed_rows_expected": reconstructed_expected,
        "reconstructed_rows_available": reconstructed_available,
        "results": str(results_path.resolve()),
        "summary": str(summary_path.resolve()),
        "results_sha256": _sha256(results_path),
        "summary_sha256": _sha256(summary_path),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "audit_script_sha256": _sha256(
            VALIDATION_DIR / "audit_classical_coefficient_recovery.py"
        ),
        "benchmark_sha256": _sha256(
            VALIDATION_DIR / "direct_rdoc_benchmark.py"
        ),
        "comparator_sha256": _sha256(
            VALIDATION_DIR / "direct_rdoc_fair_comparator.py"
        ),
    }
    manifest_path = cli.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {results_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)
    print(f"wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
