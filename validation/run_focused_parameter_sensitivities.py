"""Run the prespecified focused BALL persistence and anchor-weight sensitivities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "validation" / "direct_rdoc_benchmark.py"
CELLS = ("linear", "missingness")
PERSISTENCE_VALUES = (0.1, 0.3, 0.6, 0.9)
ANCHOR_WEIGHTS = (10.0, 20.0, 40.0, 80.0)


def _run(output: Path, common: list[str], extra: list[str]) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(BENCHMARK), *common, *extra, "--out", str(output), "--resume"]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode:
        raise RuntimeError(f"Focused sensitivity failed with exit code {completed.returncode}: {command}")
    return pd.read_csv(output / "per_run.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--t", type=int, default=84)
    parser.add_argument("--teacher-epochs", type=int, default=300)
    parser.add_argument("--student-epochs", type=int, default=300)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    common = [
        "--shares", "0.25",
        "--seeds", *[str(seed) for seed in args.seeds],
        "--n", str(args.n), "--t", str(args.t),
        "--teacher-epochs", str(args.teacher_epochs),
        "--student-epochs", str(args.student_epochs),
        "--ensemble-size", str(args.ensemble_size),
        "--device", str(args.device),
        "--s0-basis", "matched",
        "--skip-gp", "--skip-ode-rnn",
        "--skip-compute-matched-direct",
        "--skip-distillation-decomposition",
        "--skip-transition-identification",
    ]

    persistence_rows: list[pd.DataFrame] = []
    for cell in CELLS:
        for generating in PERSISTENCE_VALUES:
            for fitted in PERSISTENCE_VALUES:
                run = _run(
                    out / "runs" / f"persistence_{cell}_dgp{generating:.1f}_fit{fitted:.1f}",
                    common,
                    [
                        "--cells", cell,
                        "--delta-ar", str(generating),
                        "--delta-phi", str(fitted),
                        "--anchor-weight", "40",
                    ],
                )
                ball = run[run["method"].eq("ball_student_causal")].copy()
                ball["generating_persistence"] = generating
                ball["fitted_persistence"] = fitted
                persistence_rows.append(ball)
    persistence = pd.concat(persistence_rows, ignore_index=True)
    persistence.to_csv(out / "persistence_per_run.csv", index=False)
    selected_index = persistence.groupby(
        ["cell", "generating_persistence", "share", "seed"], sort=True
    )["val_anchor_rmse"].idxmin()
    selected = persistence.loc[selected_index].copy()
    selected["selection_rule"] = "minimum held-out validation-questionnaire RMSE"
    selected.to_csv(out / "persistence_development_selected.csv", index=False)

    weight_rows: list[pd.DataFrame] = []
    for cell in CELLS:
        for weight in ANCHOR_WEIGHTS:
            run = _run(
                out / "runs" / f"anchor_weight_{cell}_{int(weight)}",
                common,
                [
                    "--cells", cell,
                    "--delta-ar", "0.3",
                    "--delta-phi", "0.3",
                    "--anchor-weight", str(weight),
                ],
            )
            ball = run[run["method"].eq("ball_student_causal")].copy()
            ball["questionnaire_weight"] = weight
            weight_rows.append(ball)
    weights = pd.concat(weight_rows, ignore_index=True)
    weights.to_csv(out / "anchor_weight_per_run.csv", index=False)

    manifest = {
        "analysis": "focused persistence and questionnaire-weight sensitivity",
        "cells": list(CELLS),
        "generating_persistence_values": list(PERSISTENCE_VALUES),
        "fitted_persistence_values": list(PERSISTENCE_VALUES),
        "questionnaire_weights": list(ANCHOR_WEIGHTS),
        "selection_target": "held-out validation-questionnaire RMSE",
        "selection_uses_test_latent_truth": False,
        "benchmark_sha256": hashlib.sha256(BENCHMARK.read_bytes()).hexdigest(),
        "ball_py_sha256": hashlib.sha256((ROOT / "BALL.py").read_bytes()).hexdigest(),
        "args": vars(args) | {"out": str(out)},
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
