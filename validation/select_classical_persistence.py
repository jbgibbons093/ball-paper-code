'''Select one classical persistence value from development runs.'''

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        nargs=2,
        action="append",
        metavar=("PERSISTENCE", "DIRECTORY"),
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frames: list[pd.DataFrame] = []
    inputs: list[dict[str, object]] = []
    reference_keys: set[tuple[str, float, int, str]] | None = None
    for raw_persistence, raw_directory in args.candidate:
        persistence = float(raw_persistence)
        directory = Path(raw_directory).resolve()
        source = directory / "per_run.csv"
        frame = pd.read_csv(source)
        required = {
            "cell",
            "share",
            "seed",
            "method",
            "classical_validation_selection_rmse",
            "classical_converged",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{source} lacks columns: {sorted(missing)}")
        keys = set(
            frame[["cell", "share", "seed", "method"]]
            .itertuples(index=False, name=None)
        )
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError("Persistence candidates do not cover identical development runs.")
        frame = frame.copy()
        frame["persistence"] = persistence
        frames.append(frame)
        inputs.append(
            {
                "persistence": persistence,
                "per_run_csv": str(source),
                "sha256": file_sha256(source),
                "rows": int(len(frame)),
            }
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["classical_converged"] = combined["classical_converged"].map(
        lambda value: str(value).strip().lower() in {"1", "true"}
    )
    summary = (
        combined.groupby("persistence", as_index=False)
        .agg(
            validation_rmse_mean=("classical_validation_selection_rmse", "mean"),
            validation_rmse_median=("classical_validation_selection_rmse", "median"),
            validation_rmse_sd=("classical_validation_selection_rmse", "std"),
            converged_fits=("classical_converged", "sum"),
            total_fits=("classical_converged", "size"),
        )
        .sort_values("persistence")
        .reset_index(drop=True)
    )
    summary["all_fits_converged"] = summary["converged_fits"].eq(
        summary["total_fits"]
    )
    eligible = summary.loc[summary["all_fits_converged"]]
    if eligible.empty:
        eligible = summary.sort_values(
            ["converged_fits", "validation_rmse_mean", "persistence"],
            ascending=[False, True, True],
        ).head(1)
    selected_row = eligible.sort_values(
        ["validation_rmse_mean", "persistence"]
    ).iloc[0]
    selected = float(selected_row["persistence"])

    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / "selection_summary.csv", index=False)
    (args.out / "selection.json").write_text(
        json.dumps(
            {
                "created": datetime.now().isoformat(timespec="seconds"),
                "selection_rule": (
                    "lowest mean patient-balanced validation questionnaire RMSE "
                    "among candidates for which every development fit converged"
                ),
                "selected_persistence": selected,
                "development_inputs": inputs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"selected classical persistence: {selected:g}")


if __name__ == "__main__":
    main()
