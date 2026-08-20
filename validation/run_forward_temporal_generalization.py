#!/usr/bin/env python
'''Run the forward-time empirical sensitivity analysis.'''

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import traceback

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import BALL  # noqa: E402


EXPECTED_METHODS = {"ball", "locf", "interpolation", "causal_anchor_mean"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_result(frame: pd.DataFrame) -> None:
    required = {
        "validation",
        "cadence",
        "method",
        "rmse",
        "n",
        "n_train_patients",
        "n_test_patients",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Forward-time result is missing columns: {sorted(missing)}")
    if frame.empty or set(frame["validation"].astype(str)) != {"forward_2025"}:
        raise RuntimeError("Forward-time result does not contain the forward_2025 evaluation.")
    observed = set(frame["method"].astype(str))
    if observed != EXPECTED_METHODS:
        raise RuntimeError(
            f"Forward-time methods differ from the locked contract: {sorted(observed)}"
        )
    if frame["rmse"].isna().any() or (frame["n"].astype(int) <= 0).any():
        raise RuntimeError("Forward-time result contains missing RMSE values or empty targets.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--comorbidity-features", type=Path, required=True)
    parser.add_argument("--teacher-epochs", type=int, default=120)
    parser.add_argument("--student-epochs", type=int, default=120)
    parser.add_argument("--anchor-warmup", type=int, default=60)
    parser.add_argument("--kl-warmup", type=int, default=60)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--delta-phi", type=float, default=0.9)
    parser.add_argument("--anchor-weight", type=float, default=10.0)
    parser.add_argument("--student-anchor-weight", type=float, default=2.0)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every-epochs", type=int, default=10)
    parser.add_argument("--diagnostics-every-epochs", type=int, default=5)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=10)
    parser.add_argument("--max-patients", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_run = args.source_run.resolve()
    output = args.out.resolve()
    calibration_path = source_run / "measurement_calibration.json"
    source_manifest_path = source_run / "fit_run_manifest.json"
    if not calibration_path.is_file() or not source_manifest_path.is_file():
        raise FileNotFoundError("The source run lacks its calibration or fit manifest.")
    if not args.comorbidity_features.is_file():
        raise FileNotFoundError(args.comorbidity_features)

    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "empirical_generalization.csv"
    manifest_path = output / "forward_temporal_manifest.json"
    if result_path.is_file() and manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("status") == "complete":
            frame = pd.read_csv(result_path)
            _validate_result(frame)
            print(f"forward-time empirical sensitivity already complete: {result_path}", flush=True)
            return

    BALL._install_virtual_package()
    empirical = sys.modules["empirical.fit_empirical"]
    # The shared evaluation table contains an optional retrospective-teacher column.
    original_channel_eval = empirical._channel_eval

    def forward_only_channel_eval(*channel_args, **channel_kwargs):
        predictions = channel_kwargs.get("neural_predictions")
        positional_predictions = None
        if predictions is None and len(channel_args) >= 4:
            positional_predictions = channel_args[3]
            predictions = positional_predictions
        if predictions and predictions.get("student") and not predictions.get("teacher"):
            predictions = dict(predictions)
            predictions["teacher"] = predictions["student"]
            # The common table also contains a legacy linear-state-space column.
            predictions.setdefault("s0", predictions["student"])
            if positional_predictions is not None:
                channel_args = (*channel_args[:3], predictions, *channel_args[4:])
            else:
                channel_kwargs["neural_predictions"] = predictions
        return original_channel_eval(*channel_args, **channel_kwargs)

    empirical._channel_eval = forward_only_channel_eval
    sessions_full = empirical._coerce_sessions(pd.read_csv(empirical.OUT / "sessions.csv"))
    sessions = sessions_full[["id", "day"]].dropna().drop_duplicates()
    sessions["id"] = sessions["id"].astype(int)
    sessions["day"] = sessions["day"].astype(int)
    anchors = pd.read_csv(empirical.OUT / "anchors_sparse_14d.csv")
    comorbidities = empirical._load_comorbidity_sessions(args.comorbidity_features)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    split_maps = empirical._generalization_split_maps(sessions_full)
    split = split_maps.get("forward_2025", {})
    split_counts = {
        role: sum(value == role for value in split.values()) for role in ("train", "test")
    }
    if min(split_counts.values(), default=0) == 0:
        raise RuntimeError(f"Forward-time split is empty: {split_counts}")

    fit_args = argparse.Namespace(
        teacher_epochs=args.teacher_epochs,
        student_epochs=args.student_epochs,
        anchor_warmup=args.anchor_warmup,
        kl_warmup=args.kl_warmup,
        members=args.members,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        batch_size=args.batch_size,
        delta_phi=args.delta_phi,
        anchor_weight=args.anchor_weight,
        student_anchor_weight=args.student_anchor_weight,
        seed_offset=args.seed_offset,
        device=args.device,
        progress_every_epochs=args.progress_every_epochs,
        diagnostics_every_epochs=args.diagnostics_every_epochs,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        max_patients=args.max_patients,
        select_core_hyperparameters=False,
    )
    manifest = {
        "analysis": "patient-exclusive forward-2025 empirical generalization",
        "status": "validated" if args.dry_run else "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "source_run": source_run.name,
        "source_run_manifest_sha256": _sha256(source_manifest_path),
        "measurement_calibration_sha256": _sha256(calibration_path),
        "ball_py_sha256": _sha256(REPO / "BALL.py"),
        "comorbidity_features_sha256": _sha256(args.comorbidity_features),
        "split_counts": split_counts,
        "fit_args": vars(fit_args),
        "checkpoint_resume": True,
        "evaluation_compatibility": (
            "the unused retrospective-teacher column receives the student lookup; "
            "reported forward-time methods and fitted parameters are unchanged"
        ),
    }
    _atomic_json(manifest_path, manifest)
    print(f"forward-time split: {split_counts}", flush=True)
    if args.dry_run:
        print("forward-time inputs and locked split validated", flush=True)
        return

    try:
        result = empirical._fit_generalization_sensitivities(
            anchors,
            sessions_full,
            sessions,
            comorbidities,
            calibration,
            fit_args,
            output,
        )
        _validate_result(result)
        result.to_csv(result_path, index=False)
        result.to_csv(output / "empirical_generalization_complete.csv", index=False)
        manifest.update(
            {
                "status": "complete",
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "result_sha256": _sha256(result_path),
                "n_rows": int(len(result)),
            }
        )
        _atomic_json(manifest_path, manifest)
        print(result.to_string(index=False), flush=True)
        print("FORWARD TEMPORAL GENERALIZATION COMPLETE", flush=True)
    except BaseException as error:
        manifest.update(
            {
                "status": "failed",
                "failed_at": datetime.now().isoformat(timespec="seconds"),
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )
        _atomic_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
