"""Validate the aggregate BALL table and figure package before manuscript use."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image


FORBIDDEN_COLUMNS = {
    "id",
    "patientfid",
    "appointmentfid",
    "date",
    "servicedate",
    "dob",
    "facilityname",
    "note_text",
    "raw_text",
}
FORBIDDEN_FILENAMES = {
    "empirical_prediction_rows.csv",
    "empirical_ball_transition_rows.csv",
    "empirical_rdoc_transition_rows.csv",
}
LONG_INTEGER = re.compile(r"(?<![0-9])\d{7,}(?![0-9])")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_csv(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    normalized_columns = {str(column).strip().lower() for column in frame.columns}
    forbidden = sorted(normalized_columns.intersection(FORBIDDEN_COLUMNS))
    if forbidden:
        raise AssertionError(f"Patient-level columns in {path.name}: {forbidden}")
    for column in frame.columns:
        values = frame[column].astype(str)
        local_paths = values.str.contains(
            r"C:\\Users\\|Partners HealthCare Dropbox|DBH AMD",
            case=False,
            regex=True,
        )
        if local_paths.any():
            raise AssertionError(f"Local protected path in {path.name}, column {column}")
        long_identifiers = values.str.fullmatch(LONG_INTEGER)
        if long_identifiers.any():
            raise AssertionError(
                f"Possible patient identifier in {path.name}, column {column}"
            )
    return {"rows": int(len(frame)), "columns": int(len(frame.columns))}


def validate_png(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        if width < 1200 or height < 700:
            raise AssertionError(f"Figure resolution is too small: {path.name} {width}x{height}")
        return {"width_pixels": int(width), "height_pixels": int(height)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()

    package = args.package.expanduser().resolve()
    manifest_path = package / "direct_rdoc_manuscript_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = [str(name) for name in manifest.get("outputs", [])]
    if len(declared) != len(set(declared)):
        raise AssertionError("Publication manifest contains duplicate output names")
    if FORBIDDEN_FILENAMES.intersection(declared):
        raise AssertionError("Publication manifest includes a patient-level empirical file")

    validation: dict[str, dict[str, object]] = {}
    hash_rows: list[dict[str, object]] = []
    for name in declared:
        path = package / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".csv":
            validation[name] = validate_csv(path)
        elif path.suffix.lower() == ".png":
            validation[name] = validate_png(path)
        else:
            raise AssertionError(f"Unsupported declared publication artifact: {name}")
        hash_rows.append(
            {"file": name, "bytes": int(path.stat().st_size), "sha256": sha256(path)}
        )

    result = {
        "status": "pass",
        "aggregate_artifacts": len(declared),
        "patient_level_artifacts": 0,
        "files": validation,
    }
    (package / "publication_package_validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(hash_rows).to_csv(package / "publication_output_sha256.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
