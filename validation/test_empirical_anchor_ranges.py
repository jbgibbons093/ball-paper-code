"""Tests for empirical questionnaire range and measurement-status checks."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import BALL

BALL._install_virtual_package()
from empirical.build_sparse_anchors import _build_anchor_set, _genuine_masks  # noqa: E402


def test_out_of_range_questionnaire_totals_are_excluded() -> None:
    frame = pd.DataFrame(
        {
            "PatientFID": [1, 1, 1],
            "session": [0, 1, 2],
            "ServiceDate": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "day": [0, 1, 2],
            "phq9": [27.0, 28.0, 10.0],
            "phq9_is_imputed": [0, 0, 1],
            "phq9_is_interpolated": [0, 0, 0],
            "gad_filled": [21.0, 23.0, 7.0],
            "gad_measured": [1, 1, 0],
            "bdi_filled": [63.0, 64.0, 12.0],
            "bdi_measured": [1, 1, 0],
        }
    )
    masks = _genuine_masks(frame)
    assert masks["PHQ9"].tolist() == [True, False, False]
    assert masks["GAD7"].tolist() == [True, False, False]
    assert masks["BDI"].tolist() == [True, False, False]

    anchors = _build_anchor_set(frame, masks, cadence_days=14)
    assert set(anchors["value"].astype(float)) == {21.0, 27.0, 63.0}
    assert set(anchors["anchor"]) == {"PHQ9", "GAD7", "BDI"}
