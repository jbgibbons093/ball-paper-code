"""Synthetic tests for the protected comorbidity feature builder."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import BALL

BALL._install_virtual_package()
from empirical.build_comorbidities import (  # noqa: E402
    FEATURE_COLUMNS,
    _assert_protected_output,
    classify_diagnosis,
    construct_session_features,
)


class ComorbidityFeatureTests(unittest.TestCase):
    def test_icd_and_text_mapping(self) -> None:
        self.assertEqual(
            classify_diagnosis("F33.2; F41.1; F43.12"),
            {"depression", "anxiety", "ptsd"},
        )
        self.assertEqual(
            classify_diagnosis("history of bipolar disorder and ADHD"),
            {"bipolar", "adhd"},
        )
        self.assertEqual(
            classify_diagnosis("F33.2F41.1"),
            {"depression", "anxiety"},
        )
        self.assertEqual(
            classify_diagnosis("major depressive disorder, severe, without psychotic features"),
            {"depression"},
        )

    def test_strict_prior_carry_forward(self) -> None:
        sessions = pd.DataFrame(
            {
                "PatientFID": [101, 101, 101, 202],
                "session": [0, 1, 2, 0],
                "day": [0, 2, 5, 0],
                "ServiceDate": pd.to_datetime(
                    ["2025-01-10", "2025-01-12", "2025-01-15", "2025-02-01"]
                ),
            }
        )
        events = pd.DataFrame(
            {
                "PatientFID": [101, 101, 101],
                "ServiceDate": pd.to_datetime(["2025-01-09", "2025-01-12", "2025-01-16"]),
                "FieldName": ["DX_Diagnosis", "MH_diagnosis", "DX_Diagnosis"],
                "categories": [
                    {"depression"},
                    {"anxiety"},
                    {"ptsd"},
                ],
            }
        )
        out = construct_session_features(sessions, events)
        p101 = out[out["id"] == 101].set_index("session")

        self.assertEqual(int(p101.loc[0, "dx_depression"]), 1)
        self.assertEqual(int(p101.loc[0, "dx_anxiety"]), 0)
        # A diagnosis recorded on the session date is unavailable that day.
        self.assertEqual(int(p101.loc[1, "dx_anxiety"]), 0)
        # It becomes available to the next later session and is carried forward.
        self.assertEqual(int(p101.loc[2, "dx_anxiety"]), 1)
        # A diagnosis entered after the last session never moves backward in time.
        self.assertEqual(int(p101.loc[2, "dx_ptsd"]), 0)
        p202 = out[out["id"] == 202].iloc[0]
        self.assertEqual(int(p202["dx_history_available"]), 0)
        self.assertEqual(int(p202["dx_comorbidity_count"]), 0)

    def test_output_schema_has_no_raw_fields(self) -> None:
        forbidden = {
            "PatientFID", "AppointmentFID", "ServiceDate", "DOB", "Value",
            "FieldValue_UID", "DX_Diagnosis", "MH_diagnosis", "note_text",
        }
        self.assertFalse(forbidden.intersection({"id", "session", "day", *FEATURE_COLUMNS}))

    def test_builder_refuses_git_working_tree_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protected = root / "protected" / "features.csv"
            self.assertEqual(_assert_protected_output(protected), protected.resolve())
            (root / ".git").mkdir()
            with self.assertRaisesRegex(ValueError, "Git working tree"):
                _assert_protected_output(protected)


if __name__ == "__main__":
    unittest.main()
