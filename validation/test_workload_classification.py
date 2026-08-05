"""Deterministic checks for the empirical assessment-burden classification table."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import BALL

BALL._install_virtual_package()
from empirical.fit_empirical import _workload_classification_rows  # noqa: E402


class WorkloadClassificationTests(unittest.TestCase):
    def test_response_remission_and_clustered_interval(self) -> None:
        raw_prediction = {
            "ball": [5.0, 9.0, 4.0, 11.0],
            "locf": [7.0, 10.0, 5.0, 13.0],
        }
        frame = pd.DataFrame(
            {
                "id": [1, 1, 2, 2],
                "anchor": ["PHQ9"] * 4,
                "raw_true": [4.0, 8.0, 3.0, 12.0],
                "baseline_raw": [20.0, 20.0, 15.0, 15.0],
                "scale_half_range": [13.5] * 4,
                "scale_center": [13.5] * 4,
                **{
                    method: [(value - 13.5) / 13.5 for value in values]
                    for method, values in raw_prediction.items()
                },
            }
        )
        rows = _workload_classification_rows(frame, 14, ["ball", "locf"])
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertLessEqual(
                row["patient_balanced_accuracy_ci_low"],
                row["patient_balanced_accuracy"],
            )
            self.assertLessEqual(
                row["patient_balanced_accuracy"],
                row["patient_balanced_accuracy_ci_high"],
            )
            self.assertEqual(
                row["accuracy_ci_method"],
                "patient-clustered percentile bootstrap",
            )
            self.assertEqual(row["accuracy_ci_replicates"], 2000)


if __name__ == "__main__":
    unittest.main()
