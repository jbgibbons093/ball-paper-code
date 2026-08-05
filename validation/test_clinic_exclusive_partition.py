"""Synthetic checks for the clinic-exclusive empirical partition."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import BALL
BALL._install_virtual_package()
from empirical.fit_empirical import (  # noqa: E402
    _canonical_empirical_clinic,
    _generalization_split_maps,
    _primary_empirical_partitions,
)


class ClinicExclusivePartitionTests(unittest.TestCase):
    @staticmethod
    def _sessions() -> tuple[pd.DataFrame, int]:
        rows: list[dict] = []
        patient_id = 1000
        clinic_names = [f"CLINIC_{index:02d}" for index in range(19)]
        clinic_splits = ["train"] * 10 + ["val"] * 3 + ["test"] * 6
        first_train_patient = -1
        for clinic, split in zip(clinic_names, clinic_splits, strict=True):
            for _ in range(2):
                if first_train_patient < 0 and split == "train":
                    first_train_patient = patient_id
                rows.extend(
                    [
                        {"id": patient_id, "split": split, "FacilityName": clinic, "session": 0},
                        {"id": patient_id, "split": split, "FacilityName": clinic, "session": 1},
                    ]
                )
                patient_id += 1
        rows.append(
            {
                "id": first_train_patient,
                "split": "train",
                "FacilityName": clinic_names[-1],
                "session": 2,
            }
        )
        return pd.DataFrame(rows), first_train_patient

    def test_cross_partition_patient_is_excluded(self) -> None:
        sessions, crossing_patient = self._sessions()
        model_splits, interval_roles, summary = _primary_empirical_partitions(sessions)

        self.assertNotIn(crossing_patient, model_splits)
        self.assertNotIn(crossing_patient, interval_roles)
        self.assertEqual(summary["source_clinic_partitions"], {"train": 10, "val": 3, "test": 6})
        self.assertEqual(summary["canonical_clinic_count"], 19)
        self.assertEqual(summary["cross_partition_patients_excluded"], 1)
        self.assertEqual(
            summary["clinic_exclusive_patient_partitions"],
            {"train": 19, "val": 6, "test": 12},
        )
        self.assertEqual(summary["primary_model_partitions"], {"development": 25, "heldout_clinic": 12})
        self.assertEqual(summary["primary_interval_partitions"], {"calibration": 6, "test": 6})
        self.assertEqual(len(model_splits), 37)
        self.assertEqual(len(interval_roles), 12)

    def test_known_truncated_labels_reconcile(self) -> None:
        self.assertEqual(_canonical_empirical_clinic("DALLA"), "DALLAS_GRP")
        self.assertEqual(_canonical_empirical_clinic("DALLAS"), "DALLAS_GRP")
        self.assertEqual(_canonical_empirical_clinic("GRAPE"), "GRAPEVINE_GRP")
        self.assertEqual(_canonical_empirical_clinic("GRAPEVINE"), "GRAPEVINE_GRP")

    def test_forward_split_excludes_patients_spanning_the_cutoff(self) -> None:
        sessions = pd.DataFrame(
            [
                {"id": 1, "split": "train", "date": "2024-01-01"},
                {"id": 1, "split": "train", "date": "2024-12-31"},
                {"id": 2, "split": "val", "date": "2025-01-01"},
                {"id": 2, "split": "val", "date": "2025-02-01"},
                {"id": 3, "split": "train", "date": "2024-12-15"},
                {"id": 3, "split": "train", "date": "2025-01-15"},
            ]
        )
        split = _generalization_split_maps(sessions)["forward_2025"]
        self.assertEqual(split, {1: "train", 2: "test"})
        self.assertNotIn(3, split)


if __name__ == "__main__":
    unittest.main()
