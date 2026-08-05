"""Deterministic checks for patient-balanced cluster-weighted conformal scores."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import BALL


BALL._install_virtual_package()
from empirical.fit_empirical import (  # noqa: E402
    _patient_balanced_conformal_quantile as empirical_quantile,
)
from simulations.src.methods.ball_structural import (  # noqa: E402
    _patient_balanced_conformal_quantile as simulation_quantile,
)


class ClusterWeightedConformalTests(unittest.TestCase):
    @staticmethod
    def _scores(repeat_first_patient: int = 1) -> pd.DataFrame:
        first = pd.DataFrame(
            {
                "id": [1, 1] * repeat_first_patient,
                "normalized_score": [1.0, 2.0] * repeat_first_patient,
            }
        )
        other = pd.DataFrame(
            {"id": [2, 3, 3], "normalized_score": [3.0, 4.0, 5.0]}
        )
        return pd.concat([first, other], ignore_index=True)

    def test_each_patient_has_total_weight_one(self) -> None:
        q, n_patients, weighted = simulation_quantile(self._scores(), alpha=0.5)
        self.assertEqual(n_patients, 3)
        self.assertEqual(q, 3.0)
        np.testing.assert_allclose(
            weighted.groupby("id")["cluster_weight"].sum().to_numpy(),
            np.ones(3),
        )

    def test_repeating_one_patients_rows_does_not_change_its_total_weight(self) -> None:
        q_original, _, _ = simulation_quantile(self._scores(), alpha=0.5)
        q_repeated, _, weighted = simulation_quantile(
            self._scores(repeat_first_patient=4), alpha=0.5
        )
        self.assertEqual(q_original, q_repeated)
        self.assertAlmostEqual(
            float(weighted.loc[weighted["id"].eq(1), "cluster_weight"].sum()),
            1.0,
        )

    def test_empirical_and_simulation_implementations_match(self) -> None:
        scores = self._scores()
        q_simulation, n_simulation, _ = simulation_quantile(scores, alpha=0.5)
        q_empirical, n_empirical = empirical_quantile(scores, alpha=0.5)
        self.assertEqual((q_empirical, n_empirical), (q_simulation, n_simulation))

    def test_patient_count_rank_can_return_infinite_interval(self) -> None:
        q, n_patients = empirical_quantile(self._scores(), alpha=0.05)
        self.assertEqual(n_patients, 3)
        self.assertTrue(np.isinf(q))


if __name__ == "__main__":
    unittest.main()
