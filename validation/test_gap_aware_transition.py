"""Deterministic checks for elapsed-time-aware BALL transition factors."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import BALL

BALL._install_virtual_package()
from simulations.src.methods.ball_ssm import (  # noqa: E402
    SSMConfig,
    _make_models,
    _gap_transition_factors,
    _raw_elapsed_days,
    build_ssm_batch,
)
from simulations.src.dgp import generate_dataset  # noqa: E402
from simulations.src.model_utils import SimulationConfig  # noqa: E402


class GapAwareTransitionTests(unittest.TestCase):
    def test_ar_transition_changes_with_elapsed_days(self) -> None:
        gaps = torch.tensor([0.0, 1.0, 3.0])
        persistence, drift, innovation = _gap_transition_factors(gaps, 0.5)

        torch.testing.assert_close(persistence, torch.tensor([1.0, 0.5, 0.125]))
        torch.testing.assert_close(drift, torch.tensor([0.0, 1.0, 1.75]))
        torch.testing.assert_close(innovation, torch.tensor([0.0, 1.0, 1.3125]))

    def test_random_walk_limit_accumulates_linearly(self) -> None:
        gaps = torch.tensor([1.0, 3.0])
        persistence, drift, innovation = _gap_transition_factors(gaps, 1.0)

        torch.testing.assert_close(persistence, torch.ones_like(gaps))
        torch.testing.assert_close(drift, gaps)
        torch.testing.assert_close(innovation, gaps)

    def test_elapsed_days_are_not_clipped(self) -> None:
        elapsed = _raw_elapsed_days(pd.Series([0.0, 1.0, 3.0]))
        np.testing.assert_array_equal(elapsed, np.array([0.0, 1.0, 3.0], dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            _raw_elapsed_days(pd.Series([1.0, -1.0]))
        with self.assertRaisesRegex(ValueError, "cannot be missing"):
            _raw_elapsed_days(pd.Series([1.0, np.nan]))

    def test_invalid_persistence_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "delta_phi"):
            _gap_transition_factors(torch.tensor([1.0]), 0.0)
        with self.assertRaisesRegex(ValueError, "delta_phi"):
            _gap_transition_factors(torch.tensor([1.0]), 1.1)

    def test_zero_day_pair_is_masked_without_numerical_failure(self) -> None:
        data = generate_dataset(SimulationConfig(n=12, t=7, seed=91))
        train_id = int(data.individuals.loc[data.individuals["split"] == "train", "id"].iloc[0])
        data.components["dt"] = 1.0
        data.components.loc[
            (data.components["id"] == train_id) & (data.components["t"] == 0), "dt"
        ] = 0.0
        batch, _ = build_ssm_batch(data, "train", torch.device("cpu"), 12)
        generator, _ = _make_models(
            data,
            batch,
            SSMConfig(d_model=16, n_heads=4, n_layers=1, ensemble_size=1),
            torch.device("cpu"),
            causal=False,
        )
        shape = (len(batch.ids), len(batch.t_values))
        alpha = torch.zeros((*shape, data.config.q))
        delta_d = torch.zeros(shape)
        delta_p = torch.zeros(shape)

        self.assertEqual(float(batch.dt.min()), 0.0)
        self.assertTrue(torch.isfinite(generator.log_prior(alpha, delta_d, delta_p, batch)).all())


if __name__ == "__main__":
    unittest.main()
