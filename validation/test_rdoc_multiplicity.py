"""Tests for the six-test empirical RDoC multiplicity adjustment."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import BALL

BALL._install_virtual_package()
from empirical.fit_empirical import _holm_adjusted_probabilities  # noqa: E402


def test_holm_adjustment_uses_all_six_joint_tests_and_preserves_order() -> None:
    raw = np.array([0.010, 0.040, 0.005, 0.200, 0.030, 0.500])
    adjusted = _holm_adjusted_probabilities(raw)
    np.testing.assert_allclose(adjusted, np.array([0.050, 0.120, 0.030, 0.400, 0.120, 0.500]))
    assert np.all(adjusted >= raw)
