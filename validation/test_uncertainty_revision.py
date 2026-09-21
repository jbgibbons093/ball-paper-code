"""Known-answer checks for the covariance and prospective-timing corrections."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'empirical'))
from uncertainty_math import recall_moments,causal_window_positions,expected_fractional_variance_reduction
from conditional_intervals import severity_inverted_interval

class UncertaintyRevisionTests(unittest.TestCase):
    def test_shared_member_disagreement_does_not_average_away(self):
        mu,var=recall_moments([[-1,-1],[1,1]],np.zeros((2,2)),np.zeros((2,2,1)),np.zeros((2,1)),np.zeros(2))
        self.assertEqual(mu,0);self.assertEqual(var,1)

    def test_shared_weight_and_bias_variance_survives_averaging(self):
        _,var=recall_moments([[0,0]],[[0,0]],[[[1],[1]]],[[.25]],[.1])
        self.assertAlmostEqual(var,.35)

    def test_opposite_features_cancel_weight_covariance(self):
        _,var=recall_moments([[0,0]],[[0,0]],[[[1],[-1]]],[[.25]],[.1])
        self.assertAlmostEqual(var,.1)

    def test_independent_state_noise_still_averages_down(self):
        _,var=recall_moments([[0,0]],[[1,1]],[[[0],[0]]],[[0]],[0])
        self.assertAlmostEqual(var,.5)

    def test_same_day_later_sessions_are_excluded(self):
        np.testing.assert_array_equal(causal_window_positions([0,7,14,14,15],0,14,2),[0,1,2])

    def test_gain_targets_reducible_latent_uncertainty(self):
        self.assertAlmostEqual(expected_fractional_variance_reduction(1,1,1),.5)
        self.assertEqual(expected_fractional_variance_reduction(0,1,1),0)
        self.assertLess(expected_fractional_variance_reduction(1,1,10),.01)

    def test_score_band_inversion_includes_all_accepted_candidates(self):
        left,right,empty=severity_inverted_interval(np.array([-.3]),np.array([.3]),{0:.1,1:.1,2:.4})
        self.assertFalse(empty[0]);self.assertAlmostEqual(left[0],-.4);self.assertAlmostEqual(right[0],.7)
        # No test outcome is an argument to the inversion.

if __name__=='__main__':unittest.main()
