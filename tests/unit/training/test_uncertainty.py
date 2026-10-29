"""Unit tests for the conformal interval and probability calibration (FR-011, FR-012)."""
# pylint: disable=missing-function-docstring,missing-class-docstring
import unittest

import numpy as np

from training.uncertainty import (
    apply_platt,
    conformal_offset,
    fit_platt,
    interval_coverage,
    reliability_table,
)


class ConformalTests(unittest.TestCase):
    def test_fr012_offset_is_the_corrected_quantile_of_residuals(self):
        truth, pred = np.zeros(10), np.arange(1, 11, dtype=float)   # residuals 1..10
        self.assertEqual(conformal_offset(truth, pred, alpha=0.1), 10.0)   # k = ceil(11*0.9) = 10
        self.assertEqual(conformal_offset(truth, pred, alpha=0.2), 9.0)    # k = ceil(11*0.8) = 9

    def test_fr012_residual_sign_does_not_matter(self):
        truth = np.zeros(10)
        up, down = np.arange(1, 11, dtype=float), -np.arange(1, 11, dtype=float)
        self.assertEqual(conformal_offset(truth, up), conformal_offset(truth, down))

    def test_fr012_too_few_points_uses_the_largest_residual(self):
        self.assertEqual(conformal_offset([0, 0, 0], [1, 2, 3], alpha=0.1), 3.0)

    def test_fr012_empty_calibration_set_rejected(self):
        with self.assertRaises(ValueError):
            conformal_offset([], [])

    def test_fr012_interval_covers_about_ninety_percent_on_new_data(self):
        rng = np.random.default_rng(1)
        cal_true, cal_pred = rng.normal(0, 1, 3000), rng.normal(0, 0.2, 3000)
        new_true, new_pred = rng.normal(0, 1, 5000), rng.normal(0, 0.2, 5000)
        offset = conformal_offset(cal_true, cal_pred, 0.1)
        self.assertAlmostEqual(interval_coverage(new_true, new_pred, offset), 0.9, delta=0.025)

    def test_fr012_coverage_counts_points_inside_the_interval(self):
        self.assertEqual(interval_coverage([0, 0, 0, 0], [0.5, 1.0, 1.5, 2.0], 1.0), 0.5)


class PlattTests(unittest.TestCase):
    def test_fr011_fit_recovers_a_known_calibration_map(self):
        rng = np.random.default_rng(3)
        raw = rng.uniform(0.02, 0.98, 20000)
        logit = np.log(raw / (1 - raw))
        true_p = 1 / (1 + np.exp(-(2.0 * logit + 0.5)))
        labels = (rng.random(20000) < true_p).astype(int)
        slope, intercept = fit_platt(raw, labels)
        self.assertAlmostEqual(slope, 2.0, delta=0.15)
        self.assertAlmostEqual(intercept, 0.5, delta=0.1)

    def test_fr011_calibrated_values_are_probabilities_and_keep_the_order(self):
        raw = np.linspace(0.0, 1.0, 101)          # includes the extreme values 0 and 1
        calibrated = apply_platt(raw, 1.5, -0.2)
        self.assertTrue(np.all((calibrated > 0) & (calibrated < 1)))
        self.assertTrue(np.all(np.diff(calibrated) > 0))

    def test_fr011_single_class_labels_give_the_identity_map(self):
        self.assertEqual(fit_platt([0.1, 0.4, 0.8], [0, 0, 0]), (1.0, 0.0))
        calibrated = apply_platt(np.array([0.2, 0.7]), 1.0, 0.0)
        np.testing.assert_allclose(calibrated, [0.2, 0.7], atol=1e-6)


class ReliabilityTests(unittest.TestCase):
    def test_bins_hold_every_point_once(self):
        rng = np.random.default_rng(5)
        probs = rng.random(500)
        rows, _ = reliability_table((rng.random(500) < probs).astype(int), probs, n_bins=10)
        self.assertEqual(sum(row["count"] for row in rows), 500)
        self.assertEqual(len(rows), 10)

    def test_well_calibrated_probabilities_have_small_ece(self):
        rng = np.random.default_rng(6)
        probs = rng.random(20000)
        _, ece = reliability_table((rng.random(20000) < probs).astype(int), probs)
        self.assertLess(ece, 0.03)

    def test_overconfident_probabilities_have_large_ece(self):
        rng = np.random.default_rng(7)
        labels = (rng.random(5000) < 0.5).astype(int)
        probs = np.where(labels == 1, 0.99, 0.01) * 0 + 0.99   # always confident, half wrong
        _, ece = reliability_table(labels, probs)
        self.assertGreater(ece, 0.4)

    def test_fewer_points_than_bins_still_works(self):
        rows, _ = reliability_table([0, 1, 1], [0.1, 0.6, 0.9], n_bins=10)
        self.assertEqual(sum(row["count"] for row in rows), 3)


if __name__ == "__main__":
    unittest.main()
