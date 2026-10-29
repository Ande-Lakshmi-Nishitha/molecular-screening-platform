"""Unit tests for the metrics and the QA-ACC checks (MSP-22; QA-ACC)."""
# pylint: disable=missing-function-docstring,missing-class-docstring
import math
import unittest

from training import qa_acc
from training.evaluate import classification_metrics, mean_of, regression_metrics


class RegressionMetricTests(unittest.TestCase):
    def test_known_values(self):
        metrics = regression_metrics([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 6.0])
        self.assertAlmostEqual(metrics["rmse"], 1.0)     # sqrt(4 / 4)
        self.assertAlmostEqual(metrics["mae"], 0.5)
        self.assertAlmostEqual(metrics["r2"], 1.0 - 4.0 / 5.0)
        self.assertEqual(metrics["n"], 4)

    def test_perfect_prediction(self):
        metrics = regression_metrics([1.0, 5.0, 9.0], [1.0, 5.0, 9.0])
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["r2"], 1.0)

    def test_constant_target_gives_nan_r2_not_an_error(self):
        self.assertTrue(math.isnan(regression_metrics([2.0, 2.0], [1.0, 3.0])["r2"]))


class ClassificationMetricTests(unittest.TestCase):
    def test_perfect_ranking_has_auc_one(self):
        metrics = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["n_positive"], 2)
        self.assertAlmostEqual(metrics["brier"], (0.01 + 0.04 + 0.04 + 0.01) / 4)

    def test_single_class_gives_no_auc(self):
        self.assertIsNone(classification_metrics([1, 1, 1], [0.6, 0.7, 0.9])["roc_auc"])

    def test_mean_of_skips_missing_values(self):
        self.assertAlmostEqual(mean_of([0.8, None, 0.6]), 0.7)
        self.assertIsNone(mean_of([None, None]))
        self.assertIsNone(mean_of([]))


class QaAccTests(unittest.TestCase):
    def test_solubility_rmse_boundary_is_ten_percent_over_the_baseline(self):
        met = qa_acc.solubility_checks(1.10, 1.0, 0.9)[0]
        missed = qa_acc.solubility_checks(1.11, 1.0, 0.9)[0]
        self.assertTrue(met["met"])
        self.assertFalse(missed["met"])

    def test_better_than_the_baseline_passes(self):
        self.assertTrue(qa_acc.solubility_checks(0.8, 1.0, 0.9)[0]["met"])

    def test_coverage_must_be_between_85_and_95_percent(self):
        flags = {c: qa_acc.solubility_checks(1.0, 1.0, c)[1]["met"] for c in (0.84, 0.85, 0.90, 0.95, 0.96)}
        self.assertEqual(flags, {0.84: False, 0.85: True, 0.90: True, 0.95: True, 0.96: False})

    def test_tox21_auc_threshold_and_missing_value(self):
        self.assertTrue(qa_acc.tox21_checks(0.75)[0]["met"])
        self.assertFalse(qa_acc.tox21_checks(0.74)[0]["met"])
        self.assertFalse(qa_acc.tox21_checks(None)[0]["met"])

    def test_checks_are_marked_provisional_and_shortfalls_are_listed(self):
        checks = qa_acc.solubility_checks(2.0, 1.0, 0.5)
        self.assertTrue(all(c["provisional"] and c["reference"] == "TBD-C4" for c in checks))
        self.assertEqual(qa_acc.shortfalls(checks), ["QA-ACC-ML-RMSE", "QA-ACC-ML-COVERAGE"])
        self.assertEqual(qa_acc.shortfalls(qa_acc.tox21_checks(0.9)), [])


if __name__ == "__main__":
    unittest.main()
