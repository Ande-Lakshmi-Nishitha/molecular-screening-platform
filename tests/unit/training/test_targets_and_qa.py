"""Tests for the target list (FR-015, TBD-C2) and the DL QA-ACC checks."""
# pylint: disable=missing-function-docstring,missing-class-docstring
import unittest
from pathlib import Path

from backend.core.config import load_yaml_safe
from training import qa_acc
from training.fetch_chembl import ACTIVITY_TYPE

TARGETS_FILE = Path(__file__).resolve().parents[3] / "backend" / "config" / "targets.yaml"


class TargetsFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.targets = load_yaml_safe(TARGETS_FILE)["targets"]

    def test_fr015_three_targets_are_configured(self):
        self.assertEqual([t["id"] for t in self.targets], ["EGFR", "BACE1", "HIV1_PROTEASE"])

    def test_every_target_has_id_name_chembl_id_and_unit(self):
        for target in self.targets:
            with self.subTest(target=target["id"]):
                for key in ("id", "name", "chembl_id", "unit"):
                    self.assertTrue(target.get(key), key)
                self.assertRegex(target["chembl_id"], r"^CHEMBL\d+$")
                self.assertIn(target["unit"], ACTIVITY_TYPE)

    def test_ids_and_chembl_ids_are_unique(self):
        self.assertEqual(len({t["id"] for t in self.targets}), len(self.targets))
        self.assertEqual(len({t["chembl_id"] for t in self.targets}), len(self.targets))

    def test_ids_are_safe_to_use_in_file_names(self):
        for target in self.targets:
            self.assertRegex(target["id"], r"^[A-Za-z0-9_]+$")


class DlQaChecksTests(unittest.TestCase):
    def test_both_targets_met(self):
        checks = qa_acc.dl_activity_checks(0.9, 0.6)
        self.assertEqual([c["id"] for c in checks], ["QA-ACC-DL-RMSE", "QA-ACC-DL-SPEARMAN"])
        self.assertEqual(qa_acc.shortfalls(checks), [])
        self.assertTrue(all(c["provisional"] and c["reference"] == "TBD-C4" for c in checks))

    def test_shortfalls_are_reported_not_hidden(self):
        checks = qa_acc.dl_activity_checks(1.3, 0.4)
        self.assertEqual(qa_acc.shortfalls(checks), ["QA-ACC-DL-RMSE", "QA-ACC-DL-SPEARMAN"])
        self.assertEqual(checks[0]["measured"], 1.3)

    def test_boundary_values_count_as_met(self):
        self.assertEqual(qa_acc.shortfalls(qa_acc.dl_activity_checks(1.0, 0.5)), [])

    def test_missing_spearman_is_a_shortfall(self):
        self.assertEqual(qa_acc.shortfalls(qa_acc.dl_activity_checks(0.5, None)), ["QA-ACC-DL-SPEARMAN"])


if __name__ == "__main__":
    unittest.main()
