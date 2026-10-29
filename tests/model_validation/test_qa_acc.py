"""QA-ACC metric checks on the committed model cards (MSP-22; QA-ACC, BR-009).

Owner: Nishitha. These read the real cards in backend/registry/cards. A model may miss a
QA-ACC target only if the shortfall is reported in its card; this test fails if a missed
target is not listed. If no cards exist yet the tests are skipped.
"""
# pylint: disable=missing-function-docstring,missing-class-docstring
import hashlib
import json
import unittest
from pathlib import Path

from backend.workers.ml import featuriser
from training import qa_acc

ROOT = Path(__file__).resolve().parents[2]
CARDS = sorted((ROOT / "backend" / "registry" / "cards").glob("ml_*_xgb_*.json"))
MODELS = ROOT / "models" / "ml"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


@unittest.skipUnless(CARDS, "no ML model cards committed yet")
class MlModelCardTests(unittest.TestCase):
    def test_every_card_has_the_required_fields(self):
        required = {"name", "version", "tier", "unit", "trained_at", "seed", "software", "featuriser",
                    "data", "split", "model", "metrics", "qa_acc", "artifacts"}
        for path in CARDS:
            with self.subTest(card=path.name):
                self.assertTrue(required <= set(load(path)))

    def test_every_card_records_data_source_and_licence_note(self):
        for path in CARDS:
            data = load(path)["data"]
            with self.subTest(card=path.name):
                for key in ("source", "citation", "licence_note", "raw_sha256"):
                    self.assertTrue(data.get(key), key)

    def test_cards_were_built_with_the_current_featuriser(self):
        for path in CARDS:
            with self.subTest(card=path.name):
                self.assertEqual(load(path)["featuriser"]["signature"], featuriser.feature_signature())

    def test_only_native_model_formats_are_listed(self):
        for path in CARDS:
            for entry in load(path)["artifacts"]:
                with self.subTest(card=path.name, file=entry["file"]):
                    self.assertTrue(entry["file"].endswith((".json", ".npy")))
                    self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")

    def test_every_missed_qa_acc_target_is_reported_as_a_shortfall(self):
        for path in CARDS:
            block = load(path)["qa_acc"]
            with self.subTest(card=path.name):
                unmet = [check["id"] for check in block["checks"] if not check["met"]]
                self.assertEqual(block["shortfalls"], unmet)

    def test_recorded_checks_agree_with_the_recorded_metrics(self):
        for path in CARDS:
            card = load(path)
            if card["task"] == "solubility_logs":
                expected = qa_acc.solubility_checks(
                    card["metrics"]["test"]["rmse"],
                    card["metrics"]["baseline_test"]["rmse"],
                    card["uncertainty"]["test_coverage"],
                )
            else:
                expected = qa_acc.tox21_checks(card["metrics"]["mean_roc_auc_test"])
            with self.subTest(card=path.name):
                self.assertEqual([c["met"] for c in card["qa_acc"]["checks"]], [c["met"] for c in expected])

    def test_solubility_interval_offset_is_recorded(self):
        for path in CARDS:
            card = load(path)
            if card["task"] == "solubility_logs":
                self.assertGreater(card["uncertainty"]["offset"], 0.0)
                self.assertEqual(card["uncertainty"]["alpha"], 0.1)

    def test_every_tox21_assay_has_calibration_values(self):
        for path in CARDS:
            card = load(path)
            if card["task"] == "tox21_assay_signals":
                self.assertEqual(len(card["tasks"]), 12)
                for task, entry in card["tasks"].items():
                    if entry["status"] == "trained":
                        with self.subTest(task=task):
                            self.assertIn("a", entry["calibration"])
                            self.assertIn("b", entry["calibration"])


@unittest.skipUnless(CARDS and MODELS.exists(), "model files are not on this machine (models/ is git-ignored)")
class LocalModelFileIntegrityTests(unittest.TestCase):
    def test_model_files_match_the_sha256_in_their_cards(self):
        for path in CARDS:
            for entry in load(path)["artifacts"]:
                with self.subTest(file=entry["file"]):
                    file_path = MODELS / entry["file"]
                    self.assertTrue(file_path.exists())
                    self.assertEqual(hashlib.sha256(file_path.read_bytes()).hexdigest(), entry["sha256"])


if __name__ == "__main__":
    unittest.main()
