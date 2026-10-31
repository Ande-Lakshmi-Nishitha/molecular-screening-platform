"""Unit tests for the model information reader (MSP-38; FR-040, FR-064, BR-009). No Flask or database."""
import json
import tempfile
import unittest
from pathlib import Path

from backend.api.models_info.routes import CARDS_DIR, load_model_infos


def card(**changes):
    base = {
        "tier": "ml", "task": "solubility", "name": "ml_solubility_xgb", "version": "1.0.0",
        "unit": "log10(mol/L)", "trained_at": "2026-10-29T10:00:00Z", "seed": 42,
        "artifacts": [{"file": "model.json", "sha256": "abc"}],
        "requirements": {"xgboost": "3.2.0"}, "qa_acc": {"x": 1},
        "data": {"source": "AqSolDB", "citation": "Sorkun et al.", "licence_note": "Check the dataset page.",
                 "licence_verified": False, "raw_sha256": "deadbeef", "raw_file": "aqsoldb.csv", "n_clean": 9424},
        "split": {"method": "scaffold_split_bemis_murcko", "n_train": 6596, "n_calibration": 942, "n_test": 1886, "seed": 42},
        "metrics": {"test": {"mae": 0.75, "r2": 0.78, "n": 1886}},
    }
    base.update(changes)
    return base


class ModelInfoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def write(self, name, content):
        (self.folder / name).write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")

    def test_exposes_version_source_licence_and_metrics(self):
        self.write("ml_solubility_xgb_1.0.0.json", card())
        (info,) = load_model_infos(self.folder)
        self.assertEqual(info["id"], "ml_solubility_xgb_1.0.0")
        self.assertEqual(info["version"], "1.0.0")
        self.assertEqual(info["data"]["source"], "AqSolDB")
        self.assertEqual(info["data"]["licence_note"], "Check the dataset page.")
        self.assertEqual(info["metrics"]["test"]["mae"], 0.75)
        self.assertEqual(info["split"]["n_test"], 1886)

    def test_leaves_out_internal_fields(self):
        self.write("a.json", card())
        (info,) = load_model_infos(self.folder)
        text = json.dumps(info)
        for hidden in ("artifacts", "sha256", "deadbeef", "aqsoldb.csv", "requirements", "qa_acc", "xgboost"):
            self.assertNotIn(hidden, text)
        self.assertNotIn("seed", info["split"])

    def test_licence_is_unverified_unless_the_card_says_true(self):
        self.write("a.json", card())
        self.write("b.json", card(name="b", data={"source": "X", "licence_verified": True}))
        self.write("c.json", card(name="c", data={"source": "Y"}))
        flags = {i["name"]: i["data"]["licence_verified"] for i in load_model_infos(self.folder)}
        self.assertEqual(flags, {"ml_solubility_xgb": False, "b": True, "c": False})

    def test_orders_by_tier_then_name_and_skips_unreadable_cards(self):
        self.write("1.json", card(tier="dl", name="dl_b"))
        self.write("2.json", card(tier="ml", name="ml_z"))
        self.write("3.json", card(tier="dl", name="dl_a"))
        self.write("4.json", "{ not json")
        self.write("5.json", "[1, 2]")
        self.assertEqual([i["name"] for i in load_model_infos(self.folder)], ["ml_z", "dl_a", "dl_b"])

    def test_lists_assays_of_a_multi_task_card(self):
        self.write("t.json", card(tasks=["NR-AR", {"name": "SR-ARE"}, {"id": "SR-p53"}]))
        (info,) = load_model_infos(self.folder)
        self.assertEqual(info["assays"], ["NR-AR", "SR-ARE", "SR-p53"])

    def test_target_may_be_a_string_or_an_object(self):
        self.write("a.json", card(name="a", target="EGFR"))
        self.write("b.json", card(name="b", target={"id": "BACE1", "chembl_id": "CHEMBL4822"}))
        targets = {i["name"]: i["target"] for i in load_model_infos(self.folder)}
        self.assertEqual(targets, {"a": "EGFR", "b": "BACE1"})

    def test_committed_cards_all_load(self):
        infos = load_model_infos(CARDS_DIR)
        self.assertGreaterEqual(len(infos), 1)
        for info in infos:
            self.assertTrue(info["version"] and info["name"] and info["tier"])
            self.assertIn("licence_verified", info["data"])


if __name__ == "__main__":
    unittest.main()
