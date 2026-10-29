"""End-to-end test of the ML training and export pipeline on a tiny synthetic set (MSP-22).

Covers FR-010, FR-011, FR-012: native model files, SHA-256 in the card, calibration and
interval values in the card, shortfalls reported. Real datasets are not needed.
"""
# pylint: disable=missing-function-docstring,missing-class-docstring,protected-access
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import xgboost as xgb

from backend.workers.ml import featuriser
from tests.unit.training.synthetic import make_frame
from training import datasets, export_models, qa_acc

FAST_REGRESSOR = {"n_estimators": 25, "max_depth": 3}
FAST_BASELINE = {"n_estimators": 15, "max_depth": 4}
FAST_CLASSIFIER = {"n_estimators": 20, "max_depth": 3}
RAW = {"file": "synthetic.csv", "sha256": "0" * 64}
STATS = {"n_raw": 220, "n_clean": 220}


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SolubilityExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp())
        cls.dirs = {"models": cls.folder / "models" / "ml", "cards": cls.folder / "cards"}
        cls.frame = make_frame()
        cls.card = export_models.export_solubility(
            cls.frame, STATS, "aqsoldb", RAW, cls.dirs, FAST_REGRESSOR, FAST_BASELINE, n_jobs=1
        )

    def test_fr010_model_is_a_native_xgboost_json_file_that_reloads(self):
        entry = self.card["artifacts"][0]
        path = self.dirs["models"] / entry["file"]
        self.assertTrue(entry["file"].endswith(".json"))
        booster = xgb.Booster()
        booster.load_model(str(path))
        features = export_models.build_features(self.frame["smiles_std"].tolist()[:3])
        self.assertEqual(booster.predict(xgb.DMatrix(features)).shape, (3,))

    def test_every_artifact_sha256_in_the_card_matches_its_file(self):
        for entry in self.card["artifacts"]:
            with self.subTest(file=entry["file"]):
                self.assertEqual(entry["sha256"], sha256_of(self.dirs["models"] / entry["file"]))
                self.assertEqual(len(entry["sha256"]), 64)

    def test_no_pickle_files_are_written(self):
        names = [path.name for path in self.dirs["models"].iterdir()]
        self.assertTrue(all(name.endswith((".json", ".npy")) for name in names), names)

    def test_reference_fingerprints_load_without_pickle(self):
        entry = next(a for a in self.card["artifacts"] if a["file"].endswith(".npy"))
        packed = np.load(self.dirs["models"] / entry["file"], allow_pickle=False)
        self.assertEqual(packed.shape[1], featuriser.MORGAN_BITS // 8)
        self.assertEqual(packed.dtype, np.uint8)
        self.assertEqual(packed.shape[0], self.card["split"]["n_train"])

    def test_fr012_card_holds_the_interval_offset_and_measured_coverage(self):
        unc = self.card["uncertainty"]
        self.assertEqual(unc["alpha"], 0.1)
        self.assertGreater(unc["offset"], 0.0)
        self.assertTrue(0.0 <= unc["test_coverage"] <= 1.0)
        self.assertEqual(unc["n_calibration"], self.card["split"]["n_calibration"])

    def test_card_records_provenance_split_metrics_and_versions(self):
        card = self.card
        self.assertEqual(card["unit"], "log10(mol/L)")
        self.assertEqual(card["featuriser"]["signature"], featuriser.feature_signature())
        self.assertEqual(card["data"]["raw_sha256"], RAW["sha256"])
        self.assertFalse(card["data"]["licence_verified"])
        self.assertIn("rdkit", card["software"])
        self.assertEqual(card["split"]["method"], "scaffold_split_bemis_murcko")
        size = card["split"]["n_train"] + card["split"]["n_calibration"] + card["split"]["n_test"]
        self.assertEqual(size, len(self.frame))
        for key in ("rmse", "mae", "r2"):
            self.assertIn(key, card["metrics"]["test"])
            self.assertIn(key, card["metrics"]["baseline_test"])

    def test_qa_acc_shortfalls_match_the_unmet_checks(self):
        checks = self.card["qa_acc"]["checks"]
        self.assertEqual(self.card["qa_acc"]["shortfalls"], [c["id"] for c in checks if not c["met"]])
        self.assertEqual(len(checks), 2)

    def test_the_card_on_disk_is_valid_json_matching_the_returned_card(self):
        path = self.dirs["cards"] / f"ml_solubility_xgb_{export_models.MODEL_VERSION}.json"
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), json.loads(json.dumps(self.card)))

    def test_training_twice_gives_the_same_predictions(self):
        other = Path(tempfile.mkdtemp())
        dirs = {"models": other / "m", "cards": other / "c"}
        again = export_models.export_solubility(
            self.frame, STATS, "aqsoldb", RAW, dirs, FAST_REGRESSOR, FAST_BASELINE, n_jobs=1
        )
        self.assertEqual(again["metrics"]["test"], self.card["metrics"]["test"])
        self.assertEqual(again["uncertainty"]["offset"], self.card["uncertainty"]["offset"])


class Tox21ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp())
        cls.dirs = {"models": cls.folder / "models" / "ml", "cards": cls.folder / "cards"}
        cls.frame = make_frame()
        cls.card = export_models.export_tox21(cls.frame, STATS, RAW, cls.dirs, FAST_CLASSIFIER, n_jobs=1)

    def test_fr011_one_model_file_per_assay_all_twelve(self):
        self.assertEqual(list(self.card["tasks"]), list(datasets.TOX21_TASKS))
        files = [entry["file"] for entry in self.card["artifacts"] if "task" in entry]
        self.assertEqual(len(files), 12)
        self.assertEqual(len(set(files)), 12)
        self.assertTrue(all(name.endswith(".json") for name in files))

    def test_fr011_every_assay_has_calibration_numbers_and_a_curve(self):
        for task, entry in self.card["tasks"].items():
            with self.subTest(task=task):
                self.assertEqual(entry["status"], "trained")
                self.assertGreater(entry["calibration"]["a"], 0.0)   # keeps the ranking
                self.assertIn("b", entry["calibration"])
                self.assertGreater(len(entry["reliability_curve"]), 0)
                self.assertIn("roc_auc", entry["metrics_test"])
                self.assertIn("ece", entry["metrics_test"])

    def test_mean_auc_is_the_average_of_the_assay_aucs(self):
        aucs = [e["metrics_test"]["roc_auc"] for e in self.card["tasks"].values()]
        self.assertAlmostEqual(self.card["metrics"]["mean_roc_auc_test"], float(np.mean(aucs)))

    def test_artifact_hashes_match_and_qa_check_is_recorded(self):
        for entry in self.card["artifacts"]:
            self.assertEqual(entry["sha256"], sha256_of(self.dirs["models"] / entry["file"]))
        self.assertEqual(self.card["qa_acc"]["checks"][0]["id"], "QA-ACC-TOX21-AUC")

    def test_manifest_lists_every_file_with_its_hash(self):
        manifest = export_models.write_manifest([self.card], self.dirs["models"])
        on_disk = json.loads((self.dirs["models"] / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk["files"], json.loads(json.dumps(manifest["files"])))
        self.assertEqual(len(manifest["files"]), len(self.card["artifacts"]))
        self.assertTrue(all(item["tier"] == "ml" for item in manifest["files"]))

    def test_an_assay_with_one_class_is_skipped_not_hidden(self):
        frame = make_frame()
        frame["NR-AR"] = 0.0
        dirs = {"models": self.folder / "m2", "cards": self.folder / "c2"}
        card = export_models.export_tox21(frame, STATS, RAW, dirs, FAST_CLASSIFIER, n_jobs=1)
        self.assertTrue(card["tasks"]["NR-AR"]["status"].startswith("skipped"))
        self.assertNotIn("file", card["tasks"]["NR-AR"])
        self.assertEqual(sum(1 for e in card["artifacts"] if "task" in e), 11)


class SaveModelSafetyTests(unittest.TestCase):
    def test_a_reloaded_model_that_predicts_differently_is_rejected(self):
        frame = make_frame(60)
        features = export_models.build_features(frame["smiles_std"].tolist())
        model = xgb.XGBRegressor(n_estimators=5, max_depth=2).fit(features, frame["logs"].to_numpy())
        folder = Path(tempfile.mkdtemp())
        with mock.patch.object(xgb.Booster, "predict", return_value=np.zeros(5)):
            with self.assertRaises(RuntimeError):
                export_models.save_model(model, folder / "x.json", features[:5], classifier=False)

    def test_qa_acc_thresholds_are_provisional_in_every_card(self):
        for check in qa_acc.solubility_checks(1.0, 1.0, 0.9) + qa_acc.tox21_checks(0.8):
            self.assertTrue(check["provisional"])


if __name__ == "__main__":
    unittest.main()
