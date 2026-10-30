"""Unit tests for the ML prediction layer: card adapter, LogS, Tox21, predictor (MSP-23).

Covers FR-010, FR-011, FR-012, BR-006, SEC-007, SEC-008, NFR-007, NFR-008, NFR-009, NFR-011.
The model files and cards come from the project's own export code on a small synthetic set
(see ml_fixtures.py), so the real card layout is what gets read.

Run from the repo root with:  python -m unittest discover -s tests -t .
"""
# pylint: disable=missing-function-docstring,missing-class-docstring,protected-access
# pylint: disable=no-member,c-extension-no-member,too-many-public-methods
import json
import math
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import xgboost as xgb
from rdkit import Chem

from backend.core import labels
from backend.registry.loader import ModelLoadError, sha256_of
from backend.workers.ml import featuriser, model_cards, predictor, solubility, tox21
from tests.fixtures.golden_molecules import GOLDEN
from tests.unit.workers_ml import ml_fixtures as fx
from training import datasets, export_models
from training.uncertainty import apply_platt as training_apply_platt

ASPIRIN = GOLDEN["aspirin"]["smiles"]
SOL_CARD, TOX_CARD = "ml_solubility_xgb", "ml_tox21_xgb"


class CopyCase(unittest.TestCase):
    """Each test gets a private copy of the built models and cards to damage freely."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.root = fx.copy_of_built(Path(tmp.name))
        self.folder = model_cards.models_folder(fx.models_dir(self.root))
        predictor.reset_predictor()
        self.addCleanup(predictor.reset_predictor)

    def card_path(self, name):
        return fx.card_file(self.root, name)

    def edit_card(self, name, change):
        path = self.card_path(name)
        card = json.loads(path.read_text(encoding="utf-8"))
        change(card)
        path.write_text(json.dumps(card), encoding="utf-8")

    def load_solubility(self):
        return solubility.SolubilityModel.load(self.card_path(SOL_CARD), self.folder)

    def load_tox21(self):
        return tox21.Tox21Models.load(self.card_path(TOX_CARD), self.folder)

    def load_predictor(self):
        return predictor.MlPredictor.load(fx.models_dir(self.root), cards_dir=self.root / "cards")

    def model_file(self, card_name):
        card = json.loads(self.card_path(card_name).read_text(encoding="utf-8"))
        return self.folder / model_cards.model_files(card)[0]["file"]


class CardReadingTests(CopyCase):  # SEC-008
    def test_the_real_export_layout_loads(self):
        self.assertEqual(self.load_solubility().version, export_models.MODEL_VERSION)
        self.assertEqual(len(self.load_tox21().boosters), 12)

    def test_a_card_for_another_tier_or_task_is_refused(self):
        with self.assertRaises(ModelLoadError):
            model_cards.read_card(self.card_path(TOX_CARD), solubility.TASK)
        self.edit_card(SOL_CARD, lambda card: card.update(tier="dl"))
        with self.assertRaises(ModelLoadError):
            self.load_solubility()

    def test_unreadable_or_non_object_card_is_refused(self):
        with self.assertRaises(ModelLoadError):
            model_cards.read_card(self.root / "nope.json", solubility.TASK)
        self.card_path(SOL_CARD).write_text("[1, 2]", encoding="utf-8")
        with self.assertRaises(ModelLoadError):
            self.load_solubility()

    def test_a_file_entry_without_a_hash_is_refused(self):
        self.edit_card(SOL_CARD, lambda card: card["artifacts"][0].pop("sha256"))
        with self.assertRaises(ModelLoadError):
            self.load_solubility()

    def test_a_model_built_with_other_features_is_refused(self):
        self.edit_card(SOL_CARD, lambda card: card["featuriser"].update(signature="0" * 16))
        with self.assertRaises(ModelLoadError):
            self.load_solubility()
        self.edit_card(TOX_CARD, lambda card: card["featuriser"].update(n_features=10))
        with self.assertRaises(ModelLoadError):
            self.load_tox21()

    def test_a_card_without_featuriser_info_is_refused(self):
        self.edit_card(SOL_CARD, lambda card: card.pop("featuriser"))
        with self.assertRaises(ModelLoadError):
            self.load_solubility()

    def test_a_changed_model_file_fails_the_sha256_check(self):
        path = self.model_file(SOL_CARD)
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(ModelLoadError):
            self.load_solubility()

    def test_a_missing_model_file_is_refused(self):
        self.model_file(TOX_CARD).unlink()
        with self.assertRaises(ModelLoadError):
            self.load_tox21()

    def test_a_pickle_named_file_is_refused_even_with_a_matching_hash(self):
        path = self.model_file(SOL_CARD)
        pickled = path.with_name("model.pkl")
        pickled.write_bytes(path.read_bytes())

        def change(card):
            card["artifacts"][0].update(file="model.pkl", sha256=sha256_of(pickled))

        self.edit_card(SOL_CARD, change)
        with self.assertRaises(ModelLoadError):
            self.load_solubility()

    def test_a_pickle_header_is_refused_even_with_a_safe_name_and_matching_hash(self):
        path = self.model_file(SOL_CARD)
        path.write_bytes(b"\x80\x04\x95rest")
        self.edit_card(SOL_CARD, lambda card: card["artifacts"][0].update(sha256=sha256_of(path)))
        with self.assertRaises(ModelLoadError):
            self.load_solubility()

    def test_a_file_name_that_leaves_the_models_folder_is_refused(self):
        outside = self.root / "outside.json"
        outside.write_bytes(self.model_file(SOL_CARD).read_bytes())

        def change(card):
            card["artifacts"][0].update(file="../../outside.json", sha256=sha256_of(outside))

        self.edit_card(SOL_CARD, change)
        with self.assertRaises(ModelLoadError):
            self.load_solubility()

    def test_a_file_name_with_a_folder_part_is_refused_even_inside_the_models_folder(self):
        nested = self.folder / "nested"
        nested.mkdir()
        copy = nested / "model.json"
        copy.write_bytes(self.model_file(SOL_CARD).read_bytes())

        def change(card):
            card["artifacts"][0].update(file="nested/model.json", sha256=sha256_of(copy))

        self.edit_card(SOL_CARD, change)
        with self.assertRaises(ModelLoadError):
            self.load_solubility()

    def test_a_link_inside_the_models_folder_that_points_outside_is_refused(self):
        outside = self.root / "outside.json"
        outside.write_bytes(self.model_file(SOL_CARD).read_bytes())
        link = self.folder / "linked.json"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("this system does not allow creating links")

        def change(card):
            card["artifacts"][0].update(file="linked.json", sha256=sha256_of(outside))

        self.edit_card(SOL_CARD, change)
        with self.assertRaises(ModelLoadError):
            self.load_solubility()

    def test_a_file_that_is_not_a_model_is_refused_even_with_a_matching_hash(self):
        path = self.model_file(SOL_CARD)
        path.write_bytes(b'{"not": "a model"}')
        self.edit_card(SOL_CARD, lambda card: card["artifacts"][0].update(sha256=sha256_of(path)))
        with self.assertRaises(ModelLoadError):
            self.load_solubility()

    def test_a_model_with_a_different_feature_count_than_its_card_is_refused(self):
        rng = np.random.default_rng(0)
        small = xgb.train({"objective": "reg:squarederror"},
                          xgb.DMatrix(rng.random((20, 5)), label=rng.random(20)), num_boost_round=2)
        path = self.model_file(SOL_CARD)
        small.save_model(str(path))
        self.edit_card(SOL_CARD, lambda card: card["artifacts"][0].update(sha256=sha256_of(path)))
        with self.assertRaises(ModelLoadError):
            self.load_solubility()

    def test_an_invalid_interval_offset_is_refused(self):
        for bad in (-1.0, float("nan"), "wide", None):
            with self.subTest(offset=bad):
                self.edit_card(SOL_CARD, lambda card, value=bad: card["uncertainty"].update(offset=value))
                with self.assertRaises(ModelLoadError):
                    self.load_solubility()

    def test_a_card_without_interval_information_is_refused(self):
        self.edit_card(SOL_CARD, lambda card: card.pop("uncertainty"))
        with self.assertRaises(ModelLoadError):
            self.load_solubility()

    def test_tox21_card_must_describe_exactly_the_12_assays(self):
        self.edit_card(TOX_CARD, lambda card: card["tasks"].pop("SR-p53"))
        with self.assertRaises(ModelLoadError):
            self.load_tox21()

    def test_tox21_unknown_calibration_method_or_numbers_are_refused(self):
        self.edit_card(TOX_CARD, lambda card: card["tasks"]["NR-AR"]["calibration"].update(method="isotonic"))
        with self.assertRaises(ModelLoadError):
            self.load_tox21()
        self.edit_card(TOX_CARD, lambda card: card["tasks"]["NR-AR"]["calibration"].update(method="platt_logit", a=None))
        with self.assertRaises(ModelLoadError):
            self.load_tox21()

    def test_tox21_card_with_no_trained_assay_is_refused(self):
        def change(card):
            for entry in card["tasks"].values():
                entry["status"] = "skipped: one class"

        self.edit_card(TOX_CARD, change)
        with self.assertRaises(ModelLoadError):
            self.load_tox21()

    def test_tox21_untrained_assay_is_reported_unavailable_not_hidden(self):
        self.edit_card(TOX_CARD, lambda card: card["tasks"]["SR-p53"].update(status="skipped: one class"))
        models = self.load_tox21()
        signals = models.predict(featuriser.featurise_mol(
            Chem.MolFromSmiles(ASPIRIN)))["signals"]
        self.assertEqual(len(signals), 12)
        self.assertIsNone(signals["SR-p53"]["probability"])
        self.assertEqual(signals["SR-p53"]["status"], "unavailable")
        self.assertEqual(signals["SR-p53"]["label"], labels.ASSAY_SIGNAL)
        self.assertIsNotNone(signals["NR-AR"]["probability"])


def features_of(smiles):
    return export_models.build_features([smiles])[0]


class SolubilityTests(CopyCase):  # FR-010, FR-012
    def test_fr010_logs_equals_the_model_file_prediction(self):
        model = self.load_solubility()
        booster = xgb.Booster()
        booster.load_model(str(self.model_file(SOL_CARD)))
        expected = float(booster.predict(xgb.DMatrix(features_of(ASPIRIN).reshape(1, -1)))[0])
        self.assertEqual(model.predict(features_of(ASPIRIN))["logs"], expected)

    def test_fr012_interval_is_the_prediction_plus_and_minus_the_card_offset(self):
        model = self.load_solubility()
        card = json.loads(self.card_path(SOL_CARD).read_text(encoding="utf-8"))
        offset = card["uncertainty"]["offset"]
        for smiles in (ASPIRIN, "CCO", GOLDEN["caffeine"]["smiles"]):
            with self.subTest(smiles=smiles):
                block = model.predict(features_of(smiles))
                interval = block["interval"]
                self.assertAlmostEqual(interval["high"] - block["logs"], offset, places=9)
                self.assertAlmostEqual(block["logs"] - interval["low"], offset, places=9)
                self.assertAlmostEqual(interval["level"], 0.9)
                self.assertEqual(interval["method"], card["uncertainty"]["method"])
                self.assertEqual(interval["measured_test_coverage"], card["uncertainty"]["test_coverage"])

    def test_nfr007_nfr008_block_carries_unit_limitation_and_model_version(self):
        block = self.load_solubility().predict(features_of(ASPIRIN))
        self.assertEqual(labels.missing_labels(block), [])
        self.assertEqual(block["unit"], "log10(mol/L)")
        self.assertEqual(block["model_version"], export_models.MODEL_VERSION)

    def test_the_card_unit_matches_the_wording_file(self):
        card = json.loads(self.card_path(SOL_CARD).read_text(encoding="utf-8"))
        self.assertEqual(card["unit"], labels.UNIT_LOGS)

    def test_numbers_are_plain_python_floats(self):
        block = self.load_solubility().predict(features_of(ASPIRIN))
        self.assertIs(type(block["logs"]), float)
        self.assertIs(type(block["interval"]["low"]), float)
        json.dumps(block)  # must not raise


class Tox21Tests(CopyCase):  # FR-011, BR-006
    def test_fr011_twelve_signals_in_a_fixed_order_matching_the_training_assays(self):
        self.assertEqual(tox21.ASSAYS, tuple(datasets.TOX21_TASKS))
        block = self.load_tox21().predict(features_of(ASPIRIN))
        self.assertEqual(tuple(block["signals"]), tox21.ASSAYS)

    def test_br006_every_value_is_labelled_assay_signal_and_is_a_probability(self):
        block = self.load_tox21().predict(features_of(ASPIRIN))
        for assay, signal in block["signals"].items():
            with self.subTest(assay=assay):
                self.assertEqual(signal["label"], "assay_signal")
                self.assertGreaterEqual(signal["probability"], 0.0)
                self.assertLessEqual(signal["probability"], 1.0)
        self.assertEqual(block["label"], "assay_signal")

    def test_fr011_values_are_the_platt_calibrated_model_probabilities(self):
        models = self.load_tox21()
        card = json.loads(self.card_path(TOX_CARD).read_text(encoding="utf-8"))
        features = features_of(GOLDEN["ibuprofen"]["smiles"])
        block = models.predict(features)
        for assay, entry in card["tasks"].items():
            with self.subTest(assay=assay):
                booster = xgb.Booster()
                booster.load_model(str(self.folder / entry["file"]))
                raw = float(booster.predict(xgb.DMatrix(features.reshape(1, -1)))[0])
                expected = float(training_apply_platt(
                    [raw], entry["calibration"]["a"], entry["calibration"]["b"])[0])
                self.assertAlmostEqual(block["signals"][assay]["probability"], expected, places=9)

    def test_serving_platt_maths_matches_the_training_code(self):
        for raw in (0.0, 1e-9, 0.01, 0.2, 0.5, 0.9, 0.999, 1.0):
            for a, b in ((1.0, 0.0), (0.57, -0.93), (0.99, 0.89), (2.0, -1.0)):
                with self.subTest(raw=raw, a=a, b=b):
                    self.assertAlmostEqual(
                        tox21.apply_platt(raw, a, b), float(training_apply_platt([raw], a, b)[0]), places=12)

    def test_the_identity_calibration_returns_the_raw_probability(self):
        self.assertAlmostEqual(tox21.apply_platt(0.3, 1.0, 0.0), 0.3, places=9)

    def test_nfr009_limitation_names_what_tox21_does_not_cover(self):
        block = self.load_tox21().predict(features_of(ASPIRIN))
        for word in ("hERG", "hepatotoxicity", "mutagenicity"):
            self.assertIn(word, block["limitation"])
        self.assertEqual(labels.missing_labels(block), [])


class PredictorTests(CopyCase):
    def test_every_golden_molecule_gets_a_complete_labelled_payload(self):
        model = self.load_predictor()
        for name, info in GOLDEN.items():
            with self.subTest(molecule=name):
                payload = model.predict_smiles(info["smiles"]).payload
                self.assertEqual(payload["tier"], "ml")
                self.assertEqual(labels.missing_labels(payload["solubility"]), [])
                self.assertEqual(labels.missing_labels(payload["tox21"]), [])
                self.assertEqual(len(payload["tox21"]["signals"]), 12)
                json.dumps(payload)

    def test_the_same_molecule_gives_the_same_result_twice(self):
        model = self.load_predictor()
        self.assertEqual(model.predict_smiles(ASPIRIN), model.predict_smiles(ASPIRIN))

    def test_model_version_names_both_families_and_fits_the_column(self):
        result = self.load_predictor().predict_smiles("CCO")
        self.assertEqual(result.model_version, "ml_solubility_xgb@1.0.0+ml_tox21_xgb@1.0.0")
        self.assertLessEqual(len(result.model_version), predictor.MAX_MODEL_VERSION_LEN)

    def test_params_hash_is_stable_and_64_hex_characters(self):
        first, second = self.load_predictor().params_hash, self.load_predictor().params_hash
        self.assertEqual(first, second)
        self.assertRegex(first, r"^[0-9a-f]{64}$")

    def test_params_hash_changes_when_anything_that_affects_the_output_changes(self):
        base = self.load_predictor().params_hash
        self.edit_card(SOL_CARD, lambda card: card["uncertainty"].update(offset=card["uncertainty"]["offset"] + 0.5))
        self.assertNotEqual(self.load_predictor().params_hash, base)

    def test_params_hash_changes_with_the_calibration_numbers(self):
        base = self.load_predictor().params_hash
        self.edit_card(TOX_CARD, lambda card: card["tasks"]["NR-AR"]["calibration"].update(a=0.123))
        self.assertNotEqual(self.load_predictor().params_hash, base)

    def test_params_hash_changes_with_the_model_files(self):
        base = self.load_predictor().params_hash
        path = self.model_file(SOL_CARD)
        other = self.root / "other_model.json"
        rng = np.random.default_rng(1)
        booster = xgb.train({"objective": "reg:squarederror"},
                            xgb.DMatrix(rng.random((20, featuriser.N_FEATURES)), label=rng.random(20)),
                            num_boost_round=2)
        booster.save_model(str(other))
        path.write_bytes(other.read_bytes())
        self.edit_card(SOL_CARD, lambda card: card["artifacts"][0].update(sha256=sha256_of(path)))
        self.assertNotEqual(self.load_predictor().params_hash, base)

    def test_sec007_an_unparsable_structure_fails_without_echoing_it(self):
        marker = "C(C)(C)(C)(C)CXxZz"
        with self.assertRaises(ValueError) as caught:
            self.load_predictor().predict_smiles(marker)
        self.assertNotIn(marker, str(caught.exception))

    def test_nfr011_no_wording_claims_a_molecule_is_safe_or_effective(self):
        payload = self.load_predictor().predict_smiles(ASPIRIN).payload
        words = ("safe", "non-toxic", "nontoxic", "effective", "approved", "harmless")
        texts = []

        def collect(value):
            if isinstance(value, dict):
                for item in value.values():
                    collect(item)
            elif isinstance(value, str):
                texts.append(value.lower())

        collect(payload)
        texts += [labels.DISCLAIMER.lower(), labels.LIMITATION_SOLUBILITY.lower(), labels.LIMITATION_TOX21.lower()]
        for text in texts:
            for word in words:
                self.assertNotIn(word, text)

    def test_p95_prediction_time_is_under_one_second(self):  # NFR-001 budget for the ML tier
        model = self.load_predictor()
        smiles = [info["smiles"] for info in GOLDEN.values()] * 4
        model.predict_smiles(smiles[0])  # first call builds the descriptor tables
        times = []
        for item in smiles:
            started = time.perf_counter()
            model.predict_smiles(item)
            times.append(time.perf_counter() - started)
        self.assertLess(float(np.percentile(times, 95)), 1.0)


class SharedPredictorTests(CopyCase):
    def get(self):
        return predictor.get_predictor(fx.models_dir(self.root), self.root / "cards")

    def test_the_predictor_is_loaded_once_and_reused(self):
        with mock.patch.object(predictor.MlPredictor, "load", wraps=predictor.MlPredictor.load) as load:
            self.assertIs(self.get(), self.get())
        self.assertEqual(load.call_count, 1)

    def test_a_card_with_another_version_is_not_picked_up_by_accident(self):
        with mock.patch.dict("os.environ", {model_cards.MODEL_VERSION_ENV: "9.9.9"}):
            with self.assertRaises(ModelLoadError):
                self.get()

    def test_a_replaced_model_file_is_noticed_and_verified_again(self):
        first = self.get()
        path = self.model_file(SOL_CARD)
        path.write_bytes(path.read_bytes() + b" ")  # changes size: the hash no longer matches
        with self.assertRaises(ModelLoadError):
            self.get()
        path.write_bytes(path.read_bytes()[:-1])    # restore the original bytes
        second = self.get()                          # a failed load was not cached
        self.assertIsNot(first, second)
        self.assertEqual(first.params_hash, second.params_hash)

    def test_a_deleted_model_file_is_noticed(self):
        self.get()
        self.model_file(TOX_CARD).unlink()
        with self.assertRaises(ModelLoadError):
            self.get()

    def test_reset_forgets_the_cached_predictor(self):
        first = self.get()
        predictor.reset_predictor()
        self.assertIsNot(first, self.get())


class FeatureCheckTests(unittest.TestCase):
    def test_the_card_signature_in_the_committed_cards_matches_this_code(self):
        # The cards on main were trained with this featuriser; serving would refuse them otherwise.
        for name in (SOL_CARD, TOX_CARD):
            with self.subTest(card=name):
                card = json.loads(model_cards.card_path(name).read_text(encoding="utf-8"))
                self.assertEqual(card["featuriser"]["signature"], featuriser.feature_signature())
                self.assertEqual(card["featuriser"]["n_features"], featuriser.N_FEATURES)

    def test_the_committed_card_numbers_are_valid_for_serving(self):
        card = json.loads(model_cards.card_path(SOL_CARD).read_text(encoding="utf-8"))
        self.assertTrue(math.isfinite(card["uncertainty"]["offset"]))
        tox = json.loads(model_cards.card_path(TOX_CARD).read_text(encoding="utf-8"))
        self.assertEqual(set(tox["tasks"]), set(tox21.ASSAYS))
        self.assertTrue(all(entry["status"] == "trained" for entry in tox["tasks"].values()))


if __name__ == "__main__":
    unittest.main()
