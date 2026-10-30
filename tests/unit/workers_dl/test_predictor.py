"""Tests for the DL predictor: loading, verification, prediction, labels (MSP-26).

Covers FR-015..FR-020, SEC-008, NFR-007, NFR-008, NFR-011. The models are the tiny real files
from dl_fixtures.py. Skipped when torch is not installed (requirements-dl.txt).
"""
# pylint: disable=missing-function-docstring,missing-class-docstring,protected-access,import-outside-toplevel
import json
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from rdkit import Chem  # pylint: disable=no-name-in-module

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

if torch is not None:
    from backend.core import labels
    from backend.registry.loader import ModelLoadError
    from backend.workers.dl import graph, predictor
    from backend.workers.ml import featuriser, model_cards
    from tests.unit.workers_dl import dl_fixtures as fx

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
TARGET_A, TARGET_B = (fx.TARGETS if torch is not None else (None, None))


def rewrite_card(root, name, change):
    path = fx.card_file(root, name)
    card = json.loads(path.read_text(encoding="utf-8"))
    change(card)
    path.write_text(json.dumps(card), encoding="utf-8")


@unittest.skipIf(torch is None, "torch is not installed")
class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)  # pylint: disable=consider-using-with
        cls.addClassCleanup(tmp.cleanup)
        cls.root = fx.copy_of_built(Path(tmp.name))
        cls.model = predictor.DlPredictor.load(TARGET_A, fx.models_dir(cls.root), cards_dir=cls.root / "cards")
        cls.result = cls.model.predict_smiles(ASPIRIN)

    def test_fr017_fr018_activity_has_mean_sd_and_five_member_values(self):
        activity = self.result.payload["activity"]
        self.assertEqual(self.result.payload["tier"], "dl")
        self.assertEqual(activity["n_models"], 5)
        self.assertEqual(len(activity["member_values"]), 5)
        self.assertAlmostEqual(activity["mean"], float(np.mean(activity["member_values"])), places=9)
        self.assertAlmostEqual(activity["sd"], float(np.std(activity["member_values"], ddof=1)), places=9)
        self.assertGreater(activity["sd"], 0.0)

    def test_fr017_unit_comes_from_the_target_and_sits_beside_every_value(self):
        payload = self.result.payload
        self.assertEqual(payload["activity"]["unit"], "pIC50")
        self.assertEqual(payload["baseline"]["unit"], "pIC50")

    def test_fr019_baseline_value_is_returned_and_matches_the_file(self):
        import xgboost as xgb
        entry = next(a for a in self.model.card["artifacts"] if a.get("role") == "baseline")
        booster = xgb.Booster()
        booster.load_model(str(fx.models_dir(self.root) / "dl" / entry["file"]))
        expected = float(booster.predict(xgb.DMatrix(featuriser.featurise_mol(Chem.MolFromSmiles(ASPIRIN)).reshape(1, -1)))[0])
        self.assertAlmostEqual(self.result.payload["baseline"]["value"], expected, places=5)

    def test_member_values_match_an_independent_reading_of_the_files(self):
        arrays = graph.batch_graphs([graph.smiles_to_graph(ASPIRIN)])
        tensors = [torch.from_numpy(a) for a in arrays]
        values = []
        for name in sorted(a["file"] for a in self.model.card["artifacts"] if a.get("role") == "gnn_member"):
            module = torch.jit.load(str(fx.models_dir(self.root) / "dl" / name))
            with torch.no_grad():
                values.append(float(module(*tensors)[0]))
        np.testing.assert_allclose(self.result.payload["activity"]["member_values"], values, atol=1e-5)

    def test_fr020_target_and_model_version_are_stored_with_the_prediction(self):
        self.assertEqual(self.result.payload["target"],
                         {"id": "TEST_TARGET", "name": "Synthetic target A", "chembl_id": "CHEMBL1"})
        self.assertEqual(self.result.model_version, "dl_test_target_gnn@1.0.0")
        self.assertEqual(self.result.payload["activity"]["model_version"], self.result.model_version)
        self.assertEqual(self.result.payload["baseline"]["model_version"], self.result.model_version)
        self.assertLessEqual(len(self.result.model_version), 50)

    def test_nfr007_nfr008_every_block_has_unit_limitation_and_model_version(self):
        for block in ("activity", "baseline"):
            with self.subTest(block=block):
                self.assertEqual(labels.missing_labels(self.result.payload[block]), [])

    def test_nfr011_no_wording_says_safe_or_effective(self):
        text = json.dumps(self.result.payload).lower()
        for word in ("safe", "non-toxic", "effective", "approved", "will work", "proven"):
            self.assertNotRegex(text, r"\b" + re.escape(word) + r"\b")

    def test_payload_is_json_safe(self):
        self.assertEqual(json.loads(json.dumps(self.result.payload)), self.result.payload)

    def test_params_hash_is_stable_and_looks_like_sha256(self):
        again = predictor.DlPredictor.load(TARGET_A, fx.models_dir(self.root), cards_dir=self.root / "cards")
        self.assertEqual(again.params_hash, self.result.params_hash)
        self.assertRegex(self.result.params_hash, r"^[0-9a-f]{64}$")

    def test_params_hash_differs_between_targets(self):
        other = predictor.DlPredictor.load(TARGET_B, fx.models_dir(self.root), cards_dir=self.root / "cards")
        self.assertNotEqual(other.params_hash, self.result.params_hash)
        self.assertEqual(other.predict_smiles(ASPIRIN).payload["activity"]["unit"], "pKi")

    def test_params_hash_changes_with_the_sd_setting(self):
        from backend.workers.dl import ensemble
        other = predictor.DlPredictor(TARGET_A, self.model.card, ensemble.TargetEnsemble(self.model.ensemble.members, 0),
                                      self.model.baseline, self.model.files)
        self.assertNotEqual(other.params_hash, self.model.params_hash)

    def test_the_same_molecule_gives_the_same_answer_every_time(self):
        self.assertEqual(self.model.predict_smiles(ASPIRIN).payload, self.result.payload)

    def test_atom_order_in_the_stored_smiles_does_not_change_the_answer(self):
        other = self.model.predict_smiles("OC(=O)c1ccccc1OC(C)=O").payload["activity"]["mean"]
        self.assertAlmostEqual(other, self.result.payload["activity"]["mean"], places=4)

    def test_unparsable_or_unsupported_structures_raise_value_error(self):
        for smiles in ("not a molecule", "C[Si](C)(C)C", "[Na+]"):
            with self.subTest(smiles=smiles), self.assertRaises(ValueError):
                self.model.predict_smiles(smiles)

    def test_a_non_finite_member_value_is_refused(self):
        class Broken:
            def __call__(self, *_args):
                return torch.tensor([float("nan")])
        saved = self.model.ensemble.members
        self.model.ensemble.members = [Broken(), Broken()]
        self.addCleanup(setattr, self.model.ensemble, "members", saved)
        with self.assertRaises(ValueError):
            self.model.predict_smiles(ASPIRIN)

    def test_slug_matches_the_training_code(self):
        from training import train_gnn
        for value in ("EGFR", "HIV1_PROTEASE", "BACE1", "odd id (x)"):
            self.assertEqual(predictor.slug_of(value), train_gnn.slug_of(value))
        self.assertEqual(predictor.card_name("HIV1_PROTEASE"), "dl_hiv1_protease_gnn")


@unittest.skipIf(torch is None, "torch is not installed")
class RefusalTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)  # pylint: disable=consider-using-with
        self.addCleanup(tmp.cleanup)
        self.root = fx.copy_of_built(Path(tmp.name))
        self.dl = fx.models_dir(self.root) / "dl"

    def load(self, target=None):
        return predictor.DlPredictor.load(target or TARGET_A, fx.models_dir(self.root), cards_dir=self.root / "cards")

    def card(self):
        return json.loads(fx.card_file(self.root, "dl_test_target_gnn").read_text(encoding="utf-8"))

    def test_the_untouched_copy_loads(self):
        self.assertEqual(self.load().model_version, "dl_test_target_gnn@1.0.0")

    def test_sec008_a_changed_member_file_is_refused(self):
        name = next(a["file"] for a in self.card()["artifacts"] if a.get("role") == "gnn_member")
        with open(self.dl / name, "ab") as handle:
            handle.write(b"x")
        with self.assertRaises(ModelLoadError):
            self.load()

    def test_sec008_a_changed_baseline_file_is_refused(self):
        name = next(a["file"] for a in self.card()["artifacts"] if a.get("role") == "baseline")
        with open(self.dl / name, "ab") as handle:
            handle.write(b" ")
        with self.assertRaises(ModelLoadError):
            self.load()

    def test_a_missing_model_file_is_refused(self):
        name = next(a["file"] for a in self.card()["artifacts"] if a.get("role") == "gnn_member")
        (self.dl / name).unlink()
        with self.assertRaises(ModelLoadError):
            self.load()

    def test_a_missing_card_is_refused(self):
        fx.card_file(self.root, "dl_test_target_gnn").unlink()
        with self.assertRaises(ModelLoadError):
            self.load()

    def test_a_file_that_is_not_torchscript_is_refused_even_with_a_matching_hash(self):
        import hashlib
        card = self.card()
        entry = next(a for a in card["artifacts"] if a.get("role") == "gnn_member")
        path = self.dl / entry["file"]
        path.write_bytes(b"this is not a torchscript archive")
        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        fx.card_file(self.root, "dl_test_target_gnn").write_text(json.dumps(card), encoding="utf-8")
        with self.assertRaises(ModelLoadError):
            self.load()

    def test_sec008_a_pickle_suffix_is_refused(self):
        def change(card):
            entry = next(a for a in card["artifacts"] if a.get("role") == "gnn_member")
            (self.dl / "evil.pkl").write_bytes((self.dl / entry["file"]).read_bytes())
            entry["file"] = "evil.pkl"
        rewrite_card(self.root, "dl_test_target_gnn", change)
        with self.assertRaises(ModelLoadError):
            self.load()

    def test_a_file_name_that_leaves_the_models_folder_is_refused(self):
        def change(card):
            entry = next(a for a in card["artifacts"] if a.get("role") == "gnn_member")
            entry["file"] = "../escape.ts"
        rewrite_card(self.root, "dl_test_target_gnn", change)
        with self.assertRaises(ModelLoadError):
            self.load()

    def test_a_card_for_another_target_is_refused(self):
        rewrite_card(self.root, "dl_test_target_gnn", lambda card: card["target"].update(id="SOMETHING_ELSE"))
        with self.assertRaises(ModelLoadError):
            self.load()
        rewrite_card(self.root, "dl_test_target_gnn", lambda card: card["target"].update(id="TEST_TARGET", chembl_id="CHEMBL999"))
        with self.assertRaises(ModelLoadError):
            self.load()

    def test_a_unit_that_differs_from_the_configuration_is_refused(self):
        with self.assertRaises(ModelLoadError):
            self.load({**TARGET_A, "unit": "pKi"})

    def test_a_different_graph_definition_is_refused(self):
        with patch.object(graph, "ATOM_SYMBOLS", graph.ATOM_SYMBOLS + ("Se",)):
            with self.assertRaises(ModelLoadError):
                self.load()

    def test_a_card_without_graph_information_is_refused(self):
        rewrite_card(self.root, "dl_test_target_gnn", lambda card: card.pop("graph"))
        with self.assertRaises(ModelLoadError):
            self.load()

    def test_different_ml_features_are_refused(self):
        with patch.object(featuriser, "feature_signature", return_value="0" * 16):
            with self.assertRaises(ModelLoadError):
                self.load()

    def test_an_ensemble_size_that_does_not_match_the_files_is_refused(self):
        rewrite_card(self.root, "dl_test_target_gnn", lambda card: card["model"]["ensemble"].update(size=4))
        with self.assertRaises(ModelLoadError):
            self.load()

    def test_a_single_model_is_not_an_ensemble_and_is_refused(self):
        def change(card):
            members = [a for a in card["artifacts"] if a.get("role") == "gnn_member"]
            card["artifacts"] = [a for a in card["artifacts"] if a not in members[1:]]
            card["model"]["ensemble"]["size"] = 1
        rewrite_card(self.root, "dl_test_target_gnn", change)
        with self.assertRaises(ModelLoadError):
            self.load()

    def test_a_member_without_a_number_is_refused(self):
        def change(card):
            next(a for a in card["artifacts"] if a.get("role") == "gnn_member").pop("member")
        rewrite_card(self.root, "dl_test_target_gnn", change)
        with self.assertRaises(ModelLoadError):
            self.load()

    def test_the_same_member_file_listed_twice_is_refused(self):
        def change(card):
            members = [a for a in card["artifacts"] if a.get("role") == "gnn_member"]
            members[1]["file"] = members[0]["file"]
            members[1]["sha256"] = members[0]["sha256"]
        rewrite_card(self.root, "dl_test_target_gnn", change)
        with self.assertRaises(ModelLoadError):
            self.load()

    def test_an_invalid_sd_setting_is_refused(self):
        rewrite_card(self.root, "dl_test_target_gnn", lambda card: card["model"]["ensemble"].update(sd_ddof=2))
        with self.assertRaises(ModelLoadError):
            self.load()

    def test_a_card_with_no_or_two_baselines_is_refused(self):
        def two(card):
            card["artifacts"].append({**next(a for a in card["artifacts"] if a.get("role") == "baseline")})
        rewrite_card(self.root, "dl_test_target_gnn", two)
        with self.assertRaises(ModelLoadError):
            self.load()

    def test_a_model_version_too_long_to_store_is_refused(self):
        rewrite_card(self.root, "dl_test_target_gnn", lambda card: card.update(name="x" * 60))
        with self.assertRaises(ModelLoadError):
            self.load()

    def test_a_card_of_the_wrong_tier_is_refused(self):
        rewrite_card(self.root, "dl_test_target_gnn", lambda card: card.update(tier="ml"))
        with self.assertRaises(ModelLoadError):
            self.load()

    def test_the_version_can_be_chosen_with_an_environment_variable(self):
        card = self.card()
        card["version"] = "2.0.0"
        fx.card_file(self.root, "dl_test_target_gnn", "2.0.0").write_text(json.dumps(card), encoding="utf-8")
        with patch.dict(os.environ, {"DL_MODEL_VERSION": "2.0.0"}):
            self.assertEqual(predictor.model_version(), "2.0.0")
            self.assertEqual(self.load().model_version, "dl_test_target_gnn@2.0.0")
        with patch.dict(os.environ, {"DL_MODEL_VERSION": "9.9.9"}):
            with self.assertRaises(ModelLoadError):
                self.load()      # no card of that version
        self.assertEqual(predictor.model_version(), "1.0.0")


@unittest.skipIf(torch is None, "torch is not installed")
class TargetLookupAndCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)  # pylint: disable=consider-using-with
        cls.addClassCleanup(tmp.cleanup)
        cls.root = fx.copy_of_built(Path(tmp.name))

    def setUp(self):
        predictor.reset_predictor()
        self.addCleanup(predictor.reset_predictor)

    def get(self, target_id):
        return predictor.get_predictor(target_id, fx.models_dir(self.root), self.root / "cards", fx.TARGETS)

    def test_find_target_returns_the_configured_entry(self):
        self.assertEqual(predictor.find_target("OTHER_TARGET", fx.TARGETS)["unit"], "pKi")

    def test_a_job_without_a_target_gets_a_clear_message(self):
        for target_id in (None, ""):
            with self.assertRaisesRegex(ModelLoadError, "no target"):
                predictor.find_target(target_id, fx.TARGETS)

    def test_no_target_unknown_target_or_incomplete_target_is_refused(self):
        for target_id, targets in ((None, fx.TARGETS), ("", fx.TARGETS), ("NOPE", fx.TARGETS),
                                   ("X", [{"id": "X", "name": "x", "chembl_id": "CHEMBL1"}])):
            with self.subTest(target=target_id), self.assertRaises(ModelLoadError):
                predictor.find_target(target_id, targets)

    def test_find_target_reads_the_config_file_when_no_list_is_given(self):
        with patch.object(predictor, "load_targets", return_value={"targets": fx.TARGETS}):
            self.assertEqual(predictor.find_target("TEST_TARGET")["chembl_id"], "CHEMBL1")
        with patch.object(predictor, "load_targets", return_value=None):
            with self.assertRaises(ModelLoadError):
                predictor.find_target("TEST_TARGET")

    def test_a_predictor_is_loaded_once_per_target(self):
        first = self.get("TEST_TARGET")
        self.assertIs(self.get("TEST_TARGET"), first)
        self.assertIsNot(self.get("OTHER_TARGET"), first)

    def test_a_changed_file_forces_a_fresh_verified_load(self):
        first = self.get("TEST_TARGET")
        name = next(iter(first.files.paths))
        path = first.files.paths[name]
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
        self.assertFalse(first.still_current())
        self.assertIsNot(self.get("TEST_TARGET"), first)

    def test_a_failed_load_is_not_cached_so_fixing_the_file_is_enough(self):
        card = fx.card_file(self.root, "dl_other_target_gnn")
        saved = card.read_bytes()
        card.write_bytes(b"{ broken")
        try:
            with self.assertRaises(ModelLoadError):
                self.get("OTHER_TARGET")
        finally:
            card.write_bytes(saved)
        self.assertEqual(self.get("OTHER_TARGET").target["id"], "OTHER_TARGET")

    def test_a_file_that_vanishes_counts_as_changed(self):
        first = self.get("TEST_TARGET")
        with patch.object(model_cards, "take_snapshot", side_effect=OSError("gone")):
            self.assertFalse(first.still_current())


if __name__ == "__main__":
    unittest.main()
