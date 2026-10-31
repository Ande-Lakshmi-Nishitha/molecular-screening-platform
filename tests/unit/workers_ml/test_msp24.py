"""Unit tests for structural alerts, property rules and the applicability domain (MSP-24).

Covers FR-013, FR-014, NFR-007, NFR-008, NFR-010, NFR-011, SEC-008, TBD-C8. The models and the
reference data are the tiny real files from ml_fixtures.py (built with the project's export code).

Run from the repo root with:  python -m unittest discover -s tests -t .
"""
# pylint: disable=missing-function-docstring,missing-class-docstring,protected-access
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from rdkit import Chem

from backend.core import labels
from backend.registry.loader import ModelLoadError
from backend.workers.ml import alerts, domain, featuriser, model_cards, predictor
from tests.unit.training.synthetic import make_frame
from tests.unit.workers_ml import ml_fixtures as fx

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
RHODANINE = "S=C1SC(=Cc2ccccc2)C(=O)N1"            # PAINS ene_rhod_A, Brenk Michael acceptor + thiocarbonyl
NITROBENZENE = "c1ccccc1[N+](=O)[O-]"              # Brenk nitro group
LONG_CHAIN = "CCCCCCCCCCCCCCCCCCCCCCCC(=O)NC(CCCCCCCCCCCC)C(=O)O"
FORBIDDEN = ("safe", "non-toxic", "nontoxic", "harmless", "effective", "approved", "cure")


PICKLE_RAN = []


def _mark(value):
    PICKLE_RAN.append(value)
    return value


class _Evil:
    """When unpickled, it calls _mark: proof that a loader ran pickled code."""

    def __reduce__(self):
        return (_mark, ("ran",))


def mol_of(smiles):
    return Chem.MolFromSmiles(smiles)


class AlertTests(unittest.TestCase):
    def test_fr013_pains_and_brenk_matches_are_listed_by_name(self):
        result = alerts.structural_alerts(mol_of(RHODANINE))
        self.assertEqual(result["pains"], ["ene_rhod_A(235)"])
        self.assertEqual(result["brenk"], ["Michael_acceptor_1", "Thiocarbonyl_group"])
        self.assertEqual((result["n_pains"], result["n_brenk"]), (1, 2))

    def test_brenk_only_and_clean_molecules(self):
        nitro = alerts.structural_alerts(mol_of(NITROBENZENE))
        self.assertEqual((nitro["n_pains"], nitro["brenk"]), (0, ["Oxygen-nitrogen_single_bond", "nitro_group"]))
        clean = alerts.structural_alerts(mol_of("CCO"))
        self.assertEqual((clean["pains"], clean["brenk"], clean["n_pains"], clean["n_brenk"]), ([], [], 0, 0))

    def test_the_lists_are_sorted_and_repeatable(self):
        first = alerts.structural_alerts(mol_of(RHODANINE))
        self.assertEqual(first, alerts.structural_alerts(mol_of(RHODANINE)))
        for key in ("pains", "brenk"):
            self.assertEqual(first[key], sorted(first[key]))

    def test_nfr007_nfr008_the_blocks_carry_unit_limitation_and_model_version(self):
        mol = mol_of(ASPIRIN)
        for block in (alerts.structural_alerts(mol), alerts.property_rules(mol)):
            self.assertEqual(labels.missing_labels(block), [])
            self.assertTrue(block["model_version"].startswith("rdkit@"))

    def test_the_identity_names_the_rdkit_version_and_the_limits(self):
        identity = alerts.identity()
        self.assertEqual(identity["lipinski"]["mol_wt"], 500.0)
        self.assertEqual(identity["veber"], {"rotatable_bonds": 10, "tpsa": 140.0})
        self.assertEqual(identity["lipinski"], {"mol_wt": 500.0, "log_p": 5.0, "h_bond_donors": 5, "h_bond_acceptors": 10})
        self.assertTrue(identity["rdkit"])


class RuleTests(unittest.TestCase):
    def test_fr013_a_small_molecule_has_no_violations(self):
        rules = alerts.property_rules(mol_of(ASPIRIN))
        self.assertEqual((rules["lipinski"]["violations"], rules["veber"]["violations"]), (0, 0))
        self.assertEqual(rules["lipinski"]["values"]["h_bond_donors"], 1)
        self.assertEqual(rules["veber"]["values"]["rotatable_bonds"], 2)

    def test_a_large_flexible_molecule_breaks_named_rules(self):
        rules = alerts.property_rules(mol_of(LONG_CHAIN))
        self.assertEqual(rules["lipinski"]["failed"], ["mol_wt", "log_p"])
        self.assertEqual(rules["veber"]["failed"], ["rotatable_bonds"])
        self.assertEqual((rules["lipinski"]["violations"], rules["veber"]["violations"]), (2, 1))

    def test_a_value_exactly_at_the_limit_is_not_a_violation(self):
        block = alerts._rule_block({"mol_wt": 500.0, "h_bond_donors": 5}, {"mol_wt": 500.0, "h_bond_donors": 5})
        self.assertEqual((block["failed"], block["violations"]), ([], 0))
        over = alerts._rule_block({"mol_wt": 500.1}, {"mol_wt": 500.0})
        self.assertEqual(over["failed"], ["mol_wt"])

    def test_nfr011_the_new_wording_makes_no_claim_about_safety_or_effect(self):
        for text in (labels.LIMITATION_ALERTS, labels.LIMITATION_RULES, labels.LIMITATION_DOMAIN,
                     labels.WARNING_OUT_OF_DOMAIN, labels.WARNING_DOMAIN_UNAVAILABLE):
            for word in FORBIDDEN:
                self.assertNotIn(word, text.lower())


class ReferenceCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = fx.built_root()
        cls.folder = model_cards.models_folder(fx.models_dir(cls.root))
        cls.card = model_cards.read_card(fx.card_file(cls.root, "ml_solubility_xgb"), "solubility_logs")
        cls.reference = domain.DomainReference.load(cls.card, cls.folder)
        cls.training_smiles = make_frame()["smiles_std"].tolist()

    @staticmethod
    def features(smiles):
        return featuriser.featurise_mol(mol_of(smiles))


class DomainTests(ReferenceCase):
    def test_fr014_a_training_molecule_is_inside_with_similarity_one(self):
        smiles = self.training_smiles[0]
        result = self.reference.check(self.features(smiles), 0.3)
        self.assertTrue(result["available"] and result["in_domain"])
        self.assertAlmostEqual(result["max_tanimoto"], 1.0)
        self.assertEqual(result["descriptors_outside_range"], [])

    def test_nfr010_low_similarity_puts_a_molecule_outside(self):
        result = self.reference.check(self.features(RHODANINE), 0.3)
        self.assertLess(result["max_tanimoto"], 0.3)
        self.assertFalse(result["in_domain"])

    def test_the_threshold_decides_and_is_reported(self):
        features = self.features(self.training_smiles[0])
        self.assertTrue(self.reference.check(features, 1.0)["in_domain"])
        strict = self.reference.check(features, 1.01)
        self.assertFalse(strict["in_domain"])
        self.assertEqual(strict["threshold"], 1.01)

    def test_a_property_outside_the_training_range_puts_a_molecule_outside_even_if_similar(self):
        features = self.features(self.training_smiles[0]).copy()
        index = featuriser.DESCRIPTOR_NAMES.index("MolWt")
        features[index] = self.reference.ranges["MolWt"]["max"] + 1.0
        result = self.reference.check(features, 0.3)
        self.assertEqual(result["descriptors_outside_range"], ["MolWt"])
        self.assertFalse(result["in_domain"])
        features[index] = self.reference.ranges["MolWt"]["min"] - 1.0
        self.assertEqual(self.reference.check(features, 0.3)["descriptors_outside_range"], ["MolWt"])

    def test_values_at_the_range_edges_are_inside_and_a_failed_descriptor_is_not_judged(self):
        features = self.features(self.training_smiles[0]).copy()
        index = featuriser.DESCRIPTOR_NAMES.index("TPSA")
        features[index] = self.reference.ranges["TPSA"]["max"]
        self.assertEqual(self.reference.descriptors_outside_range(features), [])
        features[index] = np.nan
        self.assertEqual(self.reference.descriptors_outside_range(features), [])

    def test_a_molecule_with_no_fingerprint_bits_has_similarity_zero(self):
        features = np.zeros(featuriser.N_FEATURES)
        self.assertEqual(self.reference.max_tanimoto(features), 0.0)

    def test_two_empty_fingerprints_give_zero_not_a_division_error(self):
        empty = np.zeros((1, domain.PACKED_COLUMNS), dtype=np.uint8)
        reference = domain.DomainReference(empty, np.zeros(1, dtype=np.int32), self.reference.ranges,
                                           self.reference.files, {})
        with np.errstate(all="raise"):
            self.assertEqual(reference.max_tanimoto(np.zeros(featuriser.N_FEATURES)), 0.0)

    def test_the_overall_flag(self):
        yes, no, unknown = {"in_domain": True}, {"in_domain": False}, {"in_domain": None}
        self.assertIs(domain.overall({"a": yes, "b": yes}), True)
        self.assertIs(domain.overall({"a": yes, "b": no}), False)
        self.assertIs(domain.overall({"a": unknown, "b": no}), False)
        self.assertIsNone(domain.overall({"a": yes, "b": unknown}))

    def test_the_unavailable_result_never_claims_inside(self):
        result = domain.unavailable(0.3)
        self.assertEqual((result["available"], result["in_domain"], result["max_tanimoto"]), (False, None, None))


class ReferenceFileChecks(ReferenceCase):
    def test_sec008_a_changed_fingerprint_file_is_refused(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = fx.copy_of_built(Path(tmp))
            folder = model_cards.models_folder(fx.models_dir(root))
            target = next(folder.glob("ml_solubility_train_fp_*.npy"))
            np.save(target, np.zeros((3, domain.PACKED_COLUMNS), dtype=np.uint8), allow_pickle=False)
            with self.assertRaises(ModelLoadError):
                domain.DomainReference.load(self.card, folder)

    def test_a_missing_reference_file_is_refused(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = fx.copy_of_built(Path(tmp))
            folder = model_cards.models_folder(fx.models_dir(root))
            next(folder.glob("ml_solubility_descriptor_ranges_*.json")).unlink()
            with self.assertRaises(ModelLoadError):
                domain.DomainReference.load(self.card, folder)

    def test_a_card_without_the_two_reference_entries_is_refused(self):
        stripped = {**self.card, "artifacts": [a for a in self.card["artifacts"] if a.get("kind") != "reference_data"]}
        with self.assertRaises(ModelLoadError):
            domain.DomainReference.load(stripped, self.folder)
        doubled = {**self.card, "artifacts": self.card["artifacts"] + [
            a for a in self.card["artifacts"] if a.get("kind") == "reference_data"][:1]}
        with self.assertRaises(ModelLoadError):
            domain.DomainReference.load(doubled, self.folder)

    def test_fingerprint_files_of_the_wrong_kind_or_shape_are_refused(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "fp.npy"
            cases = {
                "wrong dtype": np.zeros((2, domain.PACKED_COLUMNS), dtype=np.float32),
                "wrong columns": np.zeros((2, 5), dtype=np.uint8),
                "one dimension": np.zeros(domain.PACKED_COLUMNS, dtype=np.uint8),
                "no rows": np.zeros((0, domain.PACKED_COLUMNS), dtype=np.uint8),
            }
            for label, array in cases.items():
                with self.subTest(label):
                    np.save(path, array, allow_pickle=False)
                    with self.assertRaises(ModelLoadError):
                        domain._read_fingerprints(path)
            np.save(path, np.array([_Evil()], dtype=object), allow_pickle=True)   # a pickled array
            del PICKLE_RAN[:]
            with self.assertRaises(ModelLoadError):
                domain._read_fingerprints(path)
            self.assertEqual(PICKLE_RAN, [])   # the pickled code was never run
            with self.assertRaises(ModelLoadError):
                domain._read_fingerprints(Path(tmp) / "missing.npy")

    def test_range_files_that_are_unreadable_or_invalid_are_refused(self):
        good = {name: {"min": 0.0, "p01": 0.0, "p99": 1.0, "max": 2.0} for name in domain.KEY_DESCRIPTORS}
        bad_cases = {
            "not json": "{",
            "not an object": json.dumps([1]),
            "missing key property": json.dumps({k: v for k, v in good.items() if k != "TPSA"}),
            "entry not an object": json.dumps({**good, "TPSA": 3}),
            "min above max": json.dumps({**good, "TPSA": {"min": 5.0, "max": 1.0}}),
            "not a number": json.dumps({**good, "TPSA": {"min": "a", "max": 1.0}}),
            "infinite": '{"MolWt": {"min": 0, "max": Infinity}}',
        }
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "ranges.json"
            for label, text in bad_cases.items():
                with self.subTest(label):
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaises(ModelLoadError):
                        domain._read_ranges(path)
            path.write_text(json.dumps(good), encoding="utf-8")
            self.assertEqual(set(domain._read_ranges(path)), set(good))
            with self.assertRaises(ModelLoadError):
                domain._read_ranges(Path(tmp) / "missing.json")


class PayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = fx.built_root()
        cls.model = predictor.MlPredictor.load(fx.models_dir(cls.root), cards_dir=cls.root / "cards")
        cls.inside = make_frame()["smiles_std"].tolist()[0]

    def test_the_payload_gains_alerts_rules_and_domain_and_keeps_its_old_fields(self):
        payload = self.model.predict_smiles(self.inside).payload
        self.assertEqual(set(payload), {"tier", "solubility", "tox21", "alerts", "rules", "domain"})
        self.assertEqual(set(payload["domain"]),
                         {"in_domain", "warning", "families", "unit", "limitation", "model_version"})
        self.assertEqual(set(payload["domain"]["families"]), {"solubility", "tox21"})
        self.assertEqual(set(payload["domain"]["families"]["tox21"]),
                         {"available", "in_domain", "max_tanimoto", "threshold", "descriptors_outside_range"})
        self.assertEqual(json.loads(json.dumps(payload)), payload)

    def test_nfr007_nfr008_every_new_block_is_labelled(self):
        payload = self.model.predict_smiles(self.inside).payload
        for key in ("alerts", "rules", "domain"):
            with self.subTest(block=key):
                self.assertEqual(labels.missing_labels(payload[key]), [])

    def test_nfr010_a_molecule_unlike_the_training_data_carries_the_warning(self):
        inside = self.model.predict_smiles(self.inside).payload["domain"]
        self.assertEqual((inside["in_domain"], inside["warning"]), (True, None))
        outside = self.model.predict_smiles(RHODANINE).payload["domain"]
        self.assertEqual((outside["in_domain"], outside["warning"]), (False, labels.WARNING_OUT_OF_DOMAIN))

    def test_the_threshold_default_comes_from_the_settings_and_changes_the_params_hash(self):
        self.assertEqual(self.model.ad_threshold, 0.3)
        other = predictor.MlPredictor(self.model.solubility, self.model.tox21, self.model.references, 0.5)
        self.assertNotEqual(other.params_hash, self.model.params_hash)
        self.assertEqual(other.predict_smiles(self.inside).payload["domain"]["families"]["tox21"]["threshold"], 0.5)

    def test_unusable_reference_data_turns_the_check_off_but_not_the_models(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = fx.copy_of_built(Path(tmp))
            folder = model_cards.models_folder(fx.models_dir(root))
            next(folder.glob("ml_tox21_train_fp_*.npy")).write_bytes(b"damaged")
            with self.assertLogs("msp.worker.ml", level="ERROR") as logs:
                model = predictor.MlPredictor.load(fx.models_dir(root), cards_dir=root / "cards")
            self.assertIn("ModelLoadError", "\n".join(logs.output))
            payload = model.predict_smiles(self.inside).payload
            self.assertIn("logs", payload["solubility"])
            families = payload["domain"]["families"]
            self.assertEqual((families["solubility"]["available"], families["tox21"]["available"]), (True, False))
            self.assertIsNone(payload["domain"]["in_domain"])
            self.assertEqual(payload["domain"]["warning"], labels.WARNING_DOMAIN_UNAVAILABLE)
            self.assertNotEqual(model.params_hash, self.model.params_hash)

    def test_one_family_outside_makes_the_overall_flag_false(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = fx.copy_of_built(Path(tmp))
            folder = model_cards.models_folder(fx.models_dir(root))
            next(folder.glob("ml_tox21_train_fp_*.npy")).write_bytes(b"damaged")
            model = predictor.MlPredictor.load(fx.models_dir(root), cards_dir=root / "cards")
            domain_block = model.predict_smiles(RHODANINE).payload["domain"]
            self.assertIs(domain_block["in_domain"], False)   # solubility says outside, tox21 unknown

    def test_a_changed_reference_file_makes_the_predictor_stale(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = fx.copy_of_built(Path(tmp))
            model = predictor.MlPredictor.load(fx.models_dir(root), cards_dir=root / "cards")
            self.assertTrue(model.still_current())
            folder = model_cards.models_folder(fx.models_dir(root))
            next(folder.glob("ml_solubility_train_fp_*.npy")).write_bytes(b"changed")
            self.assertFalse(model.still_current())

    def test_a_predictor_built_without_references_reports_unavailable(self):
        bare = predictor.MlPredictor(self.model.solubility, self.model.tox21, ad_threshold=0.4)
        families = bare.predict_smiles(self.inside).payload["domain"]["families"]
        self.assertEqual([families[name]["available"] for name in ("solubility", "tox21")], [False, False])
        self.assertEqual([families[name]["threshold"] for name in ("solubility", "tox21")], [0.4, 0.4])

    def test_the_params_hash_covers_the_alert_and_rule_settings(self):
        with patch.object(alerts, "identity", return_value={"rdkit": "other"}):
            other = predictor.MlPredictor(self.model.solubility, self.model.tox21, self.model.references)
        self.assertNotEqual(other.params_hash, self.model.params_hash)


if __name__ == "__main__":
    unittest.main()
