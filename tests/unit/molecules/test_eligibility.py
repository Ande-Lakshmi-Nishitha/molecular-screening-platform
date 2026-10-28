"""Unit tests for tier eligibility (MSP-19; FR-021, BR-002, BR-003).

Run from the repo root with:  python -m unittest discover -s tests -t .
(with PYTHONPATH set to the backend folder)
"""
# RDKit is a C extension, so pylint cannot see its members.
# pylint: disable=no-member,c-extension-no-member,missing-function-docstring,missing-class-docstring
import unittest

from rdkit import Chem

from services.molecule.eligibility import (
    SKIP_NO_FRAGMENT,
    SKIP_NO_TARGET,
    SKIP_UNSUPPORTED_ATOM,
    EligibilityConfig,
    TierDecision,
    tier_eligibility,
)
from services.molecule.standardise import standardise_smiles

TARGET = "CHEMBL203"


def mol_of(smiles):
    """Standardised RDKit molecule, the same way the routes will build it."""
    return standardise_smiles(smiles).mol


class MlTierTests(unittest.TestCase):
    def test_br002_ml_runs_for_every_valid_molecule(self):
        for smiles in ("CCO", "CC(=O)Oc1ccccc1C(=O)O", "C[Si](C)(C)C"):
            with self.subTest(smiles=smiles):
                decision = tier_eligibility(mol_of(smiles))["ml"]
                self.assertTrue(decision.eligible)
                self.assertIsNone(decision.skip_reason)

    def test_br002_all_three_tiers_are_always_reported(self):
        result = tier_eligibility(mol_of("CCO"))
        self.assertEqual(set(result), {"ml", "dl", "qml"})

    def test_none_molecule_is_a_programming_error(self):
        with self.assertRaises(ValueError):
            tier_eligibility(None)


class DlTierTests(unittest.TestCase):
    def test_fr015_dl_without_target_is_skipped(self):
        for target in (None, ""):
            with self.subTest(target=target):
                decision = tier_eligibility(mol_of("CCO"), target)["dl"]
                self.assertFalse(decision.eligible)
                self.assertEqual(decision.skip_reason, SKIP_NO_TARGET)

    def test_fr015_dl_with_target_and_supported_atoms_runs(self):
        decision = tier_eligibility(mol_of("CC(=O)Oc1ccccc1C(=O)O"), TARGET)["dl"]
        self.assertTrue(decision.eligible)

    def test_fr015_dl_unsupported_atom_is_skipped(self):
        decision = tier_eligibility(mol_of("C[Si](C)(C)C"), TARGET)["dl"]
        self.assertFalse(decision.eligible)
        self.assertEqual(decision.skip_reason, SKIP_UNSUPPORTED_ATOM)

    def test_fr015_missing_target_is_reported_before_unsupported_atom(self):
        decision = tier_eligibility(mol_of("C[Si](C)(C)C"))["dl"]
        self.assertEqual(decision.skip_reason, SKIP_NO_TARGET)

    def test_br003_dl_skip_does_not_change_ml_or_qml(self):
        result = tier_eligibility(mol_of("CCO"))
        self.assertFalse(result["dl"].eligible)
        self.assertTrue(result["ml"].eligible)
        self.assertTrue(result["qml"].eligible)


class QmlTierTests(unittest.TestCase):
    def test_fr021_small_cnof_molecule_runs(self):
        for smiles in ("CCO", "CC(=O)N", "C[N+](C)(C)C", "CF"):
            with self.subTest(smiles=smiles):
                self.assertTrue(tier_eligibility(mol_of(smiles))["qml"].eligible)

    def test_fr021_nine_heavy_atoms_is_the_limit(self):
        self.assertTrue(tier_eligibility(mol_of("C" * 9))["qml"].eligible)
        decision = tier_eligibility(mol_of("C" * 10))["qml"]
        self.assertFalse(decision.eligible)
        self.assertEqual(decision.skip_reason, SKIP_NO_FRAGMENT)

    def test_fr021_large_molecule_is_skipped_without_fragment_path(self):
        decision = tier_eligibility(mol_of("CC(=O)Oc1ccccc1C(=O)O"))["qml"]
        self.assertFalse(decision.eligible)
        self.assertEqual(decision.skip_reason, SKIP_NO_FRAGMENT)

    def test_fr021_atoms_outside_cnof_are_skipped(self):
        for smiles in ("CCS", "CCCl", "C[Si](C)C"):
            with self.subTest(smiles=smiles):
                decision = tier_eligibility(mol_of(smiles))["qml"]
                self.assertFalse(decision.eligible)
                self.assertEqual(decision.skip_reason, SKIP_NO_FRAGMENT)

    def test_fr021_explicit_hydrogens_are_not_counted(self):
        with_h = Chem.AddHs(Chem.MolFromSmiles("CCCCCCCCC"))
        self.assertTrue(tier_eligibility(with_h)["qml"].eligible)

    def test_br003_qml_skip_does_not_depend_on_target(self):
        big = mol_of("CC(=O)Oc1ccccc1C(=O)O")
        self.assertEqual(
            tier_eligibility(big, TARGET)["qml"].skip_reason, SKIP_NO_FRAGMENT
        )
        self.assertEqual(tier_eligibility(big)["qml"].skip_reason, SKIP_NO_FRAGMENT)


class ConfigTests(unittest.TestCase):
    """Limits come from config, not from the function body."""

    def test_limits_can_be_changed_through_config(self):
        config = EligibilityConfig(qml_max_heavy_atoms=12)
        self.assertTrue(tier_eligibility(mol_of("C" * 12), config=config)["qml"].eligible)
        self.assertFalse(tier_eligibility(mol_of("C" * 13), config=config)["qml"].eligible)

    def test_dl_atom_list_can_be_changed_through_config(self):
        config = EligibilityConfig(dl_supported_atoms=frozenset({"C", "N", "O", "Si"}))
        decision = tier_eligibility(mol_of("C[Si](C)(C)C"), TARGET, config)["dl"]
        self.assertTrue(decision.eligible)

    def test_defaults_follow_the_srs(self):
        config = EligibilityConfig()
        self.assertEqual(config.qml_max_heavy_atoms, 9)
        self.assertEqual(config.qml_allowed_atoms, frozenset({"C", "N", "O", "F"}))

    def test_from_env_with_nothing_set_gives_defaults(self):
        self.assertEqual(EligibilityConfig.from_env({}), EligibilityConfig())

    def test_from_env_reads_overrides(self):
        config = EligibilityConfig.from_env(
            {
                "MSP_QML_MAX_HEAVY_ATOMS": "8",
                "MSP_QML_ALLOWED_ATOMS": "C, N,O",
                "MSP_DL_SUPPORTED_ATOMS": "C,N,O,S",
            }
        )
        self.assertEqual(config.qml_max_heavy_atoms, 8)
        self.assertEqual(config.qml_allowed_atoms, frozenset({"C", "N", "O"}))
        self.assertEqual(config.dl_supported_atoms, frozenset({"C", "N", "O", "S"}))

    def test_from_env_blank_atom_list_falls_back_to_default(self):
        config = EligibilityConfig.from_env({"MSP_QML_ALLOWED_ATOMS": "  "})
        self.assertEqual(config.qml_allowed_atoms, EligibilityConfig().qml_allowed_atoms)

    def test_from_env_reads_the_real_environment_by_default(self):
        self.assertIsInstance(EligibilityConfig.from_env(), EligibilityConfig)


class TierDecisionTests(unittest.TestCase):
    def test_to_dict_matches_the_api_shape(self):
        self.assertEqual(
            TierDecision(False, SKIP_NO_TARGET).to_dict(),
            {"eligible": False, "skip_reason": "NO_TARGET_SELECTED"},
        )
        self.assertEqual(
            TierDecision(True).to_dict(), {"eligible": True, "skip_reason": None}
        )

    def test_skip_reason_codes_match_the_api_contract(self):
        self.assertEqual(SKIP_NO_TARGET, "NO_TARGET_SELECTED")
        self.assertEqual(SKIP_UNSUPPORTED_ATOM, "UNSUPPORTED_ATOM")
        self.assertEqual(SKIP_NO_FRAGMENT, "NO_FRAGMENT")


if __name__ == "__main__":
    unittest.main()
