"""Unit tests for molecule standardisation (MSP-19; FR-002, FR-003, FR-004, BR-001).

Run from the repo root with:  python -m unittest discover -s tests -t .
(with PYTHONPATH set to the backend folder)
"""
# RDKit is a C extension, so pylint cannot see its members.
# pylint: disable=no-member,c-extension-no-member,missing-function-docstring,missing-class-docstring
import json
import unittest
from unittest import mock

from rdkit import Chem

from backend.services.molecule.standardise import (
    ERROR_CODE,
    MoleculeError,
    StandardisedMolecule,
    parse_smiles,
    standardise_mol,
    standardise_smiles,
)

GOLDEN = {
    "aspirin": ("CC(=O)Oc1ccccc1C(=O)O", "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"),
    "caffeine": ("Cn1cnc2c1c(=O)n(C)c(=O)n2C", "RYYVLZVUVIJVGH-UHFFFAOYSA-N"),
    "ibuprofen": ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "HEFNNWSXXWATRW-UHFFFAOYSA-N"),
    "ethanol": ("CCO", "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
    "acetamide": ("CC(=O)N", "DLFVBJFMPXGRIB-UHFFFAOYSA-N"),
}


class GoldenMoleculeTests(unittest.TestCase):
    """FR-003: canonical isomeric SMILES and InChIKey for the golden molecules."""

    def test_fr003_aspirin_golden_values(self):
        result = standardise_smiles(GOLDEN["aspirin"][0])
        self.assertEqual(result.canonical_smiles, "CC(=O)Oc1ccccc1C(=O)O")
        self.assertEqual(result.inchikey, "BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
        self.assertEqual(result.formula, "C9H8O4")
        self.assertEqual(result.heavy_atoms, 13)
        self.assertEqual(result.warnings, [])

    def test_fr003_ethanol_golden_values(self):
        result = standardise_smiles("CCO")
        self.assertEqual(result.canonical_smiles, "CCO")
        self.assertEqual(result.inchikey, "LFQSCWFLJHTTHZ-UHFFFAOYSA-N")

    def test_fr003_all_five_golden_inchikeys(self):
        for name, (smiles, inchikey) in GOLDEN.items():
            with self.subTest(molecule=name):
                self.assertEqual(standardise_smiles(smiles).inchikey, inchikey)

    def test_fr003_standardising_twice_is_stable(self):
        for name, (smiles, _) in GOLDEN.items():
            with self.subTest(molecule=name):
                first = standardise_smiles(smiles)
                second = standardise_smiles(first.canonical_smiles)
                self.assertEqual(first.canonical_smiles, second.canonical_smiles)
                self.assertEqual(first.inchikey, second.inchikey)

    def test_fr003_different_spellings_give_one_result(self):
        a = standardise_smiles("OCC")
        b = standardise_smiles("C(O)C")
        self.assertEqual(a.canonical_smiles, b.canonical_smiles)
        self.assertEqual(a.inchikey, b.inchikey)


class StandardisationStepTests(unittest.TestCase):
    """FR-003: stereo, charge, salts, neutralisation and tautomers."""

    def test_fr003_stereocentre_is_kept(self):
        result = standardise_smiles("C[C@H](N)C(=O)O")
        self.assertIn("@", result.canonical_smiles)
        self.assertEqual(result.inchikey, "QNAYBMKLOCPYGJ-REOHCLBHSA-N")

    def test_fr003_enantiomers_stay_different(self):
        left = standardise_smiles("C[C@H](N)C(=O)O")
        right = standardise_smiles("C[C@@H](N)C(=O)O")
        self.assertNotEqual(left.canonical_smiles, right.canonical_smiles)
        self.assertNotEqual(left.inchikey, right.inchikey)

    def test_fr003_double_bond_stereo_is_kept(self):
        trans = standardise_smiles("C/C=C/C")
        cis = standardise_smiles("C/C=C\\C")
        self.assertNotEqual(trans.inchikey, cis.inchikey)

    def test_fr003_permanent_charge_is_kept(self):
        result = standardise_smiles("C[N+](C)(C)C")
        self.assertIn("+", result.canonical_smiles)

    def test_fr003_salt_is_stripped_and_listed_in_warnings(self):
        result = standardise_smiles("CC(=O)[O-].[Na+]")
        self.assertEqual(result.canonical_smiles, "CC(=O)O")
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("[Na+]", result.warnings[0])

    def test_fr003_largest_fragment_wins(self):
        result = standardise_smiles("CC(=O)O.CCO")
        self.assertEqual(result.canonical_smiles, "CCO")
        self.assertIn("CC(=O)O", result.warnings[0])

    def test_fr003_charges_are_neutralised(self):
        self.assertEqual(standardise_smiles("OC(=O)C[NH3+]").canonical_smiles, "NCC(=O)O")
        self.assertEqual(standardise_smiles("CC(=O)[O-]").canonical_smiles, "CC(=O)O")

    def test_fr003_tautomers_share_one_canonical_form(self):
        pyridinol = standardise_smiles("Oc1ccccn1")
        pyridone = standardise_smiles("O=c1cccc[nH]1")
        self.assertEqual(pyridinol.canonical_smiles, pyridone.canonical_smiles)
        self.assertEqual(pyridinol.inchikey, pyridone.inchikey)

    def test_fr003_explicit_hydrogens_are_removed(self):
        result = standardise_smiles("[H]C([H])([H])O")
        self.assertEqual(result.canonical_smiles, "CO")
        self.assertEqual(result.heavy_atoms, 2)


class InvalidInputTests(unittest.TestCase):
    """FR-002, FR-004, BR-001: invalid input is rejected with a short message."""

    def test_fr004_empty_and_blank_input_rejected(self):
        for value in ("", "   ", "\n", None):
            with self.subTest(value=value):
                with self.assertRaises(MoleculeError) as ctx:
                    standardise_smiles(value)
                self.assertEqual(ctx.exception.code, ERROR_CODE)

    def test_fr004_unparsable_smiles_rejected(self):
        for value in ("C(C", "xyz", "C1CC", "C="):
            with self.subTest(value=value):
                with self.assertRaises(MoleculeError) as ctx:
                    standardise_smiles(value)
                self.assertEqual(ctx.exception.code, ERROR_CODE)
                self.assertIn("could not be parsed", ctx.exception.message)

    def test_fr004_structure_that_cannot_be_standardised_is_rejected(self):
        # "C$C" (quadruple bond) parses but cannot be standardised.
        with self.assertRaises(MoleculeError) as ctx:
            standardise_smiles("C$C")
        self.assertEqual(ctx.exception.code, ERROR_CODE)
        self.assertIn("could not be standardised", ctx.exception.message)

    def test_fr004_missing_inchikey_is_rejected(self):
        with mock.patch("backend.services.molecule.standardise.Chem.MolToInchiKey", return_value=""):
            with self.assertRaises(MoleculeError):
                standardise_smiles("CCO")

    def test_fr002_valence_error_fails_sanitisation(self):
        with self.assertRaises(MoleculeError) as ctx:
            standardise_smiles("CC(C)(C)(C)(C)C")
        self.assertIn("sanitisation", ctx.exception.message)

    def test_fr004_message_is_short_and_has_no_traceback(self):
        with self.assertRaises(MoleculeError) as ctx:
            standardise_smiles("C(C")
        self.assertLess(len(ctx.exception.message), 100)
        self.assertNotIn("Traceback", ctx.exception.message)

    def test_br001_standardise_mol_rejects_none_and_empty(self):
        for value in (None, Chem.Mol()):
            with self.subTest(value=value):
                with self.assertRaises(MoleculeError):
                    standardise_mol(value)

    def test_fr002_parse_smiles_returns_sanitised_molecule(self):
        mol = parse_smiles("c1ccccc1")
        self.assertEqual(mol.GetNumAtoms(), 6)

    def test_fr004_molecule_error_is_a_value_error(self):
        self.assertTrue(issubclass(MoleculeError, ValueError))


class ResultShapeTests(unittest.TestCase):
    """The result is safe to return from the API and from a child process."""

    def test_to_dict_has_the_documented_fields_and_no_rdkit_object(self):
        data = standardise_smiles("CCO").to_dict()
        self.assertEqual(
            set(data),
            {"canonical_smiles", "inchikey", "formula", "heavy_atoms", "warnings"},
        )
        json.dumps(data)  # must be JSON-serialisable

    def test_result_keeps_the_rdkit_molecule_for_later_steps(self):
        result = standardise_smiles("CCO")
        self.assertIsInstance(result, StandardisedMolecule)
        self.assertEqual(result.mol.GetNumHeavyAtoms(), 3)

    def test_to_dict_copies_warnings(self):
        result = standardise_smiles("CC(=O)[O-].[Na+]")
        data = result.to_dict()
        data["warnings"].append("extra")
        self.assertEqual(len(result.warnings), 1)


if __name__ == "__main__":
    unittest.main()
