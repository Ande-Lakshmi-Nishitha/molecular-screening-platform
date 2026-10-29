"""Unit tests for the shared ML featuriser (MSP-22 / MSP-23; FR-009)."""
# pylint: disable=no-member,c-extension-no-member,missing-function-docstring,missing-class-docstring
# pylint: disable=protected-access
import unittest
from unittest import mock

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from backend.services.molecule.standardise import standardise_smiles
from backend.workers.ml import featuriser

GOLDEN = [
    "CC(=O)Oc1ccccc1C(=O)O",             # aspirin
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",        # caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",        # ibuprofen
    "CCO",                               # ethanol
    "CC(=O)N",                           # acetamide
]


def vector_of(smiles):
    return featuriser.featurise_mol(standardise_smiles(smiles).mol)


class ShapeTests(unittest.TestCase):
    def test_fr009_feature_count_is_descriptors_plus_morgan_plus_maccs(self):
        self.assertEqual(featuriser.MORGAN_BITS, 2048)
        self.assertEqual(featuriser.MACCS_BITS, 167)
        self.assertEqual(
            featuriser.N_FEATURES, featuriser.N_DESCRIPTORS + 2048 + 167
        )
        self.assertEqual(len(featuriser.FEATURE_NAMES), featuriser.N_FEATURES)
        self.assertEqual(len(set(featuriser.FEATURE_NAMES)), featuriser.N_FEATURES)

    def test_fr009_golden_molecules_give_finite_float32_vectors(self):
        for smiles in GOLDEN:
            with self.subTest(smiles=smiles):
                vector = vector_of(smiles)
                self.assertEqual(vector.shape, (featuriser.N_FEATURES,))
                self.assertEqual(vector.dtype, np.float32)
                self.assertFalse(np.isinf(vector).any())
                self.assertLess(np.isnan(vector).sum(), 5)

    def test_fr009_fingerprint_parts_are_zero_or_one(self):
        vector = vector_of(GOLDEN[0])
        bits = vector[featuriser.N_DESCRIPTORS:]
        self.assertTrue(set(np.unique(bits)) <= {0.0, 1.0})
        self.assertGreater(bits[: featuriser.MORGAN_BITS].sum(), 0)
        self.assertGreater(bits[featuriser.MORGAN_BITS:].sum(), 0)

    def test_fr009_known_descriptor_values(self):
        vector = vector_of("CCO")
        index = featuriser.DESCRIPTOR_NAMES.index("MolWt")
        self.assertAlmostEqual(float(vector[index]), 46.069, places=2)
        index = featuriser.DESCRIPTOR_NAMES.index("HeavyAtomCount")
        self.assertEqual(float(vector[index]), 3.0)


class DeterminismTests(unittest.TestCase):
    def test_same_molecule_gives_identical_vectors(self):
        first, second = vector_of(GOLDEN[1]), vector_of(GOLDEN[1])
        np.testing.assert_array_equal(first, second)

    def test_different_spellings_give_identical_vectors(self):
        np.testing.assert_array_equal(vector_of("OCC"), vector_of("C(O)C"))

    def test_different_molecules_give_different_vectors(self):
        self.assertFalse(np.array_equal(vector_of("CCO"), vector_of("CC(=O)N")))

    def test_training_and_serving_build_the_same_vector(self):
        """Training parses the canonical SMILES; serving uses the standardised molecule."""
        for smiles in GOLDEN:
            with self.subTest(smiles=smiles):
                result = standardise_smiles(smiles)
                served = featuriser.featurise_mol(result.mol)
                trained = featuriser.featurise_mol(Chem.MolFromSmiles(result.canonical_smiles))
                np.testing.assert_array_equal(served, trained)

    def test_signature_is_stable_short_hex(self):
        signature = featuriser.feature_signature()
        self.assertEqual(signature, featuriser.feature_signature())
        self.assertRegex(signature, r"^[0-9a-f]{16}$")


class RobustnessTests(unittest.TestCase):
    def test_frozen_descriptor_names_still_exist_in_rdkit(self):
        """Fails if an RDKit upgrade removes a descriptor (re-run regression, R09)."""
        available = {name for name, _ in Descriptors.descList}
        missing = [name for name in featuriser.DESCRIPTOR_NAMES if name not in available]
        self.assertEqual(missing, [])

    def test_none_and_empty_molecules_are_rejected(self):
        for value in (None, Chem.Mol()):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    featuriser.featurise_mol(value)

    def test_empty_batch_has_the_right_width(self):
        self.assertEqual(featuriser.featurise_mols([]).shape, (0, featuriser.N_FEATURES))

    def test_batch_matches_single_vectors(self):
        mols = [standardise_smiles(smiles).mol for smiles in GOLDEN[:3]]
        batch = featuriser.featurise_mols(mols)
        self.assertEqual(batch.shape, (3, featuriser.N_FEATURES))
        np.testing.assert_array_equal(batch[1], featuriser.featurise_mol(mols[1]))

    def test_one_failing_descriptor_becomes_nan_not_an_error(self):
        functions = list(featuriser._descriptor_functions())

        def broken(_mol):
            raise ValueError("boom")

        functions[0] = broken
        with mock.patch.object(featuriser, "_descriptor_functions", return_value=functions):
            vector = vector_of("CCO")
        self.assertTrue(np.isnan(vector[0]))
        self.assertFalse(np.isnan(vector[1]))

    def test_infinite_descriptor_values_become_nan(self):
        functions = list(featuriser._descriptor_functions())
        functions[0] = lambda _mol: float("inf")
        with mock.patch.object(featuriser, "_descriptor_functions", return_value=functions):
            vector = vector_of("CCO")
        self.assertTrue(np.isnan(vector[0]))

    def test_huge_descriptor_values_are_clipped(self):
        functions = list(featuriser._descriptor_functions())
        functions[0] = lambda _mol: 1e30
        with mock.patch.object(featuriser, "_descriptor_functions", return_value=functions):
            vector = vector_of("CCO")
        self.assertEqual(float(vector[0]), np.float32(featuriser.VALUE_LIMIT))

    def test_missing_rdkit_descriptor_raises_a_clear_error(self):
        with mock.patch.object(featuriser, "_DESCRIPTOR_FUNCTIONS", None):
            with mock.patch.object(Descriptors, "descList", [("MolWt", Descriptors.MolWt)]):
                with self.assertRaises(RuntimeError) as ctx:
                    featuriser._descriptor_functions()
        self.assertIn("no longer has descriptors", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
