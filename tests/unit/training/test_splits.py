"""Unit tests for the scaffold split (MSP-22; scaffold-split metrics in QA-ACC)."""
# pylint: disable=missing-function-docstring,missing-class-docstring
import unittest

import numpy as np

from tests.unit.training.synthetic import acyclic_smiles, ring_smiles
from training.splits import scaffold_groups, scaffold_split

SMILES = ring_smiles(190) + acyclic_smiles()


class ScaffoldSplitTests(unittest.TestCase):
    def setUp(self):
        self.train, self.cal, self.test = scaffold_split(SMILES, seed=42)

    def test_parts_are_disjoint_and_cover_every_molecule(self):
        everything = np.concatenate([self.train, self.cal, self.test])
        self.assertEqual(len(everything), len(SMILES))
        self.assertEqual(set(everything), set(range(len(SMILES))))

    def test_part_sizes_follow_the_fractions(self):
        total = len(SMILES)
        self.assertAlmostEqual(len(self.train) / total, 0.7, delta=0.08)
        self.assertAlmostEqual(len(self.cal) / total, 0.1, delta=0.06)
        self.assertAlmostEqual(len(self.test) / total, 0.2, delta=0.08)
        self.assertGreater(len(self.cal), 0)
        self.assertGreater(len(self.test), 0)

    def test_no_scaffold_is_shared_between_parts(self):
        part_of = {}
        for label, part in (("train", self.train), ("cal", self.cal), ("test", self.test)):
            for index in part:
                part_of[int(index)] = label
        for members in scaffold_groups(SMILES).values():
            self.assertEqual(len({part_of[index] for index in members}), 1)

    def test_acyclic_molecules_are_spread_across_parts(self):
        acyclic_rows = set(range(190, len(SMILES)))
        self.assertTrue(acyclic_rows & set(self.test.tolist()))
        self.assertTrue(acyclic_rows & set(self.train.tolist()))

    def test_same_seed_gives_the_same_split(self):
        again = scaffold_split(SMILES, seed=42)
        for first, second in zip((self.train, self.cal, self.test), again):
            np.testing.assert_array_equal(first, second)

    def test_different_seed_gives_a_different_split(self):
        _, _, other_test = scaffold_split(SMILES, seed=7)
        self.assertFalse(np.array_equal(self.test, other_test))

    def test_indices_are_sorted(self):
        for part in (self.train, self.cal, self.test):
            self.assertTrue(np.all(np.diff(part) > 0))


class InvalidInputTests(unittest.TestCase):
    def test_too_few_molecules_rejected(self):
        with self.assertRaises(ValueError):
            scaffold_split(["CCO"] * 5)

    def test_bad_fractions_rejected(self):
        with self.assertRaises(ValueError):
            scaffold_split(SMILES, fractions=(0.5, 0.2, 0.2))
        with self.assertRaises(ValueError):
            scaffold_split(SMILES, fractions=(1.0, 0.0, 0.0))

    def test_unparsable_smiles_rejected(self):
        with self.assertRaises(ValueError):
            scaffold_groups(["CCO", "not a smiles"])


if __name__ == "__main__":
    unittest.main()
