"""Tests for the shared molecule-to-graph function (MSP-25 / MSP-26; FR-016)."""
# pylint: disable=missing-function-docstring,missing-class-docstring,no-member,no-name-in-module,c-extension-no-member
import unittest
from unittest import mock

import numpy as np
from rdkit import Chem

from backend.services.molecule.eligibility import DEFAULT_DL_SUPPORTED_ATOMS
from backend.workers.dl import graph


def mol(smiles):
    return Chem.MolFromSmiles(smiles)


class GraphShapeTests(unittest.TestCase):
    def test_fr016_ethanol_has_three_atoms_and_four_directed_edges(self):
        result = graph.mol_to_graph(mol("CCO"))
        self.assertEqual(result.x.shape, (3, graph.N_ATOM_FEATURES))
        self.assertEqual(result.edge_index.shape, (2, 4))
        self.assertEqual(result.edge_attr.shape, (4, graph.N_BOND_FEATURES))
        self.assertEqual(result.n_atoms, 3)

    def test_dtypes_are_what_torch_expects(self):
        result = graph.mol_to_graph(mol("c1ccccc1O"))
        self.assertEqual(result.x.dtype, np.float32)
        self.assertEqual(result.edge_attr.dtype, np.float32)
        self.assertEqual(result.edge_index.dtype, np.int64)

    def test_every_bond_appears_in_both_directions(self):
        result = graph.mol_to_graph(mol("CC(=O)Nc1ccccc1"))
        pairs = set(map(tuple, result.edge_index.T.tolist()))
        self.assertEqual(len(pairs), result.edge_index.shape[1])
        for source, target in pairs:
            self.assertIn((target, source), pairs)

    def test_a_single_atom_has_no_edges_but_valid_empty_arrays(self):
        result = graph.mol_to_graph(mol("C"))
        self.assertEqual(result.edge_index.shape, (2, 0))
        self.assertEqual(result.edge_attr.shape, (0, graph.N_BOND_FEATURES))

    def test_each_atom_row_is_made_of_one_hot_blocks(self):
        result = graph.mol_to_graph(mol("CCO"))
        symbols = result.x[:, : len(graph.ATOM_SYMBOLS)]
        self.assertTrue(np.all(symbols.sum(axis=1) == 1.0))
        self.assertEqual(graph.ATOM_SYMBOLS[int(symbols[2].argmax())], "O")

    def test_aromatic_and_ring_flags(self):
        result = graph.mol_to_graph(mol("c1ccccc1C"))
        aromatic, in_ring = result.x[:, -2], result.x[:, -1]
        self.assertEqual(aromatic.tolist(), [1.0] * 6 + [0.0])
        self.assertEqual(in_ring.tolist(), [1.0] * 6 + [0.0])

    def test_formal_charge_is_encoded(self):
        result = graph.mol_to_graph(mol("[NH4+]"))
        start = len(graph.ATOM_SYMBOLS) + (graph.MAX_DEGREE + 1) + (graph.MAX_HYDROGENS + 1)
        self.assertEqual(result.x[0, start: start + 3].tolist(), [0.0, 0.0, 1.0])

    def test_bond_types_are_encoded(self):
        result = graph.mol_to_graph(mol("C=C"))
        self.assertEqual(result.edge_attr[0, : len(graph.BOND_TYPES)].tolist(), [0.0, 1.0, 0.0, 0.0])

    def test_atom_order_does_not_change_the_set_of_atom_rows(self):
        first = graph.smiles_to_graph("OCC(=O)N")
        second = graph.smiles_to_graph("NC(=O)CO")
        def rows(item):
            return sorted(map(tuple, item.x.tolist()))

        self.assertEqual(rows(first), rows(second))


class GraphErrorTests(unittest.TestCase):
    def test_unsupported_element_is_refused(self):
        for smiles in ("[Na+]", "C[Si](C)(C)C", "OB(O)O"):
            with self.subTest(smiles=smiles), self.assertRaises(graph.GraphError):
                graph.mol_to_graph(mol(smiles))

    def test_explicit_hydrogen_atoms_are_refused(self):
        with self.assertRaises(graph.GraphError):
            graph.mol_to_graph(Chem.AddHs(mol("CO")))

    def test_none_and_empty_molecules_are_refused(self):
        with self.assertRaises(graph.GraphError):
            graph.mol_to_graph(None)
        with self.assertRaises(graph.GraphError):
            graph.mol_to_graph(Chem.RWMol())

    def test_unparsable_smiles_is_refused(self):
        with self.assertRaises(graph.GraphError):
            graph.smiles_to_graph("this is not a molecule")


class GraphContractTests(unittest.TestCase):
    def test_graph_atoms_match_the_tier_eligibility_atoms(self):
        self.assertEqual(set(graph.ATOM_SYMBOLS), set(DEFAULT_DL_SUPPORTED_ATOMS))

    def test_signature_is_stable_and_changes_with_the_definition(self):
        first = graph.graph_signature()
        self.assertEqual(first, graph.graph_signature())
        self.assertRegex(first, r"^[0-9a-f]{16}$")
        with mock.patch.object(graph, "ATOM_SYMBOLS", graph.ATOM_SYMBOLS + ("Se",)):
            self.assertNotEqual(first, graph.graph_signature())

    def test_graph_info_describes_the_featuriser(self):
        info = graph.graph_info()
        self.assertEqual(info["signature"], graph.graph_signature())
        self.assertEqual(info["n_atom_features"], graph.N_ATOM_FEATURES)
        self.assertEqual(info["n_bond_features"], graph.N_BOND_FEATURES)


class BatchTests(unittest.TestCase):
    def test_batch_shifts_edges_and_numbers_the_molecules(self):
        first, second = graph.smiles_to_graph("CC"), graph.smiles_to_graph("CCO")
        x, edge_index, edge_attr, batch = graph.batch_graphs([first, second])
        self.assertEqual(x.shape[0], 5)
        self.assertEqual(batch.tolist(), [0, 0, 1, 1, 1])
        self.assertEqual(edge_index.shape, (2, 6))
        self.assertEqual(edge_attr.shape[0], 6)
        self.assertTrue(set(edge_index[:, 2:].ravel().tolist()) <= {2, 3, 4})
        self.assertTrue(set(edge_index[:, :2].ravel().tolist()) <= {0, 1})

    def test_batch_of_one_molecule_is_all_zeros(self):
        _, _, _, batch = graph.batch_graphs([graph.smiles_to_graph("CCO")])
        self.assertEqual(batch.tolist(), [0, 0, 0])

    def test_batch_handles_a_molecule_without_bonds(self):
        _, edge_index, _, batch = graph.batch_graphs([graph.smiles_to_graph("C"), graph.smiles_to_graph("CC")])
        self.assertEqual(edge_index.shape, (2, 2))
        self.assertEqual(batch.tolist(), [0, 1, 1])

    def test_empty_batch_is_refused(self):
        with self.assertRaises(graph.GraphError):
            graph.batch_graphs([])


if __name__ == "__main__":
    unittest.main()
