"""Molecule to graph, shared by training and serving.

Owner: Nishitha. Requirements: FR-016.

One function turns a standardised RDKit molecule into numpy arrays: one row per
heavy atom, one column block per bond direction. Training and serving both call
mol_to_graph(), so a model can never see a graph built a different way. The
feature list is frozen here and its signature is written into every model card;
serving refuses a model whose signature differs.

Pure RDKit and numpy: no torch, Flask, database or Celery imports.
"""
# RDKit is a C extension, so pylint cannot see its members.
# pylint: disable=no-member,no-name-in-module,c-extension-no-member
import hashlib
import json
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from rdkit import Chem

GRAPH_VERSION = "1"

# Must stay equal to eligibility.DEFAULT_DL_SUPPORTED_ATOMS (a unit test checks this):
# a molecule with any other heavy atom is skipped with UNSUPPORTED_ATOM before it gets here.
ATOM_SYMBOLS = ("Br", "C", "Cl", "F", "I", "N", "O", "P", "S")
MAX_DEGREE = 5          # degrees 0..5, anything larger is clipped to 5
MAX_HYDROGENS = 4       # 0..4, larger is clipped
CHARGES = (-1, 0, 1)    # other charges count as 0 (rare in the training data)
HYBRIDISATIONS = (
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
)
CHIRAL_TAGS = (
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
)
BOND_TYPES = (
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
)

N_ATOM_FEATURES = (
    len(ATOM_SYMBOLS) + (MAX_DEGREE + 1) + (MAX_HYDROGENS + 1) + len(CHARGES)
    + len(HYBRIDISATIONS) + len(CHIRAL_TAGS) + 2   # aromatic, in ring
)
N_BOND_FEATURES = len(BOND_TYPES) + 2              # conjugated, in ring


class GraphError(ValueError):
    """The molecule contains something this graph featuriser does not support."""


@dataclass(frozen=True)
class MolGraph:
    """Arrays for one molecule. Edges are directed, so each bond appears twice."""

    x: np.ndarray           # (n_atoms, N_ATOM_FEATURES) float32
    edge_index: np.ndarray  # (2, n_edges) int64, row 0 = source atom, row 1 = target atom
    edge_attr: np.ndarray   # (n_edges, N_BOND_FEATURES) float32

    @property
    def n_atoms(self) -> int:
        """Number of heavy atoms."""
        return int(self.x.shape[0])


def _one_hot(value, choices: Sequence) -> List[float]:
    return [1.0 if value == choice else 0.0 for choice in choices]


def _clip(value: int, high: int) -> int:
    return max(0, min(int(value), high))


def atom_features(atom) -> List[float]:
    """Feature vector for one atom. Raises GraphError for an unsupported element."""
    symbol = atom.GetSymbol()
    if symbol not in ATOM_SYMBOLS:
        raise GraphError("unsupported element")
    charge = atom.GetFormalCharge()
    return (
        _one_hot(symbol, ATOM_SYMBOLS)
        + _one_hot(_clip(atom.GetDegree(), MAX_DEGREE), range(MAX_DEGREE + 1))
        + _one_hot(_clip(atom.GetTotalNumHs(), MAX_HYDROGENS), range(MAX_HYDROGENS + 1))
        + _one_hot(charge if charge in CHARGES else 0, CHARGES)
        + _one_hot(atom.GetHybridization(), HYBRIDISATIONS)
        + _one_hot(atom.GetChiralTag(), CHIRAL_TAGS)
        + [float(atom.GetIsAromatic()), float(atom.IsInRing())]
    )


def bond_features(bond) -> List[float]:
    """Feature vector for one bond."""
    return _one_hot(bond.GetBondType(), BOND_TYPES) + [
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def mol_to_graph(mol) -> MolGraph:
    """Graph arrays for a standardised molecule (explicit hydrogens are not nodes)."""
    if mol is None or mol.GetNumAtoms() == 0:
        raise GraphError("empty molecule")
    atoms = [atom_features(atom) for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    if len(atoms) != mol.GetNumAtoms():
        raise GraphError("explicit hydrogen atoms are not supported")
    sources: List[int] = []
    targets: List[int] = []
    attrs: List[List[float]] = []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        features = bond_features(bond)
        sources += [begin, end]
        targets += [end, begin]
        attrs += [features, features]
    return MolGraph(
        x=np.asarray(atoms, dtype=np.float32),
        edge_index=np.asarray([sources, targets], dtype=np.int64).reshape(2, -1),
        edge_attr=np.asarray(attrs, dtype=np.float32).reshape(-1, N_BOND_FEATURES),
    )


def smiles_to_graph(smiles: str) -> MolGraph:
    """Graph for an already standardised SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise GraphError("SMILES cannot be parsed")
    return mol_to_graph(mol)


def batch_graphs(graphs: Sequence[MolGraph]):
    """Stack graphs into one batch: (x, edge_index, edge_attr, batch) as numpy arrays.

    `batch[i]` is the position of the molecule that atom i belongs to; edge_index is
    shifted so it points at rows of the stacked x. One molecule gives batch of zeros.
    """
    if not graphs:
        raise GraphError("no graphs to batch")
    offsets = np.cumsum([0] + [graph.n_atoms for graph in graphs[:-1]])
    x = np.concatenate([graph.x for graph in graphs], axis=0)
    edge_index = np.concatenate(
        [graph.edge_index + offset for graph, offset in zip(graphs, offsets)], axis=1
    )
    edge_attr = np.concatenate([graph.edge_attr for graph in graphs], axis=0)
    batch = np.concatenate(
        [np.full(graph.n_atoms, position, dtype=np.int64) for position, graph in enumerate(graphs)]
    )
    return x, edge_index, edge_attr, batch


def graph_signature() -> str:
    """Short hash of the frozen feature definition (recorded in every DL model card)."""
    description = {
        "version": GRAPH_VERSION,
        "atoms": list(ATOM_SYMBOLS),
        "max_degree": MAX_DEGREE,
        "max_hydrogens": MAX_HYDROGENS,
        "charges": list(CHARGES),
        "hybridisations": [str(item) for item in HYBRIDISATIONS],
        "chiral_tags": [str(item) for item in CHIRAL_TAGS],
        "bond_types": [str(item) for item in BOND_TYPES],
        "n_atom_features": N_ATOM_FEATURES,
        "n_bond_features": N_BOND_FEATURES,
    }
    blob = json.dumps(description, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def graph_info() -> dict:
    """What the model card records about the graph featuriser."""
    return {
        "version": GRAPH_VERSION,
        "signature": graph_signature(),
        "n_atom_features": N_ATOM_FEATURES,
        "n_bond_features": N_BOND_FEATURES,
        "atoms": list(ATOM_SYMBOLS),
    }
