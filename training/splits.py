"""Deterministic scaffold split into train / calibration / test (MSP-22).

Molecules that share a Bemis-Murcko scaffold always land in the same part, so
the test set measures performance on new chemistry, not on near-copies.
Large scaffold groups go to train first; the remaining groups are shuffled
with a fixed seed and dealt out in order. Molecules without any ring would all
share the empty scaffold, so each one is its own group and is spread across
the parts (the app will see small acyclic molecules such as ethanol).
"""
# RDKit is a C extension, so pylint cannot see its members.
# pylint: disable=no-member,no-name-in-module,c-extension-no-member
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

DEFAULT_FRACTIONS = (0.7, 0.1, 0.2)


def scaffold_groups(smiles: Sequence[str]) -> Dict[str, List[int]]:
    """Map scaffold key -> row indices. Acyclic molecules get unique keys."""
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError(f"row {index}: SMILES cannot be parsed for the scaffold split")
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        key = scaffold if scaffold else f"acyclic:{index}"
        groups[key].append(index)
    return groups


def scaffold_split(
    smiles: Sequence[str],
    fractions: Tuple[float, float, float] = DEFAULT_FRACTIONS,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, calibration_idx, test_idx) as sorted integer arrays."""
    if abs(sum(fractions) - 1.0) > 1e-9 or min(fractions) <= 0:
        raise ValueError("fractions must be positive and sum to 1")
    total = len(smiles)
    if total < 10:
        raise ValueError("need at least 10 molecules to split")
    train_size = fractions[0] * total
    cal_size = fractions[1] * total
    big_limit = min(cal_size, fractions[2] * total) / 2.0

    groups = scaffold_groups(smiles)
    ordered = sorted(groups.values(), key=lambda members: (-len(members), members[0]))
    big = [members for members in ordered if len(members) > big_limit]
    small = [members for members in ordered if len(members) <= big_limit]
    np.random.default_rng(seed).shuffle(small)

    train: List[int] = []
    calibration: List[int] = []
    test: List[int] = []
    for members in big + small:
        if len(train) + len(members) <= train_size or not train:
            train.extend(members)
        elif len(calibration) + len(members) <= cal_size or not calibration:
            calibration.extend(members)
        else:
            test.extend(members)
    return np.array(sorted(train)), np.array(sorted(calibration)), np.array(sorted(test))
