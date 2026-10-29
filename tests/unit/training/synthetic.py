"""Small synthetic molecule sets for the training tests (no real datasets needed in CI)."""
# pylint: disable=no-member,no-name-in-module,c-extension-no-member
import itertools
from typing import List

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen

from backend.services.molecule.standardise import standardise_smiles
from training.datasets import TOX21_TASKS

RINGS = [
    "c1ccccc1", "C1CCCCC1", "c1ccncc1", "c1ccoc1", "c1ccsc1", "C1CCNCC1", "C1CCOC1",
    "c1ccc2ccccc2c1",
]
LINKERS = ["", "C", "O", "N", "CC"]
END_GROUPS = ["O", "N", "C(=O)O", "Cl"]


def ring_smiles(count: int) -> List[str]:
    """Ring - linker - ring molecules, each with its own scaffold (deterministic)."""
    seen, out = set(), []
    for first, link, second in itertools.product(RINGS, LINKERS, RINGS):
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(first + link + second))
        if canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
        if len(out) == count:
            break
    return out


def acyclic_smiles() -> List[str]:
    """Chains of 1 to 10 carbons with a functional end group (no rings)."""
    return ["C" * length + end for length in range(1, 11) for end in END_GROUPS]


def make_frame(n_rings: int = 190) -> pd.DataFrame:
    """Standardised molecules with a LogS-like target and 12 Tox21-like label columns."""
    smiles = [standardise_smiles(value).canonical_smiles for value in ring_smiles(n_rings) + acyclic_smiles()]
    logp = np.array([Crippen.MolLogP(Chem.MolFromSmiles(value)) for value in smiles])
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"smiles_std": smiles})
    frame["inchikey"] = [standardise_smiles(value).inchikey for value in smiles]
    frame["logs"] = -0.6 * logp + rng.normal(0, 0.3, len(smiles))
    for task in TOX21_TASKS:
        score = logp + rng.normal(0, 0.8, len(smiles))
        labels = (score > np.quantile(score, 0.6)).astype(float)
        labels[rng.random(len(smiles)) < 0.1] = np.nan
        frame[task] = labels
    return frame
