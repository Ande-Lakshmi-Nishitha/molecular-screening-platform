"""Load and clean the public datasets used to train the ML tier (MSP-22).

Every structure goes through the MSP-19 standardiser, so training molecules are
standardised exactly like molecules arriving through the API. Duplicates (same
InChIKey after standardising) are merged. Nothing here is hidden: the returned
stats say how many rows were dropped and why.

Sources (record the licence from each source page on the download day, BR-009):
  AqSolDB  Sorkun et al., Scientific Data 2019, doi:10.1038/s41597-019-0151-1
  ESOL     Delaney 2004, as distributed with MoleculeNet / DeepChem
  Tox21    Tox21 Data Challenge 2014, as distributed with MoleculeNet / DeepChem
"""
# pylint: disable=line-too-long
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from backend.services.molecule.standardise import MoleculeError, standardise_smiles

TOX21_TASKS = (
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
)

SOURCE_INFO = {
    "aqsoldb": {
        "name": "AqSolDB",
        "citation": "Sorkun et al., Scientific Data 6, 143 (2019), doi:10.1038/s41597-019-0151-1",
        "origin": "https://raw.githubusercontent.com/mcsorkun/AqSolDB/master/results/data_curated.csv",
        "licence_note": (
            "The code repository is MIT licensed. The dataset's own licence is stated "
            "on its Harvard Dataverse page (doi:10.7910/DVN/OVHAW8) and must be checked there."
        ),
    },
    "esol": {
        "name": "ESOL (Delaney)",
        "citation": "Delaney, J. Chem. Inf. Comput. Sci. 44, 1000 (2004), doi:10.1021/ci034243x",
        "origin": "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv",
        "licence_note": "Check the licence on the MoleculeNet / DeepChem source page.",
    },
    "tox21": {
        "name": "Tox21 (Data Challenge 2014, MoleculeNet copy)",
        "citation": "Tox21 Data Challenge 2014; Wu et al., Chem. Sci. 9, 513 (2018) for MoleculeNet",
        "origin": "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/tox21.csv.gz",
        "licence_note": "Check the licence on the MoleculeNet / DeepChem source page.",
    },
}


def standardise_many(smiles: Sequence[str]) -> pd.DataFrame:
    """Standardise a list of SMILES. Columns: ok, smiles_std, inchikey (rows kept in order)."""
    rows: List[Tuple[bool, str, str]] = []
    for value in smiles:
        try:
            result = standardise_smiles(value)
            rows.append((True, result.canonical_smiles, result.inchikey))
        except (MoleculeError, ValueError, RuntimeError):
            rows.append((False, "", ""))
    return pd.DataFrame(rows, columns=["ok", "smiles_std", "inchikey"])


def load_aqsoldb(path) -> pd.DataFrame:
    """AqSolDB curated file -> columns smiles, logs (log10 mol/L), group."""
    raw = pd.read_csv(path)
    return pd.DataFrame(
        {"smiles": raw["SMILES"], "logs": raw["Solubility"].astype(float), "group": raw["Group"]}
    )


def load_esol(path) -> pd.DataFrame:
    """ESOL (Delaney) file -> columns smiles, logs (log10 mol/L)."""
    raw = pd.read_csv(path)
    column = "measured log solubility in mols per litre"
    return pd.DataFrame({"smiles": raw["smiles"], "logs": raw[column].astype(float)})


def load_tox21(path) -> pd.DataFrame:
    """Tox21 file -> columns smiles plus one 0/1/NaN column per assay."""
    raw = pd.read_csv(path)
    missing = [task for task in TOX21_TASKS if task not in raw.columns]
    if missing:
        raise ValueError("Tox21 file is missing assay columns: " + ", ".join(missing))
    return raw[["smiles", *TOX21_TASKS]].copy()


def clean_regression(frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Standardise, drop failures, merge duplicates by median. Returns (clean, stats)."""
    std = standardise_many(frame["smiles"].tolist())
    joined = pd.concat([frame.reset_index(drop=True), std], axis=1)
    kept = joined[joined["ok"] & joined["logs"].notna()]
    grouped = kept.groupby("inchikey", sort=True)
    clean = grouped.agg(smiles_std=("smiles_std", "first"), logs=("logs", "median")).reset_index()
    stats = {
        "n_raw": int(len(frame)),
        "n_dropped_unparsable": int((~joined["ok"]).sum()),
        "n_dropped_missing_value": int((joined["ok"] & joined["logs"].isna()).sum()),
        "n_duplicates_merged": int(len(kept) - len(clean)),
        "n_clean": int(len(clean)),
    }
    return clean, stats


def clean_tox21(frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Standardise and merge duplicates. Conflicting duplicate labels become missing."""
    std = standardise_many(frame["smiles"].tolist())
    joined = pd.concat([frame.reset_index(drop=True), std], axis=1)
    kept = joined[joined["ok"]]
    grouped = kept.groupby("inchikey", sort=True)
    clean = grouped.agg(
        smiles_std=("smiles_std", "first"), **{task: (task, "mean") for task in TOX21_TASKS}
    ).reset_index()
    conflicts = 0
    for task in TOX21_TASKS:
        values = clean[task].to_numpy(dtype=float, copy=True)
        mixed = ~np.isnan(values) & (values != 0.0) & (values != 1.0)
        conflicts += int(mixed.sum())
        values[mixed] = np.nan
        clean[task] = values
    stats = {
        "n_raw": int(len(frame)),
        "n_dropped_unparsable": int((~joined["ok"]).sum()),
        "n_duplicates_merged": int(len(kept) - len(clean)),
        "n_conflicting_labels_set_missing": conflicts,
        "n_clean": int(len(clean)),
    }
    return clean, stats
