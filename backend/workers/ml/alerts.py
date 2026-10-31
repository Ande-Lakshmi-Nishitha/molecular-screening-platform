"""Structural alerts (PAINS, Brenk) and property rules (Lipinski, Veber) (MSP-24).

Owner: Nishitha. Requirements: FR-013, NFR-007, NFR-008, NFR-011.

structural_alerts(mol) lists the PAINS and Brenk patterns a molecule matches, using the filter
catalogs that ship with RDKit (so the pattern lists are tied to the RDKit version, which goes
into params_hash). property_rules(mol) counts rule-of-five and Veber violations. Both only
describe the molecule: neither says it is safe, toxic or effective (NFR-011).
The limits are constants here, not magic numbers in the code, and they are part of identity().
Pure RDKit: no Flask, database or Celery imports.
"""
# RDKit is a C extension, so pylint cannot see its members.
# pylint: disable=no-member,no-name-in-module,c-extension-no-member
from typing import Dict, List

from rdkit import rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from backend.core import labels

CATALOG_PAINS = "pains"
CATALOG_BRENK = "brenk"

# Lipinski rule of five: a molecule above any of these limits has a violation.
LIPINSKI_LIMITS = {"mol_wt": 500.0, "log_p": 5.0, "h_bond_donors": 5, "h_bond_acceptors": 10}
# Veber: more rotatable bonds or a larger polar surface than these is a violation.
VEBER_LIMITS = {"rotatable_bonds": 10, "tpsa": 140.0}

_CATALOGS: Dict[str, FilterCatalog] = {}


def _catalog(name: str) -> FilterCatalog:
    """The RDKit filter catalog for 'pains' or 'brenk' (built once per process)."""
    if name not in _CATALOGS:
        params = FilterCatalogParams()
        which = FilterCatalogParams.FilterCatalogs
        params.AddCatalog(which.PAINS if name == CATALOG_PAINS else which.BRENK)
        _CATALOGS[name] = FilterCatalog(params)
    return _CATALOGS[name]


def _matches(mol, name: str) -> List[str]:
    """Names of the matching patterns, without repeats, in alphabetical order."""
    return sorted({entry.GetDescription() for entry in _catalog(name).GetMatches(mol)})


def model_version() -> str:
    """Version tag stored in the alert and rule blocks: the RDKit release that supplies the lists."""
    return f"rdkit@{rdBase.rdkitVersion}"


def identity() -> Dict[str, object]:
    """Everything that decides the output of this module (goes into params_hash)."""
    return {"rdkit": rdBase.rdkitVersion, "lipinski": LIPINSKI_LIMITS, "veber": VEBER_LIMITS,
            "catalogs": [CATALOG_PAINS, CATALOG_BRENK]}


def structural_alerts(mol) -> Dict[str, object]:
    """PAINS and Brenk matches for one molecule, labelled."""
    pains, brenk = _matches(mol, CATALOG_PAINS), _matches(mol, CATALOG_BRENK)
    return {
        "pains": pains,
        "brenk": brenk,
        "n_pains": len(pains),
        "n_brenk": len(brenk),
        **labels.alerts_labels(model_version()),
    }


def _rule_block(values: Dict[str, float], limits: Dict[str, float]) -> Dict[str, object]:
    failed = [name for name, limit in limits.items() if values[name] > limit]
    return {"values": values, "limits": dict(limits), "failed": failed, "violations": len(failed)}


def property_rules(mol) -> Dict[str, object]:
    """Lipinski and Veber results for one molecule, labelled."""
    lipinski_values = {
        "mol_wt": float(Descriptors.MolWt(mol)),
        "log_p": float(Crippen.MolLogP(mol)),
        "h_bond_donors": int(Lipinski.NumHDonors(mol)),
        "h_bond_acceptors": int(Lipinski.NumHAcceptors(mol)),
    }
    veber_values = {
        "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
    }
    return {
        "lipinski": _rule_block(lipinski_values, LIPINSKI_LIMITS),
        "veber": _rule_block(veber_values, VEBER_LIMITS),
        **labels.rules_labels(model_version()),
    }
