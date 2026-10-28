"""Molecule standardisation (MSP-19; FR-002, FR-003, BR-001).

Pure RDKit code: no Flask, database or Celery imports, so the validate route,
the submit route and the QML worker can all import the same functions.

The hard time limit and the child process that guard RDKit's C++ code belong
to the validate endpoint (MSP-20, FR-053). That endpoint calls
standardise_smiles() inside the child process.
"""
# RDKit is a C extension, so pylint cannot see its members.
# pylint: disable=no-member,no-name-in-module,c-extension-no-member
from dataclasses import dataclass, field
from typing import Any, Dict, List

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

ERROR_CODE = "VALIDATION_ERROR"


class MoleculeError(ValueError):
    """Raised when input is not a structurally valid molecule (HTTP 422)."""

    def __init__(self, message: str, code: str = ERROR_CODE):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class StandardisedMolecule:
    """Result of standardisation. `mol` is the RDKit object for later steps."""

    canonical_smiles: str
    inchikey: str
    formula: str
    heavy_atoms: int
    warnings: List[str] = field(default_factory=list)
    mol: Any = field(default=None, compare=False, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe form (no RDKit object) for API responses and processes."""
        return {
            "canonical_smiles": self.canonical_smiles,
            "inchikey": self.inchikey,
            "formula": self.formula,
            "heavy_atoms": self.heavy_atoms,
            "warnings": list(self.warnings),
        }


def parse_smiles(smiles: str):
    """Parse a SMILES string into a sanitised RDKit molecule or raise."""
    if smiles is None or not str(smiles).strip():
        raise MoleculeError("No structure was provided.")
    mol = Chem.MolFromSmiles(str(smiles).strip(), sanitize=False)
    if mol is None:
        raise MoleculeError("The SMILES string could not be parsed.")
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:  # RDKit raises several exception types
        raise MoleculeError(
            "The molecule failed sanitisation (invalid valence or aromaticity)."
        ) from exc
    return mol


def _keep_largest_fragment(mol, warnings: List[str]):
    """Salt stripping: keep the largest fragment, list the rest as a warning."""
    fragments = Chem.GetMolFrags(mol, asMols=True)
    if len(fragments) <= 1:
        return mol
    kept = rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(mol)
    removed = [Chem.MolToSmiles(fragment) for fragment in fragments]
    try:
        removed.remove(Chem.MolToSmiles(kept))
    except ValueError:
        pass  # kept fragment spelled differently; still report the others
    warnings.append(
        "Kept the largest fragment; removed: " + ", ".join(sorted(removed))
    )
    return kept


def _canonical_tautomer(mol):
    """Canonical tautomer that keeps stereochemistry (FR-003: isomeric SMILES).

    RDKit's default removes stereocentres while canonicalising tautomers,
    which would give both enantiomers of a molecule the same InChIKey.
    """
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetRemoveSp3Stereo(False)
    enumerator.SetRemoveBondStereo(False)
    return enumerator.Canonicalize(mol)


def standardise_mol(mol) -> StandardisedMolecule:
    """Sanitise, keep largest fragment, neutralise, canonical tautomer."""
    if mol is None or mol.GetNumAtoms() == 0:
        raise MoleculeError("No structure was provided.")
    warnings: List[str] = []
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = _keep_largest_fragment(mol, warnings)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        mol = _canonical_tautomer(mol)
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        inchikey = Chem.MolToInchiKey(mol)
    except Exception as exc:
        raise MoleculeError("The molecule could not be standardised.") from exc
    if not smiles or not inchikey:
        raise MoleculeError("The molecule could not be standardised.")
    return StandardisedMolecule(
        canonical_smiles=smiles,
        inchikey=inchikey,
        formula=rdMolDescriptors.CalcMolFormula(mol),
        heavy_atoms=mol.GetNumHeavyAtoms(),
        warnings=warnings,
        mol=mol,
    )


def standardise_smiles(smiles: str) -> StandardisedMolecule:
    """Single entry point used by both the validate and the submit route."""
    return standardise_mol(parse_smiles(smiles))
