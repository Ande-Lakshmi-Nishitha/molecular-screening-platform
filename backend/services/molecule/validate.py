"""Shared molecule validation (MSP-20; FR-001, FR-002, FR-004, FR-053, BR-001).

validate_smiles() and validate_molfile() are the ONLY way the validate route
and the submit route (MSP-29) turn user input into a standardised molecule, so
the two routes cannot disagree. Both do the same three things:

1. cheap checks in the API process: empty input, length / size limits (413)
2. one guarded child process for RDKit parse + standardise + tier eligibility,
   under one hard time limit (PARSE_TIMEOUT)
3. a JSON-safe ValidationResult

Errors are MoleculeError (or its subclasses in errors.py); map them to HTTP
with errors.http_status_for(). Pure Python: no Flask, database or Celery.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.services.molecule import parse_process
from backend.services.molecule.errors import InputTooLargeError
from backend.services.molecule.limits import MoleculeLimits
from backend.services.molecule.molfile import read_first_record
from backend.services.molecule.standardise import MoleculeError


@dataclass(frozen=True)
class ValidationResult:
    """A standardised molecule plus per-tier eligibility, all JSON-safe."""

    canonical_smiles: str
    inchikey: str
    formula: str
    heavy_atoms: int
    warnings: List[str] = field(default_factory=list)
    tiers: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Response body for POST /api/molecules/validate (and the submit route)."""
        return {
            "canonical_smiles": self.canonical_smiles,
            "inchikey": self.inchikey,
            "formula": self.formula,
            "heavy_atoms": self.heavy_atoms,
            "warnings": list(self.warnings),
            "tiers": {tier: dict(decision) for tier, decision in self.tiers.items()},
        }

    @classmethod
    def from_child(cls, data: Dict[str, Any], extra_warnings: Optional[List[str]] = None):
        """Build from the dict returned by parse_process.compute_validation()."""
        return cls(
            canonical_smiles=data["canonical_smiles"],
            inchikey=data["inchikey"],
            formula=data["formula"],
            heavy_atoms=data["heavy_atoms"],
            warnings=list(data.get("warnings", [])) + list(extra_warnings or []),
            tiers=data["tiers"],
        )


def validate_smiles(
    smiles: Any,
    target_id: Optional[str] = None,
    limits: Optional[MoleculeLimits] = None,
) -> ValidationResult:
    """Validate and standardise a SMILES string (FR-001, FR-002, FR-053)."""
    limits = limits or MoleculeLimits.from_env()
    if not isinstance(smiles, str) or not smiles.strip():
        raise MoleculeError("No structure was provided.")
    smiles = smiles.strip()
    if len(smiles) > limits.max_smiles_len:
        raise InputTooLargeError(
            f"The SMILES string is longer than the allowed maximum of {limits.max_smiles_len} characters."
        )
    data = parse_process.run_with_timeout(
        parse_process.compute_validation,
        (parse_process.KIND_SMILES, smiles, target_id),
        limits.parse_timeout_s,
    )
    return ValidationResult.from_child(data)


def validate_molfile(
    data: bytes,
    target_id: Optional[str] = None,
    limits: Optional[MoleculeLimits] = None,
) -> ValidationResult:
    """Validate and standardise the first record of a MOL/SDF file (FR-001)."""
    limits = limits or MoleculeLimits.from_env()
    block, record_count = read_first_record(data, limits.max_molfile_bytes)
    child_result = parse_process.run_with_timeout(
        parse_process.compute_validation,
        (parse_process.KIND_MOLBLOCK, block, target_id),
        limits.parse_timeout_s,
    )
    extra = []
    if record_count > 1:
        extra.append(
            f"The file has {record_count} records; only the first one was used."
        )
    return ValidationResult.from_child(child_result, extra)
