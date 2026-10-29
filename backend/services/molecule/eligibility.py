"""Tier eligibility (MSP-19; FR-021, BR-002, BR-003).

One function, used by the validate route, the submit route and the QML worker.
Pure RDKit logic: no Flask, database or Celery imports.

Thresholds live in EligibilityConfig (overridable from environment variables),
not in the function body.
"""
# RDKit is a C extension, so pylint cannot see its members.
# pylint: disable=no-member,no-name-in-module,c-extension-no-member
import os
from dataclasses import dataclass
from typing import Dict, FrozenSet, Mapping, Optional

TIER_ML = "ml"
TIER_DL = "dl"
TIER_QML = "qml"

SKIP_NO_TARGET = "NO_TARGET_SELECTED"
SKIP_UNSUPPORTED_ATOM = "UNSUPPORTED_ATOM"
SKIP_NO_FRAGMENT = "NO_FRAGMENT"

# FR-021: whole-molecule QML needs at most 9 heavy atoms, all from C, N, O, F.
DEFAULT_QML_MAX_HEAVY_ATOMS = 9
DEFAULT_QML_ALLOWED_ATOMS = frozenset({"C", "N", "O", "F"})

# ASSUMPTION: the SRS does not list the atoms the DL graph featuriser supports.
# Common drug-like elements are used until the DL models (MSP-25) fix the list.
DEFAULT_DL_SUPPORTED_ATOMS = frozenset(
    {"C", "N", "O", "F", "S", "P", "Cl", "Br", "I"}
)


def _atoms_from_env(value: Optional[str], default: FrozenSet[str]) -> FrozenSet[str]:
    if not value or not value.strip():
        return default
    return frozenset(part.strip() for part in value.split(",") if part.strip())


@dataclass(frozen=True)
class EligibilityConfig:
    """Eligibility limits. Defaults follow the SRS; env vars override them."""

    qml_max_heavy_atoms: int = DEFAULT_QML_MAX_HEAVY_ATOMS
    qml_allowed_atoms: FrozenSet[str] = DEFAULT_QML_ALLOWED_ATOMS
    dl_supported_atoms: FrozenSet[str] = DEFAULT_DL_SUPPORTED_ATOMS

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None):
        """Read MSP_QML_MAX_HEAVY_ATOMS, MSP_QML_ALLOWED_ATOMS, MSP_DL_SUPPORTED_ATOMS."""
        env = os.environ if environ is None else environ
        return cls(
            qml_max_heavy_atoms=int(
                env.get("MSP_QML_MAX_HEAVY_ATOMS", DEFAULT_QML_MAX_HEAVY_ATOMS)
            ),
            qml_allowed_atoms=_atoms_from_env(
                env.get("MSP_QML_ALLOWED_ATOMS"), DEFAULT_QML_ALLOWED_ATOMS
            ),
            dl_supported_atoms=_atoms_from_env(
                env.get("MSP_DL_SUPPORTED_ATOMS"), DEFAULT_DL_SUPPORTED_ATOMS
            ),
        )


@dataclass(frozen=True)
class TierDecision:
    """Whether a tier runs; if not, the skip reason code stored on the task."""

    eligible: bool
    skip_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        """JSON-safe form used in the validate and submit responses."""
        return {"eligible": self.eligible, "skip_reason": self.skip_reason}


def _heavy_atom_symbols(mol):
    """Element symbols of heavy atoms (explicit hydrogens are ignored)."""
    return [atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]


def tier_eligibility(
    mol,
    target_id: Optional[str] = None,
    config: Optional[EligibilityConfig] = None,
) -> Dict[str, TierDecision]:
    """Decide which tiers run for a valid, standardised molecule.

    ML runs for every valid molecule. DL needs a selected target and supported
    atoms. QML needs a small whole molecule (no fragment path in v1.0).
    """
    if mol is None:
        raise ValueError("tier_eligibility needs a valid molecule")
    cfg = config or EligibilityConfig()
    symbols = _heavy_atom_symbols(mol)

    if not target_id:
        dl = TierDecision(False, SKIP_NO_TARGET)
    elif not set(symbols) <= cfg.dl_supported_atoms:
        dl = TierDecision(False, SKIP_UNSUPPORTED_ATOM)
    else:
        dl = TierDecision(True)

    if len(symbols) > cfg.qml_max_heavy_atoms or not set(symbols) <= cfg.qml_allowed_atoms:
        qml = TierDecision(False, SKIP_NO_FRAGMENT)
    else:
        qml = TierDecision(True)

    return {TIER_ML: TierDecision(True), TIER_DL: dl, TIER_QML: qml}
