"""QML tier: one molecule in, one labelled electronic-stability payload out (MSP-27).

Owner: Nishitha. Requirements: FR-021, FR-023, FR-025, FR-027, BR-006, NFR-007, NFR-008.

The payload is the agreed contract with the aggregator (MSP-31); do not rename its fields
without telling Ayushi:

  {"tier": "qml",
   "is_stub": true,                                  true while the stub backend is in use
   "electronic_stability": {"metric": "homo_lumo_gap", "value": 4.5, "unit": "eV",
                            "limitation": "STUB: ...", "model_version": "qml_stub@1.0.0"},
   "convergence": {"converged": true, "iterations": 12, "final_change_mha": 0.9, "threshold_mha": 1.6},
   "backend": {"name": "stub", "version": "1.0.0", "is_stub": true},
   "active_space": {"electrons": 4, "orbitals": 4}}

  model_version  "qml_<backend name>@<backend version>"
  params_hash    SHA-256 of the backend identity, the convergence threshold and the payload version

A molecule that does not qualify for whole-molecule QML (more than 9 heavy atoms, or atoms other
than C, N, O, F) is refused here as well as at submit time (FR-021). A backend that is not known
raises ModelLoadError, which the task stores as MODEL_UNAVAILABLE (no fallback).
Pure RDKit logic: no Flask, database or Celery imports.
"""
# RDKit is a C extension, so pylint cannot see its members.
# pylint: disable=no-member,c-extension-no-member
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from rdkit import Chem

from backend.core import labels
from backend.core.config import Settings
from backend.registry.loader import ModelLoadError
from backend.services.molecule import eligibility
from backend.workers.ml.model_cards import stable_hash
from backend.workers.qml.backend import QuantumBackend
from backend.workers.qml.stub_backend import StubBackend

TIER = "qml"
BACKEND_ENV = "QML_BACKEND"
DEFAULT_BACKEND = "stub"
PAYLOAD_VERSION = 1
METRIC = "homo_lumo_gap"
BACKENDS: Dict[str, Callable[[], QuantumBackend]] = {"stub": StubBackend}


@dataclass(frozen=True)
class QmlResult:
    """What the task stores: the payload plus the two values saved beside it."""

    payload: Dict
    model_version: str
    params_hash: str


class QmlPredictor:
    """A quantum backend plus the settings that decide the result."""

    def __init__(self, backend: QuantumBackend, threshold_mha: float):
        self.backend = backend
        self.threshold_mha = float(threshold_mha)
        self.model_version = f"{TIER}_{backend.name}@{backend.version}"
        self.params_hash = stable_hash({
            "backend": backend.identity(),
            "threshold_mha": self.threshold_mha,
            "payload_version": PAYLOAD_VERSION,
        })

    def predict_smiles(self, smiles: str,
                       should_cancel: Optional[Callable[[], bool]] = None) -> QmlResult:
        """Estimate the gap of a stored canonical SMILES and build the labelled payload."""
        if not self.backend.is_stub:  # a real backend gets its own wording in MSP-28 (labels.py)
            raise ModelLoadError("no wording for a non-stub quantum backend yet")
        mol = Chem.MolFromSmiles(smiles or "")  # None for an unreadable structure: tier_eligibility refuses it
        if not eligibility.tier_eligibility(mol)[eligibility.TIER_QML].eligible:
            raise ValueError("the molecule does not qualify for whole-molecule QML")
        estimate = self.backend.estimate_gap(smiles, self.threshold_mha, should_cancel)
        stability = {"metric": METRIC, "value": estimate.gap_ev,
                     **labels.qml_stub_labels(self.model_version)}
        payload = {
            "tier": TIER,
            "is_stub": self.backend.is_stub,
            "electronic_stability": stability,
            "convergence": estimate.convergence.to_dict(),
            "backend": {"name": self.backend.name, "version": self.backend.version,
                        "is_stub": self.backend.is_stub},
            "active_space": {"electrons": estimate.active_electrons, "orbitals": estimate.active_orbitals},
        }
        return QmlResult(payload, self.model_version, self.params_hash)


def get_predictor(_target_id: Optional[str] = None, backend_name: Optional[str] = None,
                  settings: Optional[Settings] = None) -> QmlPredictor:
    """The predictor for the configured backend (QML_BACKEND, default 'stub')."""
    name = (backend_name or os.environ.get(BACKEND_ENV) or DEFAULT_BACKEND).strip().lower()
    factory = BACKENDS.get(name)
    if factory is None:
        raise ModelLoadError("Unknown quantum backend.")
    settings = settings or Settings.from_env()
    return QmlPredictor(factory(), settings.qml_conv_threshold_mha)
