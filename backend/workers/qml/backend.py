"""QuantumBackend interface: estimate_gap() (MSP-27; FR-023, TBD-C5).

Owner: Nishitha. Every quantum backend (the stub now, a real VQE backend in MSP-28) implements
this one method. The worker and the result payload only know this interface, so swapping the
backend changes no other code. TBD-C5 (which quantum framework) stays open until MSP-28.

estimate_gap() returns a GapEstimate. It calls should_cancel() once per iteration, if given, and
stops with QuantumCancelled when it returns true (FR-029, NFR-013).
Pure Python: no RDKit, Celery or database imports.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from backend.workers.qml.convergence import ConvergenceRecord


class QuantumCancelled(Exception):
    """The task was cancelled while the backend was iterating."""


@dataclass(frozen=True)
class GapEstimate:
    """One backend answer: the gap in eV, how the iterations ended and the active space."""

    gap_ev: float
    convergence: ConvergenceRecord
    active_electrons: int
    active_orbitals: int


class QuantumBackend(ABC):
    """What a quantum backend must provide."""

    name = "backend"
    version = "0"
    is_stub = False

    @abstractmethod
    def estimate_gap(self, smiles: str, threshold_mha: float,
                     should_cancel: Optional[Callable[[], bool]] = None) -> GapEstimate:
        """Estimate the HOMO-LUMO gap (eV) of a small whole molecule."""

    def identity(self) -> Dict[str, object]:
        """Fields that decide the output; they go into the params_hash."""
        return {"name": self.name, "version": self.version, "is_stub": self.is_stub}
