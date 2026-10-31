"""QuantumBackend interface: estimate_gap() (MSP-27; FR-023, TBD-C5).

Owner: Nishitha. Every quantum backend (the stub, and the PennyLane VQE backend of MSP-28) implements
this one method. The worker and the result payload only know this interface, so swapping the
backend changes no other code. TBD-C5 (which quantum framework) was decided in MSP-28: PennyLane.

estimate_gap() returns a GapEstimate. It calls should_cancel() once per iteration, if given, and
stops with QuantumCancelled when it returns true (FR-029, NFR-013).
Pure Python: no RDKit, Celery or database imports.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

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
    details: Optional[Dict[str, Any]] = None   # real backends: classical references, run settings, energy history


class QuantumBackend(ABC):
    """What a quantum backend must provide."""

    name = "backend"
    version = "0"
    is_stub = False

    @abstractmethod
    def estimate_gap(self, smiles: str, threshold_mha: float,
                     should_cancel: Optional[Callable[[], bool]] = None) -> GapEstimate:
        """Estimate the gap (eV) of a small whole molecule: the stub a fixed number, a real backend the VQE excitation gap."""

    def identity(self) -> Dict[str, object]:
        """Fields that decide the output; they go into the params_hash."""
        return {"name": self.name, "version": self.version, "is_stub": self.is_stub}
