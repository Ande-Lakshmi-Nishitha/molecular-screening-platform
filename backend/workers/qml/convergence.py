"""Convergence record and the not-converged flag (MSP-27; FR-025, FR-027).

Owner: Nishitha. A quantum run is converged when the last change in energy between two
iterations is at or below the threshold (TBD-C6: 1.6 mHa, config QML_CONV_THRESHOLD_MHA).
A run that did not converge still returns its value, but the flag is false so that the
result screen can warn about it.
Pure Python: no RDKit, Celery or database imports.
"""
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ConvergenceRecord:
    """How the iterations ended."""

    iterations: int
    final_change_mha: float
    threshold_mha: float

    @property
    def converged(self) -> bool:
        """True when the last energy change is at or below the threshold."""
        return self.final_change_mha <= self.threshold_mha

    def to_dict(self) -> Dict[str, object]:
        """JSON-safe block stored in the QML result (field names are part of the agreed payload)."""
        return {
            "converged": self.converged,
            "iterations": self.iterations,
            "final_change_mha": self.final_change_mha,
            "threshold_mha": self.threshold_mha,
        }
