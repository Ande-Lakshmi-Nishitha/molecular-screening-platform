"""Convergence record and the not-converged flag (MSP-27, MSP-28; FR-025, FR-027).

Owner: Nishitha. Threshold: TBD-C6, 1.6 mHa (config QML_CONV_THRESHOLD_MHA).

A real backend (MSP-28) knows the full-CI energies of the same active space, so it reports the error
against full CI and the run is converged when that error is at or below the threshold. The stub has no
reference, so for it the old rule applies: the last change in energy between two iterations is at or below
the threshold. Either way a run that did not converge still returns its value, but the flag is false, the
result screen warns, and the aggregator leaves the QML block out of triage (FR-027).
Pure Python: no RDKit, Celery or database imports.
"""
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ConvergenceRecord:
    """How the iterations ended. error_vs_fci_mha is None when no full-CI reference exists (the stub)."""

    iterations: int
    final_change_mha: float
    threshold_mha: float
    error_vs_fci_mha: Optional[float] = None

    @property
    def converged(self) -> bool:
        """True when the error against full CI (or, without it, the last energy change) is within the threshold."""
        measure = self.final_change_mha if self.error_vs_fci_mha is None else self.error_vs_fci_mha
        return measure <= self.threshold_mha

    def to_dict(self) -> Dict[str, object]:
        """JSON-safe block stored in the QML result (field names are part of the agreed payload)."""
        block = {
            "converged": self.converged,
            "iterations": self.iterations,
            "final_change_mha": self.final_change_mha,
            "threshold_mha": self.threshold_mha,
        }
        if self.error_vs_fci_mha is not None:
            block["error_vs_fci_mha"] = self.error_vs_fci_mha
        return block
