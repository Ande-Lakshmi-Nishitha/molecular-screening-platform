"""Stub quantum backend: returns fixed values, runs no calculation (MSP-27; QA-TST, R01, TBD-C5).

Owner: Nishitha. It exists so the whole pipeline (queue, worker, result, triage, screen) can be
built and tested before a real quantum backend (MSP-28). It returns the SAME gap for every
molecule, so its value carries no information about the molecule; every place that shows it must
say so (is_stub in the payload, "STUB:" in the limitation, "qml_stub" in the model version).
It does loop over its iterations and checks for cancellation in each one, like a real backend.
"""
from typing import Callable, Optional

from backend.workers.qml.backend import GapEstimate, QuantumBackend, QuantumCancelled
from backend.workers.qml.convergence import ConvergenceRecord

STUB_GAP_EV = 4.5
STUB_ITERATIONS = 12
STUB_FINAL_CHANGE_MHA = 0.9
STUB_ACTIVE_ELECTRONS = 4   # default active space from TBD-C6
STUB_ACTIVE_ORBITALS = 4


class StubBackend(QuantumBackend):
    """Fixed demonstration values. The final change can be raised in tests to give 'not converged'."""

    name = "stub"
    version = "1.0.0"
    is_stub = True

    def __init__(self, gap_ev: float = STUB_GAP_EV, iterations: int = STUB_ITERATIONS,
                 final_change_mha: float = STUB_FINAL_CHANGE_MHA):
        self.gap_ev = gap_ev
        self.iterations = iterations
        self.final_change_mha = final_change_mha

    def estimate_gap(self, smiles: str, threshold_mha: float,
                     should_cancel: Optional[Callable[[], bool]] = None) -> GapEstimate:
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError("a structure is required")
        for _ in range(self.iterations):
            if should_cancel is not None and should_cancel():
                raise QuantumCancelled("cancelled")
        record = ConvergenceRecord(self.iterations, self.final_change_mha, threshold_mha)
        return GapEstimate(self.gap_ev, record, STUB_ACTIVE_ELECTRONS, STUB_ACTIVE_ORBITALS)

    def identity(self):
        return {**super().identity(), "gap_ev": self.gap_ev, "iterations": self.iterations,
                "final_change_mha": self.final_change_mha}
