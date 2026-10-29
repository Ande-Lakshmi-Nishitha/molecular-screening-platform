"""Stub QML backend for tests (FR-021 to FR-027).

Deterministic and fast. Results are labelled as a stub so no test can mistake
them for a real quantum calculation. The production stub is Nishitha's
backend/workers/qml/stub_backend.py; this one only serves the test suites.
"""
# pylint: disable=too-few-public-methods
STUB_GAP_HARTREE = 0.5


class StubQmlBackend:
    """A fake quantum backend that returns a fixed excitation gap."""

    name = "stub"

    def __init__(self, fail_with=None):
        self.fail_with = fail_with

    def excitation_gap(self, smiles):
        """Return a fixed, clearly labelled result, or raise fail_with."""
        if self.fail_with is not None:
            raise self.fail_with
        return {
            "backend": self.name,
            "smiles_length": len(smiles),
            "excitation_gap_hartree": STUB_GAP_HARTREE,
            "label": "stub result, not a quantum calculation",
            "converged": True,
        }
