"""Mock PubChem / ChEMBL search source for tests (FR-005, FR-008).

mode: "ok" returns golden matches, "timeout" raises TimeoutError,
"http_500" raises MockHttpError(500), "empty" returns no results.
"""
# pylint: disable=too-few-public-methods
from tests.fixtures.golden_molecules import GOLDEN

MODES = ("ok", "timeout", "http_500", "empty")


class MockHttpError(Exception):
    """Stands in for an HTTP error response from an external service."""

    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status = status


class MockSearchSource:
    """A fake external search service with a selectable failure mode."""

    def __init__(self, name="PubChem", mode="ok"):
        if mode not in MODES:
            raise ValueError(f"unknown mode: {mode}")
        self.name = name
        self.mode = mode
        self.calls = 0

    def search(self, query):
        """Return matching compounds or fail according to the mode."""
        self.calls += 1
        if self.mode == "timeout":
            raise TimeoutError(f"{self.name} timed out")
        if self.mode == "http_500":
            raise MockHttpError(500)
        if self.mode == "empty":
            return []
        needle = query.strip().lower()
        return [
            {"name": name, "smiles": data["smiles"], "source": self.name}
            for name, data in GOLDEN.items()
            if needle in name
        ]
