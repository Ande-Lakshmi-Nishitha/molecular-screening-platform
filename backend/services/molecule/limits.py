"""Input limits for molecule validation (MSP-20; FR-053, TBD-C12, SEC-004).

The values are open SRS items (TBD-C12), so they are read from configuration
and never written into the validation code:

* MAX_SMILES_LEN   longest SMILES accepted        (Settings, default 500)
* PARSE_TIMEOUT_S  hard limit for parse + standardise (Settings, default 5)
* MAX_MOLFILE_BYTES largest MOL/SDF upload accepted  (read here, default 100000)

ASSUMPTION: the SRS names no size limit for MOL/SDF files, so
MAX_MOLFILE_BYTES is a new setting with a conservative default. Log it in
docs/decisions/decision-log.md under TBD-C12.
"""
import os
from dataclasses import dataclass
from typing import Optional

from backend.core.config import Settings

DEFAULT_MAX_MOLFILE_BYTES = 100_000


@dataclass(frozen=True)
class MoleculeLimits:
    """Limits applied to every molecule input, whatever the route."""

    max_smiles_len: int
    parse_timeout_s: float
    max_molfile_bytes: int = DEFAULT_MAX_MOLFILE_BYTES

    def __post_init__(self):
        if self.max_smiles_len < 1:
            raise ValueError("max_smiles_len must be at least 1")
        if self.parse_timeout_s <= 0:
            raise ValueError("parse_timeout_s must be greater than 0")
        if self.max_molfile_bytes < 1:
            raise ValueError("max_molfile_bytes must be at least 1")

    @classmethod
    def from_env(cls, settings: Optional[Settings] = None) -> "MoleculeLimits":
        """Build the limits from the central settings and the environment."""
        settings = settings or Settings.from_env()
        molfile_bytes = os.environ.get("MAX_MOLFILE_BYTES") or DEFAULT_MAX_MOLFILE_BYTES
        return cls(
            max_smiles_len=settings.max_smiles_len,
            parse_timeout_s=settings.parse_timeout_s,
            max_molfile_bytes=int(molfile_bytes),
        )
