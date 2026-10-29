"""Error types for molecule input (MSP-20; FR-004, FR-053, SEC-004, SEC-007).

All three codes are MoleculeError subclasses or instances, so a route needs one
`except MoleculeError` and one lookup in HTTP_STATUS. Messages are fixed
sentences: they never contain the submitted structure (SEC-007).

Pure Python: no Flask, RDKit, database or Celery imports.
"""
from backend.services.molecule.standardise import ERROR_CODE, MoleculeError

CODE_VALIDATION = ERROR_CODE  # "VALIDATION_ERROR"
CODE_TOO_LARGE = "INPUT_TOO_LARGE"
CODE_PARSE_TIMEOUT = "PARSE_TIMEOUT"

# SAD section 9.1: 413 INPUT_TOO_LARGE, 422 VALIDATION_ERROR / PARSE_TIMEOUT.
HTTP_STATUS = {
    CODE_VALIDATION: 422,
    CODE_TOO_LARGE: 413,
    CODE_PARSE_TIMEOUT: 422,
}


class InputTooLargeError(MoleculeError):
    """The SMILES string or the uploaded file is over the configured limit (413)."""

    def __init__(self, message: str):
        super().__init__(message, CODE_TOO_LARGE)


class ParseTimeoutError(MoleculeError):
    """Parsing or standardising ran past the hard time limit (422)."""

    def __init__(self, message: str = "Processing the structure took too long and was stopped."):
        super().__init__(message, CODE_PARSE_TIMEOUT)


def http_status_for(error: MoleculeError) -> int:
    """HTTP status for a MoleculeError; anything unknown is a 422."""
    return HTTP_STATUS.get(error.code, 422)
