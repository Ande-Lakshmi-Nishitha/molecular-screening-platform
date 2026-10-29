"""MOL/SDF input (MSP-20; FR-001, FR-004, BR-001).

Two steps, kept apart on purpose:

* read_first_record() is plain string handling: size check, decoding and
  picking the first record. It runs in the API process and never touches RDKit.
* parse_molblock() turns that record into a sanitised RDKit molecule. It calls
  RDKit's C++ parser, so it only runs inside the guarded child process
  (see parse_process.py).

Only the first record is used; the caller adds a warning if there were more.
"""
# RDKit is a C extension, so pylint cannot see its members.
# pylint: disable=no-member,no-name-in-module,c-extension-no-member,broad-exception-caught
from typing import Tuple

from rdkit import Chem

from backend.services.molecule.errors import InputTooLargeError
from backend.services.molecule.standardise import MoleculeError

SDF_RECORD_END = "$$$$"


def read_first_record(data: bytes, max_bytes: int) -> Tuple[str, int]:
    """Return (first MOL block as text, number of records in the file).

    Raises InputTooLargeError over max_bytes and MoleculeError for an empty,
    binary or undecodable file.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise MoleculeError("No structure was provided.")
    if len(data) > max_bytes:
        raise InputTooLargeError(
            f"The file is larger than the allowed maximum of {max_bytes} bytes."
        )
    if not bytes(data).strip():
        raise MoleculeError("The file is empty.")
    try:
        text = bytes(data).decode("utf-8")
    except UnicodeDecodeError:
        raise MoleculeError("The file is not a text MOL/SDF file.") from None
    if "\x00" in text:
        raise MoleculeError("The file is not a text MOL/SDF file.")

    records, current = [], []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line.startswith(SDF_RECORD_END):
            records.append(current)
            current = []
        else:
            current.append(line)
    if any(line.strip() for line in current):  # last record without a terminator
        records.append(current)
    records = [record for record in records if any(line.strip() for line in record)]
    if not records:
        raise MoleculeError("The file is empty.")
    return "\n".join(records[0]) + "\n", len(records)


def parse_molblock(block: str):
    """Parse one MOL block into a sanitised RDKit molecule or raise."""
    try:
        mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
    except Exception:
        mol = None
    if mol is None:
        raise MoleculeError("The file could not be parsed as a MOL/SDF record.")
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:  # RDKit raises several exception types
        raise MoleculeError(
            "The molecule failed sanitisation (invalid valence or aromaticity)."
        ) from exc
    return mol
