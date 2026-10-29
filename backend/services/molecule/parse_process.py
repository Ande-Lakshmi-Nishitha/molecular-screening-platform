"""Guarded child process for RDKit work (MSP-20; FR-053, SEC-004, SEC-007).

RDKit's parser is C++, so a Python signal or thread timeout cannot interrupt a
call that hangs or crashes. Instead the work runs in a short-lived child
process with ONE hard time limit that covers parsing, standardising and the
eligibility decision together. If the limit is hit the child is killed.

* The child answers with JSON bytes, never pickled objects, so nothing is
  unpickled from a process that just parsed untrusted input.
* A crash inside the child (exit without an answer) is reported as a normal
  validation error; the structure is never logged (SEC-007). Only the exit
  code and the exception class are.
* The default start method is "spawn": it behaves the same on Windows (the
  only method there), CI and the API container, and costs about 0.1 s. Set
  MSP_PARSE_START_METHOD=fork on Linux to trade that safety for speed.

Pure Python plus RDKit: no Flask, database or Celery imports.
"""
import json
import logging
import multiprocessing
import os
from typing import Any, Callable, Dict, Optional, Tuple

from backend.services.molecule.eligibility import tier_eligibility
from backend.services.molecule.errors import ParseTimeoutError
from backend.services.molecule.molfile import parse_molblock
from backend.services.molecule.standardise import (
    MoleculeError,
    parse_smiles,
    standardise_mol,
)

LOGGER = logging.getLogger("msp.molecule")

KIND_SMILES = "smiles"
KIND_MOLBLOCK = "molblock"

START_METHOD_ENV = "MSP_PARSE_START_METHOD"
DEFAULT_START_METHOD = "spawn"

_GENERIC_FAILURE = "The structure could not be processed."


def compute_validation(
    kind: str, payload: str, target_id: Optional[str] = None
) -> Dict[str, Any]:
    """Parse, standardise and decide tier eligibility. JSON-safe result.

    This is the function that runs inside the child process. It is the single
    code path behind both the validate and the submit route.
    """
    if kind == KIND_SMILES:
        mol = parse_smiles(payload)
    elif kind == KIND_MOLBLOCK:
        mol = parse_molblock(payload)
    else:
        raise ValueError(f"unknown input kind: {kind}")
    standardised = standardise_mol(mol)
    tiers = tier_eligibility(standardised.mol, target_id)
    result = standardised.to_dict()
    result["tiers"] = {tier: decision.to_dict() for tier, decision in tiers.items()}
    return result


def _child_main(conn, func: Callable[..., Dict[str, Any]], args: Tuple[Any, ...]) -> None:
    """Child process entry point (module level so 'spawn' can import it)."""
    try:
        message = {"status": "ok", "result": func(*args)}
    except MoleculeError as exc:
        message = {"status": "error", "code": exc.code, "message": exc.message}
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # The message of an unexpected exception could contain the structure.
        message = {"status": "crash", "exception": type(exc).__name__}
    try:
        conn.send_bytes(json.dumps(message).encode("utf-8"))
    finally:
        conn.close()


def start_method() -> str:
    """Start method for the child process (env override, default 'spawn')."""
    method = os.environ.get(START_METHOD_ENV) or DEFAULT_START_METHOD
    if method not in multiprocessing.get_all_start_methods():
        raise ValueError(f"{START_METHOD_ENV}={method!r} is not available on this system")
    return method


def run_with_timeout(
    func: Callable[..., Dict[str, Any]],
    args: Tuple[Any, ...],
    timeout_s: float,
) -> Dict[str, Any]:
    """Run func(*args) in a child process and return its JSON-safe result.

    Raises ParseTimeoutError if the child is still running after timeout_s (it
    is killed), and MoleculeError if the child reports a validation error or
    dies without answering.
    """
    ctx = multiprocessing.get_context(start_method())
    receiver, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_child_main, args=(sender, func, args), daemon=True)
    process.start()
    sender.close()  # the parent only reads; lets recv see EOF if the child dies
    raw = None
    timed_out = False
    try:
        if receiver.poll(timeout_s):
            try:
                raw = receiver.recv_bytes()
            except EOFError:
                raw = None  # child ended without an answer (crash)
        else:
            timed_out = True
    finally:
        receiver.close()
        if process.is_alive():
            process.kill()
        process.join(timeout=5)

    if timed_out:
        raise ParseTimeoutError()
    if raw is None:
        LOGGER.warning("molecule child process ended without a result exitcode=%s", process.exitcode)
        raise MoleculeError(_GENERIC_FAILURE)

    message = json.loads(raw.decode("utf-8"))
    status = message.get("status")
    if status == "ok":
        return message["result"]
    if status == "error":
        raise MoleculeError(message["message"], message["code"])
    LOGGER.warning("molecule child process raised %s", message.get("exception"))
    raise MoleculeError(_GENERIC_FAILURE)
