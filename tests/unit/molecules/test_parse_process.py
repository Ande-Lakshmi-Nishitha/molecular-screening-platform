"""Unit tests for the guarded child process (MSP-20; FR-053, SEC-004, SEC-007).

These tests start real child processes (default start method: spawn), so the
helper functions below are defined at module level: a spawned child must be
able to import them by name.

Run from the repo root with:  python -m unittest discover -s tests -t .
"""
# RDKit is a C extension, so pylint cannot see its members.
# pylint: disable=no-member,c-extension-no-member,missing-function-docstring,missing-class-docstring
import os
import tempfile
import time
import unittest
from unittest import mock

from backend.services.molecule import parse_process
from backend.services.molecule.errors import CODE_PARSE_TIMEOUT, ParseTimeoutError, http_status_for
from backend.services.molecule.standardise import ERROR_CODE, MoleculeError

SECRET = "SECRET-STRUCTURE-TEXT"


def echo(value):
    return {"echo": value}


def sleep_forever(_value):
    time.sleep(60)
    return {}


def heartbeat(path):
    """Write a counter to `path` every 50 ms until killed (shows if the child is alive)."""
    count = 0
    while True:
        count += 1
        # write-then-rename, so a kill mid-write can never leave an empty file
        with open(path + ".tmp", "w", encoding="utf-8") as handle:
            handle.write(str(count))
        os.replace(path + ".tmp", path)
        time.sleep(0.05)


def raise_with_secret(_value):
    raise RuntimeError(SECRET)


def die_without_answer(_value):
    os._exit(1)  # pylint: disable=protected-access


class ComputeValidationTests(unittest.TestCase):
    """compute_validation runs in the child; test it in-process (fast)."""

    def test_fr002_smiles_result_has_molecule_and_all_three_tiers(self):
        result = parse_process.compute_validation(
            parse_process.KIND_SMILES, "CC(=O)Oc1ccccc1C(=O)O", "CHEMBL203"
        )
        self.assertEqual(result["inchikey"], "BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
        self.assertEqual(set(result["tiers"]), {"ml", "dl", "qml"})
        self.assertTrue(result["tiers"]["ml"]["eligible"])
        self.assertTrue(result["tiers"]["dl"]["eligible"])

    def test_fr021_aspirin_is_too_big_for_qml(self):
        result = parse_process.compute_validation(parse_process.KIND_SMILES, "CC(=O)Oc1ccccc1C(=O)O")
        self.assertEqual(result["tiers"]["qml"], {"eligible": False, "skip_reason": "NO_FRAGMENT"})

    def test_fr001_unknown_kind_is_a_programming_error(self):
        with self.assertRaises(ValueError):
            parse_process.compute_validation("nonsense", "CCO")


class RunWithTimeoutTests(unittest.TestCase):
    def test_fr053_child_result_is_returned(self):
        self.assertEqual(
            parse_process.run_with_timeout(echo, ("x",), 20), {"echo": "x"}
        )

    def test_fr053_hard_timeout_raises_at_the_configured_limit(self):
        started = time.monotonic()
        with self.assertRaises(ParseTimeoutError) as ctx:
            parse_process.run_with_timeout(sleep_forever, ("x",), 0.5)
        elapsed = time.monotonic() - started
        self.assertEqual(ctx.exception.code, CODE_PARSE_TIMEOUT)
        self.assertEqual(http_status_for(ctx.exception), 422)
        self.assertLess(elapsed, 2.5, "the call did not stop at the 0.5 s limit")

    def test_fr053_hard_timeout_really_kills_the_child(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "beat.txt")
            with self.assertRaises(ParseTimeoutError):
                parse_process.run_with_timeout(heartbeat, (path,), 1.0)
            with open(path, encoding="utf-8") as handle:
                after_timeout = handle.read()
            self.assertGreater(int(after_timeout), 0, "the child never started")
            time.sleep(0.5)  # a living child would add about ten more beats
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), after_timeout, "the child is still running")

    def test_fr004_validation_error_in_the_child_keeps_its_code(self):
        with self.assertRaises(MoleculeError) as ctx:
            parse_process.run_with_timeout(
                parse_process.compute_validation,
                (parse_process.KIND_SMILES, "C(C", None),
                20,
            )
        self.assertEqual(ctx.exception.code, ERROR_CODE)

    def test_sec007_unexpected_exception_text_never_reaches_the_caller_or_the_log(self):
        with self.assertLogs("msp.molecule", level="WARNING") as logs:
            with self.assertRaises(MoleculeError) as ctx:
                parse_process.run_with_timeout(raise_with_secret, ("x",), 20)
        self.assertNotIn(SECRET, str(ctx.exception))
        self.assertNotIn(SECRET, ctx.exception.message)
        self.assertNotIn(SECRET, "\n".join(logs.output))
        self.assertIn("RuntimeError", "\n".join(logs.output))

    def test_fr053_child_that_dies_without_answering_is_a_validation_error(self):
        with self.assertLogs("msp.molecule", level="WARNING"):
            with self.assertRaises(MoleculeError) as ctx:
                parse_process.run_with_timeout(die_without_answer, ("x",), 20)
        self.assertEqual(ctx.exception.code, ERROR_CODE)


class StartMethodTests(unittest.TestCase):
    def test_default_start_method_is_spawn(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(parse_process.START_METHOD_ENV, None)
            self.assertEqual(parse_process.start_method(), "spawn")

    def test_unknown_start_method_is_rejected(self):
        with mock.patch.dict(os.environ, {parse_process.START_METHOD_ENV: "teleport"}):
            with self.assertRaises(ValueError):
                parse_process.start_method()


if __name__ == "__main__":
    unittest.main()
