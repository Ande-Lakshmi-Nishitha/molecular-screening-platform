"""Unit tests for shared molecule validation (MSP-20).

Covers FR-001, FR-002, FR-004, FR-053, BR-001: the code behind
POST /api/molecules/validate and the submit route. Real child processes are
used for the end-to-end cases (default start method: spawn).

Run from the repo root with:  python -m unittest discover -s tests -t .
"""
# RDKit is a C extension, so pylint cannot see its members.
# pylint: disable=no-member,c-extension-no-member,missing-function-docstring,missing-class-docstring
import json
import os
import time
import unittest
from unittest import mock

from rdkit import Chem
from rdkit.Chem import AllChem

from backend.services.molecule import parse_process
from backend.services.molecule.errors import (
    CODE_PARSE_TIMEOUT,
    CODE_TOO_LARGE,
    InputTooLargeError,
    ParseTimeoutError,
    http_status_for,
)
from backend.services.molecule.limits import DEFAULT_MAX_MOLFILE_BYTES, MoleculeLimits
from backend.services.molecule.standardise import ERROR_CODE, MoleculeError, standardise_smiles
from backend.services.molecule.validate import ValidationResult, validate_molfile, validate_smiles

TARGET = "CHEMBL203"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ASPIRIN_KEY = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"

LIMITS = MoleculeLimits(max_smiles_len=50, parse_timeout_s=30, max_molfile_bytes=100_000)


def molblock_of(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    return Chem.MolToMolBlock(mol)


def mol_file(smiles: str) -> bytes:
    return molblock_of(smiles).encode("utf-8")


def sdf_file(*smiles_list: str) -> bytes:
    return "".join(molblock_of(s) + "$$$$\n" for s in smiles_list).encode("utf-8")


def slow_compute(*_args):
    """Stands in for compute_validation to force the time limit (module level for spawn)."""
    time.sleep(60)
    return {}


class ValidateSmilesTests(unittest.TestCase):
    def test_fr002_aspirin_golden_values_and_tiers(self):
        result = validate_smiles(ASPIRIN, TARGET, LIMITS)
        self.assertEqual(result.canonical_smiles, ASPIRIN)
        self.assertEqual(result.inchikey, ASPIRIN_KEY)
        self.assertEqual(result.formula, "C9H8O4")
        self.assertEqual(result.heavy_atoms, 13)
        self.assertEqual(result.warnings, [])
        self.assertEqual(result.tiers["ml"], {"eligible": True, "skip_reason": None})
        self.assertEqual(result.tiers["dl"], {"eligible": True, "skip_reason": None})
        self.assertEqual(
            result.tiers["qml"], {"eligible": False, "skip_reason": "NO_FRAGMENT"}
        )

    def test_fr021_ethanol_is_eligible_for_every_tier(self):
        result = validate_smiles("CCO", TARGET, LIMITS)
        self.assertEqual(result.inchikey, "LFQSCWFLJHTTHZ-UHFFFAOYSA-N")
        for tier in ("ml", "dl", "qml"):
            self.assertTrue(result.tiers[tier]["eligible"], tier)

    def test_br002_no_target_skips_dl_with_a_reason(self):
        result = validate_smiles("CCO", None, LIMITS)
        self.assertEqual(
            result.tiers["dl"], {"eligible": False, "skip_reason": "NO_TARGET_SELECTED"}
        )
        self.assertTrue(result.tiers["ml"]["eligible"])

    def test_fr003_salt_is_stripped_and_listed_as_a_warning(self):
        result = validate_smiles("CC(=O)[O-].[Na+]", None, LIMITS)
        self.assertEqual(result.canonical_smiles, "CC(=O)O")
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("removed", result.warnings[0])

    def test_fr003_stereocentre_is_kept(self):
        result = validate_smiles("C[C@H](N)C(=O)O", None, LIMITS)
        self.assertIn("@", result.canonical_smiles)

    def test_fr003_permanent_charge_is_kept(self):
        result = validate_smiles("C[N+](C)(C)C", None, LIMITS)
        self.assertIn("+", result.canonical_smiles)

    def test_whitespace_around_the_smiles_is_ignored(self):
        self.assertEqual(
            validate_smiles("  CCO \n", None, LIMITS).canonical_smiles, "CCO"
        )

    def test_to_dict_is_json_safe_and_has_the_documented_keys(self):
        body = validate_smiles("CCO", TARGET, LIMITS).to_dict()
        self.assertEqual(
            set(body),
            {"canonical_smiles", "inchikey", "formula", "heavy_atoms", "warnings", "tiers"},
        )
        self.assertEqual(json.loads(json.dumps(body)), body)


class LimitTests(unittest.TestCase):
    def test_fr053_over_length_smiles_is_413_and_never_reaches_rdkit(self):
        limits = MoleculeLimits(max_smiles_len=10, parse_timeout_s=30)
        with mock.patch.object(parse_process, "run_with_timeout") as runner:
            with self.assertRaises(InputTooLargeError) as ctx:
                validate_smiles("C" * 11, None, limits)
        runner.assert_not_called()
        self.assertEqual(ctx.exception.code, CODE_TOO_LARGE)
        self.assertEqual(http_status_for(ctx.exception), 413)
        self.assertNotIn("CCCC", ctx.exception.message)

    def test_fr053_smiles_exactly_at_the_limit_is_accepted(self):
        limits = MoleculeLimits(max_smiles_len=10, parse_timeout_s=30)
        result = validate_smiles("C" * 10, None, limits)
        self.assertEqual(result.heavy_atoms, 10)

    def test_fr053_hard_time_limit_stops_a_slow_parse(self):
        limits = MoleculeLimits(max_smiles_len=50, parse_timeout_s=0.5)
        started = time.monotonic()
        with mock.patch.object(parse_process, "compute_validation", slow_compute):
            with self.assertRaises(ParseTimeoutError) as ctx:
                validate_smiles("CCO", None, limits)
        self.assertLess(time.monotonic() - started, 10)
        self.assertEqual(ctx.exception.code, CODE_PARSE_TIMEOUT)
        self.assertEqual(http_status_for(ctx.exception), 422)

    def test_fr053_limits_come_from_the_environment(self):
        env = {"MAX_SMILES_LEN": "123", "PARSE_TIMEOUT_S": "7", "MAX_MOLFILE_BYTES": "999"}
        with mock.patch.dict(os.environ, env):
            limits = MoleculeLimits.from_env()
        self.assertEqual(
            (limits.max_smiles_len, limits.parse_timeout_s, limits.max_molfile_bytes),
            (123, 7, 999),
        )

    def test_fr053_limits_have_defaults_when_the_environment_is_empty(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for name in ("MAX_SMILES_LEN", "PARSE_TIMEOUT_S", "MAX_MOLFILE_BYTES"):
                os.environ.pop(name, None)
            limits = MoleculeLimits.from_env()
        self.assertEqual(limits.max_smiles_len, 500)
        self.assertEqual(limits.parse_timeout_s, 5)
        self.assertEqual(limits.max_molfile_bytes, DEFAULT_MAX_MOLFILE_BYTES)

    def test_fr053_nonsense_limits_are_refused(self):
        for kwargs in (
            {"max_smiles_len": 0, "parse_timeout_s": 5},
            {"max_smiles_len": 10, "parse_timeout_s": 0},
            {"max_smiles_len": 10, "parse_timeout_s": 5, "max_molfile_bytes": 0},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    MoleculeLimits(**kwargs)


class InvalidInputTests(unittest.TestCase):
    def test_fr004_invalid_smiles_is_422_with_a_short_message(self):
        for smiles in ("C(C", "not a smiles", "C(C)(C)(C)(C)C", "c1ccc"):
            with self.subTest(smiles=smiles):
                with self.assertRaises(MoleculeError) as ctx:
                    validate_smiles(smiles, None, LIMITS)
                self.assertEqual(ctx.exception.code, ERROR_CODE)
                self.assertEqual(http_status_for(ctx.exception), 422)
                self.assertLess(len(ctx.exception.message), 120)
                self.assertNotIn(smiles, ctx.exception.message)

    def test_br001_missing_or_empty_input_is_rejected(self):
        for value in (None, "", "   ", 123, ["CCO"]):
            with self.subTest(value=value):
                with self.assertRaises(MoleculeError) as ctx:
                    validate_smiles(value, None, LIMITS)
                self.assertEqual(ctx.exception.code, ERROR_CODE)


class ValidateMolfileTests(unittest.TestCase):
    def test_fr001_ethanol_file_gives_the_same_result_as_the_string(self):
        from_file = validate_molfile(mol_file("CCO"), TARGET, LIMITS)
        from_string = validate_smiles("CCO", TARGET, LIMITS)
        self.assertEqual(from_file.to_dict(), from_string.to_dict())

    def test_fr001_acetamide_file_gives_the_same_result_as_the_string(self):
        from_file = validate_molfile(mol_file("CC(=O)N"), TARGET, LIMITS)
        from_string = validate_smiles("CC(=O)N", TARGET, LIMITS)
        self.assertEqual(from_file.canonical_smiles, from_string.canonical_smiles)
        self.assertEqual(from_file.inchikey, from_string.inchikey)
        self.assertEqual(from_file.tiers, from_string.tiers)

    def test_fr001_file_result_matches_standardise_smiles(self):
        direct = standardise_smiles("CC(=O)N")
        result = validate_molfile(mol_file("CC(=O)N"), None, LIMITS)
        self.assertEqual(result.canonical_smiles, direct.canonical_smiles)

    def test_fr001_sdf_uses_the_first_record_and_says_so(self):
        result = validate_molfile(sdf_file("CCO", "CC(=O)N"), None, LIMITS)
        self.assertEqual(result.canonical_smiles, "CCO")
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("2 records", result.warnings[0])
        self.assertIn("first", result.warnings[0])

    def test_fr001_single_record_sdf_has_no_warning(self):
        self.assertEqual(validate_molfile(sdf_file("CCO"), None, LIMITS).warnings, [])

    def test_fr004_invalid_file_is_422(self):
        for data in (b"not a molecule file\n", b"", b"\x00\x01\x02"):
            with self.subTest(data=data):
                with self.assertRaises(MoleculeError) as ctx:
                    validate_molfile(data, None, LIMITS)
                self.assertEqual(ctx.exception.code, ERROR_CODE)
                self.assertEqual(http_status_for(ctx.exception), 422)

    def test_fr053_oversized_file_is_413_and_never_reaches_rdkit(self):
        data = mol_file("CCO")
        limits = MoleculeLimits(max_smiles_len=50, parse_timeout_s=30, max_molfile_bytes=len(data) - 1)
        with mock.patch.object(parse_process, "run_with_timeout") as runner:
            with self.assertRaises(InputTooLargeError) as ctx:
                validate_molfile(data, None, limits)
        runner.assert_not_called()
        self.assertEqual(http_status_for(ctx.exception), 413)

    def test_fr053_slow_file_parse_hits_the_same_time_limit(self):
        limits = MoleculeLimits(max_smiles_len=50, parse_timeout_s=0.5)
        with mock.patch.object(parse_process, "compute_validation", slow_compute):
            with self.assertRaises(ParseTimeoutError):
                validate_molfile(mol_file("CCO"), None, limits)


class ValidationResultTests(unittest.TestCase):
    def test_from_child_appends_extra_warnings_after_the_child_ones(self):
        data = {
            "canonical_smiles": "CCO",
            "inchikey": "k",
            "formula": "C2H6O",
            "heavy_atoms": 3,
            "warnings": ["first"],
            "tiers": {},
        }
        result = ValidationResult.from_child(data, ["second"])
        self.assertEqual(result.warnings, ["first", "second"])


if __name__ == "__main__":
    unittest.main()
