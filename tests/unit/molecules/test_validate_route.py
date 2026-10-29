"""API tests for POST /api/molecules/validate (MSP-20).

Covers FR-001, FR-002, FR-004, FR-053, SEC-001, SEC-004, SEC-007, BR-001 through
the real Flask app from create_app(), with a real signed JWT and an in-memory
stand-in for the Redis session store. Needs the MSP-14 API skeleton and the
MSP-17 sessions on main.

Run from the repo root with:  python -m unittest discover -s tests -t .
"""
# RDKit is a C extension, so pylint cannot see its members.
# pylint: disable=no-member,c-extension-no-member,missing-function-docstring,missing-class-docstring
import io
import json
import os
import time
import types
import unittest
import uuid
from unittest import mock

from rdkit import Chem
from rdkit.Chem import AllChem

from backend.api.app import create_app
from backend.api.auth import sessions
from backend.api.molecules.validate_routes import LIMITS_CONFIG_KEY
from backend.core.config import Settings
from backend.services.molecule import parse_process
from backend.services.molecule.limits import MoleculeLimits

URL = "/api/molecules/validate"
TARGET = "CHEMBL203"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
LIMITS = MoleculeLimits(max_smiles_len=60, parse_timeout_s=30, max_molfile_bytes=50_000)


def molblock_of(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    return Chem.MolToMolBlock(mol)


def slow_compute(*_args):
    """Stands in for compute_validation to force the time limit (module level for spawn)."""
    time.sleep(60)
    return {}


class FakeSessionStore:
    """In-memory stand-in for the Redis SessionStore (same create/touch/delete)."""

    def __init__(self):
        self.records = {}

    def create(self, jti, user_id):
        self.records[jti] = str(user_id)

    def touch(self, jti):
        return jti in self.records

    def delete(self, jti):
        self.records.pop(jti, None)


class AuthMixin:
    """Patches the session store and secret, and issues a real signed token."""

    def start_auth(self):
        self.store = FakeSessionStore()
        env = mock.patch.dict(os.environ, {"JWT_SECRET": "unit-test-secret-not-real-0123456789abcdef"})
        env.start()
        self.addCleanup(env.stop)
        patch = mock.patch.object(sessions, "get_session_store", return_value=self.store)
        patch.start()
        self.addCleanup(patch.stop)
        user = types.SimpleNamespace(id=uuid.uuid4(), role="researcher")
        self.token = sessions.issue_token(self.store, user, Settings.from_env())
        self.auth = {"Authorization": "Bearer " + self.token}


class RouteTestCase(AuthMixin, unittest.TestCase):
    limits = LIMITS

    def setUp(self):
        app = create_app({"TESTING": True, LIMITS_CONFIG_KEY: self.limits})
        self.client = app.test_client()
        self.start_auth()

    def post_json(self, payload):
        return self.client.post(URL, json=payload, headers=self.auth)

    def post_file(self, data: bytes, name="molecule.mol", target_id=None):
        form = {"file": (io.BytesIO(data), name)}
        if target_id:
            form["target_id"] = target_id
        return self.client.post(
            URL, data=form, content_type="multipart/form-data", headers=self.auth
        )

    def assert_error(self, resp, status, code):
        self.assertEqual(resp.status_code, status, resp.get_data(as_text=True))
        error = resp.get_json()["error"]
        self.assertEqual(set(error), {"code", "message", "details", "request_id"})
        self.assertEqual(error["code"], code)
        self.assertEqual(error["request_id"], resp.headers["X-Request-ID"])
        return error


class LoginRequiredTests(RouteTestCase):
    """SEC-001: the route needs a live session (MSP-17)."""

    def assert_rejected_before_rdkit(self, resp, status, code):
        self.assertEqual(resp.status_code, status, resp.get_data(as_text=True))
        self.assertEqual(resp.get_json()["error"]["code"], code)

    def test_sec001_no_token_is_401_and_nothing_is_parsed(self):
        with mock.patch.object(parse_process, "run_with_timeout") as runner:
            resp = self.client.post(URL, json={"smiles": "CCO"})
        runner.assert_not_called()
        self.assert_rejected_before_rdkit(resp, 401, "UNAUTHENTICATED")

    def test_sec001_garbage_token_is_401(self):
        resp = self.client.post(
            URL, json={"smiles": "CCO"}, headers={"Authorization": "Bearer not-a-real-token"}
        )
        self.assert_rejected_before_rdkit(resp, 401, "UNAUTHENTICATED")

    def test_sec001_logged_out_session_is_401(self):
        claims = sessions.authenticate(self.auth["Authorization"], self.store, Settings.from_env())
        self.store.delete(claims["jti"])
        resp = self.post_json({"smiles": "CCO"})
        self.assert_rejected_before_rdkit(resp, 401, "SESSION_EXPIRED")

    def test_sec001_upload_also_needs_login(self):
        form = {"file": (io.BytesIO(molblock_of("CCO").encode()), "ethanol.mol")}
        resp = self.client.post(URL, data=form, content_type="multipart/form-data")
        self.assert_rejected_before_rdkit(resp, 401, "UNAUTHENTICATED")

    def test_sec001_valid_session_is_accepted(self):
        self.assertEqual(self.post_json({"smiles": "CCO"}).status_code, 200)


class WiringTests(unittest.TestCase):
    def test_route_is_registered_in_create_app(self):
        rules = {rule.rule for rule in create_app({"TESTING": True}).url_map.iter_rules()}
        self.assertIn(URL, rules)


class JsonInputTests(RouteTestCase):
    def test_fr002_aspirin_gives_golden_values_warnings_and_tiers(self):
        resp = self.post_json({"smiles": ASPIRIN, "target_id": TARGET})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("X-Request-ID", resp.headers)
        body = resp.get_json()
        self.assertEqual(body["canonical_smiles"], ASPIRIN)
        self.assertEqual(body["inchikey"], "BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
        self.assertEqual(body["warnings"], [])
        self.assertEqual(body["tiers"]["ml"], {"eligible": True, "skip_reason": None})
        self.assertEqual(body["tiers"]["dl"], {"eligible": True, "skip_reason": None})
        self.assertEqual(body["tiers"]["qml"], {"eligible": False, "skip_reason": "NO_FRAGMENT"})

    def test_br002_no_target_skips_dl(self):
        body = self.post_json({"smiles": "CCO"}).get_json()
        self.assertEqual(body["tiers"]["dl"], {"eligible": False, "skip_reason": "NO_TARGET_SELECTED"})
        self.assertTrue(body["tiers"]["qml"]["eligible"])

    def test_fr003_salt_gives_a_warning(self):
        body = self.post_json({"smiles": "CC(=O)[O-].[Na+]"}).get_json()
        self.assertEqual(body["canonical_smiles"], "CC(=O)O")
        self.assertEqual(len(body["warnings"]), 1)

    def test_fr053_over_length_smiles_is_413(self):
        smiles = "C" * (LIMITS.max_smiles_len + 1)
        resp = self.post_json({"smiles": smiles})
        error = self.assert_error(resp, 413, "INPUT_TOO_LARGE")
        self.assertNotIn(smiles, resp.get_data(as_text=True))
        self.assertEqual(error["details"], [{"field": "smiles", "type": "too_large"}])

    def test_fr053_smiles_at_the_limit_is_accepted(self):
        resp = self.post_json({"smiles": "C" * LIMITS.max_smiles_len})
        self.assertEqual(resp.status_code, 200)

    def test_fr004_invalid_smiles_is_422_with_a_short_message(self):
        for smiles in ("C(C", "CCOXx", "C(C)(C)(C)(C)C"):
            with self.subTest(smiles=smiles):
                resp = self.post_json({"smiles": smiles})
                error = self.assert_error(resp, 422, "VALIDATION_ERROR")
                self.assertLess(len(error["message"]), 120)
                self.assertEqual(error["details"], [{"field": "smiles", "type": "invalid_structure"}])

    def test_sec007_submitted_structure_is_never_echoed_in_an_error(self):
        marker = "CCOXxZz"
        text = self.post_json({"smiles": marker}).get_data(as_text=True)
        self.assertNotIn(marker, text)

    def test_br001_empty_smiles_is_422(self):
        self.assert_error(self.post_json({"smiles": "   "}), 422, "VALIDATION_ERROR")

    def test_sec004_missing_unknown_or_mistyped_fields_are_400(self):
        for payload in ({}, {"smiles": 123}, {"smiles": "CCO", "extra": 1}, {"smiles": "CCO", "target_id": "x" * 65}):
            with self.subTest(payload=payload):
                self.assert_error(self.post_json(payload), 400, "validation_error")

    def test_sec004_json_that_is_not_an_object_is_400(self):
        resp = self.client.post(
            URL, data=json.dumps(["CCO"]), content_type="application/json", headers=self.auth
        )
        self.assert_error(resp, 400, "invalid_json")

    def test_sec004_other_content_types_are_415(self):
        resp = self.client.post(URL, data="CCO", content_type="text/plain", headers=self.auth)
        self.assertEqual(resp.status_code, 415)
        self.assertIn("error", resp.get_json())

    def test_fr053_slow_parse_is_stopped_with_parse_timeout(self):
        slow = MoleculeLimits(max_smiles_len=60, parse_timeout_s=0.5)
        app = create_app({"TESTING": True, LIMITS_CONFIG_KEY: slow})
        started = time.monotonic()
        with mock.patch.object(parse_process, "compute_validation", slow_compute):
            resp = app.test_client().post(URL, json={"smiles": "CCO"}, headers=self.auth)
        self.assertLess(time.monotonic() - started, 10)
        self.assert_error(resp, 422, "PARSE_TIMEOUT")


class UploadInputTests(RouteTestCase):
    def test_fr001_ethanol_file_gives_the_same_result_as_the_string(self):
        from_file = self.post_file(molblock_of("CCO").encode(), "ethanol.mol", TARGET)
        from_string = self.post_json({"smiles": "CCO", "target_id": TARGET})
        self.assertEqual(from_file.status_code, 200)
        self.assertEqual(from_file.get_json(), from_string.get_json())

    def test_fr001_acetamide_file_gives_the_same_canonical_smiles_as_the_string(self):
        from_file = self.post_file(molblock_of("CC(=O)N").encode(), "acetamide.mol")
        from_string = self.post_json({"smiles": "CC(=O)N"})
        self.assertEqual(
            from_file.get_json()["canonical_smiles"], from_string.get_json()["canonical_smiles"]
        )

    def test_fr001_sdf_reads_the_first_record_and_warns(self):
        sdf = (molblock_of("CCO") + "$$$$\n" + molblock_of("CC(=O)N") + "$$$$\n").encode()
        body = self.post_file(sdf, "two.sdf").get_json()
        self.assertEqual(body["canonical_smiles"], "CCO")
        self.assertIn("2 records", body["warnings"][0])

    def test_fr004_invalid_file_is_422(self):
        resp = self.post_file(b"this is not a molecule file\n")
        error = self.assert_error(resp, 422, "VALIDATION_ERROR")
        self.assertEqual(error["details"], [{"field": "file", "type": "invalid_structure"}])

    def test_fr004_binary_file_is_422(self):
        self.assert_error(self.post_file(b"\x00\x01\x02\xff"), 422, "VALIDATION_ERROR")

    def test_fr053_oversized_file_is_413(self):
        big = molblock_of("CCO").encode() + b"\n" * (LIMITS.max_molfile_bytes + 10)
        self.assert_error(self.post_file(big), 413, "INPUT_TOO_LARGE")

    def test_sec004_missing_file_part_is_400(self):
        resp = self.client.post(
            URL, data={"target_id": TARGET}, content_type="multipart/form-data", headers=self.auth
        )
        error = self.assert_error(resp, 400, "validation_error")
        self.assertEqual(error["details"], [{"field": "file", "type": "missing"}])

    def test_sec004_overlong_target_id_is_400(self):
        resp = self.post_file(molblock_of("CCO").encode(), target_id="x" * 65)
        self.assert_error(resp, 400, "validation_error")


if __name__ == "__main__":
    unittest.main()
