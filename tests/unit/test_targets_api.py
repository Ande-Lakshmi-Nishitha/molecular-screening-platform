"""Tests for GET /api/targets (MSP-26; FR-015, FR-064)."""
# pylint: disable=missing-function-docstring,missing-class-docstring
import unittest
from unittest import mock

from backend.api.app import create_app
from backend.api.auth.service import AuthError

TARGETS = {"targets": [
    {"id": "EGFR", "name": "EGFR", "chembl_id": "CHEMBL203", "unit": "pIC50", "secret_note": "do not show"},
    {"id": "HIV1_PROTEASE", "name": "HIV-1 protease", "chembl_id": "CHEMBL243", "unit": "pKi"},
]}


class TargetsEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app({"TESTING": True}).test_client()

    def logged_in(self):
        patches = [
            mock.patch("backend.api.auth.sessions.authenticate", return_value={"sub": "u1"}),
            mock.patch("backend.api.auth.sessions.get_session_store", return_value=object()),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def get(self):
        return self.client.get("/api/targets", headers={"Authorization": "Bearer x"})

    def test_fr015_lists_every_configured_target_in_file_order(self):
        self.logged_in()
        with mock.patch("backend.api.targets.routes.load_targets", return_value=TARGETS):
            response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertEqual([t["id"] for t in response.get_json()["targets"]], ["EGFR", "HIV1_PROTEASE"])

    def test_each_entry_has_id_name_chembl_id_and_unit_only(self):
        self.logged_in()
        with mock.patch("backend.api.targets.routes.load_targets", return_value=TARGETS):
            first = self.get().get_json()["targets"][0]
        self.assertEqual(set(first), {"id", "name", "chembl_id", "unit"})
        self.assertEqual(first["unit"], "pIC50")

    def test_an_empty_or_missing_config_gives_an_empty_list(self):
        self.logged_in()
        for value in ({"targets": []}, {}, None):
            with self.subTest(value=value), mock.patch("backend.api.targets.routes.load_targets", return_value=value):
                self.assertEqual(self.get().get_json(), {"targets": []})

    def test_the_real_config_file_lists_the_three_targets(self):
        self.logged_in()
        body = self.get().get_json()
        self.assertEqual([t["id"] for t in body["targets"]], ["EGFR", "BACE1", "HIV1_PROTEASE"])

    def test_a_request_without_a_session_is_401(self):
        error = AuthError("UNAUTHENTICATED", "Authentication required.", 401)
        with mock.patch("backend.api.auth.sessions.authenticate", side_effect=error):
            self.assertEqual(self.client.get("/api/targets").status_code, 401)

    def test_the_route_is_registered_in_the_real_app(self):
        rules = {(rule.rule, method) for rule in create_app({"TESTING": True}).url_map.iter_rules() for method in rule.methods}
        self.assertIn(("/api/targets", "GET"), rules)


if __name__ == "__main__":
    unittest.main()
