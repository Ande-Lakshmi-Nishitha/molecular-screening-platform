"""Smoke suite (MSP-14): fast checks that run in CI in well under 3 minutes."""
# pylint: disable=missing-function-docstring,missing-class-docstring
import unittest
from unittest.mock import patch

from backend.api.app import create_app
from tests.fixtures.golden_molecules import GOLDEN
from tests.mocks.qml_stub import StubQmlBackend
from tests.mocks.search_mocks import MockHttpError, MockSearchSource

GOLDEN_NAMES = {"aspirin", "caffeine", "ibuprofen", "ethanol", "acetamide"}


class GoldenSetTests(unittest.TestCase):
    def test_golden_set_has_the_five_molecules(self):
        self.assertEqual(set(GOLDEN), GOLDEN_NAMES)

    def test_every_entry_has_required_fields(self):
        for name, data in GOLDEN.items():
            for key in ("smiles", "inchikey", "formula", "heavy_atoms", "qml_eligible"):
                self.assertIn(key, data, name)

    def test_inchikeys_have_standard_shape(self):
        for name, data in GOLDEN.items():
            parts = data["inchikey"].split("-")
            self.assertEqual([len(p) for p in parts], [14, 10, 1], name)

    def test_fr021_qml_eligibility_follows_the_nine_atom_rule(self):
        for name, data in GOLDEN.items():
            self.assertEqual(data["qml_eligible"], data["heavy_atoms"] <= 9, name)


class MockSearchTests(unittest.TestCase):
    def test_ok_returns_golden_match(self):
        results = MockSearchSource(mode="ok").search("aspirin")
        self.assertEqual(results[0]["smiles"], GOLDEN["aspirin"]["smiles"])

    def test_fr008_timeout_mode(self):
        with self.assertRaises(TimeoutError):
            MockSearchSource(mode="timeout").search("aspirin")

    def test_fr008_http_500_mode(self):
        with self.assertRaises(MockHttpError) as ctx:
            MockSearchSource(name="ChEMBL", mode="http_500").search("aspirin")
        self.assertEqual(ctx.exception.status, 500)

    def test_fr008_empty_mode(self):
        self.assertEqual(MockSearchSource(mode="empty").search("aspirin"), [])

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            MockSearchSource(mode="nonsense")


class StubQmlTests(unittest.TestCase):
    def test_result_is_labelled_as_stub(self):
        result = StubQmlBackend().excitation_gap("CCO")
        self.assertEqual(result["backend"], "stub")
        self.assertIn("stub", result["label"])

    def test_can_be_told_to_fail(self):
        with self.assertRaises(TimeoutError):
            StubQmlBackend(fail_with=TimeoutError("qml")).excitation_gap("CCO")


class AppSmokeTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()

    def test_fr063_healthz(self):
        self.assertEqual(self.client.get("/healthz").status_code, 200)

    @patch("backend.api.health.check_redis", return_value=True)
    @patch("backend.api.health.check_postgres", return_value=True)
    def test_fr063_readyz_ready(self, _pg, _rd):
        self.assertEqual(self.client.get("/readyz").status_code, 200)

    def test_gunicorn_entry_point_serves_the_app(self):
        # pylint: disable=import-outside-toplevel
        from backend import wsgi

        self.assertEqual(wsgi.app.test_client().get("/healthz").status_code, 200)

    def test_openapi_served(self):
        self.assertEqual(self.client.get("/api/openapi.json").status_code, 200)

    def test_unknown_route_uses_envelope(self):
        body = self.client.get("/api/does-not-exist").get_json()
        self.assertEqual(body["error"]["code"], "not_found")

    def test_every_response_has_request_id(self):
        for path in ("/healthz", "/api/openapi.json", "/api/does-not-exist"):
            self.assertIn("X-Request-ID", self.client.get(path).headers, path)


if __name__ == "__main__":
    unittest.main()
