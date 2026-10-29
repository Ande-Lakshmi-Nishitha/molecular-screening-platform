"""Unit tests for the API skeleton (MSP-14): SEC-004, SEC-007, QA-TST."""
# pylint: disable=missing-function-docstring,missing-class-docstring
import logging
import sys
import unittest

from backend.api.app import create_app
from backend.api.errors import ApiError
from backend.api.schemas import StrictModel, validate_body
from backend.core.logging_filter import REDACTED, RedactionFilter, redact


class Body(StrictModel):
    name: str
    count: int


def make_app(config=None):
    app = create_app(config)

    @app.post("/test/body")
    def body_route():
        data = validate_body(Body)
        return {"name": data.name, "count": data.count}

    @app.get("/test/api-error")
    def api_error_route():
        raise ApiError(403, "forbidden", "Not allowed.", [{"field": "x", "type": "t"}])

    @app.get("/test/boom")
    def boom_route():
        raise RuntimeError("SELECT * FROM users WHERE smiles='CCO' secret-detail")

    return app


class ErrorEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.client = make_app().test_client()

    def assert_envelope(self, resp, status, code):
        self.assertEqual(resp.status_code, status)
        err = resp.get_json()["error"]
        self.assertEqual(set(err), {"code", "message", "details", "request_id"})
        self.assertEqual(err["code"], code)
        self.assertEqual(err["request_id"], resp.headers["X-Request-ID"])
        return err

    def test_sec004_404_uses_envelope(self):
        self.assert_envelope(self.client.get("/nope"), 404, "not_found")

    def test_405_uses_envelope_and_keeps_allow_header(self):
        resp = self.client.post("/healthz")
        self.assert_envelope(resp, 405, "method_not_allowed")
        self.assertIn("GET", resp.headers["Allow"])

    def test_api_error_uses_envelope_and_details(self):
        err = self.assert_envelope(self.client.get("/test/api-error"), 403, "forbidden")
        self.assertEqual(err["details"], [{"field": "x", "type": "t"}])

    def test_sec007_500_leaks_nothing(self):
        resp = self.client.get("/test/boom")
        err = self.assert_envelope(resp, 500, "internal_error")
        text = resp.get_data(as_text=True)
        for leaked in ("SELECT", "secret-detail", "Traceback", "RuntimeError", "CCO"):
            self.assertNotIn(leaked, text)
        self.assertEqual(err["message"], "An internal error occurred.")

    def test_sec004_oversized_body_gets_413(self):
        client = make_app({"MAX_CONTENT_LENGTH": 50}).test_client()
        resp = client.post("/test/body", json={"name": "x" * 500, "count": 1})
        self.assert_envelope(resp, 413, "payload_too_large")


class RequestIdTests(unittest.TestCase):
    def setUp(self):
        self.client = make_app().test_client()

    def test_header_on_success(self):
        self.assertTrue(self.client.get("/healthz").headers["X-Request-ID"])

    def test_valid_incoming_id_is_echoed(self):
        resp = self.client.get("/healthz", headers={"X-Request-ID": "abc-123"})
        self.assertEqual(resp.headers["X-Request-ID"], "abc-123")

    def test_invalid_incoming_id_is_replaced(self):
        resp = self.client.get("/healthz", headers={"X-Request-ID": "bad id <x>"})
        self.assertRegex(resp.headers["X-Request-ID"], r"^[0-9a-f]{32}$")

    def test_ids_differ_between_requests(self):
        first = self.client.get("/healthz").headers["X-Request-ID"]
        second = self.client.get("/healthz").headers["X-Request-ID"]
        self.assertNotEqual(first, second)


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.client = make_app().test_client()

    def post(self, payload):
        return self.client.post("/test/body", json=payload)

    def test_sec004_valid_body_accepted(self):
        resp = self.post({"name": "a", "count": 2})
        self.assertEqual(resp.status_code, 200)

    def test_sec004_unknown_field_rejected(self):
        resp = self.post({"name": "a", "count": 2, "extra": 1})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()["error"]["code"], "validation_error")

    def test_sec004_wrong_type_rejected_without_coercion(self):
        self.assertEqual(self.post({"name": "a", "count": "2"}).status_code, 400)

    def test_missing_field_rejected(self):
        self.assertEqual(self.post({"name": "a"}).status_code, 400)

    def test_non_object_body_rejected(self):
        resp = self.post([1, 2])
        self.assertEqual(resp.get_json()["error"]["code"], "invalid_json")

    def test_not_json_rejected(self):
        resp = self.client.post("/test/body", data="name=a", content_type="text/plain")
        self.assertEqual(resp.get_json()["error"]["code"], "invalid_json")

    def test_sec007_validation_error_does_not_echo_input(self):
        resp = self.post({"name": "CC(=O)Oc1ccccc1C(=O)O", "count": "bad"})
        self.assertNotIn("CC(=O)Oc1ccccc1C(=O)O", resp.get_data(as_text=True))
        details = resp.get_json()["error"]["details"]
        self.assertEqual(details, [{"field": "count", "type": "int_type"}])


class CorsTests(unittest.TestCase):
    def setUp(self):
        self.client = make_app({"FRONTEND_ORIGIN": "https://app.example"}).test_client()

    def test_frontend_origin_allowed(self):
        resp = self.client.get("/healthz", headers={"Origin": "https://app.example"})
        self.assertEqual(resp.headers["Access-Control-Allow-Origin"], "https://app.example")

    def test_other_origin_gets_no_cors_headers(self):
        resp = self.client.get("/healthz", headers={"Origin": "https://evil.example"})
        self.assertNotIn("Access-Control-Allow-Origin", resp.headers)

    def test_no_origin_header_gets_no_cors_headers(self):
        self.assertNotIn("Access-Control-Allow-Origin", self.client.get("/healthz").headers)

    def test_preflight_from_frontend(self):
        resp = self.client.options(
            "/api/molecules/validate", headers={"Origin": "https://app.example"}
        )
        self.assertEqual(resp.headers["Access-Control-Allow-Origin"], "https://app.example")


class OpenApiTests(unittest.TestCase):
    def setUp(self):
        self.spec = make_app().test_client().get("/api/openapi.json").get_json()

    def test_served_as_openapi_3(self):
        self.assertTrue(self.spec["openapi"].startswith("3."))

    def test_lists_every_srs_endpoint(self):
        for path in ("/api/auth/login", "/api/molecules/submit", "/api/jobs/{id}/results",
                     "/api/targets", "/api/models", "/healthz", "/readyz"):
            self.assertIn(path, self.spec["paths"])

    def test_error_schema_matches_envelope(self):
        required = self.spec["components"]["schemas"]["Error"]["properties"]["error"]["required"]
        self.assertEqual(set(required), {"code", "message", "details", "request_id"})


class LogRedactionTests(unittest.TestCase):
    def test_sec007_redacts_key_value_forms(self):
        for text in ("smiles=CCO", 'smiles: "CCO"', "{'smiles': 'CCO'}", '{"smiles": "CCO"}'):
            out = redact(text)
            self.assertNotIn("CCO", out, text)
            self.assertIn(REDACTED, out)

    def test_sec007_redacts_all_four_keys(self):
        text = "smiles=CCO structure=X molfile=Y inchikey=LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
        out = redact(text)
        for secret in ("CCO", "LFQSCWFLJHTTHZ", "=X", "=Y"):
            self.assertNotIn(secret, out)

    def test_other_text_is_untouched(self):
        self.assertEqual(redact("job 7 finished in 3s"), "job 7 finished in 3s")

    def test_sec007_redacts_multiline_molfile(self):
        out = redact('molfile: "line1\nline2 M  END"')
        self.assertNotIn("line2", out)

    def test_filter_redacts_args_and_exception_text(self):
        exc_info = None
        try:
            raise ValueError("bad smiles=CCO")
        except ValueError:
            exc_info = sys.exc_info()
        record = logging.LogRecord(
            "t", logging.ERROR, __file__, 1, "got %s", ({"smiles": "CCO"},), exc_info
        )
        RedactionFilter().filter(record)
        self.assertNotIn("CCO", record.getMessage())
        self.assertNotIn("CCO", record.exc_text)
        self.assertIsNone(record.exc_info)

    def test_installed_factory_redacts_real_log_calls(self):
        make_app()
        with self.assertLogs("msp.test", level="INFO") as captured:
            logging.getLogger("msp.test").info("submitted smiles=CCO for user 4")
        self.assertNotIn("CCO", captured.output[0])
        self.assertIn("user 4", captured.output[0])


if __name__ == "__main__":
    unittest.main()
