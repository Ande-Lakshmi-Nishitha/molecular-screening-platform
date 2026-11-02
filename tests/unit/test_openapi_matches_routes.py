"""The OpenAPI document must list exactly the routes the app serves (MSP-47, IF-3.4, QA-TST)."""
# pylint: disable=missing-function-docstring,missing-class-docstring
import re
import unittest

from backend.api.app import create_app
from backend.api.openapi import build_spec

FLASK_OWN_ROUTES = {"/static/<path:filename>"}  # added by Flask itself, not part of the API


def to_openapi_path(rule):
    """/api/jobs/<job_id> -> /api/jobs/{job_id} (also drops converters such as <int:id>)."""
    return re.sub(r"<(?:[^:>]+:)?([^>]+)>", r"{\1}", rule)


def app_operations():
    app = create_app({"TESTING": True})
    return {(to_openapi_path(rule.rule), method.lower())
            for rule in app.url_map.iter_rules() if rule.rule not in FLASK_OWN_ROUTES
            for method in rule.methods - {"HEAD", "OPTIONS"}}


def spec_operations():
    return {(path, method) for path, methods in build_spec()["paths"].items() for method in methods}


def refs(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref":
                yield value
            else:
                yield from refs(value)
    elif isinstance(node, list):
        for item in node:
            yield from refs(item)


class OpenApiMatchesRoutesTests(unittest.TestCase):
    def test_every_route_is_documented(self):
        missing = sorted(app_operations() - spec_operations())
        self.assertEqual(missing, [], f"routes without an OpenAPI entry: {missing}")

    def test_every_documented_operation_has_a_route(self):
        extra = sorted(spec_operations() - app_operations())
        self.assertEqual(extra, [], f"OpenAPI entries without a route: {extra}")

    def test_logout_jobs_and_report_are_documented(self):  # MSP-47 acceptance criteria
        documented = spec_operations()
        for operation in (("/api/auth/logout", "post"), ("/api/jobs", "get"),
                          ("/api/jobs/{job_id}", "get"), ("/api/jobs/{job_id}", "delete"),
                          ("/api/jobs/{job_id}/report", "get")):
            self.assertIn(operation, documented)

    def test_every_reference_resolves(self):
        spec = build_spec()
        for ref in refs(spec):
            node = spec
            for part in ref.lstrip("#/").split("/"):
                self.assertIn(part, node, f"{ref} does not resolve")
                node = node[part]

    def test_protected_operations_declare_the_bearer_scheme(self):
        spec = build_spec()
        public = {"/api/auth/register", "/api/auth/login", "/api/openapi.json",
                  "/healthz", "/readyz"}
        for path, methods in spec["paths"].items():
            for method, operation in methods.items():
                self.assertEqual("security" in operation, path not in public, f"{method} {path}")


if __name__ == "__main__":
    unittest.main()
