import unittest

from backend.api.app import create_app


class BlueprintRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app({"TESTING": True})
        self.client = self.app.test_client()

    def test_routes_exist_in_the_real_app(self):
        rules = {(rule.rule, method) for rule in self.app.url_map.iter_rules()
                 for method in rule.methods}
        for expected in (("/api/auth/register", "POST"), ("/api/auth/login", "POST"),
                         ("/api/auth/logout", "POST"), ("/api/molecules/submit", "POST"),
                         ("/api/jobs/<job_id>", "GET")):
            self.assertIn(expected, rules)

    def test_protected_routes_refuse_requests_without_a_token(self):
        self.assertEqual(self.client.post("/api/molecules/submit", json={"smiles": "CCO"}).status_code, 401)
        self.assertEqual(self.client.get("/api/jobs/00000000-0000-0000-0000-000000000000").status_code, 401)
        self.assertEqual(self.client.post("/api/auth/logout").status_code, 401)

    def test_register_rejects_a_non_json_body_before_touching_the_database(self):
        resp = self.client.post("/api/auth/register", data="x", content_type="text/plain")
        self.assertEqual(resp.status_code, 422)


if __name__ == "__main__":
    unittest.main()
    