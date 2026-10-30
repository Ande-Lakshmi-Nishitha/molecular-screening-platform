import os
import re
import unittest
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.api.app import create_app
from backend.api.auth.sessions import SessionStore, issue_token
from backend.core.config import Settings
from backend.db.models import Base, Job, Molecule, User

SECRET = "unit-test-secret-unit-test-secret-1234"
PUBLIC_MODULES = ("backend.api.health", "backend.api.openapi")


class FakeRedis:
    def __init__(self):
        self.data = {}

    def set(self, key, value, ex=None):
        self.data[key] = value

    def expire(self, key, seconds):
        return key in self.data

    def delete(self, key):
        self.data.pop(key, None)


class AccessControlTests(unittest.TestCase):  # FR-048..051, SEC-001, SEC-002, SEC-005
    def setUp(self):
        engine = create_engine("sqlite://", poolclass=StaticPool,
                               connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)

        @contextmanager
        def scope():
            session = Session(engine)
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        with Session(engine) as session:
            owner = User(email="a@example.com", password_hash="x")
            other = User(email="b@example.com", password_hash="x")
            admin = User(email="c@example.com", password_hash="x", role="administrator")
            molecule = Molecule(inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N", canonical_smiles="CCO")
            session.add_all([owner, other, admin, molecule])
            session.flush()
            job = Job(molecule_id=molecule.id, user_id=owner.id, options={}, rules_version="0.1")
            session.add(job)
            session.commit()
            self.job_id = job.id
            self.users = {"owner": (owner.id, "researcher"), "other": (other.id, "researcher"),
                          "admin": (admin.id, "administrator")}

        env = patch.dict(os.environ, {"JWT_SECRET": SECRET})
        env.start()
        self.addCleanup(env.stop)
        self.sessions = SessionStore(FakeRedis(), 1800)
        for target, value in (("backend.api.jobs.routes.session_scope", scope),
                              ("backend.api.auth.sessions.get_session_store", lambda: self.sessions)):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.app = self.make_app()
        self.client = self.app.test_client()
        settings = Settings.from_env()
        self.tokens = {name: issue_token(self.sessions, SimpleNamespace(id=uid, role=role), settings)
                       for name, (uid, role) in self.users.items()}

    @staticmethod
    def make_app():
        app = create_app({"TESTING": True})
        app.add_url_rule("/api/admin/ping", "admin_ping", lambda: ("ok", 200))
        return app

    def auth(self, who):
        return {"Authorization": f"Bearer {self.tokens[who]}"}

    def test_every_protected_route_rejects_anonymous_requests(self):  # SEC-001
        public = {"auth.register", "auth.login", "static"}
        checked = 0
        for rule in self.app.url_map.iter_rules():
            view = self.app.view_functions[rule.endpoint]
            if (rule.endpoint in public or view.__module__ in PUBLIC_MODULES
                    or not rule.rule.startswith("/api/")):
                continue
            path = re.sub(r"<[^>]+>", "x", rule.rule)
            for method in rule.methods - {"HEAD", "OPTIONS"}:
                resp = self.client.open(path, method=method)
                self.assertEqual(resp.status_code, 401, f"{method} {rule.rule}")
                checked += 1
        self.assertGreater(checked, 4)

    def test_register_login_and_health_stay_public(self):
        for path in ("/api/auth/register", "/api/auth/login"):
            self.assertEqual(self.client.post(path, data="x", content_type="text/plain").status_code, 422)
        health = [r for r in self.app.url_map.iter_rules()
                  if self.app.view_functions[r.endpoint].__module__ == "backend.api.health"
                  and "GET" in r.methods and not r.arguments]
        self.assertTrue(health)
        for rule in health:
            self.assertNotEqual(self.client.get(rule.rule).status_code, 401)

    def test_admin_routes_are_forbidden_to_researchers(self):  # FR-050, FR-051
        self.assertEqual(self.client.get("/api/admin/ping").status_code, 401)
        resp = self.client.get("/api/admin/ping", headers=self.auth("owner"))
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.get_json()["error"]["code"], "FORBIDDEN")
        self.assertEqual(self.client.get("/api/admin/ping", headers=self.auth("admin")).status_code, 200)

    def test_owner_sees_job_but_others_get_403_and_unknown_gets_404(self):  # FR-049, SEC-002
        url = f"/api/jobs/{self.job_id}"
        self.assertEqual(self.client.get(url, headers=self.auth("owner")).status_code, 200)
        for who in ("other", "admin"):
            resp = self.client.get(url, headers=self.auth(who))
            self.assertEqual(resp.status_code, 403)
            self.assertEqual(resp.get_json()["error"]["code"], "FORBIDDEN")
        resp = self.client.get(f"/api/jobs/{uuid.uuid4()}", headers=self.auth("owner"))
        self.assertEqual(resp.status_code, 404)

    def test_no_route_lets_anyone_write_results(self):  # FR-050
        for rule in self.app.url_map.iter_rules():
            if "result" in rule.rule:
                self.assertTrue(rule.methods <= {"GET", "HEAD", "OPTIONS"}, rule.rule)

    def test_rate_limit_gives_429_with_retry_after(self):  # SEC-005
        with patch.dict(os.environ, {"RATE_LIMIT": "3/minute"}):
            client = self.make_app().test_client()
        url = f"/api/jobs/{uuid.uuid4()}"
        for _ in range(3):
            self.assertEqual(client.get(url, headers=self.auth("owner")).status_code, 404)
        resp = client.get(url, headers=self.auth("owner"))
        self.assertEqual(resp.status_code, 429)
        self.assertEqual(resp.get_json()["error"]["code"], "RATE_LIMITED")
        self.assertGreaterEqual(int(resp.headers["Retry-After"]), 1)
        self.assertEqual(client.get(url, headers=self.auth("admin")).status_code, 404)  # other user, own count


if __name__ == "__main__":
    unittest.main()
    