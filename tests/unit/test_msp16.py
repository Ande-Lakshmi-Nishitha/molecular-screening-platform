import unittest
from contextlib import contextmanager
from unittest.mock import patch

from flask import Flask
from redis.exceptions import RedisError
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.api.auth.lockout import LockoutStore
from backend.api.auth.routes import auth_bp
from backend.db.models import AuditLog, Base, User

GOOD = {"email": "researcher1@example.com", "password": "Str0ng-pass!"}


class FakeRedis:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def incr(self, key):
        self.data[key] = int(self.data.get(key, 0)) + 1
        return self.data[key]

    def expire(self, key, seconds):
        return True

    def delete(self, key):
        self.data.pop(key, None)


class BrokenRedis(FakeRedis):
    def get(self, key):
        raise RedisError("down")


class AuthTests(unittest.TestCase):  # FR-041..044, FR-052, FR-065
    def setUp(self):
        self.engine = create_engine("sqlite://", poolclass=StaticPool,
                                    connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        engine = self.engine

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

        self.store = LockoutStore(FakeRedis(), 3, 15)
        for target, value in (("backend.api.auth.routes.session_scope", scope),
                              ("backend.api.auth.routes.get_lockout_store", lambda: self.store)):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        app = Flask(__name__)
        app.register_blueprint(auth_bp)
        self.client = app.test_client()

    def post(self, path, body):
        return self.client.post(f"/api/auth/{path}", json=body)

    def test_register_creates_researcher_with_argon2id_hash(self):
        resp = self.post("register", GOOD)
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(resp.get_json()["role"], "researcher")
        with Session(self.engine) as s:
            user = s.execute(select(User)).scalar_one()
            self.assertTrue(user.password_hash.startswith("$argon2id$"))
            self.assertNotIn(GOOD["password"], user.password_hash)

    def test_duplicate_email_gives_409(self):
        self.post("register", GOOD)
        resp = self.post("register", GOOD)
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.get_json()["error"]["code"], "EMAIL_TAKEN")

    def test_invalid_bodies_give_422(self):
        for body in ({"email": "not-an-email", "password": "Str0ng-pass!"},
                     {"email": "a@b.com", "password": "short"}, {"email": "a@b.com"}):
            self.assertEqual(self.post("register", body).status_code, 422)
        resp = self.client.post("/api/auth/register", data="not json", content_type="text/plain")
        self.assertEqual(resp.status_code, 422)

    def test_login_success(self):
        self.post("register", GOOD)
        self.assertEqual(self.post("login", GOOD).status_code, 200)

    def test_wrong_password_and_unknown_email_look_identical(self):
        self.post("register", GOOD)
        wrong_pw = self.post("login", {**GOOD, "password": "wrong-password"})
        unknown = self.post("login", {"email": "nobody@example.com", "password": "wrong-password"})
        self.assertEqual((wrong_pw.status_code, unknown.status_code), (401, 401))
        self.assertEqual(wrong_pw.get_json()["error"]["code"], "INVALID_CREDENTIALS")
        self.assertEqual(wrong_pw.get_json()["error"]["message"], unknown.get_json()["error"]["message"])

    def test_lockout_after_configured_failures(self):
        self.post("register", GOOD)
        for _ in range(3):
            self.post("login", {**GOOD, "password": "wrong-password"})
        resp = self.post("login", GOOD)  # correct password, but locked
        self.assertEqual(resp.status_code, 429)
        self.assertEqual(resp.get_json()["error"]["code"], "ACCOUNT_LOCKED")

    def test_audit_rows_written_for_login_failed_and_lockout(self):
        self.post("register", GOOD)
        self.post("login", GOOD)
        for _ in range(3):
            self.post("login", {**GOOD, "password": "wrong-password"})
        with Session(self.engine) as s:
            events = [row.event_type for row in s.execute(select(AuditLog)).scalars()]
        for expected in ("login", "login_failed", "account_locked"):
            self.assertIn(expected, events)

    def test_audit_log_holds_no_plain_email(self):
        self.post("register", GOOD)
        self.post("login", {**GOOD, "password": "wrong-password"})
        with Session(self.engine) as s:
            details = str([row.detail for row in s.execute(select(AuditLog)).scalars()])
        self.assertNotIn(GOOD["email"], details)

    def test_redis_down_gives_503(self):
        self.store = LockoutStore(BrokenRedis(), 3, 15)
        self.post("register", GOOD)
        self.assertEqual(self.post("login", GOOD).status_code, 503)


if __name__ == "__main__":
    unittest.main()
    