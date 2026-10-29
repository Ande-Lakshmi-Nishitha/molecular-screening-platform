import os
import time
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import jwt
from flask import Flask, g, jsonify
from redis.exceptions import RedisError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.api.auth.lockout import LockoutStore
from backend.api.auth.routes import auth_bp
from backend.api.auth.sessions import SessionStore, login_required
from backend.db.models import Base

SECRET = "unit-test-secret-unit-test-secret-1234"
OTHER_SECRET = "another-test-secret-another-test-secret-99"
IDLE = 1800  # 30 minutes, the SESSION_IDLE_MIN default
GOOD = {"email": "researcher1@example.com", "password": "Str0ng-pass!"}


class FakeRedis:
    def __init__(self):
        self.data = {}
        self.ttl = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self.data[key] = value
        self.ttl[key] = ex

    def incr(self, key):
        self.data[key] = int(self.data.get(key, 0)) + 1
        return self.data[key]

    def expire(self, key, seconds):
        if key not in self.data:
            return False
        self.ttl[key] = seconds
        return True

    def delete(self, key):
        self.data.pop(key, None)
        self.ttl.pop(key, None)


class BrokenRedis(FakeRedis):
    def expire(self, key, seconds):
        raise RedisError("down")


class SessionTests(unittest.TestCase):  # FR-045, FR-046, FR-047, SEC-001
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

        self.redis = FakeRedis()
        self.sessions = SessionStore(self.redis, IDLE)
        lockout = LockoutStore(self.redis, 5, 15)
        env = patch.dict(os.environ, {"JWT_SECRET": SECRET})
        env.start()
        self.addCleanup(env.stop)
        for target, value in (("backend.api.auth.routes.session_scope", scope),
                              ("backend.api.auth.routes.get_lockout_store", lambda: lockout),
                              ("backend.api.auth.routes.get_session_store", lambda: self.sessions),
                              ("backend.api.auth.sessions.get_session_store", lambda: self.sessions)):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        app = Flask(__name__)
        app.register_blueprint(auth_bp)

        @app.get("/api/ping")
        @login_required
        def ping():
            return jsonify({"user": g.claims["sub"]})

        self.client = app.test_client()

    def login_token(self):
        self.client.post("/api/auth/register", json=GOOD)
        return self.client.post("/api/auth/login", json=GOOD).get_json()["token"]

    def ping(self, token=None):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        return self.client.get("/api/ping", headers=headers)

    def test_login_issues_jwt_and_redis_session(self):
        claims = jwt.decode(self.login_token(), SECRET, algorithms=["HS256"])
        self.assertEqual(claims["role"], "researcher")
        for field in ("sub", "jti", "exp"):
            self.assertIn(field, claims)
        key = f"session:{claims['jti']}"
        self.assertIn(key, self.redis.data)
        self.assertEqual(self.redis.ttl[key], IDLE)

    def test_every_request_resets_idle_ttl(self):
        token = self.login_token()
        key = f"session:{jwt.decode(token, SECRET, algorithms=['HS256'])['jti']}"
        self.redis.ttl[key] = 5
        self.assertEqual(self.ping(token).status_code, 200)
        self.assertEqual(self.redis.ttl[key], IDLE)

    def test_logout_deletes_record_and_token_then_gives_401(self):
        token = self.login_token()
        jti = jwt.decode(token, SECRET, algorithms=["HS256"])["jti"]
        resp = self.client.post("/api/auth/logout", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(resp.status_code, 204)
        self.assertNotIn(f"session:{jti}", self.redis.data)
        self.assertEqual(self.ping(token).status_code, 401)

    def test_missing_or_garbage_token_gives_401(self):
        resp = self.ping()
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.get_json()["error"]["code"], "UNAUTHENTICATED")
        self.assertEqual(self.ping("not-a-token").status_code, 401)

    def test_idle_expiry_gives_401_session_expired(self):
        token = self.login_token()
        self.redis.data.clear()  # Redis drops the record when the idle TTL runs out
        resp = self.ping(token)
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.get_json()["error"]["code"], "SESSION_EXPIRED")

    def test_absolute_expiry_gives_401(self):
        expired = jwt.encode({"sub": "x", "role": "researcher", "jti": "abc",
                              "exp": int(time.time()) - 10}, SECRET, algorithm="HS256")
        self.redis.set("session:abc", "x", ex=IDLE)
        resp = self.ping(expired)
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.get_json()["error"]["code"], "SESSION_EXPIRED")

    def test_wrong_signature_and_alg_none_are_rejected(self):
        claims = {"sub": "x", "role": "researcher", "jti": "abc", "exp": int(time.time()) + 600}
        self.redis.set("session:abc", "x", ex=IDLE)
        forged = jwt.encode(claims, OTHER_SECRET, algorithm="HS256")
        unsigned = jwt.encode(claims, None, algorithm="none")
        self.assertEqual(self.ping(forged).status_code, 401)
        self.assertEqual(self.ping(unsigned).status_code, 401)

    def test_redis_down_gives_503(self):
        token = self.login_token()
        self.sessions = SessionStore(BrokenRedis(), IDLE)
        self.assertEqual(self.ping(token).status_code, 503)


if __name__ == "__main__":
    unittest.main()
    