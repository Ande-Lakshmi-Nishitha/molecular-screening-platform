"""Unit tests for the health endpoints (FR-063)."""
# pylint: disable=missing-function-docstring
import unittest
from unittest.mock import patch

from flask import Flask

from backend.api.health import check_postgres, check_redis, health_bp


def make_client():
    """Build a minimal app with the health blueprint."""
    app = Flask(__name__)
    app.register_blueprint(health_bp)
    return app.test_client()


class HealthEndpointTests(unittest.TestCase):
    """FR-063: /healthz and /readyz."""

    def test_fr063_healthz_is_always_200(self):
        resp = make_client().get("/healthz")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(), {"status": "ok"})

    @patch("backend.api.health.check_redis", return_value=True)
    @patch("backend.api.health.check_postgres", return_value=True)
    def test_fr063_readyz_200_when_all_up(self, _pg, _rd):
        resp = make_client().get("/readyz")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["status"], "ready")

    @patch("backend.api.health.check_redis", return_value=True)
    @patch("backend.api.health.check_postgres", return_value=False)
    def test_fr063_readyz_503_when_postgres_down(self, _pg, _rd):
        resp = make_client().get("/readyz")
        self.assertEqual(resp.status_code, 503)
        body = resp.get_json()
        self.assertEqual(body["checks"]["postgres"], "unavailable")
        self.assertEqual(body["checks"]["redis"], "ok")

    @patch("backend.api.health.check_redis", return_value=False)
    @patch("backend.api.health.check_postgres", return_value=True)
    def test_fr063_readyz_503_when_redis_down(self, _pg, _rd):
        resp = make_client().get("/readyz")
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.get_json()["checks"]["redis"], "unavailable")


class DependencyCheckTests(unittest.TestCase):
    """The checks fail closed and leak nothing when config is missing."""

    @patch.dict("os.environ", {}, clear=True)
    def test_sec007_missing_env_means_unavailable(self):
        self.assertFalse(check_postgres())
        self.assertFalse(check_redis())

    @patch.dict("os.environ", {"REDIS_URL": "redis://x"}, clear=True)
    @patch("backend.api.health.redis.Redis.from_url")
    def test_redis_ping_true(self, from_url):
        from_url.return_value.ping.return_value = True
        self.assertTrue(check_redis())

    @patch.dict("os.environ", {"DATABASE_URL": "postgresql://x"}, clear=True)
    @patch("backend.api.health.psycopg2.connect")
    def test_postgres_select_ok(self, connect):
        self.assertTrue(check_postgres())
        connect.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
