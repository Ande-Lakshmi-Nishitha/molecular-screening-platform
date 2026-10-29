"""Health endpoints (FR-063, SEC-007).

/healthz is liveness: the process is up.
/readyz is readiness: PostgreSQL and Redis both answer. It returns 200 when
ready and 503 otherwise. Responses never include exception text, hostnames or
connection strings.
"""
import os

import psycopg2
import redis
from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)


def _timeout_seconds():
    """Dependency check timeout, read from the environment (not hard-coded)."""
    return int(os.environ.get("HEALTH_CHECK_TIMEOUT", "2"))


def check_postgres():
    """Return True if PostgreSQL accepts a trivial query."""
    try:
        conn = psycopg2.connect(
            os.environ["DATABASE_URL"], connect_timeout=_timeout_seconds()
        )
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        finally:
            conn.close()
        return True
    except (psycopg2.Error, KeyError):
        return False


def check_redis():
    """Return True if Redis answers PING."""
    try:
        client = redis.Redis.from_url(
            os.environ["REDIS_URL"],
            socket_connect_timeout=_timeout_seconds(),
            socket_timeout=_timeout_seconds(),
        )
        return bool(client.ping())
    except (redis.RedisError, KeyError):
        return False


@health_bp.get("/healthz")
def healthz():
    """Liveness probe."""
    return jsonify({"status": "ok"}), 200


@health_bp.get("/readyz")
def readyz():
    """Readiness probe: 200 if every dependency is up, else 503."""
    checks = {
        "postgres": "ok" if check_postgres() else "unavailable",
        "redis": "ok" if check_redis() else "unavailable",
    }
    ready = all(v == "ok" for v in checks.values())
    body = {"status": "ready" if ready else "not_ready", "checks": checks}
    return jsonify(body), 200 if ready else 503
