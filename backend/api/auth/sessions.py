"""JWT with a Redis session record (FR-045, FR-046, FR-047, SEC-001).

Owner: Ayushi. A JWT alone cannot expire on inactivity or be revoked at logout, so each
login also writes `session:{jti}` to Redis with a TTL equal to the idle limit. Every
request checks the signature, then that the record exists, then resets its TTL.
"""
import time
import uuid
from functools import wraps

import jwt
import redis
from flask import g, jsonify, request
from redis.exceptions import RedisError

from backend.api.auth.service import AuthError
from backend.core.config import Settings

ALGORITHM = "HS256"


def error_response(code, message, status, details=None):
    """Build the uniform error body."""
    body = {"error": {"code": code, "message": message, "details": details or {},
                      "request_id": request.headers.get("X-Request-ID") or str(uuid.uuid4())}}
    return jsonify(body), status


def get_secret(settings):
    """Return the signing secret; refuse to run without one (SEC-003)."""
    if not settings.jwt_secret:
        raise RuntimeError("JWT_SECRET is not set")
    return settings.jwt_secret


class SessionStore:
    """Session records in Redis."""

    def __init__(self, client, idle_seconds):
        self.client = client
        self.idle_seconds = idle_seconds

    @staticmethod
    def _key(jti):
        return f"session:{jti}"

    def create(self, jti, user_id):
        """Write the record with the idle TTL."""
        self.client.set(self._key(jti), str(user_id), ex=self.idle_seconds)

    def touch(self, jti):
        """Reset the idle TTL. Returns False if the record is gone."""
        return bool(self.client.expire(self._key(jti), self.idle_seconds))

    def delete(self, jti):
        """Remove the record (logout)."""
        self.client.delete(self._key(jti))


def get_session_store():
    """Build the store from config and the Redis URL."""
    settings = Settings.from_env()
    client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    return SessionStore(client, settings.session_idle_min * 60)


def issue_token(store, user, settings):
    """Create the session record and return a signed JWT (user id, role, jti, expiry)."""
    secret = get_secret(settings)
    jti = uuid.uuid4().hex
    now = int(time.time())
    claims = {"sub": str(user.id), "role": user.role, "jti": jti, "iat": now,
              "exp": now + settings.session_max_hours * 3600}
    store.create(jti, user.id)
    return jwt.encode(claims, secret, algorithm=ALGORITHM)


def authenticate(header, store, settings):
    """Return the token claims, or raise AuthError (401)."""
    if not header or not header.startswith("Bearer "):
        raise AuthError("UNAUTHENTICATED", "Authentication required.", 401)
    token = header[len("Bearer "):].strip()
    try:
        claims = jwt.decode(token, get_secret(settings), algorithms=[ALGORITHM],
                            options={"require": ["exp", "jti", "sub"]})
    except jwt.ExpiredSignatureError:
        raise AuthError("SESSION_EXPIRED", "Session expired. Please log in again.", 401) from None
    except jwt.InvalidTokenError:
        raise AuthError("UNAUTHENTICATED", "Authentication required.", 401) from None
    if not store.touch(claims["jti"]):
        raise AuthError("SESSION_EXPIRED", "Session expired. Please log in again.", 401)
    return claims


def login_required(view):
    """Reject the request unless it carries a live session. Redis down gives 503."""
    @wraps(view)
    def wrapper(*args, **kwargs):
        try:
            g.claims = authenticate(request.headers.get("Authorization"),
                                    get_session_store(), Settings.from_env())
        except AuthError as err:
            return error_response(err.code, err.message, err.status)
        except RedisError:
            return error_response("SERVICE_UNAVAILABLE", "Service temporarily unavailable.", 503)
        return view(*args, **kwargs)
    return wrapper
