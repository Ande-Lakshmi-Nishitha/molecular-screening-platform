"""Auth routes: register and login (FR-041, FR-042, FR-044).

Owner: Ayushi. Tokens and sessions are added in MSP-17.
"""
import uuid

from flask import Blueprint, jsonify, request
from redis.exceptions import RedisError

from backend.api.auth import service
from backend.api.auth.lockout import get_lockout_store
from backend.db.session import session_scope

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")


def _error(code, message, status, details=None):
    body = {"error": {"code": code, "message": message, "details": details or {},
                      "request_id": request.headers.get("X-Request-ID") or str(uuid.uuid4())}}
    return jsonify(body), status


def _fields():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise service.AuthError("VALIDATION_ERROR", "Request body must be JSON.", 422)
    return data.get("email"), data.get("password")


def _user_body(user):
    return {"id": str(user.id), "email": user.email, "role": user.role}


@auth_bp.post("/register")
def register():
    """Create an account (201)."""
    try:
        email, password = _fields()
        with session_scope() as session:
            body = _user_body(service.register(session, email, password))
        return jsonify(body), 201
    except service.AuthError as err:
        return _error(err.code, err.message, err.status, err.details)


@auth_bp.post("/login")
def login():
    """Check credentials (200). Failed attempts still commit their audit rows."""
    try:
        email, password = _fields()
        store = get_lockout_store()
        with session_scope() as session:
            try:
                body = _user_body(service.login(session, store, email, password))
            except service.AuthError as err:
                return _error(err.code, err.message, err.status, err.details)
        return jsonify(body), 200
    except service.AuthError as err:
        return _error(err.code, err.message, err.status, err.details)
    except RedisError:
        return _error("SERVICE_UNAVAILABLE", "Service temporarily unavailable.", 503)
    