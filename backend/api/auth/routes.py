"""Auth routes: register, login and logout (FR-041, FR-042, FR-044, FR-045, FR-047).

Owner: Ayushi.
"""
from flask import Blueprint, g, jsonify, request
from redis.exceptions import RedisError

from backend.api.auth import service
from backend.api.auth.lockout import get_lockout_store
from backend.api.auth.sessions import (error_response, get_session_store,
                                       issue_token, login_required)
from backend.core.config import Settings
from backend.db.session import session_scope

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")


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
        return error_response(err.code, err.message, err.status, err.details)


@auth_bp.post("/login")
def login():
    """Check credentials and return a token (200). Failed attempts still commit their audit rows."""
    try:
        email, password = _fields()
        store = get_lockout_store()
        with session_scope() as session:
            try:
                user = service.login(session, store, email, password)
            except service.AuthError as err:
                return error_response(err.code, err.message, err.status, err.details)
            body = _user_body(user)
            token = issue_token(get_session_store(), user, Settings.from_env())
        return jsonify({"token": token, "token_type": "Bearer", "user": body}), 200
    except service.AuthError as err:
        return error_response(err.code, err.message, err.status, err.details)
    except RedisError:
        return error_response("SERVICE_UNAVAILABLE", "Service temporarily unavailable.", 503)


@auth_bp.post("/logout")
@login_required
def logout():
    """End the session: delete the Redis record so the token stops working (FR-047)."""
    try:
        get_session_store().delete(g.claims["jti"])
    except RedisError:
        return error_response("SERVICE_UNAVAILABLE", "Service temporarily unavailable.", 503)
    return "", 204
