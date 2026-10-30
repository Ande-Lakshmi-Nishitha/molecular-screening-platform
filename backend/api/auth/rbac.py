"""One access check for the whole API (FR-048, FR-049, FR-050, FR-051, SEC-001, SEC-002).

Owner: Ayushi. A single before_request hook authenticates every route except register, login
and the health/openapi modules, so a new route is protected by default. Paths under
/api/admin also need the administrator role (403 for a Researcher). Administrators can read
system data and manage accounts, but no route writes to results: only workers do.
"""
from flask import current_app, g, request
from redis.exceptions import RedisError

from backend.api.auth import sessions
from backend.api.auth.service import AuthError
from backend.core.config import Settings

ROLE_RESEARCHER = "researcher"
ROLE_ADMINISTRATOR = "administrator"
PUBLIC_ENDPOINTS = {"auth.register", "auth.login", "static"}
PUBLIC_MODULES = {"backend.api.health", "backend.api.openapi"}
ADMIN_PREFIX = "/api/admin"


def is_public():
    """True for requests that do not need a session."""
    if request.method == "OPTIONS" or request.endpoint is None:
        return True  # CORS preflight, or an unknown URL that Flask will answer with 404
    if request.endpoint in PUBLIC_ENDPOINTS:
        return True
    view = current_app.view_functions.get(request.endpoint)
    return getattr(view, "__module__", "") in PUBLIC_MODULES


def check_access():
    """before_request hook: 401 without a live session, 403 for admin paths without the role."""
    if not current_app.config.get("ACCESS_CONTROL", True) or is_public():
        return None
    try:
        g.claims = sessions.authenticate(request.headers.get("Authorization"),
                                         sessions.get_session_store(), Settings.from_env())
    except AuthError as err:
        return sessions.error_response(err.code, err.message, err.status)
    except RedisError:
        return sessions.error_response("SERVICE_UNAVAILABLE", "Service temporarily unavailable.", 503)
    if request.path.startswith(ADMIN_PREFIX) and g.claims.get("role") != ROLE_ADMINISTRATOR:
        return sessions.error_response("FORBIDDEN", "Administrator access required.", 403)
    return None


def install_access_control(app):
    """Register the hook. Call it before install_rate_limit so the limiter sees the user."""
    app.before_request(check_access)
    