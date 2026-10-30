"""API rate limiting with Redis counters (SEC-005).

Owner: Ayushi. The limit comes from config (RATE_LIMIT, TBD-C12). Logged-in callers are counted
per user, anonymous callers per IP. A breach returns 429 with a Retry-After header.
"""
from flask import current_app, g, request
from flask_limiter import Limiter
from flask_limiter.errors import RateLimitExceeded
from flask_limiter.util import get_remote_address

from backend.api.auth import sessions
from backend.core.config import Settings


def _key():
    claims = getattr(g, "claims", None)
    return f"user:{claims['sub']}" if claims else f"ip:{get_remote_address()}"


def _is_health():
    view = current_app.view_functions.get(request.endpoint)
    return getattr(view, "__module__", "") == "backend.api.health"


def _retry_after(err):
    try:
        return max(1, int(err.limit.limit.get_expiry()))
    except (AttributeError, TypeError, ValueError):
        return 60


def install_rate_limit(app, settings=None):
    """Attach the limiter and the 429 handler to the app."""
    settings = settings or Settings.from_env()
    storage = app.config.get("RATELIMIT_STORAGE_URI") or (
        "memory://" if app.config.get("TESTING") else settings.redis_url)
    limiter = Limiter(_key, app=app, default_limits=[settings.rate_limit], storage_uri=storage,
                      headers_enabled=True, swallow_errors=True,
                      default_limits_exempt_when=_is_health)

    @app.errorhandler(RateLimitExceeded)
    def too_many_requests(err):
        response, status = sessions.error_response(
            "RATE_LIMITED", "Too many requests. Try again later.", 429)
        response.headers["Retry-After"] = str(_retry_after(err))
        return response, status

    return limiter
