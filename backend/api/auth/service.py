"""Registration and login logic (FR-041, FR-042, FR-043, FR-044, FR-052, FR-065).

Owner: Ayushi.
"""
import re

from sqlalchemy.exc import IntegrityError

from backend.api.auth.lockout import email_key
from backend.api.auth.passwords import hash_password, verify_missing_user, verify_password
from backend.core import audit
from backend.db.repositories import users

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LEN = 8


class AuthError(Exception):
    """An error that maps to the uniform API error body."""

    def __init__(self, code, message, status, details=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = details or {}


def _normalise(email):
    return email.strip().lower()


def _require_strings(email, password):
    if not isinstance(email, str) or not isinstance(password, str) or not email or not password:
        raise AuthError("VALIDATION_ERROR", "Email and password are required.", 422)


def register(session, email, password):
    """Create a Researcher account. 409 if the email exists, 422 for a bad body."""
    _require_strings(email, password)
    email = _normalise(email)
    if not EMAIL_RE.match(email):
        raise AuthError("VALIDATION_ERROR", "Enter a valid email address.", 422, {"field": "email"})
    if len(password) < MIN_PASSWORD_LEN:
        raise AuthError("VALIDATION_ERROR", f"Password must be at least {MIN_PASSWORD_LEN} characters.",
                        422, {"field": "password"})
    if users.get_by_email(session, email):
        raise AuthError("EMAIL_TAKEN", "An account with this email already exists.", 409)
    try:
        return users.create(session, email, hash_password(password))
    except IntegrityError as exc:
        raise AuthError("EMAIL_TAKEN", "An account with this email already exists.", 409) from exc


def login(session, store, email, password):
    """Check credentials. Same 401 message for a wrong email or a wrong password."""
    _require_strings(email, password)
    email = _normalise(email)
    key = email_key(email)
    if store.is_locked(email):
        audit.record(session, "login_blocked", detail={"email_key": key})
        raise AuthError("ACCOUNT_LOCKED", "Too many failed attempts. Try again later.", 429)

    user = users.get_by_email(session, email)
    if user is not None:
        ok = verify_password(user.password_hash, password)
    else:
        ok = verify_missing_user(password)

    if not ok:
        locked = store.register_failure(email)
        user_id = user.id if user is not None else None
        audit.record(session, "login_failed", user_id, {"email_key": key})
        if locked:
            audit.record(session, "account_locked", user_id, {"email_key": key})
        raise AuthError("INVALID_CREDENTIALS", "Invalid email or password.", 401)

    store.reset(email)
    audit.record(session, "login", user.id, {"email_key": key})
    return user
