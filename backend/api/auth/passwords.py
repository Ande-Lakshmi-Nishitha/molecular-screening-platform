"""Argon2id password hashing (FR-043).

Owner: Ayushi.
"""
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

_HASHER = PasswordHasher()  # Argon2id with a random salt per password
_DUMMY_HASH = _HASHER.hash("dummy-password-for-timing")


def hash_password(password):
    """Return a salted Argon2id hash."""
    return _HASHER.hash(password)


def verify_password(stored_hash, password):
    """Return True only if the password matches the stored hash."""
    try:
        return _HASHER.verify(stored_hash, password)
    except (VerificationError, InvalidHashError):
        return False


def verify_missing_user(password):
    """Do the same hashing work as a real check when the email is unknown (FR-044)."""
    verify_password(_DUMMY_HASH, password)
    return False
