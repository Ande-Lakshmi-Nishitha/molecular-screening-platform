"""Failed-login counter per email string (FR-052).

Owner: Ayushi. Counter is kept per email text, so lockout does not reveal whether an
email is registered. Limits come from config (TBD-C12).
"""
import hashlib

import redis

from backend.core.config import Settings


def email_key(email):
    """Return a hash of the normalised email, used as the counter key and in the audit log."""
    return hashlib.sha256(email.strip().lower().encode("utf-8")).hexdigest()


class LockoutStore:
    """Counts failed logins in Redis and says when an account is locked."""

    def __init__(self, client, max_failed, lockout_min):
        self.client = client
        self.max_failed = max_failed
        self.lockout_min = lockout_min

    def _key(self, email):
        return f"lockout:{email_key(email)}"

    def is_locked(self, email):
        """True when the failed-login count has reached the limit."""
        value = self.client.get(self._key(email))
        return value is not None and int(value) >= self.max_failed

    def register_failure(self, email):
        """Count one failure; return True if this failure locks the account."""
        key = self._key(email)
        count = self.client.incr(key)
        self.client.expire(key, self.lockout_min * 60)
        return int(count) >= self.max_failed

    def reset(self, email):
        """Clear the counter after a successful login."""
        self.client.delete(self._key(email))


def get_lockout_store():
    """Build the store from config and the Redis URL."""
    settings = Settings.from_env()
    client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    return LockoutStore(client, settings.max_failed_logins, settings.lockout_min)
