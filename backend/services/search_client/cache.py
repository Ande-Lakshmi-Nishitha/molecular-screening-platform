"""24 h Redis cache; failures and empty results are not cached (MSP-21; FR-055).

If Redis is down this behaves as a permanent miss, so search still works uncached.
"""
import hashlib
import json
import logging

from backend.core.config import Settings

LOGGER = logging.getLogger("msp.search")

_client = None


def get_redis():
    """One lazily created Redis client, shared by the cache and the limiter."""
    global _client  # pylint: disable=global-statement
    if _client is None:
        import redis  # pylint: disable=import-outside-toplevel

        _client = redis.Redis.from_url(
            Settings.from_env().redis_url,
            socket_connect_timeout=1, socket_timeout=1, decode_responses=True,
        )
    return _client


class SearchCache:
    def __init__(self, client_getter=get_redis, ttl_seconds=None):
        self._get_client = client_getter
        self.ttl = ttl_seconds if ttl_seconds is not None else Settings.from_env().cache_ttl_hours * 3600

    @staticmethod
    def key(source, query):
        digest = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()
        return f"msp:search:{source}:{digest}"

    def get(self, source, query):
        try:
            raw = self._get_client().get(self.key(source, query))
            return json.loads(raw) if raw else None
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("search cache read failed: %s", type(exc).__name__)
            return None

    def set(self, source, query, payload):
        try:
            self._get_client().setex(self.key(source, query), self.ttl, json.dumps(payload))
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("search cache write failed: %s", type(exc).__name__)
