"""PubChem limiter: PUBCHEM_RPS requests per second across ALL processes (MSP-21; FR-055).

SharedLimiter keeps a sliding window in Redis (one Lua script, using Redis's own clock
so processes cannot disagree about time). If Redis is unreachable it falls back to a
per-process limiter, so search degrades instead of failing.
"""
import logging
import threading
import time
import uuid
from collections import deque

from backend.core.config import Settings
from backend.services.search_client.errors import RateLimited

LOGGER = logging.getLogger("msp.search")

KEY = "msp:ratelimit:pubchem"
WINDOW_MS = 1000

_LUA = """
local t = redis.call('TIME')
local now = t[1] * 1000 + math.floor(t[2] / 1000)
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
redis.call('ZREMRANGEBYSCORE', KEYS[1], 0, now - window)
if redis.call('ZCARD', KEYS[1]) < limit then
  redis.call('ZADD', KEYS[1], now, now .. '-' .. ARGV[3])
  redis.call('PEXPIRE', KEYS[1], window)
  return 0
end
local oldest = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
return (tonumber(oldest[2]) + window) - now
"""


class InProcessLimiter:
    def __init__(self, limit, window=1.0, clock=time.monotonic, sleep=time.sleep):
        self.limit, self.window, self._clock, self._sleep = limit, window, clock, sleep
        self._stamps = deque()
        self._lock = threading.Lock()

    def acquire(self, service="PubChem", max_wait=5.0):
        deadline = self._clock() + max_wait
        while True:
            with self._lock:
                now = self._clock()
                while self._stamps and self._stamps[0] <= now - self.window:
                    self._stamps.popleft()
                if len(self._stamps) < self.limit:
                    self._stamps.append(now)
                    return
                wait = self._stamps[0] + self.window - now
            if self._clock() + wait > deadline:
                raise RateLimited(service, f"{service} request limit reached. Try again shortly.")
            self._sleep(wait)


class SharedLimiter:
    def __init__(self, client_getter, limit=None, clock=time.monotonic, sleep=time.sleep):
        self._get_client = client_getter
        self.limit = limit if limit is not None else Settings.from_env().pubchem_rps
        self._clock, self._sleep = clock, sleep
        self._fallback = InProcessLimiter(self.limit, WINDOW_MS / 1000, clock, sleep)

    def _wait_ms(self):
        return int(self._get_client().eval(_LUA, 1, KEY, self.limit, WINDOW_MS, uuid.uuid4().hex))

    def acquire(self, service="PubChem", max_wait=5.0):
        deadline = self._clock() + max_wait
        while True:
            try:
                wait_ms = self._wait_ms()
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("redis limiter unavailable (%s); using per-process limiter", type(exc).__name__)
                return self._fallback.acquire(service, max_wait)
            if wait_ms <= 0:
                return
            if self._clock() + wait_ms / 1000 > deadline:
                raise RateLimited(service, f"{service} request limit reached. Try again shortly.")
            self._sleep(wait_ms / 1000)
