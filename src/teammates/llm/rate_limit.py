"""Redis sliding-window rate limiter: at most N requests in ANY rolling window of W seconds.

The limit is shared by every thread, process and server that uses the same Redis key.
"""

import logging
import os
import socket
import threading
import time
from collections.abc import Callable
from itertools import count
from typing import Protocol

import redis

from teammates.llm.client import LLMError

logger = logging.getLogger(__name__)

DEFAULT_MAX_REQUESTS = 35
DEFAULT_WINDOW_S = 60.0
DEFAULT_KEY = "ratelimit:sarvam"

# Atomic check-and-record. A sorted set holds one member per granted request, scored by
# grant time in ms from the Redis server clock (so all hosts agree on "now").
# Returns 0 if granted, else milliseconds until the oldest grant leaves the window.
_SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local member = ARGV[3]
local t = redis.call('TIME')
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
redis.call('ZREMRANGEBYSCORE', key, '-inf', now_ms - window_ms)
if redis.call('ZCARD', key) < limit then
  redis.call('ZADD', key, now_ms, member)
  redis.call('PEXPIRE', key, window_ms)
  return 0
end
local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
return math.max(1, tonumber(oldest[2]) + window_ms - now_ms)
"""


class RateLimiter(Protocol):
    """Blocks until a request may be sent (or raises LLMError if it may not)."""

    def acquire(self) -> float:
        """Wait for a slot; return seconds waited."""
        ...


class MeteredRateLimiter:
    """Wraps a limiter and records wall-clock time spent waiting for a slot, per thread.

    Lets callers report queueing separately from provider latency; take() returns this
    thread's total wait since its last take() and resets it.
    """

    def __init__(
        self, inner: RateLimiter, clock: Callable[[], float] = time.perf_counter
    ) -> None:
        self._inner = inner
        self._clock = clock
        self._local = threading.local()

    def acquire(self) -> float:
        """Delegate to the wrapped limiter, timing how long this thread was blocked."""
        start = self._clock()
        try:
            return self._inner.acquire()
        finally:
            self._local.wait_s = self._current() + (self._clock() - start)

    def take(self) -> float:
        """Return seconds this thread waited since the last take(), then reset."""
        waited = self._current()
        self._local.wait_s = 0.0
        return waited

    def _current(self) -> float:
        return getattr(self._local, "wait_s", 0.0)


class RateLimitExceeded(LLMError):
    """Waiting for a free slot would take longer than the caller allows."""


class RateLimiterUnavailable(LLMError):
    """Redis could not be reached, so no request is allowed (fail closed)."""


class RedisSlidingWindowRateLimiter:
    """Sliding-window log in a Redis sorted set.

    A request is granted only if fewer than max_requests grants fall inside the last
    window_s seconds; otherwise the caller sleeps until the oldest grant expires. Unlike
    fixed per-minute buckets this never allows a burst of 2N across a minute boundary.
    If Redis is down the limiter fails closed: it raises instead of letting calls through.
    """

    def __init__(
        self,
        client: redis.Redis,
        *,
        key: str = DEFAULT_KEY,
        max_requests: int = DEFAULT_MAX_REQUESTS,
        window_s: float = DEFAULT_WINDOW_S,
        max_wait_s: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_requests < 1:
            raise ValueError("max_requests must be >= 1")
        if window_s * 1000 < 1:
            raise ValueError("window_s must be at least 1 ms")
        if max_wait_s is not None and max_wait_s < 0:
            raise ValueError("max_wait_s must be >= 0")
        self._script = client.register_script(_SLIDING_WINDOW_LUA)
        self._key = key
        self._max_requests = max_requests
        self._window_ms = round(window_s * 1000)
        self._max_wait_s = max_wait_s
        self._sleep = sleep
        # Unique member per grant across hosts, processes and limiter instances.
        self._member_prefix = f"{socket.gethostname()}:{os.getpid()}:{id(self)}"
        self._counter = count(1)

    def _try_grant(self) -> float:
        """Grant a slot and return 0.0, or return seconds until the next slot frees up."""
        member = f"{self._member_prefix}:{next(self._counter)}"
        try:
            wait_ms = self._script(
                keys=[self._key], args=[self._max_requests, self._window_ms, member]
            )
        except redis.RedisError as exc:
            raise RateLimiterUnavailable(f"Rate limiter Redis error: {type(exc).__name__}") from exc
        return int(wait_ms) / 1000

    def acquire(self) -> float:
        """Block until a request may be sent; return total seconds waited.

        Raises RateLimitExceeded if the wait would exceed max_wait_s, and
        RateLimiterUnavailable if Redis cannot be reached.
        """
        waited = 0.0
        while (wait_s := self._try_grant()) > 0:
            if self._max_wait_s is not None and waited + wait_s > self._max_wait_s:
                raise RateLimitExceeded(
                    f"Rate limit {self._max_requests}/{self._window_ms / 1000:g}s: next slot "
                    f"in {wait_s:.1f}s exceeds max wait {self._max_wait_s:g}s"
                )
            logger.info("rate limit reached, waiting %.2fs", wait_s)
            self._sleep(wait_s)
            waited += wait_s
        return waited
