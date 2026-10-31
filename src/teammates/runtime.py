"""Process-wide clients shared by every teammate in the bot process (FSE, collections)."""

import logging
import os
from functools import cache
from pathlib import Path

import redis

from teammates.config import load_settings, read_env_file
from teammates.llm.adapters.sarvam import SarvamClient
from teammates.llm.client import LLMClient
from teammates.llm.rate_limit import (
    DEFAULT_KEY,
    DEFAULT_MAX_REQUESTS,
    DEFAULT_WINDOW_S,
    RateLimiter,
    RedisSlidingWindowRateLimiter,
)

log = logging.getLogger(__name__)
_FILE_ENV = read_env_file(Path(".env"))
MODEL = "sarvam-105b-conversations"  # same as orchestrator planner


def env(key: str, default: str = "") -> str:
    return os.environ.get(key) or _FILE_ENV.get(key, default)


@cache
def redis_client() -> redis.Redis | None:
    url = env("REDIS_URL")
    if not url:
        return None
    return redis.Redis.from_url(url, socket_timeout=2, socket_connect_timeout=2,
                                health_check_interval=30)


class _Unlimited:
    def acquire(self) -> float:
        return 0.0


@cache
def rate_limiter(key: str, max_requests: int, window_s: float,
                 max_wait_s: float | None = None) -> RateLimiter:
    """Shared Redis sliding-window limiter for one key (all threads, processes, servers).
    Redis unreachable at startup -> unlimited, logged; Redis lost later -> fails closed."""
    r = redis_client()
    try:
        if r is None:
            raise redis.ConnectionError("REDIS_URL not set")
        r.ping()
        return RedisSlidingWindowRateLimiter(r, key=key, max_requests=max_requests,
                                             window_s=window_s, max_wait_s=max_wait_s)
    except redis.RedisError:
        # ponytail: a few calls per morning; fine unlimited until volume grows.
        log.warning("Redis unreachable; %s is not rate limited", key)
        return _Unlimited()


@cache
def llm() -> LLMClient:
    s = load_settings()
    limiter = rate_limiter(DEFAULT_KEY, DEFAULT_MAX_REQUESTS, DEFAULT_WINDOW_S)
    return SarvamClient(s.sarvam_api_key, limiter, base_url=s.sarvam_base_url)
