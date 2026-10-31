import threading
import time

import fakeredis
import pytest
import redis

from teammates.llm.client import LLMError
from teammates.llm.rate_limit import (
    DEFAULT_KEY,
    MeteredRateLimiter,
    RateLimiterUnavailable,
    RateLimitExceeded,
    RedisSlidingWindowRateLimiter,
)


@pytest.fixture
def server() -> fakeredis.FakeServer:
    """One in-memory Redis server; every client made from it shares the same data."""
    return fakeredis.FakeServer()


def _client(server: fakeredis.FakeServer) -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(server=server)


def _limiter(server: fakeredis.FakeServer, **kwargs: object) -> RedisSlidingWindowRateLimiter:
    return RedisSlidingWindowRateLimiter(_client(server), **kwargs)


def test_defaults_are_35_per_60s_and_36th_must_wait(server: fakeredis.FakeServer) -> None:
    sleeps: list[float] = []
    limiter = _limiter(server, max_wait_s=0, sleep=sleeps.append)
    assert [limiter.acquire() for _ in range(35)] == [0.0] * 35
    with pytest.raises(RateLimitExceeded) as info:
        limiter.acquire()
    assert "35/60s" in str(info.value)
    assert sleeps == []


def test_waits_until_oldest_grant_leaves_window(server: fakeredis.FakeServer) -> None:
    limiter = _limiter(server, max_requests=2, window_s=0.2)
    start = time.monotonic()
    limiter.acquire()
    limiter.acquire()
    waited = limiter.acquire()  # third must wait ~0.2 s for the first to expire
    elapsed = time.monotonic() - start
    assert 0.15 <= waited <= 0.25
    assert elapsed >= 0.18


def test_limit_is_shared_across_limiter_instances(server: fakeredis.FakeServer) -> None:
    """Two processes = two limiters on the same Redis key: they share one budget."""
    a = _limiter(server, max_requests=3, window_s=60, max_wait_s=0)
    b = _limiter(server, max_requests=3, window_s=60, max_wait_s=0)
    a.acquire()
    b.acquire()
    a.acquire()
    with pytest.raises(RateLimitExceeded):
        b.acquire()
    assert _client(server).zcard(DEFAULT_KEY) == 3


def test_different_keys_have_separate_budgets(server: fakeredis.FakeServer) -> None:
    a = _limiter(server, key="ratelimit:a", max_requests=1, max_wait_s=0)
    b = _limiter(server, key="ratelimit:b", max_requests=1, max_wait_s=0)
    a.acquire()
    b.acquire()
    with pytest.raises(RateLimitExceeded):
        a.acquire()


def test_key_expires_so_idle_limiters_leave_no_data(server: fakeredis.FakeServer) -> None:
    _limiter(server, window_s=60).acquire()
    ttl_ms = _client(server).pttl(DEFAULT_KEY)
    assert 0 < ttl_ms <= 60_000


def test_thread_safe_under_real_concurrency(server: fakeredis.FakeServer) -> None:
    """9 threads against 3 per 0.2 s: grants never exceed 3 in any 0.2 s window."""
    limiter = _limiter(server, max_requests=3, window_s=0.2)
    grants: list[float] = []
    lock = threading.Lock()

    def worker() -> None:
        limiter.acquire()
        with lock:
            grants.append(time.monotonic())

    threads = [threading.Thread(target=worker) for _ in range(9)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    grants.sort()
    assert len(grants) == 9
    for i, start in enumerate(grants):
        assert len([g for g in grants[i:] if g - start < 0.18]) <= 3
    assert grants[-1] - grants[0] >= 0.35  # 9 requests at 3 per 0.2 s need >= 2 windows


def test_redis_down_fails_closed() -> None:
    class DownRedis:
        def register_script(self, script: str):
            def run(**_: object) -> int:
                raise redis.ConnectionError("refused")

            return run

    limiter = RedisSlidingWindowRateLimiter(DownRedis())  # type: ignore[arg-type]
    with pytest.raises(RateLimiterUnavailable, match="ConnectionError"):
        limiter.acquire()


def test_limiter_errors_are_llm_errors_so_planner_escalates() -> None:
    assert issubclass(RateLimitExceeded, LLMError)
    assert issubclass(RateLimiterUnavailable, LLMError)


@pytest.mark.parametrize(
    "kwargs",
    [{"max_requests": 0}, {"window_s": 0}, {"window_s": 0.0001}, {"max_wait_s": -1}],
)
def test_invalid_settings_rejected(server: fakeredis.FakeServer, kwargs: dict) -> None:
    with pytest.raises(ValueError):
        _limiter(server, **kwargs)


class _FixedWait:
    """Limiter stub; each acquire advances a fake clock by `step` seconds."""

    def __init__(self, clock: list[float], step: float, fail: bool = False) -> None:
        self.clock, self.step, self.fail = clock, step, fail

    def acquire(self) -> float:
        self.clock[0] += self.step
        if self.fail:
            raise RateLimitExceeded("full")
        return self.step


def test_metered_limiter_sums_wait_per_thread_and_resets() -> None:
    now = [0.0]
    metered = MeteredRateLimiter(_FixedWait(now, 1.5), clock=lambda: now[0])
    metered.acquire()
    metered.acquire()
    assert metered.take() == 3.0
    assert metered.take() == 0.0


def test_metered_limiter_counts_wait_even_when_refused() -> None:
    now = [0.0]
    metered = MeteredRateLimiter(_FixedWait(now, 2.0, fail=True), clock=lambda: now[0])
    with pytest.raises(RateLimitExceeded):
        metered.acquire()
    assert metered.take() == 2.0


def test_metered_limiter_isolates_threads(server: fakeredis.FakeServer) -> None:
    metered = MeteredRateLimiter(_limiter(server, max_requests=100))
    seen: dict[int, float] = {}

    def work(i: int) -> None:
        metered.acquire()
        seen[i] = metered.take()

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(seen) == 4 and all(0 <= w < 1 for w in seen.values())
    assert metered.take() == 0.0  # main thread never waited
