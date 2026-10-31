from datetime import UTC, datetime, timedelta

import fakeredis
import pytest
import redis

from teammates.support import risk
from teammates.support.tools import ContextCache, MockTools, call, parse_signals, seed

NOW = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)
TTLS = risk.config()["cache"]


@pytest.fixture
def tools():
    return MockTools(seed()["merchants"], NOW)


def test_only_allowlisted_tools_with_typed_parameters(tools):
    assert call(tools, "find_merchant", "RaviStores@example.com ") == 102
    assert call(tools, "get_user_details", 102)["tier"] == "key"
    with pytest.raises(PermissionError):
        call(tools, "delete_merchant", 102)
    with pytest.raises(PermissionError):
        call(tools, "__init__", 102)
    for name, bad in (("get_user_details", "102 OR 1=1"), ("find_merchant", 102),
                      ("get_txn_status", True)):
        with pytest.raises(TypeError):
            call(tools, name, bad)


def test_unknown_contact_and_merchant(tools):
    assert tools.find_merchant("919811112222") is None
    assert tools.get_user_details(999) is None


def test_security_signals_relative_to_now(tools):
    s = parse_signals(tools.get_security_signals(104))
    assert s["sim_change_at"] == NOW - timedelta(days=3)
    assert len(s["failed_logins"]) == 3 and s["blocked_now"] is False
    clean = parse_signals(tools.get_security_signals(101))
    assert clean["sim_change_at"] is None and clean["failed_logins"] == []


def test_txn_status_has_dates(tools):
    assert tools.get_txn_status(101)[0] | {} == {
        "ref": "S-40101", "kind": "settlement", "amount": 12480, "status": "settled",
        "days_ago": 1, "date": "2026-10-29"}


def test_cache_hit_after_miss_with_part_ttl_and_invalidate():
    r = fakeredis.FakeRedis()
    c, calls = ContextCache(r, TTLS), []

    def fetch():
        calls.append(1)
        return {"two_fa": True}

    assert c.get_or_fetch(1, 7, "security", fetch) == ({"two_fa": True}, False)
    assert c.get_or_fetch(1, 7, "security", fetch) == ({"two_fa": True}, True)
    assert len(calls) == 1 and 0 < r.ttl("support:1:7:security") <= TTLS["security_ttl_s"]
    c.get_or_fetch(1, 7, "profile", lambda: {"x": 1})
    assert r.ttl("support:1:7:profile") > TTLS["security_ttl_s"]
    assert c.get_or_fetch(1, 8, "security", fetch)[1] is False  # another ticket: own key
    c.invalidate(1, 7)
    assert c.get_or_fetch(1, 7, "security", fetch)[1] is False and len(calls) == 3


def test_cache_down_calls_the_tool():
    class Down:
        def get(self, *a):
            raise redis.ConnectionError("down")

        setex = delete = get

    c = ContextCache(Down(), TTLS)
    assert c.get_or_fetch(1, 1, "txns", lambda: [1]) == ([1], False)
    c.invalidate(1, 1)  # no exception
    assert ContextCache(None, TTLS).get_or_fetch(1, 1, "txns", lambda: [2]) == ([2], False)


def test_cached_any_key_with_its_own_ttl():
    r = fakeredis.FakeRedis()
    c, calls = ContextCache(r, TTLS), []

    def fetch():
        calls.append(1)
        return [{"a": 1}]

    assert c.cached("kam:x:2026-10-31:meetings", 900, fetch) == ([{"a": 1}], False)
    assert c.cached("kam:x:2026-10-31:meetings", 900, fetch) == ([{"a": 1}], True)
    assert len(calls) == 1 and 0 < r.ttl("kam:x:2026-10-31:meetings") <= 900
