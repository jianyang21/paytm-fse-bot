from datetime import date, datetime, timedelta

import pytest
from pydantic import ValidationError

from teammates.kam.digest import config
from teammates.kam.tools import (
    ALLOWED,
    MockKamTools,
    UnknownKam,
    call,
    fixed_source_items,
    seed,
)
from teammates.support.tools import seed as support_seed

CFG = config()
NOW = datetime(2026, 10, 31, 9, 0, tzinfo=CFG.tz)


@pytest.fixture
def tools():
    s = seed()
    return MockKamTools(support_seed()["merchants"] | s["merchants"], s["kams"], s["meetings"],
                        s["feed"], CFG.sources, NOW, CFG.tz)


def test_only_allowlisted_tools(tools):
    assert set(ALLOWED) == {"get_key_accounts", "get_txn_status", "get_meetings",
                            "get_competitor_updates"}
    for name in ("get_security_signals", "find_merchant", "__class__", "migrate"):
        with pytest.raises(PermissionError):
            call(tools, name, kam_id="kam@example.com")


@pytest.mark.parametrize("name,kwargs", [
    ("get_meetings", {"kam_id": "not-an-email", "day": date(2026, 10, 31)}),
    ("get_meetings", {"kam_id": "kam@example.com"}),                          # missing day
    ("get_meetings", {"kam_id": "kam@example.com", "day": "tomorrow"}),
    ("get_meetings", {"kam_id": "kam@example.com", "day": date(2026, 10, 31), "sql": "1=1"}),
    ("get_competitor_updates", {"since": datetime(2026, 10, 1)}),            # naive datetime
    ("get_txn_status", {"merchant_id": "102"}),                               # strict int
    ("get_key_accounts", {"kam_id": "' OR 1=1 --"}),
])
def test_arguments_are_validated_before_the_tool_runs(tools, name, kwargs):
    with pytest.raises(ValidationError):
        call(tools, name, **kwargs)
    assert tools.calls == []


def test_key_accounts_are_the_kams_own(tools):
    got = call(tools, "get_key_accounts", kam_id=" KAM@example.com ")
    assert [a.merchant_id for a in got] == [102, 104, 108]
    assert [a.merchant_id for a in call(tools, "get_key_accounts", kam_id="kam2@example.com")
            ] == [201]
    assert call(tools, "get_key_accounts", kam_id="kam3@example.com") == []


def test_declared_fields_only(tools):
    a = call(tools, "get_key_accounts", kam_id="kam@example.com")[0]
    assert set(a.model_dump()) == {"merchant_id", "business", "name", "language"}  # no contacts
    u = call(tools, "get_competitor_updates", since=NOW - timedelta(days=7))[0]
    assert set(u.model_dump()) == {"competitor", "source", "title", "body", "url",
                                   "published_at"}  # no internal_id / author_email
    t = call(tools, "get_txn_status", merchant_id=102)[0]
    assert set(t.model_dump()) == {"ref", "kind", "amount", "status", "date"}


def test_unknown_kam_is_an_error_not_an_empty_calendar(tools):
    with pytest.raises(UnknownKam):
        call(tools, "get_meetings", kam_id="nobody@example.com", day=NOW.date())
    with pytest.raises(UnknownKam):
        call(tools, "get_key_accounts", kam_id="nobody@example.com")


def test_meetings_for_one_local_day(tools):
    today = call(tools, "get_meetings", kam_id="kam@example.com", day=NOW.date())
    assert [(m.starts_at.astimezone(CFG.tz).strftime("%H:%M"), m.merchant_id) for m in today
            ] == [("10:30", 102), ("15:00", 104), ("17:00", None)]
    tomorrow = call(tools, "get_meetings", kam_id="kam@example.com",
                    day=NOW.date() + timedelta(days=1))
    assert [m.title for m in tomorrow] == ["Soundbox rollout plan"]
    assert call(tools, "get_meetings", kam_id="kam4@example.com", day=NOW.date()) == []


def test_competitor_updates_fixed_sources_with_links(tools):
    got = call(tools, "get_competitor_updates", since=NOW - timedelta(days=7))
    assert [u.competitor for u in got] == ["ZetaPay", "OrbitPOS", "KiranaCloud"]  # newest first
    for u in got:
        assert u.url.startswith("https://")
        assert any(u.url.startswith(s.url) for s in CFG.sources)
    # not a fixed source, or older than `since`: never returned
    assert all("random-blog" not in u.url and "old-announcement" not in u.url for u in got)


def test_an_update_without_a_https_link_is_rejected():
    feed = [{"url": "https://news.zetapay.example/x", "title": "t", "body": "b",
             "published_days_ago": 0}]
    items = fixed_source_items(feed, CFG.sources, NOW)
    assert items[0]["source"] == "ZetaPay newsroom"
    t = MockKamTools({}, ["kam@example.com"], [], feed, CFG.sources, NOW, CFG.tz)
    t.feed[0]["url"] = "http://news.zetapay.example/x"
    with pytest.raises(ValidationError):
        call(t, "get_competitor_updates", since=NOW - timedelta(days=1))


@pytest.fixture
def pg():
    import uuid

    import fakeredis
    import psycopg
    from psycopg import sql

    from teammates.support.store import PgStore
    from tests.fse.conftest import DSN
    if not DSN:
        pytest.skip("TEST_DATABASE_URL not set")
    schema = f"t_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(DSN, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    s = PgStore(DSN, cache=fakeredis.FakeRedis(), schema=schema)
    try:
        s.migrate()
        yield s
    finally:
        s.pool.close()
        with psycopg.connect(DSN, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def test_pg_tools_read_the_kam_tables(pg):
    from teammates.kam.tools import PgKamTools
    s = seed()
    pg.upsert_reviewer("kam@example.com", "pw", "Anita", "kam")
    t = PgKamTools(pg, support_seed()["merchants"], NOW, CFG.tz)
    t.migrate()
    t.migrate()  # idempotent
    for _ in range(2):  # idempotent seed: no duplicate meetings or updates
        t.seed_demo(s["meetings"], s["feed"], CFG.sources, NOW)
    got = call(t, "get_meetings", kam_id="kam@example.com", day=NOW.date())
    assert [m.title for m in got] == ["Quarterly business review",
                                      "Settlement hold follow-up", "Weekly KAM sync"]
    ups = call(t, "get_competitor_updates", since=NOW - timedelta(days=7))
    assert [u.competitor for u in ups] == ["ZetaPay", "OrbitPOS", "KiranaCloud"]
    with pytest.raises(UnknownKam):
        call(t, "get_meetings", kam_id="kam2@example.com", day=NOW.date())
    with pg.pool.connection() as c:
        idx = {r["indexname"] for r in c.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename IN"
            " ('kam_meeting', 'competitor_update')")}
    assert {"kam_meeting_slot_idx", "competitor_update_published_idx"} <= idx
