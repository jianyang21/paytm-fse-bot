import uuid
from datetime import UTC, datetime

import fakeredis
import psycopg
import pytest
from psycopg import sql

from teammates.support.store import MemoryStore, PgStore
from tests.fse.conftest import DSN

NOW = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)


def new(s, mid=101, **kw):
    return s.create({"merchant_id": mid, "channel": "whatsapp", "contact": "919900000101",
                     "text": "hi", "received_at": NOW, "last_contact_at": NOW} | kw)


@pytest.fixture
def pg():
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


@pytest.fixture(params=["memory", "pg"])
def store(request):
    return MemoryStore() if request.param == "memory" else request.getfixturevalue("pg")


def test_create_get_update(store):
    t = new(store, amount=450.0, risk_breakdown={"score": 2.5, "overrides": []})
    assert t["status"] == "new" and t["contacts"] == 1 and t["guard_flags"] == []
    store.update(101, t["id"], status="pending_review", band="low")
    got = store.get(101, t["id"])
    assert got["status"] == "pending_review" and got["risk_breakdown"]["score"] == 2.5
    assert store.get(999, t["id"]) is None  # merchant_id is part of the key


def test_unknown_columns_rejected(store):
    with pytest.raises(ValueError, match="unknown ticket columns"):
        new(store, **{"status; DROP TABLE ticket": "x"})
    t = new(store)
    with pytest.raises(ValueError):
        store.update(101, t["id"], password="x")


def test_open_for_latest_open_ticket(store):
    assert store.open_for(101) is None
    a, b = new(store), new(store)
    store.update(101, a["id"], status="pending_review")
    store.update(101, b["id"], status="resolved")
    assert store.open_for(101)["id"] == a["id"] and store.open_for(102) is None


def test_act_is_all_or_nothing(store):
    t = new(store, status="pending_review", band="low")
    entry = {"merchant_id": 101, "ticket_id": t["id"], "actor": "x", "action": "approve",
             "band": "low", "payload": {"diff": ""}}

    def boom(_):
        raise RuntimeError("send failed")

    with pytest.raises(RuntimeError):
        store.act(101, t["id"], boom)
    assert store.get(101, t["id"])["status"] == "pending_review" and store.audits() == []
    after = store.act(101, t["id"], lambda row: ({"status": "resolved"}, entry))
    assert after["status"] == "resolved"
    assert [(a["action"], a["band"]) for a in store.audits()] == [("approve", "low")]
    with pytest.raises(LookupError):
        store.act(101, 10**9, lambda row: ({}, entry))


def test_queue_by_role_assignee_and_status(store):
    kam = {"role": "kam", "email": "kam@example.com"}
    agent = {"role": "support_agent", "email": "agent@example.com"}
    r1 = new(store, status="pending_review", assigned_role="kam", assigned_to="kam@example.com")
    new(store, status="pending_review", assigned_role="kam", assigned_to="other@example.com")
    r3 = new(store, status="pending_review", assigned_role="support_agent")
    m = new(store, status="manual", assigned_role="support_agent", assigned_to=agent["email"])
    f = new(store, status="resolved", route="auto", assigned_role="kam",
            assigned_to="kam@example.com")
    ids = lambda q: {k: [t["id"] for t in v] for k, v in q.items()}  # noqa: E731
    assert ids(store.queue(kam)) == {"review": [r1["id"]], "manual": [], "fyi": [f["id"]]}
    assert ids(store.queue(agent)) == {"review": [r3["id"]], "manual": [m["id"]], "fyi": []}


def test_friction_accumulates(store):
    store.add_friction(101, 1, 1, "auto_resolved")
    store.add_friction(101, 2, 3, "manual")
    if isinstance(store, MemoryStore):
        s = store.signals[101]
    else:
        with store.pool.connection() as c:
            s = c.execute("SELECT * FROM merchant_signal WHERE merchant_id = 101").fetchone()
    assert (s["friction"], s["tickets"], s["last_ticket_id"], s["last_outcome"]) == (
        4, 2, 2, "manual")


def test_reviewer_login(store):
    store.upsert_reviewer("KAM@example.com", "s3cret", "Anita", "kam")
    assert store.authenticate("kam@example.com", "s3cret")["role"] == "kam"
    assert store.authenticate("kam@example.com", "wrong") is None
    assert store.authenticate("nobody@example.com", "s3cret") is None


def test_pg_stores_sql_payload_verbatim_and_has_indexes(pg):
    payload = "'; DROP TABLE ticket; -- 1' OR '1'='1"
    t = new(pg, text=payload)
    assert pg.get(101, t["id"])["text"] == payload
    with pg.pool.connection() as c:
        idx = {r["indexname"] for r in c.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename = 'ticket'")}
    assert {"ticket_merchant_status_idx", "ticket_status_idx"} <= idx


def test_kam_digest_reads(store):
    """queue(since) keeps every AI-resolved ticket since then (no cap), signal and by_role."""
    kam = {"role": "kam", "email": "kam@example.com"}
    fresh = [new(store, status="resolved", route="auto", assigned_role="kam",
                 assigned_to="kam@example.com") for _ in range(25)]
    old = new(store, status="resolved", route="auto", assigned_role="kam",
              assigned_to="kam@example.com")
    then = datetime(2026, 10, 1, tzinfo=UTC)
    if isinstance(store, MemoryStore):
        store.tickets[101, old["id"]]["updated_at"] = then
    else:
        with store.pool.connection() as c:
            c.execute("UPDATE ticket SET updated_at = %s WHERE id = %s", (then, old["id"]))
    since = datetime(2026, 10, 20, tzinfo=UTC)
    assert sorted(t["id"] for t in store.queue(kam, since=since)["fyi"]) == sorted(
        t["id"] for t in fresh)
    assert len(store.queue(kam)["fyi"]) == (26 if isinstance(store, MemoryStore) else 20)
    assert store.signal(101) is None
    store.add_friction(101, 1, 2, "auto_resolved")
    assert (store.signal(101)["friction"], store.signal(101)["tickets"]) == (2, 1)
    store.upsert_reviewer("kam@example.com", "pw", "Anita", "kam")
    store.upsert_reviewer("agent@example.com", "pw", "Agent", "support_agent")
    assert [r["email"] for r in store.by_role("kam")] == ["kam@example.com"]


def test_memory_store_message_log_masks_recipient():
    s = MemoryStore()
    s.log_message("out", "kam@example.com", "email", "Digest", False, http_status=503)
    assert s.messages == [{"direction": "out", "to_masked": "ka***@example.com", "kind": "email",
                           "body": "Digest", "ok": False, "http_status": 503}]
