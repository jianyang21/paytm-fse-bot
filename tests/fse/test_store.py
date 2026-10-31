import json

import redis

from teammates.fse.store import ist_today
from tests.fse.conftest import PHONE


def test_reads_hit_redis_first_and_writes_invalidate(store):
    fse = store.fse_by_phone(PHONE)
    key = f"plan:{fse['id']}:{ist_today()}"
    plan = store.plan(fse["id"])
    assert json.loads(store.cache.get(key)) == plan  # cached after the first (DB) read

    with store.pool.connection() as conn:  # change the DB behind the cache's back
        conn.execute("UPDATE task SET kind = 'changed'")
    assert store.plan(fse["id"]) == plan  # still served from Redis

    assert store.mark_done(fse["id"], ist_today(), plan[0]["id"])
    assert store.cache.get(key) is None  # write deleted the cached plan
    assert store.plan(fse["id"])[0]["status"] == "done"


def test_redis_down_falls_back_to_postgres(store):
    store.cache = redis.Redis(port=1, socket_connect_timeout=0.2, socket_timeout=0.2)
    fse = store.fse_by_phone(PHONE)
    plan = store.plan(fse["id"])
    assert store.mark_done(fse["id"], ist_today(), plan[1]["id"])
    assert store.plan(fse["id"])[1]["status"] == "done"


def test_write_updates_task_event_and_summary_together(store):
    fse = store.fse_by_phone(PHONE)
    plan = store.plan(fse["id"])
    store.mark_done(fse["id"], ist_today(), plan[0]["id"])
    store.set_reason(fse["id"], ist_today(), plan[1]["id"], "closed")
    assert not store.set_reason(fse["id"], ist_today(), plan[0]["id"], "closed")  # done stays done

    with store.pool.connection() as conn:
        events = [r["type"] for r in conn.execute("SELECT type FROM task_event ORDER BY id")]
        day = conn.execute("SELECT total, done, not_done, reasons FROM fse_day").fetchone()
    assert events == ["done", "reason"]
    assert day == {"total": 5, "done": 1, "not_done": 1, "reasons": {"closed": 1}}


def test_plan_read_touches_only_this_months_partition(store):
    fse = store.fse_by_phone(PHONE)
    today = ist_today()
    with store.pool.connection() as conn:
        parts = {r["relname"] for r in conn.execute(
            "SELECT c.relname FROM pg_inherits i JOIN pg_class c ON c.oid = i.inhrelid"
            " WHERE i.inhparent = 'task'::regclass")}
        plan = "\n".join(r["QUERY PLAN"] for r in conn.execute(
            f"EXPLAIN SELECT * FROM task WHERE fse_id = {int(fse['id'])} AND day = '{today}'"))
    this_month = f"task_y{today:%Y}m{today:%m}"
    assert len(parts) == 2 and this_month in parts  # this month + next month
    assert this_month in plan
    assert not (parts - {this_month}) & set(plan.split())  # next month pruned away
