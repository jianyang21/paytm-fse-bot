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


def test_allots_open_projects_in_area_once_and_tracks_lifecycle(store):
    fse = store.fse_by_phone(PHONE)
    assert fse["address"] and fse["lat"] is not None  # location comes with the FSE lookup...
    assert json.loads(store.cache.get(f"fse:phone:{PHONE}")) == fse  # ...and is cached
    with store.pool.connection() as conn:
        conn.execute("INSERT INTO fse (name, phone, area) VALUES"
                     " ('Same area', '911', 'Bellandur'), ('Other area', '912', 'Whitefield')")
        m = conn.execute("INSERT INTO merchant (name, area) VALUES ('WF Shop', 'Whitefield')"
                         " RETURNING id").fetchone()
        conn.execute("INSERT INTO project (area, merchant_id, kind, priority, effort)"
                     " VALUES ('Whitefield', %s, 'QR replacement', 1, 1)", (m["id"],))

    plan = store.plan(fse["id"])
    assert len(plan) == 5 and all(t["address"] for t in plan)
    assert "WF Shop" not in {t["merchant"] for t in plan}  # only this FSE's area
    assert store.plan(store.fse_by_phone("911")["id"]) == []  # already allotted, not twice
    assert [t["merchant"] for t in store.plan(store.fse_by_phone("912")["id"])] == ["WF Shop"]

    store.mark_done(fse["id"], ist_today(), plan[0]["id"])
    store.set_reason(fse["id"], ist_today(), plan[1]["id"], "closed")
    with store.pool.connection() as conn:
        status = {r["name"]: r["status"] for r in conn.execute(
            "SELECT m.name, p.status FROM project p JOIN merchant m ON m.id = p.merchant_id")}
    assert status.pop(plan[0]["merchant"]) == "closed"  # done -> closed
    assert status.pop(plan[1]["merchant"]) == "open"  # not done -> back in the pool
    assert set(status.values()) == {"allotted"}


def test_open_projects_read_uses_partial_index(store):
    with store.pool.connection() as conn:
        conn.execute("SET LOCAL enable_seqscan = off")  # tiny table: show the path used at volume
        plan = "\n".join(r["QUERY PLAN"] for r in conn.execute(
            "EXPLAIN SELECT id FROM project WHERE status = 'open' AND area = 'Bellandur'"
            " ORDER BY priority"))
    assert "project_open_area_idx" in plan


def test_fse_location_cache_reconciles_with_the_table(store):
    key = f"fse:phone:{PHONE}"
    fse = store.fse_by_phone(PHONE)

    # Copy cached by an older deploy (no location fields): treated as a miss, re-read, re-cached.
    store.cache.set(key, json.dumps({"id": fse["id"], "name": "old", "phone": PHONE}))
    assert store.fse_by_phone(PHONE) == fse
    assert json.loads(store.cache.get(key)) == fse

    # Location write goes to the table, then the next read re-reads it and re-caches it.
    store.set_fse_location(fse["id"], 12.95, 77.70, "New base")
    assert store.cache.get(key) is None
    with store.pool.connection() as conn:
        row = conn.execute("SELECT address, lat, lng FROM fse WHERE id = %s",
                           (fse["id"],)).fetchone()
    assert row == {"address": "New base", "lat": 12.95, "lng": 77.70}
    served = store.fse_by_phone(PHONE)
    assert {k: served[k] for k in row} == row  # served value == table
    assert json.loads(store.cache.get(key))["address"] == "New base"

    # Deleted from the table: the stale cached copy is cleared, not served.
    with store.pool.connection() as conn:
        conn.execute("DELETE FROM fse_day; DELETE FROM task; DELETE FROM fse")
    store.cache.delete(key)
    assert store.fse_by_phone(PHONE) is None and store.cache.get(key) is None


def test_fse_list_for_jobs_reads_redis_then_table_and_caches(store):
    assert store.cache.get("fse:all") is None
    fses = store.fses()  # miss: read from the fse table...
    assert [f["phone"] for f in fses] == [PHONE] and fses[0]["area"] == "Bellandur"
    assert json.loads(store.cache.get("fse:all")) == fses  # ...and added to Redis

    with store.pool.connection() as conn:  # change the table behind the cache's back
        conn.execute("UPDATE fse SET name = 'changed'")
    assert store.fses() == fses  # hit: served from Redis, no DB read

    store.cache.set("fse:all", json.dumps([{"id": 1}]))  # wrong shape: re-read the table
    assert store.fses()[0]["name"] == "changed"

    store.set_fse_location(fses[0]["id"], 12.95, 77.70, "New base")  # write drops the copy
    assert store.cache.get("fse:all") is None
    assert store.fses()[0]["address"] == "New base"


def test_plan_ranks_priority_then_effort_then_distance_capped_at_5(store):
    # FSE at (0, 0). Each pair differs in one rule only, so the order proves each rule.
    projects = [  # (merchant, lat, kind, priority, effort)
        ("P1 small near", 0.01, "QR replacement", 1, 1),
        ("P1 big far", 0.09, "Multi-Soundbox install x3", 1, 3),  # effort beats distance
        ("P2 near", 0.01, "KYC re-verification", 2, 2),
        ("P2 far", 0.05, "KYC re-verification", 2, 2),  # same priority+effort: nearer first
        ("P3 tie a", 0.02, "Soundbox service visit", 3, 1),
        ("P3 tie b", 0.02, "Soundbox service visit", 3, 1),  # full tie: lower id first; cut
        ("P3 no location", None, "Soundbox service visit", 3, 1),  # unknown distance: last
    ]
    with store.pool.connection() as conn:
        fse = conn.execute("INSERT INTO fse (name, phone, area, lat, lng)"
                           " VALUES ('Ranker', '913', 'Testarea', 0, 0) RETURNING id").fetchone()
        for name, lat, kind, prio, effort in projects:
            m = conn.execute(
                "INSERT INTO merchant (name, area, lat, lng) VALUES (%s, 'Testarea', %s, %s)"
                " RETURNING id", (name, lat, None if lat is None else 0.0)).fetchone()
            conn.execute("INSERT INTO project (area, merchant_id, kind, priority, effort)"
                         " VALUES ('Testarea', %s, %s, %s, %s)", (m["id"], kind, prio, effort))

    plan = store.plan(fse["id"])
    assert [t["merchant"] for t in plan] == [
        "P1 big far", "P1 small near", "P2 near", "P2 far", "P3 tie a"]
    with store.pool.connection() as conn:
        left = {r["name"] for r in conn.execute(
            "SELECT m.name FROM project p JOIN merchant m ON m.id = p.merchant_id"
            " WHERE p.area = 'Testarea' AND p.status = 'open'")}
    assert left == {"P3 tie b", "P3 no location"}  # over the cap of 5: stay open for tomorrow


def test_review_only_for_own_done_tasks_and_saved_once(store):
    fse = store.fse_by_phone(PHONE)
    today = ist_today()
    plan = store.plan(fse["id"])
    assert store.request_review(fse["id"], today) is None  # nothing done yet: nothing to ask
    store.mark_done(fse["id"], today, plan[2]["id"])
    r = store.request_review(fse["id"], today)
    assert r["merchant"] == plan[2]["merchant"]
    assert store.request_review(fse["id"], today) == r  # asking again doesn't duplicate

    with store.pool.connection() as conn:
        other = conn.execute("INSERT INTO fse (name, phone, area) VALUES ('Other', '914', 'X')"
                             " RETURNING id").fetchone()["id"]
    assert not store.save_review(other, today, r["id"], "fake", None)  # not their review
    assert store.save_review(fse["id"], today, r["id"], "happy", None)
    assert not store.save_review(fse["id"], today, r["id"], "again", None)  # saved once
    assert store.request_review(fse["id"], today) is None  # all done tasks reviewed
