"""Postgres (source of truth) + Redis (cache-aside) for the FSE bot.

Reads:  Redis first; on a miss, one indexed query on a single monthly partition, then cached.
Writes: one transaction updates task + appends task_event + refreshes the fse_day read row,
        then the cached plan is deleted. Redis being down never blocks a read or a write.
"""

import json
import logging
from datetime import date, datetime, timedelta, timezone
from importlib.resources import files
from typing import Any

import redis
from psycopg import Connection, sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from teammates.fse import tracing
from teammates.fse.tracing import Span, span

log = logging.getLogger(__name__)
IST = timezone(timedelta(hours=5, minutes=30))
# ponytail: a stale read can be re-cached right after a write's DEL; the short TTL bounds it.
PLAN_TTL_S = 300
# FSE rows (incl. location) are written through set_fse_location(), which drops the cache.
# The TTL only bounds edits made straight in the DB (e.g. Neon SQL editor): visible in <= 5 min.
FSE_TTL_S = 300

REASONS = {  # code -> WhatsApp list row title (max 24 chars)
    "unavailable": "Merchant not available",
    "closed": "Shop closed",
    "stock": "Device/stock unavailable",
    "not_interested": "Merchant not interested",
    "no_time": "Ran out of time",
    "rescheduled": "Rescheduled",
    "other": "Other (type it)",
}

# ponytail: fixed daily cap; switch to an effort budget (sum of effort) if FSEs get overloaded.
DAILY_TASKS = 5

# The one ranking rule, used by both the allotment and the plan read (SQL only, never the LLM):
# priority (1 = highest) first, then effort (largest job first), then distance (nearest first),
# then id so ties are deterministic. Distance is squared lat/lng difference: enough to order
# stops inside one area; no coordinates sort last.
RANK = "{p}.priority, {p}.effort DESC, {dist} NULLS LAST, {p}.id"
DIST = "((m.lat - f.lat) ^ 2 + (m.lng - f.lng) ^ 2)"

# Demo rows written into Postgres by seed_demo(); real data is inserted into the same tables.
DEMO_AREA = "Bellandur"
DEMO_FSE_LOCATION = ("Bellandur Gate, Outer Ring Road", 12.9260, 77.6762)  # (address, lat, lng)
DEMO_PROJECTS = [  # (merchant, address, lat, lng, kind, priority 1=highest, effort 1-3)
    ("Sri Medicals", "Green Glen Layout, 1st Main", 12.9279, 77.6790, "QR replacement", 2, 1),
    ("Ravi Stores", "Kaikondrahalli, Sarjapur Rd", 12.9135, 77.6731,
     "Multi-Soundbox install x3", 1, 3),
    ("Green Leaf Cafe", "Ecospace, Outer Ring Road", 12.9254, 77.6847,
     "Soundbox service visit", 3, 1),
    ("Lakshmi Bakery", "Bellandur Main Road", 12.9301, 77.6787, "New merchant onboarding", 1, 2),
    ("Anand Kirana", "Ambalipura, Sarjapur Rd", 12.9180, 77.6750, "KYC re-verification", 2, 2),
]

PLAN_SQL = f"""
SELECT t.id, t.day::text AS day, t.kind, t.priority, t.effort, t.status,
       t.reason_code, t.reason_text, m.name AS merchant, m.address, m.lat, m.lng
FROM task t JOIN merchant m ON m.id = t.merchant_id JOIN fse f ON f.id = t.fse_id
WHERE t.fse_id = %(fse)s AND t.day = %(day)s
ORDER BY {RANK.format(p="t", dist=DIST)}
"""

# One statement: rank the open projects in the FSE's area with RANK (one per merchant), take
# the top DAILY_TASKS, mark them allotted, and create the day's tasks from them.
# `p.status = 'open'` in the UPDATE is re-checked after a row lock, so two FSEs allotting at
# once can never take the same project; the loser just gets fewer rows.
ALLOT_SQL = f"""
WITH f AS (SELECT area, lat, lng FROM fse WHERE id = %(fse)s),
cand AS (
    SELECT DISTINCT ON (p.merchant_id) p.id, p.priority, p.effort, {DIST} AS dist
    FROM project p JOIN merchant m ON m.id = p.merchant_id CROSS JOIN f
    WHERE p.status = 'open' AND p.area = f.area
    ORDER BY p.merchant_id, {RANK.format(p="p", dist=DIST)}
),
pick AS (SELECT c.id FROM cand c ORDER BY {RANK.format(p="c", dist="c.dist")}
         LIMIT %(n)s),
upd AS (
    UPDATE project p SET status = 'allotted', fse_id = %(fse)s, allotted_on = %(day)s,
                         updated_at = now()
    FROM pick WHERE p.id = pick.id AND p.status = 'open'
    RETURNING p.id, p.merchant_id, p.kind, p.priority, p.effort
)
INSERT INTO task (day, fse_id, merchant_id, project_id, kind, priority, effort)
SELECT %(day)s, %(fse)s, merchant_id, id, kind, priority, effort FROM upd
"""

PENDING_REVIEW_SQL = """
SELECT r.id, r.day::text AS day, r.task_id, m.name AS merchant
FROM review r JOIN merchant m ON m.id = r.merchant_id
WHERE r.fse_id = %(fse)s AND r.day = %(day)s AND r.answered_at IS NULL
ORDER BY r.id LIMIT 1
"""

# Next done task (in the order they were done) that hasn't been asked about yet.
NEXT_REVIEW_SQL = """
INSERT INTO review (day, task_id, fse_id, merchant_id)
SELECT t.day, t.id, t.fse_id, t.merchant_id FROM task t
WHERE t.fse_id = %(fse)s AND t.day = %(day)s AND t.status = 'done'
  AND NOT EXISTS (SELECT 1 FROM review r WHERE r.day = t.day AND r.task_id = t.id)
ORDER BY t.done_at, t.id LIMIT 1
ON CONFLICT DO NOTHING
"""

FSE_COLS = "id, name, phone, area, address, lat, lng"
FSE_FIELDS = set(FSE_COLS.split(", "))
FSE_ALL_KEY = "fse:all"

REFRESH_DAY_SQL = """
WITH t AS (SELECT status, reason_code FROM task WHERE fse_id = %(fse)s AND day = %(day)s),
     r AS (SELECT reason_code, count(*) AS n FROM t WHERE reason_code IS NOT NULL GROUP BY 1)
INSERT INTO fse_day (fse_id, day, total, done, not_done, reasons, updated_at)
SELECT %(fse)s, %(day)s,
       (SELECT count(*) FROM t),
       (SELECT count(*) FROM t WHERE status = 'done'),
       (SELECT count(*) FROM t WHERE status = 'not_done'),
       coalesce((SELECT jsonb_object_agg(reason_code, n) FROM r), '{}'),
       now()
ON CONFLICT (fse_id, day) DO UPDATE SET
    total = EXCLUDED.total, done = EXCLUDED.done, not_done = EXCLUDED.not_done,
    reasons = EXCLUDED.reasons, updated_at = EXCLUDED.updated_at
"""


def ist_today() -> date:
    return datetime.now(IST).date()


def _next_month(d: date) -> date:
    return (d.replace(day=1) + timedelta(days=32)).replace(day=1)


class Store:
    def __init__(self, dsn: str, cache: redis.Redis | None = None, schema: str | None = None):
        kwargs: dict[str, Any] = {"row_factory": dict_row}
        if schema:  # tests: isolated schema (needs a direct, non-pgbouncer connection)
            kwargs["options"] = f"-c search_path={schema}"
        self.pool = ConnectionPool(dsn, min_size=1, max_size=5, kwargs=kwargs,
                                   check=ConnectionPool.check_connection, open=True)
        self.cache = cache

    # --- setup ---------------------------------------------------------------------------

    def migrate(self) -> None:
        """Create tables, reasons and this + next month's partitions. Idempotent."""
        with self.pool.connection() as conn:
            conn.execute(files("teammates.fse").joinpath("schema.sql").read_text("utf-8"))
            conn.cursor().executemany(
                "INSERT INTO reason (code, label) VALUES (%s, %s)"
                " ON CONFLICT (code) DO UPDATE SET label = EXCLUDED.label",
                list(REASONS.items()),
            )
        self.ensure_partitions()

    def ensure_partitions(self, day: date | None = None) -> None:
        start = (day or ist_today()).replace(day=1)
        with self.pool.connection() as conn:
            for lo in (start, _next_month(start)):
                hi = _next_month(lo)
                bounds = {
                    "task": (lo.isoformat(), hi.isoformat()),
                    "review": (lo.isoformat(), hi.isoformat()),
                    "task_event": (f"{lo} 00:00+05:30", f"{hi} 00:00+05:30"),
                    "feedback": (f"{lo} 00:00+05:30", f"{hi} 00:00+05:30"),
                    "trace_span": (f"{lo} 00:00+05:30", f"{hi} 00:00+05:30"),
                    "message_log": (f"{lo} 00:00+05:30", f"{hi} 00:00+05:30"),
                }
                for table, (a, b) in bounds.items():
                    conn.execute(sql.SQL(
                        "CREATE TABLE IF NOT EXISTS {} PARTITION OF {} FOR VALUES FROM ({}) TO ({})"
                    ).format(sql.Identifier(f"{table}_y{lo:%Y}m{lo:%m}"), sql.Identifier(table),
                             sql.Literal(a), sql.Literal(b)))

    def seed_demo(self, phone: str, name: str = "Anbuselvan") -> None:
        with self.pool.connection() as conn:
            lead = conn.execute(
                "INSERT INTO team_lead (name, email) VALUES ('Demo Lead', 'lead@example.com')"
                " ON CONFLICT (email) DO UPDATE SET name = EXCLUDED.name RETURNING id"
            ).fetchone()
            conn.execute(
                "INSERT INTO fse (name, phone, area, lead_id, address, lat, lng)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (phone) DO UPDATE SET"
                " address = EXCLUDED.address, lat = EXCLUDED.lat, lng = EXCLUDED.lng",
                (name, phone, DEMO_AREA, lead["id"], *DEMO_FSE_LOCATION),
            )
            cur = conn.cursor()
            cur.executemany(
                "INSERT INTO merchant (name, area, address, lat, lng) VALUES (%s, %s, %s, %s, %s)"
                " ON CONFLICT (area, name) DO UPDATE SET"
                " address = EXCLUDED.address, lat = EXCLUDED.lat, lng = EXCLUDED.lng",
                [(m, DEMO_AREA, addr, lat, lng) for m, addr, lat, lng, *_ in DEMO_PROJECTS],
            )
            cur.executemany(
                "INSERT INTO project (area, merchant_id, kind, priority, effort)"
                " SELECT m.area, m.id, %s, %s, %s FROM merchant m WHERE m.area = %s AND m.name = %s"
                " ON CONFLICT (merchant_id, kind) WHERE status <> 'closed' DO NOTHING",
                [(kind, p, e, DEMO_AREA, m) for m, _, _, _, kind, p, e in DEMO_PROJECTS],
            )
        self._drop(f"fse:phone:{phone}")  # location may have changed
        self._drop(FSE_ALL_KEY)

    # --- reads (cache-aside) ---------------------------------------------------------------

    def fse_by_phone(self, phone: str) -> dict[str, Any] | None:
        """FSE incl. location. Redis first; a miss, or a cached copy whose fields don't match
        the fse table (e.g. cached by an older deploy), is re-read from the table and re-cached."""
        key = f"fse:phone:{phone}"
        hit = self._get(key)
        if isinstance(hit, dict) and hit.keys() == FSE_FIELDS:
            return hit
        with self.pool.connection() as conn:
            row = conn.execute(f"SELECT {FSE_COLS} FROM fse WHERE phone = %s",
                               (phone,)).fetchone()
        if row:
            self._set(key, row, FSE_TTL_S)
        else:  # not in the table: clear any leftover copy; unknown numbers aren't cached
            self._drop(key)
        return row

    def fses(self) -> list[dict[str, Any]]:
        """All FSEs with area and location, for the 8:30 / evening jobs. Redis first; on a miss
        (or a cached copy whose fields don't match the fse table) read the table and cache it."""
        hit = self._get(FSE_ALL_KEY)
        if isinstance(hit, list) and all(
                isinstance(f, dict) and f.keys() == FSE_FIELDS for f in hit):
            return hit
        with self.pool.connection() as conn:
            rows = conn.execute(f"SELECT {FSE_COLS} FROM fse ORDER BY id").fetchall()
        self._set(FSE_ALL_KEY, rows, FSE_TTL_S)
        return rows

    def plan(self, fse_id: int, day: date | None = None) -> list[dict[str, Any]]:
        """The day's tasks, highest priority/hardest first; allots open projects on first read.
        Ranking is SQL, never the LLM."""
        day = day or ist_today()
        key = f"plan:{fse_id}:{day}"
        with span("plan", fse_id=fse_id) as sp:
            if (hit := self._get(key)) is not None:
                sp.attrs.update(cache="hit", tasks=len(hit))
                return hit
            sp.attrs["cache"] = "miss"
            with self.pool.connection() as conn:
                params = {"fse": fse_id, "day": day}
                rows = conn.execute(PLAN_SQL, params).fetchall()
                if not rows:
                    self._allot_day(conn, fse_id, day)
                    rows = conn.execute(PLAN_SQL, params).fetchall()
            sp.attrs["tasks"] = len(rows)
            self._set(key, rows, PLAN_TTL_S)
            return rows

    # --- writes (one transaction each, then invalidate) ------------------------------------

    def mark_done(self, fse_id: int, day: date, task_id: int) -> bool:
        with self.pool.connection() as conn:
            hit = conn.execute(
                "UPDATE task SET status = 'done', reason_code = NULL, reason_text = NULL,"
                " done_at = now(), updated_at = now()"
                " WHERE day = %s AND id = %s AND fse_id = %s RETURNING project_id",
                (day, task_id, fse_id),
            ).fetchone()
            if hit:
                conn.execute("UPDATE project SET status = 'closed', updated_at = now()"
                             " WHERE id = %s", (hit["project_id"],))
                self._event(conn, fse_id, task_id, "done", {})
                conn.execute(REFRESH_DAY_SQL, {"fse": fse_id, "day": day})
        self._invalidate(fse_id, day)
        return hit is not None

    def set_reason(self, fse_id: int, day: date, task_id: int, code: str,
                   text: str | None = None) -> bool:
        """Record why a task wasn't done. code 'other' with no text = waiting for typed text."""
        with self.pool.connection() as conn:
            hit = conn.execute(
                "UPDATE task SET status = 'not_done', reason_code = %s, reason_text = %s,"
                " updated_at = now()"
                " WHERE day = %s AND id = %s AND fse_id = %s AND status <> 'done'"
                " RETURNING project_id",
                (code, text, day, task_id, fse_id),
            ).fetchone()
            if hit:  # not done: back to the open pool, so tomorrow's allotment picks it up
                conn.execute(
                    "UPDATE project SET status = 'open', fse_id = NULL, allotted_on = NULL,"
                    " updated_at = now() WHERE id = %s AND status = 'allotted'",
                    (hit["project_id"],))
                self._event(conn, fse_id, task_id, "reason", {"code": code, "text": text})
                conn.execute(REFRESH_DAY_SQL, {"fse": fse_id, "day": day})
        self._invalidate(fse_id, day)
        return hit is not None

    def set_fse_location(self, fse_id: int, lat: float, lng: float,
                         address: str | None = None) -> dict[str, Any] | None:
        """Write the location to the fse table first, then drop the cached copy so the next
        read re-reads the table. Coordinates and address are replaced together, so an old
        address never sits next to new coordinates."""
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            raise ValueError(f"bad coordinates: {lat}, {lng}")
        with self.pool.connection() as conn:
            row = conn.execute(
                f"UPDATE fse SET lat = %s, lng = %s, address = %s WHERE id = %s"
                f" RETURNING {FSE_COLS}",
                (lat, lng, address, fse_id),
            ).fetchone()
        if row:
            self._drop(f"fse:phone:{row['phone']}")
            self._drop(FSE_ALL_KEY)
        return row

    def claim_job(self, job: str, day: date) -> bool:
        """True for exactly one caller per job per day, across restarts and instances."""
        with self.pool.connection() as conn:
            return conn.execute(
                "INSERT INTO job_run (job, day) VALUES (%s, %s) ON CONFLICT DO NOTHING"
                " RETURNING job", (job, day)).fetchone() is not None

    # --- customer reviews ------------------------------------------------------------------

    def pending_review(self, fse_id: int, day: date) -> dict[str, Any] | None:
        """The review question this FSE hasn't answered yet today, with the merchant name."""
        with self.pool.connection() as conn:
            return conn.execute(PENDING_REVIEW_SQL, {"fse": fse_id, "day": day}).fetchone()

    def request_review(self, fse_id: int, day: date) -> dict[str, Any] | None:
        """Return the open review question, or open one for the next done task without a
        review. None when every done task has been asked about."""
        params = {"fse": fse_id, "day": day}
        with self.pool.connection() as conn:
            if row := conn.execute(PENDING_REVIEW_SQL, params).fetchone():
                return row
            conn.execute(NEXT_REVIEW_SQL, params)
            return conn.execute(PENDING_REVIEW_SQL, params).fetchone()

    def save_review(self, fse_id: int, day: date, review_id: int, body: str | None,
                    audio_url: str | None) -> bool:
        """Store the customer's words against the asked merchant. Only the FSE it was asked to,
        and only once: a duplicate or someone else's reply changes nothing."""
        with self.pool.connection() as conn:
            hit = conn.execute(
                "UPDATE review SET body = %s, audio_url = %s, answered_at = now()"
                " WHERE day = %s AND id = %s AND fse_id = %s AND answered_at IS NULL"
                " RETURNING id",
                (body, audio_url, day, review_id, fse_id),
            ).fetchone()
        return hit is not None

    def drop_unanswered_reviews(self, fse_id: int, day: date) -> None:
        """Close the review window so later free text isn't filed as a review."""
        with self.pool.connection() as conn:
            conn.execute("DELETE FROM review WHERE fse_id = %s AND day = %s"
                         " AND answered_at IS NULL", (fse_id, day))

    def add_feedback(self, fse_id: int, body: str | None, audio_url: str | None) -> None:
        with self.pool.connection() as conn:
            conn.execute("INSERT INTO feedback (fse_id, body, audio_url) VALUES (%s, %s, %s)",
                         (fse_id, body, audio_url))
            self._event(conn, fse_id, None, "feedback", {"audio": audio_url is not None})

    # --- observability -----------------------------------------------------------------------

    def record_span(self, s: Span) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "INSERT INTO trace_span (started_at, id, trace_id, parent_id, name, path, fse_id,"
                " status, error, duration_ms, attrs) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,"
                " %s, %s)",
                (s.started_at, s.id, s.trace_id, s.parent_id, s.name, s.path, s.fse_id, s.status,
                 s.error, s.duration_ms, Jsonb(s.attrs)),
            )

    def log_message(self, direction: str, phone: str, kind: str, body: str | None, ok: bool,
                    fse_id: int | None = None, http_status: int | None = None,
                    error: str | None = None, message_uuid: str | None = None) -> None:
        """One row per WhatsApp message in or out; the phone number is stored masked."""
        sp = tracing.current()
        with self.pool.connection() as conn:
            conn.execute(
                "INSERT INTO message_log (direction, fse_id, phone_masked, kind, body, trace_id,"
                " ok, http_status, error, message_uuid) VALUES"
                " (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (direction, fse_id if fse_id is not None else (sp.fse_id if sp else None),
                 tracing.mask(phone), kind, (body or "")[:2000] or None,
                 sp.trace_id if sp else None, ok, http_status, error, message_uuid),
            )

    def record_delivery(self, message_uuid: str, status: str,
                        error: str | None) -> dict[str, Any] | None:
        """Vonage delivery callback -> the matching message_log row (last 3 days only, so the
        lookup stays on the newest partitions)."""
        with self.pool.connection() as conn:
            return conn.execute(
                "UPDATE message_log SET delivery = %s, delivery_error = %s, delivery_at = now()"
                " WHERE message_uuid = %s AND at > now() - interval '3 days'"
                " RETURNING fse_id, trace_id, kind",
                (status, error, message_uuid),
            ).fetchone()

    def health(self, day: date, expected_jobs: list[str]) -> dict[str, Any]:
        """Everything needed to tell whether the day ran properly, and if not, why."""
        with self.pool.connection() as conn:
            metrics = conn.execute("SELECT * FROM v_daily_metrics WHERE day = %s",
                                   (day,)).fetchone()
            jobs = conn.execute(
                "SELECT job, started_ist::text AS started, minutes_late, fses, failed_fses,"
                " status FROM v_job_runs WHERE day = %s", (day,)).fetchall()
            causes = conn.execute(
                "SELECT failed_step, error, count(*) AS times, count(DISTINCT fse_id) AS fses"
                " FROM v_root_causes WHERE at_ist::date = %s GROUP BY 1, 2 ORDER BY 3 DESC",
                (day,)).fetchall()
        ran = {j["job"].removeprefix("job.") for j in jobs}
        missing = [j for j in expected_jobs if j not in ran]
        problems = (
            [f"job {j} did not run" for j in missing]
            + [f"{j['job']}: {j['failed_fses']} FSE(s) failed" for j in jobs if j["failed_fses"]]
            + [f"{c['failed_step']} failed {c['times']}x: {c['error']}" for c in causes]
        )
        if metrics and metrics["undelivered"]:
            problems.append(f"{metrics['undelivered']} message(s) not delivered by WhatsApp")
        return {"day": str(day), "ok": not problems, "problems": problems,
                "metrics": {k: (float(v) if hasattr(v, "is_finite") else v)
                            for k, v in (metrics or {}).items() if k != "day"},
                "jobs": jobs, "root_causes": causes}

    # --- internals -------------------------------------------------------------------------

    def _allot_day(self, conn: Connection, fse_id: int, day: date) -> None:
        # Lock the FSE row so two first-reads of the same day (morning job + a reply) can't
        # both allot; the second waits, then sees the first one's tasks and stops.
        with span("allot") as sp:
            conn.execute("SELECT 1 FROM fse WHERE id = %s FOR UPDATE", (fse_id,))
            if conn.execute("SELECT 1 FROM task WHERE fse_id = %s AND day = %s LIMIT 1",
                            (fse_id, day)).fetchone():
                sp.attrs["skipped"] = "already allotted"
                return
            sp.attrs["allotted"] = conn.execute(
                ALLOT_SQL, {"fse": fse_id, "day": day, "n": DAILY_TASKS}).rowcount
            conn.execute(REFRESH_DAY_SQL, {"fse": fse_id, "day": day})

    @staticmethod
    def _event(conn: Connection, fse_id: int, task_id: int | None, type_: str,
               payload: dict[str, Any]) -> None:
        conn.execute(
            "INSERT INTO task_event (fse_id, task_id, type, payload) VALUES (%s, %s, %s, %s)",
            (fse_id, task_id, type_, Jsonb(payload)),
        )

    def _get(self, key: str) -> Any:
        if self.cache is None:
            return None
        try:
            raw = self.cache.get(key)
        except redis.RedisError as exc:
            log.warning("cache get failed, reading DB: %s", exc)
            if sp := tracing.current():  # degraded, not failed: the DB answered instead
                sp.attrs["cache_error"] = str(exc)[:200]
            return None
        return None if raw is None else json.loads(raw)

    def _set(self, key: str, value: Any, ttl_s: int) -> None:
        if self.cache is None:
            return
        try:
            self.cache.setex(key, ttl_s, json.dumps(value))
        except redis.RedisError as exc:
            log.warning("cache set failed: %s", exc)

    def _invalidate(self, fse_id: int, day: date) -> None:
        self._drop(f"plan:{fse_id}:{day}")  # on failure: stale for at most PLAN_TTL_S

    def _drop(self, key: str) -> None:
        if self.cache is None:
            return
        try:
            self.cache.delete(key)
        except redis.RedisError as exc:
            log.warning("cache delete failed: %s", exc)
