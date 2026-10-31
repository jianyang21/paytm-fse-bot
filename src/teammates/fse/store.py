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

log = logging.getLogger(__name__)
IST = timezone(timedelta(hours=5, minutes=30))
# ponytail: a stale read can be re-cached right after a write's DEL; the short TTL bounds it.
PLAN_TTL_S = 300
FSE_TTL_S = 86_400

REASONS = {  # code -> WhatsApp list row title (max 24 chars)
    "unavailable": "Merchant not available",
    "closed": "Shop closed",
    "stock": "Device/stock unavailable",
    "not_interested": "Merchant not interested",
    "no_time": "Ran out of time",
    "rescheduled": "Rescheduled",
    "other": "Other (type it)",
}

# ponytail: demo tasks seeded on the first read of the day; replace with the real task feed.
DEMO_AREA = "Bellandur"
DEMO_TASKS = [  # (merchant, kind, priority 1=highest, effort 1-3)
    ("Sri Medicals", "QR replacement", 2, 1),
    ("Ravi Stores", "Multi-Soundbox install x3", 1, 3),
    ("Green Leaf Cafe", "Soundbox service visit", 3, 1),
    ("Lakshmi Bakery", "New merchant onboarding", 1, 2),
    ("Anand Kirana", "KYC re-verification", 2, 2),
]

PLAN_SQL = """
SELECT t.id, t.day::text AS day, t.kind, t.priority, t.effort, t.status,
       t.reason_code, t.reason_text, m.name AS merchant
FROM task t JOIN merchant m ON m.id = t.merchant_id
WHERE t.fse_id = %(fse)s AND t.day = %(day)s
ORDER BY t.priority, t.effort DESC, t.id
"""

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
                    "task_event": (f"{lo} 00:00+05:30", f"{hi} 00:00+05:30"),
                    "feedback": (f"{lo} 00:00+05:30", f"{hi} 00:00+05:30"),
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
                "INSERT INTO fse (name, phone, area, lead_id) VALUES (%s, %s, %s, %s)"
                " ON CONFLICT (phone) DO NOTHING",
                (name, phone, DEMO_AREA, lead["id"]),
            )
            conn.cursor().executemany(
                "INSERT INTO merchant (name, area) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                [(m, DEMO_AREA) for m, *_ in DEMO_TASKS],
            )

    # --- reads (cache-aside) ---------------------------------------------------------------

    def fse_by_phone(self, phone: str) -> dict[str, Any] | None:
        key = f"fse:phone:{phone}"
        if (hit := self._get(key)) is not None:
            return hit
        with self.pool.connection() as conn:
            row = conn.execute("SELECT id, name, phone, area FROM fse WHERE phone = %s",
                               (phone,)).fetchone()
        if row:  # unknown numbers aren't cached, so a newly added FSE works at once
            self._set(key, row, FSE_TTL_S)
        return row

    def fses(self) -> list[dict[str, Any]]:
        with self.pool.connection() as conn:
            return conn.execute("SELECT id, name, phone, area FROM fse ORDER BY id").fetchall()

    def plan(self, fse_id: int, day: date | None = None) -> list[dict[str, Any]]:
        """Today's tasks, hardest/highest priority first. Ranking is SQL, never the LLM."""
        day = day or ist_today()
        key = f"plan:{fse_id}:{day}"
        if (hit := self._get(key)) is not None:
            return hit
        with self.pool.connection() as conn:
            params = {"fse": fse_id, "day": day}
            rows = conn.execute(PLAN_SQL, params).fetchall()
            if not rows:
                self._seed_day(conn, fse_id, day)
                rows = conn.execute(PLAN_SQL, params).fetchall()
        self._set(key, rows, PLAN_TTL_S)
        return rows

    # --- writes (one transaction each, then invalidate) ------------------------------------

    def mark_done(self, fse_id: int, day: date, task_id: int) -> bool:
        with self.pool.connection() as conn:
            hit = conn.execute(
                "UPDATE task SET status = 'done', reason_code = NULL, reason_text = NULL,"
                " done_at = now(), updated_at = now()"
                " WHERE day = %s AND id = %s AND fse_id = %s RETURNING id",
                (day, task_id, fse_id),
            ).fetchone()
            if hit:
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
                " WHERE day = %s AND id = %s AND fse_id = %s AND status <> 'done' RETURNING id",
                (code, text, day, task_id, fse_id),
            ).fetchone()
            if hit:
                self._event(conn, fse_id, task_id, "reason", {"code": code, "text": text})
                conn.execute(REFRESH_DAY_SQL, {"fse": fse_id, "day": day})
        self._invalidate(fse_id, day)
        return hit is not None

    def add_feedback(self, fse_id: int, body: str | None, audio_url: str | None) -> None:
        with self.pool.connection() as conn:
            conn.execute("INSERT INTO feedback (fse_id, body, audio_url) VALUES (%s, %s, %s)",
                         (fse_id, body, audio_url))
            self._event(conn, fse_id, None, "feedback", {"audio": audio_url is not None})

    # --- internals -------------------------------------------------------------------------

    def _seed_day(self, conn: Connection, fse_id: int, day: date) -> None:
        conn.cursor().executemany(
            "INSERT INTO task (day, fse_id, merchant_id, kind, priority, effort)"
            " SELECT %s, %s, m.id, %s, %s, %s FROM merchant m"
            " WHERE m.area = %s AND m.name = %s ON CONFLICT DO NOTHING",
            [(day, fse_id, kind, p, e, DEMO_AREA, m) for m, kind, p, e in DEMO_TASKS],
        )
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
        if self.cache is None:
            return
        try:
            self.cache.delete(f"plan:{fse_id}:{day}")
        except redis.RedisError as exc:  # stale for at most PLAN_TTL_S
            log.warning("cache invalidate failed: %s", exc)
