"""Ticket persistence: PgStore (Postgres, production) and MemoryStore (same behaviour in
memory, for offline tests and evals). Business rules (RBAC, review actions, metrics) live in
desk.py as pure functions; the stores only persist, and act() applies a change atomically.
"""

import itertools
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.resources import files
from typing import Any

from psycopg import Connection, sql
from psycopg.types.json import Jsonb

from teammates.collections.store import check_password, hash_password
from teammates.fse import tracing
from teammates.fse.store import Store
from teammates.guard import mask_email

COLS = ("merchant_id", "channel", "contact", "text", "amount", "received_at", "last_contact_at",
        "contacts", "status", "type", "confidence", "guard_flags", "faq_id", "score", "band",
        "risk_breakdown", "route", "assigned_role", "assigned_to", "merchant_name", "tier",
        "language", "draft", "final_reply", "reject_reason", "trace_id")
OPEN = ("pending_review", "manual")
# (changes to the ticket, audit entry) <- fn(ticket); fn raises to abort without changes.
Change = Callable[[dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]]


def _check(fields: dict[str, Any]) -> None:
    if bad := set(fields) - set(COLS):
        raise ValueError(f"unknown ticket columns: {sorted(bad)}")


def _queue_filter(t: dict[str, Any], r: dict[str, Any],
                  since: datetime | None = None) -> str | None:
    """Which list of reviewer r's page ticket t belongs on, if any. `since` limits the
    AI-resolved (fyi) list to tickets resolved since then."""
    if t["status"] == "pending_review" and t["assigned_role"] == r["role"] \
            and t["assigned_to"] in (None, r["email"]):
        return "review"
    if t["status"] == "manual" and t["assigned_to"] == r["email"]:
        return "manual"
    if (t["status"] == "resolved" and t["route"] == "auto" and t["assigned_to"] == r["email"]
            and r["role"] == "kam" and (since is None or t["updated_at"] >= since)):
        return "fyi"  # key-account ticket the AI resolved: reported to the KAM
    return None


class MemoryStore:
    def __init__(self) -> None:
        self.tickets: dict[tuple[int, int], dict[str, Any]] = {}
        self.audit_rows: list[dict[str, Any]] = []
        self.signals: dict[int, dict[str, Any]] = {}
        self.reviewers: dict[str, dict[str, Any]] = {}
        self.messages: list[dict[str, Any]] = []  # message_log rows
        self._ids = itertools.count(1)
        self._lock = threading.RLock()

    def create(self, fields: dict[str, Any]) -> dict[str, Any]:
        _check(fields)
        with self._lock:
            t = dict.fromkeys(COLS) | {"contacts": 1, "status": "new", "guard_flags": []} \
                | fields | {"id": next(self._ids), "updated_at": datetime.now(UTC)}
            self.tickets[t["merchant_id"], t["id"]] = t
            return dict(t)

    def get(self, merchant_id: int, ticket_id: int) -> dict[str, Any] | None:
        t = self.tickets.get((merchant_id, ticket_id))
        return dict(t) if t else None

    def update(self, merchant_id: int, ticket_id: int, **fields: Any) -> None:
        _check(fields)
        with self._lock:
            self.tickets[merchant_id, ticket_id].update(fields, updated_at=datetime.now(UTC))

    def open_for(self, merchant_id: int) -> dict[str, Any] | None:
        with self._lock:
            hits = [t for (m, _), t in self.tickets.items()
                    if m == merchant_id and t["status"] in OPEN]
        return dict(max(hits, key=lambda t: t["id"])) if hits else None

    def act(self, merchant_id: int, ticket_id: int, fn: Change) -> dict[str, Any]:
        with self._lock:
            t = self.get(merchant_id, ticket_id)
            if t is None:
                raise LookupError("no such ticket")
            changes, entry = fn(t)
            self.update(merchant_id, ticket_id, **changes)
            self.audit(entry)
            return self.get(merchant_id, ticket_id)

    def audit(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self.audit_rows.append(entry | {"at": datetime.now(UTC)})

    def audits(self) -> list[dict[str, Any]]:
        return list(self.audit_rows)

    def add_friction(self, merchant_id: int, ticket_id: int, delta: int, outcome: str) -> None:
        with self._lock:
            s = self.signals.setdefault(merchant_id, {"friction": 0, "tickets": 0})
            s.update(friction=s["friction"] + delta, tickets=s["tickets"] + 1,
                     last_ticket_id=ticket_id, last_outcome=outcome)

    def queue(self, reviewer: dict[str, Any],
              since: datetime | None = None) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {"review": [], "manual": [], "fyi": []}
        with self._lock:
            for t in sorted(self.tickets.values(), key=lambda t: t["id"]):
                if (where := _queue_filter(t, reviewer, since)) is not None:
                    out[where].append(dict(t))
        return out

    def signal(self, merchant_id: int) -> dict[str, Any] | None:
        s = self.signals.get(merchant_id)
        return dict(s) if s else None

    def by_role(self, role: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.reviewers.values() if r["role"] == role and r["active"]]

    def log_message(self, direction: str, to: str, kind: str, body: str | None, ok: bool,
                    **kw: Any) -> None:
        """Same row as Store.log_message (recipient masked)."""
        self.messages.append({"direction": direction, "to_masked": mask_email(to) if "@" in to
                              else tracing.mask(to), "kind": kind, "body": body, "ok": ok, **kw})

    def upsert_reviewer(self, email: str, password: str, name: str, role: str) -> None:
        self.reviewers[email.lower()] = {"id": len(self.reviewers) + 1, "email": email.lower(),
                                         "name": name, "role": role, "active": True,
                                         "password_hash": hash_password(password)}

    def authenticate(self, email: str, password: str) -> dict[str, Any] | None:
        r = self.reviewers.get(email.strip().lower())
        return r if r and r["active"] and check_password(password, r["password_hash"]) else None


def _db(fields: dict[str, Any]) -> dict[str, Any]:
    return {k: Jsonb(v) if k == "risk_breakdown" and v is not None else v
            for k, v in fields.items()}


class PgStore(Store):
    """Same interface as MemoryStore over the support schema (schema.sql)."""

    def migrate(self) -> None:
        with self.pool.connection() as conn:
            conn.execute(files("teammates.support").joinpath("schema.sql").read_text("utf-8"))
        super().migrate()  # trace_span etc.

    def create(self, fields: dict[str, Any]) -> dict[str, Any]:
        _check(fields)
        cols = list(fields)
        q = sql.SQL("INSERT INTO ticket ({}) VALUES ({}) RETURNING *").format(
            sql.SQL(", ").join(map(sql.Identifier, cols)),
            sql.SQL(", ").join(sql.Placeholder() * len(cols)))
        with self.pool.connection() as conn:
            return conn.execute(q, list(_db(fields).values())).fetchone()

    def get(self, merchant_id: int, ticket_id: int) -> dict[str, Any] | None:
        with self.pool.connection() as conn:
            return self._get_row(conn, merchant_id, ticket_id, lock=False)

    @staticmethod
    def _get_row(conn: Connection, merchant_id: int, ticket_id: int,
                 lock: bool) -> dict[str, Any] | None:
        return conn.execute("SELECT * FROM ticket WHERE merchant_id = %s AND id = %s"
                            + (" FOR UPDATE" if lock else ""),
                            (merchant_id, ticket_id)).fetchone()

    def update(self, merchant_id: int, ticket_id: int, **fields: Any) -> None:
        with self.pool.connection() as conn:
            self._update(conn, merchant_id, ticket_id, fields)

    @staticmethod
    def _update(conn: Connection, merchant_id: int, ticket_id: int,
                fields: dict[str, Any]) -> None:
        _check(fields)
        if not fields:
            return
        sets = sql.SQL(", ").join(sql.SQL("{} = %s").format(sql.Identifier(k)) for k in fields)
        conn.execute(sql.SQL("UPDATE ticket SET {}, updated_at = now()"
                             " WHERE merchant_id = %s AND id = %s").format(sets),
                     [*_db(fields).values(), merchant_id, ticket_id])

    def open_for(self, merchant_id: int) -> dict[str, Any] | None:
        with self.pool.connection() as conn:
            return conn.execute(
                "SELECT * FROM ticket WHERE merchant_id = %s AND status = ANY(%s)"
                " ORDER BY id DESC LIMIT 1", (merchant_id, list(OPEN))).fetchone()

    def act(self, merchant_id: int, ticket_id: int, fn: Change) -> dict[str, Any]:
        """Row locked for the whole change, so two reviewers can't both act (or send)."""
        with self.pool.connection() as conn, conn.transaction():
            t = self._get_row(conn, merchant_id, ticket_id, lock=True)
            if t is None:
                raise LookupError("no such ticket")
            changes, entry = fn(dict(t))
            self._update(conn, merchant_id, ticket_id, changes)
            self._audit(conn, entry)
            return self._get_row(conn, merchant_id, ticket_id, lock=False)

    def audit(self, entry: dict[str, Any]) -> None:
        with self.pool.connection() as conn:
            self._audit(conn, entry)

    @staticmethod
    def _audit(conn: Connection, e: dict[str, Any]) -> None:
        conn.execute(
            "INSERT INTO audit_log (merchant_id, ticket_id, actor, action, band, payload,"
            " trace_id) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (e["merchant_id"], e["ticket_id"], e["actor"], e["action"], e.get("band"),
             Jsonb(e.get("payload") or {}), e.get("trace_id")))

    def audits(self) -> list[dict[str, Any]]:
        with self.pool.connection() as conn:
            return conn.execute("SELECT * FROM audit_log ORDER BY at, id").fetchall()

    def add_friction(self, merchant_id: int, ticket_id: int, delta: int, outcome: str) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "INSERT INTO merchant_signal (merchant_id, friction, tickets, last_ticket_id,"
                " last_outcome) VALUES (%(m)s, %(d)s, 1, %(t)s, %(o)s)"
                " ON CONFLICT (merchant_id) DO UPDATE SET"
                " friction = merchant_signal.friction + %(d)s,"
                " tickets = merchant_signal.tickets + 1, last_ticket_id = %(t)s,"
                " last_outcome = %(o)s, updated_at = now()",
                {"m": merchant_id, "d": delta, "t": ticket_id, "o": outcome})

    def queue(self, reviewer: dict[str, Any],
              since: datetime | None = None) -> dict[str, list[dict[str, Any]]]:
        """Without `since` the fyi list is the latest 20 (the review page); with it, every
        ticket the AI resolved since then (the KAM digest: none may be missing)."""
        p = {"role": reviewer["role"], "me": reviewer["email"], "since": since,
             "n": None if since else 20}
        with self.pool.connection() as conn:
            return {
                "review": conn.execute(
                    "SELECT * FROM ticket WHERE status = 'pending_review'"
                    " AND assigned_role = %(role)s AND (assigned_to IS NULL OR assigned_to ="
                    " %(me)s) ORDER BY created_at", p).fetchall(),
                "manual": conn.execute(
                    "SELECT * FROM ticket WHERE status = 'manual' AND assigned_to = %(me)s"
                    " ORDER BY updated_at DESC", p).fetchall(),
                "fyi": conn.execute(
                    "SELECT * FROM ticket WHERE status = 'resolved' AND route = 'auto'"
                    " AND assigned_to = %(me)s AND %(role)s = 'kam'"
                    " AND (%(since)s::timestamptz IS NULL OR updated_at >= %(since)s)"
                    " ORDER BY updated_at DESC LIMIT %(n)s", p).fetchall(),
            }

    def signal(self, merchant_id: int) -> dict[str, Any] | None:
        with self.pool.connection() as conn:
            return conn.execute("SELECT friction, tickets, last_outcome FROM merchant_signal"
                                " WHERE merchant_id = %s", (merchant_id,)).fetchone()

    def by_role(self, role: str) -> list[dict[str, Any]]:
        with self.pool.connection() as conn:
            return conn.execute("SELECT * FROM support_reviewer WHERE role = %s AND active"
                                " ORDER BY id", (role,)).fetchall()

    def upsert_reviewer(self, email: str, password: str, name: str, role: str) -> None:
        email = email.strip().lower()
        with self.pool.connection() as conn:
            row = conn.execute("SELECT password_hash FROM support_reviewer WHERE email = %s",
                               (email,)).fetchone()
            h = (row["password_hash"] if row and check_password(password, row["password_hash"])
                 else hash_password(password))
            conn.execute(
                "INSERT INTO support_reviewer (name, email, password_hash, role)"
                " VALUES (%s, %s, %s, %s) ON CONFLICT (email) DO UPDATE SET name = EXCLUDED.name,"
                " password_hash = EXCLUDED.password_hash, role = EXCLUDED.role, active = true",
                (name, email, h, role))

    def authenticate(self, email: str, password: str) -> dict[str, Any] | None:
        """Unknown emails cost the same scrypt time as wrong passwords (no email probing)."""
        with self.pool.connection() as conn:
            r = conn.execute("SELECT * FROM support_reviewer WHERE email = %s AND active",
                             (email.strip().lower(),)).fetchone()
        if r is None:
            hash_password(password)
            return None
        return r if check_password(password, r["password_hash"]) else None
