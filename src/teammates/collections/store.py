"""Collections data on the FSE Store: same Postgres pool, Redis helpers, job_run, trace_span and
message_log. Brief reads are cache-first with version reconciliation (see schema.sql);
applications and decisions read and write Postgres directly and are audited in
application_event, in the same transaction as the change.
"""

import hashlib
import hmac
import os
from collections.abc import Callable
from datetime import date, timedelta
from importlib.resources import files
from typing import Any

from psycopg import Connection, sql
from psycopg.types.json import Jsonb

from teammates.fse import tracing
from teammates.fse.store import Store, _next_month, ist_today

# Freshness comes from the version stamp, not the TTL; the TTL only clears old days' keys.
BRIEF_TTL_S = 6 * 3600

DEMO_AREA_TARGETS = {"Bellandur": 60000, "Koramangala": 70000, "HSR Layout": 30000}
# (name, business, phone, area, credit score, background flags, monthly revenue, expenses,
#  loan principal, EMI, missed: how many of the latest past-due EMIs are unpaid; None = no loan)
DEMO_BORROWERS = [
    ("Meena Iyer", "Meena Tailoring", "919800000001", "Bellandur", 762, [], 180000, 120000,
     300000, 14500, 0),
    ("Rakesh Gowda", "Gowda Hardware", "919800000002", "Bellandur", 705, [], 400000, 310000,
     500000, 24000, 1),
    ("Salma Khan", "Khan Mobiles", "919800000003", "Bellandur", 640, [], 220000, 180000,
     250000, 12500, 4),
    ("Vinod Shetty", "Shetty Tiffins", "919800000004", "Koramangala", 781, [], 150000, 95000,
     200000, 9800, 0),
    ("Priya Nair", "Nair Pharmacy", "919800000005", "Koramangala", 690, [], 350000, 280000,
     400000, 19500, 2),
    ("Arjun Rao", "Rao Auto Spares", "919800000006", "Koramangala", 735, [], 500000, 390000,
     600000, 28500, 0),
    ("Fatima Sheikh", "Sheikh Bakery", "919800000007", "HSR Layout", 668, [], 160000, 125000,
     150000, 7600, 1),
    ("Kiran Kumar", "Kumar Electricals", "919800000008", "HSR Layout", 720, [], 260000, 200000,
     350000, 16800, 2),
    # Applicants with no loan yet
    ("Deepa Menon", "Menon Organics", "919800000009", "HSR Layout", 768, [], 240000, 150000,
     None, None, None),
    ("Suresh Pillai", "Pillai Traders", "919800000010", "Bellandur", 612, ["wilful_default"],
     300000, 270000, None, None, None),
]
# Instalment due dates relative to the seed day: 5 past (oldest first) + 1 upcoming.
DEMO_DUE_OFFSETS = [-123, -93, -63, -33, -3, 12]

# Days past due = age of the oldest unpaid instalment that is already due.
LOANS_SQL = """
SELECT l.borrower_id, l.id AS loan_id, b.name, b.business, b.area, b.phone, l.principal, l.emi,
       l.status, l.disbursed_on::text AS disbursed_on,
       coalesce(%(today)s - min(i.due_on) FILTER (WHERE i.paid_on IS NULL
                                                  AND i.due_on < %(today)s), 0) AS dpd,
       coalesce(sum(i.amount) FILTER (WHERE i.paid_on IS NULL
                                      AND i.due_on < %(today)s), 0)::bigint AS overdue,
       count(i.id) FILTER (WHERE i.paid_on IS NOT NULL) AS paid_count,
       count(i.id) AS instalment_count
FROM loan l JOIN borrower b ON b.id = l.borrower_id
LEFT JOIN instalment i ON i.borrower_id = l.borrower_id AND i.loan_id = l.id
WHERE {where}
GROUP BY l.borrower_id, l.id, b.id
ORDER BY dpd DESC, overdue DESC, b.name
"""

PENDING_SQL = """
SELECT b.name, b.business, b.area, b.phone, i.amount, i.due_on::text AS due_on,
       greatest(%(today)s - i.due_on, 0) AS days_late
FROM instalment i JOIN borrower b ON b.id = i.borrower_id
JOIN loan l ON l.borrower_id = i.borrower_id AND l.id = i.loan_id
WHERE b.area = %(area)s AND i.paid_on IS NULL AND i.due_on < %(next)s AND l.status = 'active'
ORDER BY i.due_on, b.name
"""

TARGET_SQL = """
SELECT t.area, t.target,
       coalesce((SELECT sum(i.amount) FROM instalment i JOIN borrower b ON b.id = i.borrower_id
                 WHERE b.area = t.area AND i.paid_on >= %(month)s AND i.paid_on < %(next)s),
                0)::bigint AS collected
FROM area_target t WHERE t.area = %(area)s AND t.month = %(month)s
"""

MANAGER_SQL = """
SELECT m.id, m.name, m.email, m.role, m.email_verified, m.password_hash,
       coalesce(array_agg(a.area ORDER BY a.area) FILTER (WHERE a.area IS NOT NULL), '{{}}')
           AS areas
FROM collection_manager m LEFT JOIN manager_area a ON a.manager_id = m.id
WHERE m.active AND {where}
GROUP BY m.id ORDER BY m.id
"""

BORROWER_COLS = ("id, name, business, phone, area, credit_score, background_flags,"
                 " monthly_revenue, monthly_expenses")
APP_SQL = """
SELECT a.borrower_id, a.id, a.amount, a.tenure_months, a.purpose, a.checks, a.summary,
       a.recommendation, a.overridden, a.status, m.email AS decided_by, a.note,
       a.created_at::text AS created_at, a.decided_at::text AS decided_at
FROM loan_application a JOIN borrower b ON b.id = a.borrower_id
LEFT JOIN collection_manager m ON m.id = a.decided_by
WHERE b.area = ANY(%(areas)s) AND {where}
"""


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    h = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${salt.hex()}${h.hex()}"


def check_password(password: str, stored: str) -> bool:
    try:
        _, salt, _ = stored.split("$")
        return hmac.compare_digest(hash_password(password, bytes.fromhex(salt)), stored)
    except ValueError:
        return False


class LoanStore(Store):
    # --- setup ---------------------------------------------------------------------------

    def migrate(self) -> None:
        """Collections tables, then the shared FSE schema (job_run, trace_span, message_log),
        then this + next month's partitions for both."""
        with self.pool.connection() as conn:
            conn.execute(files("teammates.collections").joinpath("schema.sql").read_text("utf-8"))
        super().migrate()

    def ensure_partitions(self, day: date | None = None) -> None:
        super().ensure_partitions(day)
        start = (day or ist_today()).replace(day=1)
        with self.pool.connection() as conn:
            for lo in (start, _next_month(start)):
                conn.execute(sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {} PARTITION OF application_event"
                    " FOR VALUES FROM ({}) TO ({})"
                ).format(sql.Identifier(f"application_event_y{lo:%Y}m{lo:%m}"),
                         sql.Literal(f"{lo} 00:00+05:30"),
                         sql.Literal(f"{_next_month(lo)} 00:00+05:30")))

    def seed_demo(self, today: date) -> None:
        """Areas and this month's targets every start; demo borrowers/loans once, with dates
        relative to the first start. ponytail: demo loans age with real time (healthy ones go
        overdue next month); TRUNCATE borrower CASCADE to reseed."""
        month = today.replace(day=1)
        with self.pool.connection() as conn:
            cur = conn.cursor()
            cur.executemany("INSERT INTO area (name) VALUES (%s) ON CONFLICT DO NOTHING",
                            [(a,) for a in DEMO_AREA_TARGETS])
            cur.executemany(
                "INSERT INTO area_target (area, month, target) VALUES (%s, %s, %s)"
                " ON CONFLICT DO NOTHING",
                [(a, month, t) for a, t in DEMO_AREA_TARGETS.items()])
            if conn.execute("SELECT 1 FROM borrower LIMIT 1").fetchone():
                return
            for (name, business, phone, area, score, flags, revenue, expenses,
                 principal, emi, missed) in DEMO_BORROWERS:
                bid = conn.execute(
                    "INSERT INTO borrower (name, business, phone, area, credit_score,"
                    " background_flags, monthly_revenue, monthly_expenses)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                    (name, business, phone, area, score, flags, revenue, expenses),
                ).fetchone()["id"]
                if principal is None:
                    continue
                lid = conn.execute(
                    "INSERT INTO loan (borrower_id, principal, emi, disbursed_on)"
                    " VALUES (%s, %s, %s, %s) RETURNING id",
                    (bid, principal, emi, today - timedelta(days=150)),
                ).fetchone()["id"]
                past = [o for o in DEMO_DUE_OFFSETS if o < 0]
                rows = []
                for o in DEMO_DUE_OFFSETS:
                    due = today + timedelta(days=o)
                    paid = o < 0 and past.index(o) < len(past) - missed
                    rows.append((bid, lid, due, emi, due if paid else None))
                conn.cursor().executemany(
                    "INSERT INTO instalment (borrower_id, loan_id, due_on, amount, paid_on)"
                    " VALUES (%s, %s, %s, %s, %s)", rows)

    def upsert_manager(self, email: str, password: str, name: str, role: str,
                       areas: list[str], verified: bool) -> int:
        """Create or update a manager login. The hash is kept when the password is unchanged,
        so restarts don't invalidate open dashboard forms. Areas are only ever added."""
        email = email.strip().lower()
        with self.pool.connection() as conn:
            row = conn.execute("SELECT password_hash FROM collection_manager WHERE email = %s",
                               (email,)).fetchone()
            pw_hash = (row["password_hash"] if row and check_password(password,
                                                                      row["password_hash"])
                       else hash_password(password))
            mid = conn.execute(
                "INSERT INTO collection_manager (name, email, password_hash, role, email_verified)"
                " VALUES (%s, %s, %s, %s, %s) ON CONFLICT (email) DO UPDATE SET"
                " name = EXCLUDED.name, password_hash = EXCLUDED.password_hash,"
                " role = EXCLUDED.role, email_verified = EXCLUDED.email_verified, active = true"
                " RETURNING id",
                (name, email, pw_hash, role, verified)).fetchone()["id"]
            conn.cursor().executemany(
                "INSERT INTO manager_area (manager_id, area) VALUES (%s, %s)"
                " ON CONFLICT DO NOTHING", [(mid, a) for a in areas])
        return mid

    # --- RBAC ----------------------------------------------------------------------------

    def authenticate(self, email: str, password: str) -> dict[str, Any] | None:
        """Active manager with role and areas, or None. Unknown emails cost the same scrypt
        time as wrong passwords, so login timing doesn't reveal which emails exist."""
        with self.pool.connection() as conn:
            m = conn.execute(MANAGER_SQL.format(where="m.email = %s"),
                             (email.strip().lower(),)).fetchone()
        if m is None:
            hash_password(password)
            return None
        return m if check_password(password, m["password_hash"]) else None

    def brief_recipients(self) -> list[dict[str, Any]]:
        """Every active collection manager. The output guard, not this query, decides who
        may actually receive a brief (verified email, areas assigned)."""
        with self.pool.connection() as conn:
            return conn.execute(MANAGER_SQL.format(where="m.role = 'collection_manager'")
                                ).fetchall()

    # --- brief reads: Redis first, reconciled against area.version --------------------------

    def brief_area(self, area: str, today: date) -> tuple[dict[str, Any] | None, str]:
        """One area's loans, pending instalments and target -> (data, cache state).
        State: hit (Redis, version matches), stale (Redis had an older version), miss."""
        month = today.replace(day=1)

        def load(conn: Connection) -> dict[str, Any]:
            p = {"today": today, "area": area, "month": month, "next": _next_month(month)}
            return {
                "loans": conn.execute(LOANS_SQL.format(
                    where="b.area = %(area)s AND l.status = 'active'"), p).fetchall(),
                "pending": conn.execute(PENDING_SQL, p).fetchall(),
                "target": conn.execute(TARGET_SQL, p).fetchone(),
            }

        return self._versioned(area, f"coll:brief:{area}:{today}", load)

    def _versioned(self, area: str, key: str,
                   load: Callable[[Connection], Any]) -> tuple[Any, str]:
        with self.pool.connection() as conn:
            row = conn.execute("SELECT version FROM area WHERE name = %s", (area,)).fetchone()
            if row is None:
                return None, "unknown_area"
            # Version read before the data: whatever we load is at least this new, so a
            # concurrent write can only make the stamp too old (-> re-read), never too new.
            hit = self._get(key)
            if isinstance(hit, dict) and hit.get("v") == row["version"]:
                return hit["data"], "hit"
            data = load(conn)
        self._set(key, {"v": row["version"], "data": data}, BRIEF_TTL_S)
        return data, "stale" if hit is not None else "miss"

    def names_outside(self, names: list[str], areas: list[str]) -> list[str]:
        """Which of these strings are borrowers outside the given areas (output guard)."""
        if not names:
            return []
        with self.pool.connection() as conn:
            rows = conn.execute("SELECT DISTINCT name FROM borrower WHERE name = ANY(%s)"
                                " AND area <> ALL(%s) ORDER BY name", (names, areas)).fetchall()
        return [r["name"] for r in rows]

    # --- loan applications -----------------------------------------------------------------

    def borrower_by_phone(self, phone: str) -> dict[str, Any] | None:
        with self.pool.connection() as conn:
            return conn.execute(f"SELECT {BORROWER_COLS} FROM borrower WHERE phone = %s",
                                (phone,)).fetchone()

    def applicant_record(self, borrower_id: int, today: date) -> dict[str, Any]:
        """Everything the application tools read, straight from Postgres (credit decisions
        never use cached data): the borrower row and every loan with its repayment record."""
        with self.pool.connection() as conn:
            return {
                "borrower": conn.execute(f"SELECT {BORROWER_COLS} FROM borrower WHERE id = %s",
                                         (borrower_id,)).fetchone(),
                "loans": conn.execute(LOANS_SQL.format(where="l.borrower_id = %(b)s"),
                                      {"today": today, "b": borrower_id}).fetchall(),
            }

    def create_application(self, borrower_id: int, amount: int, tenure_months: int,
                           purpose: str, a: dict[str, Any], actor: str) -> int:
        """Application row + 'submitted' and 'assessed' audit events, one transaction."""
        sp = tracing.current()
        trace_id = sp.trace_id if sp else None
        with self.pool.connection() as conn:
            app_id = conn.execute(
                "INSERT INTO loan_application (borrower_id, amount, tenure_months, purpose,"
                " checks, summary, recommendation, overridden, trace_id)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (borrower_id, amount, tenure_months, purpose, Jsonb(a["checks"]),
                 a["summary"], a["recommendation"], a["overridden"], trace_id),
            ).fetchone()["id"]
            self._audit(conn, borrower_id, app_id, "submitted", actor,
                        {"amount": amount, "tenure_months": tenure_months}, trace_id)
            self._audit(conn, borrower_id, app_id, "assessed", "ai",
                        {"recommendation": a["recommendation"], "overridden": a["overridden"],
                         "checks": {k: c["ok"] for k, c in a["checks"].items()}}, trace_id)
        return app_id

    def applications_for(self, manager: dict[str, Any], today: date,
                         decided_limit: int = 10) -> list[dict[str, Any]]:
        """Applications in the manager's areas: pending oldest first, then the latest decided,
        each with the applicant, every loan and repayment record, earlier applications and
        the audit trail."""
        p = {"areas": manager["areas"], "n": decided_limit}
        with self.pool.connection() as conn:
            apps = conn.execute(APP_SQL.format(where="a.status = 'pending'")
                                + " ORDER BY a.created_at", p).fetchall()
            apps += conn.execute(APP_SQL.format(where="a.status <> 'pending'")
                                 + " ORDER BY a.decided_at DESC LIMIT %(n)s", p).fetchall()
            # ponytail: 4 single-shard reads per card; batch them past ~50 cards.
            for a in apps:
                b = {"b": a["borrower_id"], "id": a["id"], "today": today}
                a["borrower"] = conn.execute(
                    f"SELECT {BORROWER_COLS} FROM borrower WHERE id = %(b)s", b).fetchone()
                a["loans"] = conn.execute(LOANS_SQL.format(where="l.borrower_id = %(b)s"),
                                          b).fetchall()
                a["history"] = conn.execute(
                    "SELECT id, amount, status, created_at::text AS created_at"
                    " FROM loan_application WHERE borrower_id = %(b)s AND id <> %(id)s"
                    " ORDER BY created_at DESC", b).fetchall()
                a["events"] = conn.execute(
                    "SELECT at::text AS at, type, actor FROM application_event"
                    " WHERE borrower_id = %(b)s AND application_id = %(id)s ORDER BY at, id",
                    b).fetchall()
        return apps

    def decide(self, manager: dict[str, Any], borrower_id: int, app_id: int, approve: bool,
               note: str | None) -> str:
        """RBAC is in the UPDATE itself: only an active collection_manager assigned to the
        applicant's area can move a pending application. Decision + audit event commit
        together. Returns ok / not_found / forbidden / already_decided."""
        status = "approved" if approve else "rejected"
        with self.pool.connection() as conn:
            row = conn.execute(
                "UPDATE loan_application a SET status = %(s)s, decided_by = m.id,"
                " decided_at = now(), note = %(note)s"
                " FROM borrower b, collection_manager m"
                " WHERE a.borrower_id = %(b)s AND a.id = %(id)s AND a.status = 'pending'"
                "   AND b.id = a.borrower_id AND m.id = %(m)s AND m.active"
                "   AND m.role = 'collection_manager'"
                "   AND EXISTS (SELECT 1 FROM manager_area ma"
                "               WHERE ma.manager_id = m.id AND ma.area = b.area)"
                " RETURNING a.id",
                {"s": status, "note": note or None, "b": borrower_id, "id": app_id,
                 "m": manager["id"]}).fetchone()
            if row:
                sp = tracing.current()
                self._audit(conn, borrower_id, app_id, status, manager["email"],
                            {"note": note or None}, sp.trace_id if sp else None)
                return "ok"
            cur = conn.execute(
                "SELECT a.status, b.area FROM loan_application a JOIN borrower b"
                " ON b.id = a.borrower_id WHERE a.borrower_id = %s AND a.id = %s",
                (borrower_id, app_id)).fetchone()
        if cur is None:
            return "not_found"
        if manager["role"] != "collection_manager" or cur["area"] not in manager["areas"]:
            return "forbidden"
        return "already_decided"

    @staticmethod
    def _audit(conn: Connection, borrower_id: int, app_id: int, type_: str, actor: str,
               payload: dict[str, Any], trace_id: str | None) -> None:
        conn.execute(
            "INSERT INTO application_event (borrower_id, application_id, type, actor, payload,"
            " trace_id) VALUES (%s, %s, %s, %s, %s, %s)",
            (borrower_id, app_id, type_, actor, Jsonb(payload), trace_id))
