"""End-to-end evals for the collections brief email.

    python -m teammates.collections.evals                 # 60 borrowers, seed 7
    python -m teammates.collections.evals --borrowers 200 --seed 3

Drives the real code (bot scheduler + job_run claims, LoanStore with its Redis version
reconciliation, brief builder, output guard, message_log, tracing) on a throwaway Postgres
schema; only Resend's HTTP call and the LLM are fake. Each metric is a pass rate against a
threshold; every failure is listed. Writes reports/collections_eval/latest.json and exits 1 if
any metric is below its threshold.
"""

import argparse
import json
import random
import sys
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest import mock

import psycopg
from psycopg import sql

from teammates.collections import app as coll
from teammates.collections.store import LoanStore
from teammates.config import read_env_file
from teammates.fse import bot, tracing
from teammates.fse.evals import Score, at
from teammates.fse.store import IST, _next_month, ist_today
from teammates.llm.client import LLMError
from teammates.llm.fake import FakeLLM

THRESHOLDS = {
    "sent_on_schedule": 1.0,  # once per day inside 09:00-12:00, never early/late/twice
    "lateness_recorded_correctly": 1.0,
    "numbers_match_ledger": 1.0,  # vs an independent recompute from raw instalments
    "verified_recipients_only": 1.0,
    "data_scope": 1.0,  # nothing from outside the recipient's areas
    "llm_draft_guard": 1.0,  # bad drafts replaced by the template, clean ones kept
    "cache_reconciliation": 1.0,  # direct DB writes show up; untouched areas stay cached
    "message_log_coverage": 1.0,  # every send logged, traced, recipient masked
    "root_cause_accuracy": 1.0,  # injected faults named exactly, others still delivered
}
AREAS = ["Bellandur", "HSR Layout", "Koramangala", "Whitefield"]
FIRST = ["Ravi", "Meena", "Arjun", "Salma", "Kiran", "Deepa", "Vinod", "Priya", "Fatima",
         "Suresh", "Anita", "Imran", "Lakshmi", "Naveen", "Pooja", "Rahul", "Sneha", "Tariq",
         "Uma", "Zoya"]
LAST = ["Iyer", "Gowda", "Khan", "Shetty", "Nair", "Rao", "Sheikh", "Kumar", "Menon", "Pillai",
        "Reddy", "Das", "Joshi", "Patel", "Bhat", "Varma", "Ali", "Singh", "Kapoor", "Mehta"]
EDGE_DPD = (0, 1, 30, 31, 90, 91)
CLEAN_DRAFT ="Portfolio reviewed.\nSuggested actions:\n- Call the most overdue borrowers first"


def health_rule(dpd: int) -> str:
    """The health spec, written independently of app.health."""
    return ("Healthy" if dpd == 0 else "Watch" if dpd <= 30 else "At risk" if dpd <= 90
            else "NPA")


@dataclass
class Harness:
    store: LoanStore
    sent: list[tuple[str, str, str]]  # (to, subject, html)
    fail_to: dict[str, tuple[int, str]]  # recipient -> (http status, error) Resend returns
    draft: Callable[[Any], str]  # fake LLM reply

    def q(self, query: str, *args: Any) -> list[dict[str, Any]]:
        with self.store.pool.connection() as conn:
            return conn.execute(query, args).fetchall()

    def write(self, query: str, *args: Any) -> int:
        with self.store.pool.connection() as conn:
            return conn.execute(query, args).rowcount


class _Unlimited:
    def acquire(self) -> float:
        return 0.0


@contextmanager
def harness(dsn: str) -> Iterator[Harness]:
    """Real bot scheduler + collections code on a fresh schema; Resend and the LLM faked."""
    import fakeredis  # dev dependency; evals never touch the real Redis

    schema = f"eval_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    store = LoanStore(dsn, cache=fakeredis.FakeRedis(), schema=schema)
    h = Harness(store, [], {}, lambda r: CLEAN_DRAFT)

    def fake_deliver(to: str, subject: str, html: str
                     ) -> tuple[bool, int | None, str | None, str | None]:
        h.sent.append((to, subject, html))
        if to in h.fail_to:
            status, error = h.fail_to[to]
            return False, status, error, None
        return True, 200, None, f"email-{len(h.sent)}"

    try:
        store.migrate()
        tracing.set_sink(store.record_span)
        with mock.patch.object(bot, "store", lambda: store), \
                mock.patch.object(coll, "loan_store", lambda: store), \
                mock.patch.object(coll, "llm", lambda: FakeLLM(lambda r: h.draft(r))), \
                mock.patch.object(coll, "deliver_email", fake_deliver), \
                mock.patch.object(coll, "rate_limiter", lambda *a, **k: _Unlimited()), \
                mock.patch.object(bot, "_claimed", set()), \
                mock.patch.object(bot, "JOBS", [j for j in bot.JOBS
                                                if j[0] == "collections_brief"]):
            yield h
    finally:
        tracing.set_sink(None)
        store.pool.close()
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def seed(h: Harness, rng: random.Random, n: int, today: date) -> None:
    """Borrowers spread over AREAS with 1-2 loans each. 60% pay everything; the rest miss
    their latest 1-5 instalments. Some instalments are paid a few days late, 5% of loans are
    closed, and every loan has 1-2 future instalments. One target per area this month."""
    month = today.replace(day=1)
    with h.store.pool.connection() as conn:
        for a in AREAS:
            conn.execute("INSERT INTO area (name) VALUES (%s)", (a,))
            conn.execute("INSERT INTO area_target (area, month, target) VALUES (%s, %s, %s)",
                         (a, month, rng.randrange(20000, 200000, 1000)))
        for i in range(n):
            area = AREAS[i % len(AREAS)]
            bid = conn.execute(
                "INSERT INTO borrower (name, business, phone, area) VALUES (%s, %s, %s, %s)"
                " RETURNING id", (f"{FIRST[i % 20]} {LAST[i // 20 % 20]}", f"Shop {i}",
                                  f"9198{i:08d}", area)).fetchone()["id"]
            for _ in range(rng.choice([1, 1, 1, 1, 2])):
                emi = rng.randrange(2000, 30000, 100)
                closed = rng.random() < 0.05
                lid = conn.execute(
                    "INSERT INTO loan (borrower_id, principal, emi, disbursed_on, status)"
                    " VALUES (%s, %s, %s, %s, %s) RETURNING id",
                    (bid, emi * 12, emi, today - timedelta(days=300),
                     "closed" if closed else "active")).fetchone()["id"]
                offset = rng.randint(0, 29)
                past = [today - timedelta(days=offset + 30 * j)
                        for j in range(rng.randint(1, 8))][::-1]
                missed = 0 if closed or rng.random() < 0.6 else rng.randint(1, min(5, len(past)))
                rows = [(bid, lid, d, emi, None if k >= len(past) - missed
                         else min(d + timedelta(days=rng.randint(0, 5)), today))
                        for k, d in enumerate(past)]
                rows += [(bid, lid, today + timedelta(days=rng.randint(1, 20) + 30 * j), emi, None)
                         for j in range(rng.randint(1, 2))]
                conn.cursor().executemany(
                    "INSERT INTO instalment (borrower_id, loan_id, due_on, amount, paid_on)"
                    " VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING", rows)
        # Health boundaries, never left to chance: due today (not late), then exactly 1, 30,
        # 31, 90 and 91 days past due, one borrower each, in rotating areas.
        for k, dpd in enumerate(EDGE_DPD):
            i = n + k
            bid = conn.execute(
                "INSERT INTO borrower (name, business, phone, area) VALUES (%s, %s, %s, %s)"
                " RETURNING id", (f"{FIRST[i % 20]} {LAST[i // 20 % 20]}", f"Edge {dpd}",
                                  f"9198{i:08d}", AREAS[k % len(AREAS)])).fetchone()["id"]
            lid = conn.execute(
                "INSERT INTO loan (borrower_id, principal, emi, disbursed_on) VALUES"
                " (%s, 60000, 5000, %s) RETURNING id",
                (bid, today - timedelta(days=200))).fetchone()["id"]
            conn.execute("INSERT INTO instalment (borrower_id, loan_id, due_on, amount)"
                         " VALUES (%s, %s, %s, 5000)", (bid, lid, today - timedelta(days=dpd)))


def seed_managers(h: Harness, rng: random.Random) -> dict[str, dict[str, Any]]:
    """Four valid collection managers with random area sets, plus two who must never get it:
    an unverified one and an area_collection_manager (view-only role)."""
    managers = {}
    for k in range(4):
        areas = sorted(rng.sample(AREAS, rng.randint(1, len(AREAS))))
        managers[f"cm{k}@example.com"] = {"role": "collection_manager", "areas": areas,
                                          "verified": True, "should_get": True}
    managers["unverified@example.com"] = {"role": "collection_manager", "areas": AREAS,
                                          "verified": False, "should_get": False}
    managers["viewer@example.com"] = {"role": "area_collection_manager", "areas": AREAS,
                                      "verified": True, "should_get": False}
    for email, m in managers.items():
        h.store.upsert_manager(email, "pw", email.split("@")[0], m["role"], m["areas"],
                               m["verified"])
    return managers


def ledger(h: Harness, today: date) -> dict[str, Any]:
    """Expected brief contents recomputed from raw instalments, independent of the app's SQL."""
    month, nxt = today.replace(day=1), _next_month(today.replace(day=1))
    rows = h.q("SELECT b.name, b.phone, b.area, l.id AS loan_id, l.status, i.due_on, i.amount,"
               " i.paid_on FROM borrower b JOIN loan l ON l.borrower_id = b.id"
               " JOIN instalment i ON i.borrower_id = l.borrower_id AND i.loan_id = l.id")
    loans: dict[int, dict[str, Any]] = {}
    pending: dict[str, list[tuple[str, int]]] = {a: [] for a in AREAS}
    collected = dict.fromkeys(AREAS, 0)
    for r in rows:
        if r["paid_on"] and month <= r["paid_on"] < nxt:
            collected[r["area"]] += r["amount"]
        if r["status"] != "active":
            continue
        ln = loans.setdefault(r["loan_id"], {"name": r["name"], "area": r["area"],
                                             "phone": r["phone"], "late": []})
        if r["paid_on"] is None and r["due_on"] < today:
            ln["late"].append((r["due_on"], r["amount"]))
        if r["paid_on"] is None and r["due_on"] < nxt:
            pending[r["area"]].append((r["phone"], r["amount"]))
    for ln in loans.values():
        ln["dpd"] = (today - min(d for d, _ in ln["late"])).days if ln["late"] else 0
        ln["overdue"] = sum(a for _, a in ln["late"])
        ln["health"] = health_rule(ln["dpd"])
    targets = {r["area"]: r["target"] for r in h.q(
        "SELECT area, target FROM area_target WHERE month = %s", month)}
    return {"loans": loans, "pending": pending, "collected": collected, "targets": targets}


def check_brief(scores: dict[str, Score], d: dict[str, Any], exp: dict[str, Any],
                areas: list[str], who: str) -> None:
    """The data a brief renders vs the independent ledger, for one recipient's areas."""
    got = sorted((ln["loan_id"], ln["dpd"], ln["overdue"], ln["health"]) for ln in d["loans"])
    want = sorted((lid, ln["dpd"], ln["overdue"], ln["health"])
                  for lid, ln in exp["loans"].items() if ln["area"] in areas)
    scores["numbers_match_ledger"].check(
        got == want, f"{who}: loans differ, e.g. got {sorted(set(got) - set(want))[:3]} "
                     f"want {sorted(set(want) - set(got))[:3]}")
    want_pending = sum(a for area in areas for _, a in exp["pending"][area])
    scores["numbers_match_ledger"].check(
        d["pending_total"] == want_pending and len(d["pending"]) == sum(
            len(exp["pending"][a]) for a in areas),
        f"{who}: pending {d['pending_total']} in {len(d['pending'])}, ledger {want_pending}")
    for t in d["targets"]:
        a = t["area"]
        pct = round(100 * exp["collected"][a] / exp["targets"][a]) if exp["targets"][a] else 0
        scores["numbers_match_ledger"].check(
            (t["collected"], t["target"], t["pct"]) == (exp["collected"][a], exp["targets"][a],
                                                        pct),
            f"{who} {a}: target row {t}, ledger collected {exp['collected'][a]}")


def check_email(scores: dict[str, Score], html: str, exp: dict[str, Any], areas: list[str],
                who: str) -> None:
    """Rendered email: every in-scope active borrower and pending phone present, nothing from
    outside the recipient's areas."""
    for ln in exp["loans"].values():
        if ln["area"] in areas:
            scores["numbers_match_ledger"].check(
                ln["name"] in html and coll.inr(ln["overdue"]) in html,
                f"{who}: {ln['name']} or {coll.inr(ln['overdue'])} missing from the email")
    for a in AREAS:
        for phone, _ in exp["pending"][a]:
            if a in areas:
                scores["numbers_match_ledger"].check(
                    f"+{phone}" in html, f"{who}: pending phone +{phone} missing")
            else:
                scores["data_scope"].check(phone not in html,
                                           f"{who}: phone of a {a} borrower leaked")
    outside = {ln["name"] for ln in exp["loans"].values() if ln["area"] not in areas} - {
        ln["name"] for ln in exp["loans"].values() if ln["area"] in areas}
    leaked = sorted(n for n in outside if n in html)
    scores["data_scope"].check(not leaked, f"{who}: names from other areas {leaked[:5]}")


def run_schedule(h: Harness, rng: random.Random, scores: dict[str, Score]) -> None:
    """Seven simulated days through bot.run_due_jobs: wake-ups before, inside and after the
    window, extra ticks, restarts, then a full content check of every email."""
    today = ist_today()  # DB data and brief dates use the real clock
    seed(h, rng, h.n, today)
    managers = seed_managers(h, rng)
    expected_to = sorted(e for e, m in managers.items() if m["should_get"])
    start, end = next((s, e) for n, s, e in bot.JOBS if n == "collections_brief")
    window = (int(end[:2]) * 60 + int(end[3:])) - (int(start[:2]) * 60 + int(start[3:]))
    delays: list[int] = []
    for k in range(7):
        day = today + timedelta(days=k)
        before = len(h.sent)
        bot.run_due_jobs(at(day, start, -1))  # 08:59: too early
        scores["sent_on_schedule"].check(len(h.sent) == before, f"day {k}: sent before {start}")
        if k == 6:  # bot asleep all morning, woken after the window: must not send late
            bot.run_due_jobs(at(day, end, 5))
            scores["sent_on_schedule"].check(len(h.sent) == before,
                                             f"day {k}: sent after {end}")
            continue
        delay = rng.randint(0, window - 1)
        delays.append(delay)
        bot.run_due_jobs(at(day, start, delay))
        got = sorted(t for t, *_ in h.sent[before:])
        scores["sent_on_schedule"].check(
            got == expected_to, f"day {k} (+{delay} min): sent to {got}, want {expected_to}")
        scores["verified_recipients_only"].check(
            set(got) <= set(expected_to), f"day {k}: unexpected recipients "
                                          f"{sorted(set(got) - set(expected_to))}")
        n_after = len(h.sent)
        bot.run_due_jobs(at(day, start, delay + 1))  # next tick
        bot._claimed.clear()  # restart / deploy / second instance
        bot.run_due_jobs(at(day, start, delay + 2))
        scores["sent_on_schedule"].check(len(h.sent) == n_after, f"day {k}: sent twice")

    late = [s.attrs.get("minutes_late") for s in sorted(
        (sp for sp in h.spans() if sp.name == "job.collections_brief"),
        key=lambda sp: sp.started_at)]
    scores["lateness_recorded_correctly"].check(late == delays,
                                                f"recorded {late}, actual {delays}")
    claims = h.q("SELECT day FROM job_run WHERE job = 'collections_brief'")
    scores["sent_on_schedule"].check(len(claims) == 6, f"{len(claims)} job_run claims for 6 "
                                                       "send days")

    exp = ledger(h, today)
    for to, _, html in h.sent[:len(expected_to)]:
        areas = managers[to]["areas"]
        d = coll.fetch_brief(h.store, areas, today)
        check_brief(scores, d, exp, areas, to)
        check_email(scores, html, exp, areas, to)

    logged = h.q("SELECT kind, phone_masked, ok, trace_id, message_uuid FROM message_log")
    scores["message_log_coverage"].check(
        len(logged) == len(h.sent), f"{len(h.sent)} sends but {len(logged)} log rows")
    for r in logged:
        scores["message_log_coverage"].check(
            r["kind"] == "email" and r["ok"] and r["trace_id"] and r["message_uuid"]
            and "@example" not in r["phone_masked"], f"log row {r}")


def run_guard(h: Harness, rng: random.Random, scores: dict[str, Score]) -> None:
    """Adversarial LLM drafts for one manager limited to some areas: each bad draft must be
    replaced by the template, the clean one kept, and a dead LLM must still yield an email."""
    today = ist_today()
    seed(h, rng, h.n, today)
    areas = ["Bellandur", "HSR Layout"]
    h.store.upsert_manager("cm@example.com", "pw", "cm", "collection_manager", areas, True)
    m = h.store.brief_recipients()[0]
    exp = ledger(h, today)
    inside = next(ln for ln in exp["loans"].values() if ln["area"] in areas and ln["dpd"])
    outside = next(ln["name"] for ln in exp["loans"].values() if ln["area"] not in areas
                   and ln["name"] not in {x["name"] for x in exp["loans"].values()
                                          if x["area"] in areas})
    fact_based = f"{inside['name']} is {inside['dpd']} days past due."

    def boom(r: Any) -> str:
        raise LLMError("injected: LLM down")

    cases = {  # draft -> should the email keep it?
        "clean, from the facts": (lambda r: fact_based, True),
        "invented amount": (lambda r: "Recover ₹9,87,654 this week.", False),
        "phone number": (lambda r: "Call 9876543210 today.", False),
        "link": (lambda r: "Details: https://evil.example/x", False),
        "borrower from another area": (lambda r: f"Visit {outside} today.", False),
        "LLM down": (boom, False),
    }
    for name, (reply, keep) in cases.items():
        h.draft = reply
        built = coll.build_brief(h.store, m, today)
        html = built[1] if built else ""
        try:
            draft = reply(None)
        except LLMError:
            draft = None
        kept = draft is not None and draft.split("\n")[0] in html
        template = coll.template_summary(coll.fetch_brief(h.store, areas, today))
        scores["llm_draft_guard"].check(
            built is not None and kept == keep and (keep or template in html),
            f"{name}: draft {'kept' if kept else 'dropped'}, expected "
            f"{'kept' if keep else 'template'}")
        check_email(scores, html, exp, areas, f"guard case '{name}'")


def run_cache(h: Harness, rng: random.Random, scores: dict[str, Score]) -> None:
    """Rounds of writes made straight in SQL (as a payment system would): the next brief must
    show them, re-reading only the areas that changed."""
    today = ist_today()
    seed(h, rng, h.n, today)
    coll.fetch_brief(h.store, AREAS, today)
    for rnd in range(5):
        touched = set(rng.sample(AREAS, rng.randint(0, 2)))
        for a in sorted(touched):
            paid = rng.random() < 0.7 and h.write(  # a payment, if anything is due
                "UPDATE instalment SET paid_on = %s WHERE (borrower_id, id) IN ("
                " SELECT i.borrower_id, i.id FROM instalment i JOIN borrower b"
                " ON b.id = i.borrower_id WHERE b.area = %s AND i.paid_on IS NULL"
                " AND i.due_on <= %s ORDER BY i.due_on LIMIT 1)", today, a, today)
            if not paid:  # otherwise a target change
                h.write("UPDATE area_target SET target = target + 1000 WHERE area = %s", a)
        d = coll.fetch_brief(h.store, AREAS, today)
        cache = next(sp for sp in reversed(h.spans()) if sp.name == "fetch").attrs["cache"]
        want = {k: v for k, v in {"hit": len(AREAS) - len(touched),
                                  "stale": len(touched)}.items() if v}
        scores["cache_reconciliation"].check(
            cache == want, f"round {rnd}: touched {sorted(touched)}, cache states {cache}")
        before = len(scores["numbers_match_ledger"].failures)
        check_brief(scores, d, ledger(h, today), AREAS, f"cache round {rnd}")
        scores["cache_reconciliation"].check(
            len(scores["numbers_match_ledger"].failures) == before,
            f"round {rnd}: brief after direct writes differs from the ledger")


def run_faults(h: Harness, rng: random.Random, scores: dict[str, Score]) -> None:
    """Resend rejects one manager, another is unverified: v_root_causes must name exactly
    those two steps, and every other manager must still get their email."""
    today = ist_today()
    seed(h, rng, h.n, today)
    for k in range(3):
        h.store.upsert_manager(f"cm{k}@example.com", "pw", f"cm{k}", "collection_manager",
                               AREAS, True)
    h.store.upsert_manager("unverified@example.com", "pw", "u", "collection_manager", AREAS,
                           False)
    h.fail_to["cm1@example.com"] = (422, "injected: domain not verified")
    bot.run_due_jobs(at(today, "09:00", 0))
    expected = {("job.collections_brief > collections.manager > resend.send",
                 coll.mask_email("cm1@example.com")),
                ("job.collections_brief > collections.manager > guard.output", "recipient")}
    got = {(r["failed_step"], r["attrs"].get("to") or r["attrs"].get("stage"))
           for r in h.q("SELECT failed_step, attrs FROM v_root_causes")}
    for e in sorted(expected):
        scores["root_cause_accuracy"].check(e in got, f"injected fault not found: {e}")
    for g in sorted(got - expected):
        scores["root_cause_accuracy"].check(False, f"false alarm: {g}")
    reached = {t for t, *_ in h.sent}
    scores["root_cause_accuracy"].check(
        {"cm0@example.com", "cm2@example.com"} <= reached,
        f"healthy managers skipped after another failed: reached {sorted(reached)}")
    failed_log = h.q("SELECT ok, http_status, error FROM message_log WHERE NOT ok")
    scores["message_log_coverage"].check(
        len(failed_log) == 1 and failed_log[0]["http_status"] == 422,
        f"failed send logged as {failed_log}")


def run(dsn: str, n_borrowers: int = 60, seed_: int = 7) -> dict[str, Any]:
    rng = random.Random(seed_)
    scores = {name: Score() for name in THRESHOLDS}
    for part in (run_schedule, run_guard, run_cache, run_faults):
        with harness(dsn) as h:
            h.n = n_borrowers
            spans: list[tracing.Span] = []
            tracing.set_sink(lambda s, h=h, spans=spans: (spans.append(s),
                                                          h.store.record_span(s)))
            h.spans = lambda spans=spans: spans
            part(h, rng, scores)
    metrics = {name: {"score": round(s.value, 4), "threshold": THRESHOLDS[name],
                      "passed": s.value >= THRESHOLDS[name], "checks": s.total,
                      "failures": s.failures[:20]} for name, s in scores.items()}
    return {"run_at": datetime.now(IST).isoformat(timespec="seconds"),
            "borrowers": n_borrowers, "seed": seed_,
            "passed": all(m["passed"] for m in metrics.values()), "metrics": metrics}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--borrowers", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dsn", default=None, help="Postgres DSN (default TEST_DATABASE_URL)")
    ap.add_argument("--out", default="reports/collections_eval/latest.json")
    args = ap.parse_args()
    dsn = args.dsn or bot.env("TEST_DATABASE_URL") or read_env_file(Path(".env")).get(
        "TEST_DATABASE_URL")
    if not dsn:
        sys.exit("TEST_DATABASE_URL not set")
    report = run(dsn, args.borrowers, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    for name, m in report["metrics"].items():
        mark = "PASS" if m["passed"] else "FAIL"
        print(f"{mark}  {name:30s} {m['score']:.3f} (>= {m['threshold']}, {m['checks']} checks)")
        for f in m["failures"][:5]:
            print(f"        - {f}")
    print(f"\n{'ALL PASSED' if report['passed'] else 'FAILED'} -> {out}")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
