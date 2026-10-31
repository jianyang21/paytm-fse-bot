from datetime import datetime, time

import pytest

from teammates.fse import bot
from teammates.fse.store import IST, ist_today
from teammates.llm.fake import FakeLLM
from tests.fse.conftest import PHONE

MORNING = datetime.combine(ist_today(), time(8, 30), IST)  # DB timestamps use the real day


@pytest.fixture
def wired(store, monkeypatch):
    """Real bot code against the test store; only the Vonage HTTP call and the LLM are fake."""
    monkeypatch.setattr(bot, "store", lambda: store)
    monkeypatch.setattr(bot, "llm", lambda: FakeLLM(lambda r: "Morning!"))
    monkeypatch.setattr(bot, "_claimed", set())
    sent: list[tuple[str, dict]] = []

    def fake_deliver(to, msg):
        sent.append((to, msg))
        return True, 202, None, f"uuid-{len(sent)}"

    monkeypatch.setattr(bot, "deliver", fake_deliver)
    with store.pool.connection() as conn:  # a second FSE, so one can fail and one succeed
        conn.execute("INSERT INTO fse (name, phone, area) VALUES ('Second', '915', 'Nowhere')")
    store.cache.flushall()
    return store, sent


def q(store, sql, *args):
    with store.pool.connection() as conn:
        return conn.execute(sql, args).fetchall()


def test_one_fse_failing_is_isolated_and_root_cause_is_pinpointed(wired, monkeypatch):
    store, sent = wired
    second = store.fse_by_phone("915")["id"]
    real_allot = store._allot_day

    def boom(conn, fse_id, day):
        if fse_id == second:
            raise RuntimeError("injected allot failure")
        return real_allot(conn, fse_id, day)

    monkeypatch.setattr(store, "_allot_day", boom)
    bot.run_due_jobs(MORNING)

    assert [to for to, _ in sent] == [PHONE]  # the healthy FSE still got the plan
    cause = q(store, "SELECT failed_step, error, fse_id FROM v_root_causes")
    assert len(cause) == 1
    assert cause[0]["failed_step"] == "job.morning > fse > plan"
    assert cause[0]["fse_id"] == second
    assert cause[0]["error"].startswith("RuntimeError: injected allot failure at test_tracing.py:")
    job = q(store, "SELECT job, minutes_late, fses, failed_fses, status FROM v_job_runs")
    assert job == [{"job": "job.morning", "minutes_late": 0, "fses": 2, "failed_fses": 1,
                    "status": "ok"}]


def test_every_send_is_logged_and_vonage_rejection_is_the_root_cause(wired, monkeypatch):
    store, sent = wired
    monkeypatch.setattr(bot, "deliver",
                        lambda to, msg: (False, 401, '{"title":"Unauthorized"}', None))
    bot.run_due_jobs(MORNING)

    log = q(store, "SELECT direction, phone_masked, kind, ok, http_status, trace_id"
                   " FROM message_log ORDER BY id")
    assert {r["phone_masked"] for r in log} == {"91******0000", "***"}  # never the raw number
    assert all(r["direction"] == "out" and not r["ok"] and r["http_status"] == 401 for r in log)
    assert all(r["trace_id"] for r in log)  # each message links to the trace that sent it
    steps = {r["failed_step"] for r in q(store, "SELECT failed_step FROM v_root_causes")}
    assert steps == {"job.morning > fse > vonage.send"}
    assert "Unauthorized" in q(store, "SELECT error FROM v_root_causes LIMIT 1")[0]["error"]


def test_failed_whatsapp_delivery_lands_in_the_sending_trace(wired):
    store, sent = wired
    bot.run_due_jobs(MORNING)
    trace = q(store, "SELECT trace_id FROM message_log WHERE message_uuid = 'uuid-1'")[0]
    bot.record_status({"message_uuid": "uuid-1", "status": "rejected",
                       "error": {"title": "Outside 24h window", "detail": "use a template"}})

    row = q(store, "SELECT delivery, delivery_error FROM message_log"
                   " WHERE message_uuid = 'uuid-1'")[0]
    assert row == {"delivery": "rejected", "delivery_error": "Outside 24h window - use a template"}
    cause = q(store, "SELECT failed_step, error, trace_id FROM v_root_causes")[0]
    assert cause["failed_step"] == "vonage.delivery"
    assert cause["error"] == "WhatsApp rejected: Outside 24h window - use a template"
    assert cause["trace_id"] == trace["trace_id"]  # same trace as the morning job that sent it


def test_inbound_is_traced_and_strangers_are_logged_masked(wired):
    store, sent = wired
    bot.handle_inbound({"from": "919812345678", "message_type": "text", "text": "hi"})
    bot.handle_inbound({"from": PHONE, "message_type": "text", "text": "Ravi wants 2 more"})

    spans = q(store, "SELECT fse_id, attrs FROM trace_span WHERE name = 'inbound'"
                     " ORDER BY started_at")
    assert spans[0]["attrs"]["rejected"] == "unregistered sender"
    assert spans[0]["attrs"]["sender"] == "91******5678"
    assert spans[1]["attrs"]["action"] == "feedback" and spans[1]["fse_id"] is not None
    inbound = q(store, "SELECT phone_masked, ok FROM message_log WHERE direction = 'in'"
                       " ORDER BY id")
    assert inbound == [{"phone_masked": "91******5678", "ok": False},
                       {"phone_masked": "91******0000", "ok": True}]


def test_health_report_names_missing_jobs_and_root_causes_and_alerts(wired, monkeypatch):
    store, sent = wired
    monkeypatch.setattr(bot, "deliver", lambda to, msg: (
        sent.append((to, msg)) or (to != PHONE, 202 if to != PHONE else 500,
                                   None if to != PHONE else "boom", None)))
    monkeypatch.setattr(bot, "env", lambda k, d="": "919000000001" if k == "ALERT_PHONE" else d)
    bot.run_due_jobs(MORNING)  # morning runs, its send to PHONE fails
    bot.run_due_jobs(MORNING.replace(hour=22, minute=5))  # health; reviews/evening never ran

    report = store.health(MORNING.date(), ["morning", "reviews", "evening"])
    assert not report["ok"]
    assert "job reviews did not run" in report["problems"]
    assert "job evening did not run" in report["problems"]
    assert any(p.startswith("job.morning > fse > vonage.send failed") for p in report["problems"])
    assert report["metrics"]["sent"] >= 2 and report["metrics"]["send_failed"] >= 1
    alert_to, alert = sent[-1]
    assert alert_to == "919000000001" and alert["text"].startswith("⚠️ FSE bot health")


def test_daily_metrics_view(wired):
    store, sent = wired
    bot.run_due_jobs(MORNING)
    bot.handle_inbound({"from": PHONE, "message_type": "text", "text": "1"})
    m = q(store, "SELECT * FROM v_daily_metrics WHERE day = %s", MORNING.date())[0]
    assert m["fses_planned"] == 2 and m["tasks"] == 5 and m["done"] == 1
    assert float(m["done_pct"]) == 20.0
    assert m["sent"] == len(sent) and m["send_failed"] == 0 and m["received"] == 1
    assert m["inbound_p50_ms"] is not None
