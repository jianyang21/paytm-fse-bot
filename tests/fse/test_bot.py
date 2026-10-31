import pytest

from teammates.fse import bot
from teammates.llm.fake import FakeLLM
from tests.fse.conftest import PHONE


@pytest.fixture
def sent(store, monkeypatch):
    monkeypatch.setattr(bot, "store", lambda: store)
    monkeypatch.setattr(bot, "llm", lambda: FakeLLM(lambda r: "Morning Anbu!"))
    out: list[dict] = []
    monkeypatch.setattr(bot, "send", lambda to, msg: out.append({"to": to, **msg}) or True)
    return out


def rows(msg):
    return msg["custom"]["interactive"]["action"]["sections"][0]["rows"]


def body(msg):
    return msg["custom"]["interactive"]["body"]["text"]


def test_morning_list_is_ranked_and_tap_marks_done(sent):
    bot.send_morning()
    plan = sent[-1]
    assert plan["to"] == PHONE and plan["message_type"] == "custom"
    assert bot.REMINDER in body(plan)
    assert [r["title"] for r in rows(plan)] == [
        "Ravi Stores", "Lakshmi Bakery", "Anand Kirana", "Sri Medicals", "Green Leaf Cafe"]

    bot.handle_inbound({"from": PHONE, "message_type": "reply", "reply": rows(plan)[0]})
    assert body(sent[-1]) == "✅ Ravi Stores done. 4 left."
    assert len(rows(sent[-1])) == 4


def test_text_fallback_and_number_reply(sent, monkeypatch):
    def reject_custom(to, msg):
        sent.append(msg)
        return msg["message_type"] == "text"

    monkeypatch.setattr(bot, "send", reject_custom)
    bot.send_morning()
    assert "⬜ 2. New merchant onboarding - Lakshmi Bakery" in sent[-1]["text"]

    bot.handle_inbound({"from": PHONE, "message_type": "text", "text": "done 2"})
    assert sent[-1]["text"].startswith("✅ Lakshmi Bakery done. 4 left.")
    assert "✅ 2. New merchant onboarding" in sent[-1]["text"]


def test_free_text_is_feedback_and_strangers_are_ignored(sent, store):
    bot.handle_inbound({"from": "910000000000", "message_type": "text", "text": "hi"})
    assert sent == []
    bot.handle_inbound({"from": PHONE, "message_type": "text", "text": "Ravi wants 2 more"})
    assert sent[-1]["text"] == "📝 Noted, thanks!"
    with store.pool.connection() as conn:
        assert conn.execute("SELECT body FROM feedback").fetchone()["body"] == "Ravi wants 2 more"


def test_evening_asks_why_per_unfinished_task(sent, store):
    for n in ("1", "2", "3"):  # Ravi, Lakshmi, Anand done; Sri Medicals, Green Leaf left
        bot.handle_inbound({"from": PHONE, "message_type": "text", "text": n})
    bot.send_evening()
    assert body(sent[-1]) == "❓ QR replacement at Sri Medicals wasn't done today. Why?"
    closed = next(r for r in rows(sent[-1]) if r["title"] == "Shop closed")

    bot.handle_inbound({"from": PHONE, "message_type": "reply", "reply": closed})
    assert "Green Leaf Cafe" in body(sent[-1])
    other = next(r for r in rows(sent[-1]) if r["id"].endswith(":other"))

    bot.handle_inbound({"from": PHONE, "message_type": "reply", "reply": other})
    assert sent[-1]["text"] == "✍️ Type why Green Leaf Cafe wasn't done."
    bot.handle_inbound({"from": PHONE, "message_type": "text", "text": "owner travelling"})
    assert sent[-1]["text"].startswith("🙏 Thanks! Any merchant feedback")

    with store.pool.connection() as conn:
        got = conn.execute("SELECT m.name, t.reason_code, t.reason_text FROM task t"
                           " JOIN merchant m ON m.id = t.merchant_id"
                           " WHERE t.status = 'not_done' ORDER BY m.name").fetchall()
        assert [tuple(r.values()) for r in got] == [
            ("Green Leaf Cafe", "other", "owner travelling"), ("Sri Medicals", "closed", None)]
        day = conn.execute("SELECT total, done, not_done, reasons FROM fse_day").fetchone()
        assert day == {"total": 5, "done": 3, "not_done": 2,
                       "reasons": {"closed": 1, "other": 1}}
        assert conn.execute("SELECT count(*) AS n FROM feedback").fetchone()["n"] == 0


def test_shared_location_updates_fse_table(sent, store):
    bot.handle_inbound({"from": PHONE, "message_type": "location",
                        "location": {"lat": 12.95, "long": 77.70, "name": "HSR Layout"}})
    assert sent[-1]["text"] == "📍 Location saved."
    assert {k: store.fse_by_phone(PHONE)[k] for k in ("address", "lat", "lng")} == {
        "address": "HSR Layout", "lat": 12.95, "lng": 77.70}
    bot.handle_inbound({"from": PHONE, "message_type": "location", "location": {"lat": "x"}})
    assert sent[-1]["text"].startswith("⚠️ Couldn't read that location")


def test_done_tasks_and_customer_reviews_are_saved_per_merchant(sent, store):
    for n in ("1", "2"):  # Ravi Stores, Lakshmi Bakery done
        bot.handle_inbound({"from": PHONE, "message_type": "text", "text": n})
    bot.send_reviews()
    assert sent[-1]["text"].startswith("⭐ What did Ravi Stores say")
    bot.handle_inbound({"from": PHONE, "message_type": "text", "text": "Wants 2 more Soundboxes"})
    assert sent[-1]["text"].startswith("⭐ What did Lakshmi Bakery say")
    bot.handle_inbound({"from": PHONE, "message_type": "audio",
                        "audio": {"url": "https://example.com/v.ogg"}})
    assert sent[-1]["text"] == "🙏 Thanks, all customer reviews saved."
    bot.handle_inbound({"from": "910000000000", "message_type": "text", "text": "fake review"})

    with store.pool.connection() as conn:
        done = conn.execute(
            "SELECT m.name, t.done_at IS NOT NULL AS has_time, p.status AS project"
            " FROM task t JOIN merchant m ON m.id = t.merchant_id"
            " JOIN project p ON p.id = t.project_id WHERE t.status = 'done' ORDER BY t.id"
        ).fetchall()
        events = conn.execute("SELECT count(*) AS n FROM task_event WHERE type = 'done'"
                              ).fetchone()["n"]
        reviews = conn.execute(
            "SELECT m.name, r.body, r.audio_url FROM review r"
            " JOIN merchant m ON m.id = r.merchant_id ORDER BY r.id").fetchall()
    assert {r["name"] for r in done} == {"Ravi Stores", "Lakshmi Bakery"}
    assert all(r["has_time"] and r["project"] == "closed" for r in done) and events == 2
    assert [tuple(r.values()) for r in reviews] == [
        ("Ravi Stores", "Wants 2 more Soundboxes", None),
        ("Lakshmi Bakery", None, "https://example.com/v.ogg")]


def test_review_window_closes_at_8pm(sent, store):
    bot.handle_inbound({"from": PHONE, "message_type": "text", "text": "1"})
    bot.send_reviews()  # asks about Ravi Stores, FSE doesn't answer
    bot.send_evening()  # 8 pm: unanswered question dropped, asks why instead
    assert "wasn't done today. Why?" in body(sent[-1])
    with store.pool.connection() as conn:
        assert conn.execute("SELECT count(*) AS n FROM review").fetchone()["n"] == 0


def test_jobs_run_on_their_own_once_per_day_with_catch_up(store, monkeypatch):
    from datetime import datetime

    from teammates.fse.store import IST
    monkeypatch.setattr(bot, "store", lambda: store)
    monkeypatch.setattr(bot, "_claimed", set())
    ran: list[str] = []
    monkeypatch.setattr(bot, "job_fn", lambda name: lambda: ran.append(name))

    def at(hh, mm, day=1):
        bot.run_due_jobs(datetime(2026, 11, day, hh, mm, tzinfo=IST))

    at(8, 29)
    assert ran == []  # before the window
    at(8, 30)
    at(8, 31)
    assert ran == ["morning"]  # on time, once
    bot._claimed.clear()  # restart/deploy: in-memory state gone, DB claim still holds
    at(9, 0)
    # morning not re-run; the 9 am collections brief and KAM digest are due
    assert ran == ["morning", "collections_brief", "kam_digest"]
    at(19, 10)  # bot was asleep at 18:30 and woke at 19:10: catch up
    assert ran == ["morning", "collections_brief", "kam_digest", "reviews"]
    at(20, 0)
    at(23, 0)  # evening window over: not sent late; the 22:00-23:59 health check runs
    full_day = ["morning", "collections_brief", "kam_digest", "reviews", "evening", "health"]
    assert ran == full_day
    at(13, 0, day=2)  # asleep all morning (window ends 12:00): no stale plan at 1 pm
    assert ran == full_day
    at(8, 30, day=3)
    assert ran[-1] == "morning"  # next day runs again


def test_kam_digest_fires_at_nine_in_its_configured_timezone(monkeypatch):
    """Fake clock, in-memory job claims: no database needed."""
    from datetime import UTC, datetime
    from zoneinfo import ZoneInfo

    claims: set = set()

    class FakeStore:
        @staticmethod
        def claim_job(job, day):
            if (job, day) in claims:
                return False
            claims.add((job, day))
            return True

    monkeypatch.setattr(bot, "store", lambda: FakeStore)
    monkeypatch.setattr(bot, "_claimed", set())
    monkeypatch.setattr(bot, "JOBS", [j for j in bot.JOBS if j[0] == "kam_digest"])
    ran: list[datetime] = []
    clock: list[datetime] = []
    monkeypatch.setattr(bot, "job_fn", lambda name: lambda: ran.append(clock[-1]))
    for day, tz in ((1, "Asia/Kolkata"), (2, "Asia/Dubai")):
        monkeypatch.setitem(bot.JOB_TZ, "kam_digest", ZoneInfo(tz))
        ran.clear()
        for hh, mm in ((8, 59), (9, 0), (9, 1), (11, 59), (12, 0)):
            clock.append(datetime(2026, 11, day, hh, mm, tzinfo=ZoneInfo(tz)).astimezone(UTC))
            bot.run_due_jobs(clock[-1])  # the bot's clock may be in any zone
        assert [r.astimezone(ZoneInfo(tz)).strftime("%H:%M") for r in ran] == ["09:00"], tz
