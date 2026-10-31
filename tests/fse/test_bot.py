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
