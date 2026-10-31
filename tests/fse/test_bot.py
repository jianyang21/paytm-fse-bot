import pytest

from teammates.fse import bot
from teammates.llm.fake import FakeLLM

PHONE = "919999900000"


@pytest.fixture
def sent(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "fse.db"))
    monkeypatch.setenv("FSE_PHONE", "+91 99999 00000")
    monkeypatch.setattr(bot, "llm", lambda: FakeLLM(lambda r: "Morning Anbu!"))
    out: list[dict] = []
    monkeypatch.setattr(bot, "send", lambda to, msg: out.append({"to": to, **msg}) or True)
    return out


def rows(msg):
    return msg["custom"]["interactive"]["action"]["sections"][0]["rows"]


def test_morning_list_is_ranked_and_tap_marks_done(sent):
    bot.send_morning()
    plan = sent[-1]
    assert plan["to"] == PHONE and plan["message_type"] == "custom"
    assert bot.REMINDER in plan["custom"]["interactive"]["body"]["text"]
    assert [r["title"] for r in rows(plan)] == [
        "Ravi Stores", "Lakshmi Bakery", "Anand Kirana", "Sri Medicals", "Green Leaf Cafe"]

    bot.handle_inbound({"from": PHONE, "message_type": "reply", "reply": rows(plan)[0]})
    after = sent[-1]
    assert after["custom"]["interactive"]["body"]["text"] == "✅ Ravi Stores done. 4 left."
    assert len(rows(after)) == 4


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


def test_free_text_is_feedback_and_strangers_are_ignored(sent):
    bot.handle_inbound({"from": "910000000000", "message_type": "text", "text": "hi"})
    assert sent == []
    bot.handle_inbound({"from": PHONE, "message_type": "text", "text": "Ravi wants 2 more"})
    assert sent[-1]["text"] == "📝 Noted, thanks!"
    with bot.db() as con:
        assert con.execute("SELECT body FROM feedback").fetchone()[0] == "Ravi wants 2 more"
