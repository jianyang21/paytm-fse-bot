from teammates.llm.client import LLMError, TokenUsage
from teammates.llm.fake import FakeLLM
from teammates.support import ab
from teammates.support.evals import fake_reply


def test_compare_models_on_the_same_tickets():
    def reply(req):
        if req.model == "bad-model":
            raise LLMError("model not found")
        return fake_reply(req)

    rows = ab.compare(["sarvam-m", "bad-model"], FakeLLM(reply, TokenUsage(
        prompt_tokens=1000, completion_tokens=100)))
    good, bad = rows
    assert good["model"] == "sarvam-m" and good["tickets"] == 36 and good["errors"] == 0
    assert good["cost_inr_per_ticket"] > 0 and good["type_accuracy"] > 0.9
    assert good["escalation_recall"] == 1.0 and good["llm_draft_kept"] == 1.0
    # A failing model still completes every ticket (fallbacks), and the table shows the cost.
    assert bad["errors"] == 0 and bad["llm_draft_kept"] == 0.0 and bad["cost_inr_per_ticket"] == 0
    assert bad["escalation_recall"] == 1.0  # low confidence -> every ticket to a human


def test_main_prints_a_table(monkeypatch, capsys):
    import teammates.runtime

    monkeypatch.setattr(teammates.runtime, "llm", lambda: FakeLLM(fake_reply))
    monkeypatch.setattr("sys.argv", ["ab", "--models", "a,b"])
    ab.main()
    out = capsys.readouterr().out.splitlines()
    assert "type_accuracy" in out[0] and len(out) == 3
