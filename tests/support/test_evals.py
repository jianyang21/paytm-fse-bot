from teammates.support import evals


def test_offline_eval_passes_every_threshold():
    report = evals.run(live=False)
    failed = {k: m["failures"] for k, m in report["metrics"].items() if not m["passed"]}
    assert report["passed"], failed
    assert report["metrics"]["attack_suite"]["checks"] > 10
    info = report["info"]
    assert info["cost_inr_per_ticket"] > 0 and info["latency_ms_p50"] is not None
    assert set(info["reviewer_rates_per_band"]) <= {"low", "medium", "high"}


def test_fake_reply_classifies_and_drafts_from_facts():
    from teammates.llm.client import ChatRequest, Message

    req = ChatRequest(model="m", schema_name="classify", messages=[
        Message(role="user", content='<untrusted source="email">\nRefund ₹500\n</untrusted>')])
    assert '"type": "refund"' in evals.fake_reply(req)
    req = ChatRequest(model="m", messages=[Message(role="user", content=(
        "FACTS:\n- refund R-1: ₹5, pending, 2026-10-30\n\nCUSTOMER MESSAGE:\nx"))])
    assert evals.fake_reply(req) == ("Thank you for writing to us. Your refund R-1: ₹5, "
                                     "pending, 2026-10-30.")
