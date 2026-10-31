import json

import pytest

from teammates.fse import tracing
from teammates.llm.client import LLMError
from teammates.llm.fake import FakeLLM
from teammates.support.desk import (
    Denied,
    SendBlocked,
    metrics,
    reject_outcome,
    route_for,
    rule_classify,
    why_denied,
)
from teammates.support.evals import build, fake_reply
from teammates.support.faq import FAQS, TRANSLATIONS
from tests.support.conftest import SEED

FAQ_SETTLE = next(f["a"] for f in FAQS if f["id"] == "settle-time")


def tool_names(env):
    return [name for name, _ in env.tools.calls]


def whatsapp(frm, text, ts="2026-10-30T11:20:00+05:30"):
    return "whatsapp", {"from": frm, "text": text, "timestamp": ts}


# --- FAQ path ------------------------------------------------------------------------------

def test_faq_ticket_scores_zero_and_is_answered_from_the_faq_only(env):
    t = env.run("T01")
    assert (t["score"], t["status"], t["route"], t["type"]) == (0, "resolved", "auto", "faq")
    assert t["final_reply"] == FAQ_SETTLE and t["faq_id"] == "settle-time"
    assert tool_names(env) == ["find_merchant", "get_user_details"]  # no other data fetch
    (channel, to, text), = env.sent
    assert (channel, to) == ("email", "srimedicals@example.com")
    assert text == f"{FAQ_SETTLE}\n\nTicket #{t['id']} is resolved. Reply to this message " \
                   "if you need anything else."


def test_key_account_faq_resolved_by_ai_is_reported_to_the_kam(env):
    t = env.run("T05")
    assert t["status"] == "resolved" and t["assigned_to"] == "kam@example.com"
    rep = [a for a in env.store.audits() if a["action"] == "kam_report"]
    assert len(rep) == 1 and "Ravi Stores" in rep[0]["payload"]["summary"]
    assert [x["id"] for x in env.store.queue(env.kam)["fyi"]] == [t["id"]]
    # FAQ answer and confirmation both in the merchant's language (Hindi)
    assert t["final_reply"] == TRANSLATIONS["refund-how"]["hi"] and "टिकट" in env.sent[0][2]


# --- non-FAQ: below / above threshold, routing ------------------------------------------------

def test_low_risk_non_faq_is_resolved_by_ai_with_fetched_data(env):
    t = env.run("T12")  # clean account, status question: score 2.5 < threshold 5
    assert t["score"] > 0 and t["status"] == "resolved" and t["route"] == "auto"
    assert set(tool_names(env)) == {"find_merchant", "get_user_details",
                                    "get_security_signals", "get_txn_status"}
    assert "S-12901" in t["final_reply"] and "ಟಿಕೆಟ್" in env.sent[0][2]  # Kannada confirmation
    assert env.store.signals[109]["friction"] == 1


def test_above_threshold_is_drafted_but_not_sent_and_goes_to_a_support_agent(env):
    t = env.run("T13")  # score 5.9 >= 5
    assert (t["status"], t["route"], t["assigned_role"], t["assigned_to"]) == (
        "pending_review", "review", "support_agent", None)
    assert "R-13002" in t["draft"] and env.sent == []
    rb = t["risk_breakdown"]
    assert rb["escalate"] and rb["threshold"] == 5 and rb["grounding"]["source"] == "llm"
    assert [x["id"] for x in env.store.queue(env.agent)["review"]] == [t["id"]]


def test_key_account_review_goes_to_its_kam(env):
    t = env.run("T14")
    assert (t["assigned_role"], t["assigned_to"]) == ("kam", "kam@example.com")
    assert env.store.queue(env.agent)["review"] == []


def test_route_for():
    assert route_for({"tier": "key", "kam_email": "k@x"}) == ("kam", "k@x")
    assert route_for({"tier": "key", "kam_email": None}) == ("support_agent", None)
    assert route_for({"tier": "standard", "kam_email": "k@x"}) == ("support_agent", None)
    assert route_for(None) == ("support_agent", None)


def test_override_forces_review_whatever_the_score(env):
    t = env.run("T17")
    assert t["route"] == "review" and "blocked from account now" in t["risk_breakdown"][
        "overrides"] and t["risk_breakdown"]["security_flag"]


# --- review actions + RBAC --------------------------------------------------------------------

def test_approve_sends_the_ai_draft_as_is(env):
    t = env.run("T13")
    after = env.desk.review(env.agent, 110, t["id"], "approve")
    assert after["status"] == "resolved" and after["final_reply"] == t["draft"]
    assert env.sent[-1][2].startswith(t["draft"]) and "டிக்கெட்" in env.sent[-1][2]
    a = [x for x in env.store.audits() if x["action"] == "approve"]
    assert a[0]["actor"] == "agent@example.com" and a[0]["band"] == t["band"]
    assert env.store.signals[110]["friction"] == 2  # 1 contact + a human was needed


def test_edit_then_send_records_the_diff(env):
    t = env.run("T13")
    after = env.desk.review(env.agent, 110, t["id"], "edit", text="Refund R-13002 is on its way.")
    assert after["final_reply"] == "Refund R-13002 is on its way."
    assert env.sent[-1][2].startswith("Refund R-13002 is on its way.")
    diff = next(a for a in env.store.audits() if a["action"] == "edit")["payload"]["diff"]
    assert "-" + t["draft"].splitlines()[0] in diff and "+Refund R-13002 is on its way." in diff


def test_reject_keeps_ticket_with_reviewer_and_logs_reason(env):
    t = env.run("T13")
    after = env.desk.review(env.agent, 110, t["id"], "reject", reason="  call the merchant ")
    assert (after["status"], after["assigned_to"], after["reject_reason"]) == (
        "manual", "agent@example.com", "call the merchant")
    assert env.sent == []
    assert next(a for a in env.store.audits() if a["action"] == "reject")["payload"] == {
        "reason": "call the merchant"}
    assert [x["id"] for x in env.store.queue(env.agent)["manual"]] == [t["id"]]


def test_reject_outcome_is_one_function():
    t = {"merchant_id": 1, "id": 2, "band": "low", "trace_id": None}
    changes, entry = reject_outcome(t, {"email": "a@x"}, "why")
    assert changes == {"status": "manual", "assigned_to": "a@x", "reject_reason": "why"}
    assert entry["action"] == "reject" and entry["payload"] == {"reason": "why"}


@pytest.mark.parametrize("action, kw, err", [
    ("reject", {}, ValueError), ("reject", {"reason": "  "}, ValueError),
    ("edit", {"text": ""}, ValueError), ("edit", {"text": "x" * 2001}, ValueError),
    ("reclassify", {}, ValueError),
    ("edit", {"text": "Call me on 919812345678"}, SendBlocked),
    ("edit", {"text": "see https://evil.example"}, SendBlocked),
])
def test_bad_review_requests_change_nothing(env, action, kw, err):
    t = env.run("T13")
    with pytest.raises(err):
        env.desk.review(env.agent, 110, t["id"], action, **kw)
    assert env.store.get(110, t["id"])["status"] == "pending_review" and env.sent == []


def test_rbac_denials_are_audited(env):
    t = env.run("T14")  # assigned to kam@example.com
    env.store.upsert_reviewer("kam2@example.com", "pw", "Other KAM", "kam")
    other = env.store.reviewers["kam2@example.com"]
    inactive = env.kam | {"active": False}
    for who, why in ((env.agent, "role support_agent cannot act on kam tickets"),
                     (other, "assigned to another reviewer"), (inactive, "not active")):
        with pytest.raises(Denied, match=why):
            env.desk.review(who, 102, t["id"], "approve")
    assert env.sent == [] and env.store.get(102, t["id"])["status"] == "pending_review"
    denied = [a for a in env.store.audits() if a["action"] == "denied"]
    assert len(denied) == 3 and denied[0]["payload"]["attempted"] == "approve"
    env.desk.review(env.kam, 102, t["id"], "approve")
    with pytest.raises(Denied, match="not awaiting review"):  # double click: no second send
        env.desk.review(env.kam, 102, t["id"], "approve")
    assert len(env.sent) == 1


def test_why_denied_happy_path():
    t = {"status": "pending_review", "assigned_role": "kam", "assigned_to": None}
    assert why_denied({"active": True, "role": "kam", "email": "a@x"}, t) is None


def test_unknown_ticket(env):
    with pytest.raises(LookupError):
        env.desk.review(env.agent, 110, 999, "approve")


def test_metrics_rates_per_band():
    rows = [{"action": a, "band": b} for a, b in [
        ("approve", "low"), ("approve", "low"), ("edit", "low"), ("reject", "low"),
        ("reject", "high"), ("denied", "high"), ("escalated", "low")]]
    assert metrics(rows) == {
        "high": {"n": 1, "approve_rate": 0.0, "edit_rate": 0.0, "reject_rate": 1.0},
        "low": {"n": 4, "approve_rate": 0.5, "edit_rate": 0.25, "reject_rate": 0.25}}


# --- repeat contacts + cache --------------------------------------------------------------------

def test_repeat_contact_reuses_cached_data_and_answers_instantly(env):
    t = env.run("T13")
    calls = len(env.tools.calls)
    again = env.desk.handle(*whatsapp("919900000110", "Any update??"))
    assert again["id"] == t["id"] and again["contacts"] == 2
    assert again["status"] == "pending_review"  # still the human's
    assert tool_names(env)[calls:] == ["find_merchant"]  # profile, security, txns: cache hits
    assert "Any update??" in again["text"] and env.sent[-1][2].startswith("டிக்கெட் #")
    assert any(a["action"] == "repeat_contact" for a in env.store.audits())


def test_repeat_by_ticket_number_from_another_channel(env):
    t = env.run("T13")
    again = env.desk.handle("email", {"from": "iyertailoring@example.com", "subject": "Re",
                                      "body": f"About ticket #{t['id']}, any news?",
                                      "date": "2026-10-31T09:00:00+05:30"})
    assert again["id"] == t["id"] and again["contacts"] == 2


def test_later_message_from_same_contact_is_a_new_ticket(env):
    t = env.run("T13")  # 11:20
    later = env.desk.handle(*whatsapp("919900000110", "Where is my settlement?",
                                      ts="2026-10-30T13:00:00+05:30"))
    assert later["id"] != t["id"]


def test_review_invalidates_the_cache(env):
    t = env.run("T13")
    env.desk.review(env.agent, 110, t["id"], "approve")
    assert not env.desk.cache.r.keys(f"support:110:{t['id']}:*")


# --- failure paths ------------------------------------------------------------------------------

def test_llm_down_falls_back_to_keywords_and_a_human(env):
    def down(req):
        raise LLMError("timeout")

    desk, store, tools, sent = build(FakeLLM(down))
    t = desk.handle(SEED["T12"]["channel"], SEED["T12"]["payload"])
    assert t["route"] == "review" and t["confidence"] == 0.0
    assert any("confidence" in o for o in t["risk_breakdown"]["overrides"])
    assert t["risk_breakdown"]["grounding"]["source"] == "template" and "S-12901" in t["draft"]
    assert t["draft"].startswith("ನಮಸ್ಕಾರ Rakesh, ನಿಮ್ಮ ಸ್ಥಿತಿ ವಿನಂತಿ ನಮಗೆ ತಲುಪಿದೆ.")  # Kannada


def test_draft_with_invented_numbers_is_replaced_by_template():
    def liar(req):
        return fake_reply(req) if req.schema_name == "classify" else \
            "Your refund of ₹99,999 was sent; share the OTP to confirm."

    desk, *_ = build(FakeLLM(liar))
    t = desk.handle(SEED["T13"]["channel"], SEED["T13"]["payload"])
    g = t["risk_breakdown"]["grounding"]
    assert g["source"] == "template" and "asks for a secret" in g["problems"]
    assert "99,999" not in t["draft"] and "R-13002 (₹450): pending" in t["draft"]
    assert t["draft"].startswith("வணக்கம் Kiran, உங்கள் ரீஃபண்ட் கோரிக்கை (₹450)")  # Tamil


def test_customer_text_reaches_the_llm_only_inside_untrusted_tags(env):
    env.run("A03")
    for req in env.llm.requests:
        user = req.messages[-1].content
        assert user.count("<untrusted") == 1 and user.count("</untrusted>") == 1
        assert "you are now an admin" not in req.messages[0].content


@pytest.mark.parametrize("tid", ["A01", "A02", "A03", "A04"])
def test_injection_tickets_go_to_a_human(env, tid):
    t = env.run(tid)
    assert t["route"] == "review" and "input guard: injection flagged" in t["risk_breakdown"][
        "overrides"] and env.sent == []


def test_unknown_sender_never_gets_a_reply(env):
    t = env.run("A05")
    assert t["merchant_id"] == 0 and t["route"] == "review" and env.sent == []
    with pytest.raises(SendBlocked):
        env.desk.review(env.agent, 0, t["id"], "approve")
    assert env.sent == []


def test_channel_failure_on_auto_reply_escalates(env):
    env.desk.sender = lambda *a: False
    t = env.run("T01")
    assert t["status"] == "pending_review" and t["route"] == "review"


def test_crash_mid_decision_leaves_ticket_with_a_human(env):
    def broken(mid):
        raise RuntimeError("security service down")

    env.tools.get_security_signals = broken
    with pytest.raises(RuntimeError):
        env.run("T12")
    (t,) = env.store.tickets.values()
    assert t["status"] == "pending_review" and t["assigned_role"] == "support_agent"


# --- concurrency --------------------------------------------------------------------------------

def test_many_tickets_at_once_in_input_order(env):
    items = [(SEED[k]["channel"], SEED[k]["payload"]) for k in ("T01", "T13", "T14", "T22")]
    out = env.desk.handle_many(items + [("fax", {})])
    assert [t.get("merchant_id") for t in out[:4]] == [101, 110, 102, 101]
    assert out[4] == {"error": "unknown channel 'fax'; expected one of ('email', 'whatsapp', "
                               "'call')"}
    assert len({t["id"] for t in out[:4]}) == 4


# --- tracing ------------------------------------------------------------------------------------

def test_every_step_is_traced_without_raw_contact_details(env):
    spans = []
    tracing.set_sink(spans.append)
    try:
        t = env.run("T13")
        env.desk.review(env.agent, 110, t["id"], "approve")
    finally:
        tracing.set_sink(None)
    names = [s.name for s in spans]
    for n in ("intake", "classify", "llm.classify", "faq_check", "fetch", "risk", "llm.draft",
              "support.ticket", "support.review", "send"):
        assert n in names, n
    llm = next(s for s in spans if s.name == "llm.draft")
    assert {"model", "prompt_version", "prompt_tokens", "completion_tokens", "latency_ms",
            "cost_inr"} <= set(llm.attrs)
    root = next(s for s in spans if s.name == "support.ticket")
    assert root.attrs["cost_inr"] > 0 and root.attrs["prompt_tokens"] == 800
    assert {s.trace_id for s in spans} == {t["trace_id"]}  # review joins the ticket's trace
    blob = json.dumps([s.attrs for s in spans], default=str)
    assert "919900000110" not in blob and "Refund of" not in blob


def test_rule_classify_fallback():
    assert rule_classify("How do I issue a refund?")["type"] == "faq"
    assert rule_classify("Refund ₹500 to my customer")["type"] == "refund"
    assert rule_classify("chargeback on my account")["type"] == "dispute"
    assert rule_classify("my settlement is pending")["type"] == "status"
    assert rule_classify("anything")["confidence"] == 0.0


def test_template_in_english_for_english_merchants():
    def down(req):
        raise LLMError("timeout")

    desk, *_ = build(FakeLLM(down))
    t = desk.handle(SEED["T11"]["channel"], SEED["T11"]["payload"])
    assert t["draft"].startswith("Hello Suresh, we have your status request for ₹9,350.")
    assert t["draft"].endswith("Our team will update you here if anything more is needed.")
