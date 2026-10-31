"""Support desk evals on the labeled seed tickets (seed.json) and the attack suite.

    python -m teammates.support.evals            # offline: fake LLM, in-memory store
    python -m teammates.support.evals --live     # real Sarvam calls (needs SARVAM_API_KEY)

Drives the real Desk (intake, guard, classify, FAQ, tools, cache, risk, threshold, draft,
outbound guard, review, audit_log, tracing) with MockTools, MemoryStore and a fake Redis; only
the channel sender is fake. Each metric is a pass rate against a threshold; reviewer rates per
band, latency, cost and cache hit rate are reported alongside. Writes
reports/support_eval/latest.json and exits 1 if any metric is below its threshold.
"""

import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from teammates.fse import tracing
from teammates.fse.evals import Score
from teammates.guard import text_problems
from teammates.llm.client import ChatRequest, TokenUsage
from teammates.llm.fake import FakeLLM
from teammates.support import risk
from teammates.support.desk import Denied, Desk, rule_classify
from teammates.support.faq import FAQS, answer
from teammates.support.store import MemoryStore
from teammates.support.threshold import SEED_NOW, reference_risk
from teammates.support.tools import ContextCache, MockTools, seed

THRESHOLDS = {
    "faq_zero_nonfaq_positive": 1.0,
    "score_matches_label": 0.9,       # pipeline score (LLM classification) vs labeled inputs
    "overrides_fire": 1.0,
    "escalation_recall": 0.95,        # risky tickets that reached a human
    "grounded_answers": 1.0,          # FAQ verbatim, or every number from the fetched data
    "review_actions": 1.0,            # approve / edit / reject applied and audited
    "cache_hit_on_repeat": 1.0,
    "attack_suite": 1.0,
    "verified_recipients_only": 1.0,
}
REVIEWERS = {"kam": {"email": "kam@example.com", "name": "Anita (KAM)"},
             "support_agent": {"email": "agent@example.com", "name": "Support agent"}}
UNTRUSTED_RE = re.compile(r"<untrusted[^>]*>\n(.*)\n</untrusted>", re.S)


def fake_reply(req: ChatRequest) -> str:
    """Deterministic stand-in: keyword classification, a draft that restates the records."""
    user = req.messages[-1].content
    if req.schema_name == "classify":
        m = UNTRUSTED_RE.search(user)
        return json.dumps(rule_classify(m.group(1) if m else user) | {"confidence": 0.9})
    facts = user.split("FACTS:\n", 1)[-1].split("\n\nCUSTOMER MESSAGE", 1)[0]
    recs = [ln[2:] for ln in facts.splitlines() if ln.startswith("- ")]
    return "Thank you for writing to us. " + (
        " ".join(f"Your {r}." for r in recs) or "Our team is checking this for you.")


def build(llm: Any, now: datetime = SEED_NOW, model: str | None = None
          ) -> tuple[Desk, MemoryStore, MockTools, list[tuple[str, str, str]]]:
    import fakeredis  # dev dependency: evals and tests only

    s, store = seed(), MemoryStore()
    for role, r in REVIEWERS.items():
        store.upsert_reviewer(r["email"], "pw", r["name"], role)
    tools, sent = MockTools(s["merchants"], now), []
    cfg = risk.config()
    desk = Desk(store=store, tools=tools, llm=llm, model=model,
                cache=ContextCache(fakeredis.FakeRedis(), cfg["cache"]),
                sender=lambda ch, to, text: sent.append((ch, to, text)) or True,
                now=lambda: now)
    return desk, store, tools, sent


def grounding_facts(desk: Desk, tools: MockTools, t: dict[str, Any]) -> str:
    """The facts a reply may use, rebuilt from the tools (not from what the desk stored)."""
    if t["risk_breakdown"]["grounding"]["source"].startswith("faq:"):
        return answer(next(f for f in FAQS if f["id"] == t["faq_id"]), t["language"])
    mid = t["merchant_id"]
    profile = tools.get_user_details(mid) if mid else None
    faq = next((f for f in FAQS if f["id"] == t["faq_id"]), None)
    return desk.facts(profile, t["type"], t["amount"], tools.get_txn_status(mid) if mid else [],
                      faq) + (f"\n{profile['name']}" if profile else "")


def run(live: bool = False) -> dict[str, Any]:
    spans: list[tracing.Span] = []
    tracing.set_sink(spans.append)
    try:
        return _run(live, spans)
    finally:
        tracing.set_sink(None)


def _run(live: bool, spans: list[tracing.Span]) -> dict[str, Any]:
    if live:
        from teammates.runtime import llm
        client: Any = llm()
    else:
        client = FakeLLM(fake_reply, TokenUsage(prompt_tokens=420, completion_tokens=90))
    desk, store, tools, sent = build(client)
    cfg, sc = desk.cfg, {name: Score() for name in THRESHOLDS}
    s = seed()
    labeled, attacks = s["tickets"], s["attacks"]

    # 1. every seed + attack ticket, concurrently
    results = desk.handle_many([(t["channel"], t["payload"]) for t in labeled + attacks])
    by_id = dict(zip([t["id"] for t in labeled + attacks], results, strict=True))
    first_risk = {k: dict(v.get("risk_breakdown") or {}) for k, v in by_id.items()}
    for t in labeled:
        got, e = by_id[t["id"]], t["expected"]
        if "error" in got:
            for m in ("faq_zero_nonfaq_positive", "score_matches_label"):
                sc[m].check(False, f"{t['id']}: {got['error']}")
            continue
        sc["faq_zero_nonfaq_positive"].check(
            (got["score"] == 0) == e["faq"],
            f"{t['id']}: faq={e['faq']} but score {got['score']}")
        ref = reference_risk(t, MockTools(s["merchants"], SEED_NOW), cfg)
        sc["score_matches_label"].check(
            abs(got["score"] - ref["score"]) < 0.05,
            f"{t['id']}: score {got['score']} (type {got['type']}, amount {got['amount']}), "
            f"label gives {ref['score']} (type {e['type']}, amount {e['amount']})")
        if e["risky"]:
            sc["escalation_recall"].check(got["route"] == "review",
                                          f"{t['id']}: risky but auto-resolved "
                                          f"(score {got['score']}, threshold "
                                          f"{first_risk[t['id']]['threshold']})")
    for t in labeled + attacks:
        got = by_id[t["id"]]
        for o in t["expected"].get("overrides", []):
            fired = first_risk[t["id"]].get("overrides", [])
            sc["overrides_fire"].check(any(o in f for f in fired),
                                       f"{t['id']}: expected override '{o}', fired {fired}")
        if "draft" in got and got["draft"]:
            facts = grounding_facts(desk, tools, got)
            source = got["risk_breakdown"]["grounding"]["source"]
            ok = got["draft"] == facts if source.startswith("faq:") else \
                not text_problems(got["draft"], facts)
            sc["grounded_answers"].check(ok, f"{t['id']}: {source} draft not grounded: "
                                             f"{text_problems(got['draft'], facts)}")

    # 2. attack suite: injection/SQL text never auto-sent, stored verbatim, store intact
    for t in attacks:
        got = by_id[t["id"]]
        sc["attack_suite"].check(got.get("route") == "review",
                                 f"{t['id']}: attack ticket was auto-resolved")
    a02 = by_id["A02"]
    sc["attack_suite"].check("DROP TABLE ticket" in a02["text"] and len(store.tickets) >= len(
        labeled), "SQL payload not stored verbatim, or tickets lost")

    # 3. repeat contact on every ticket a human owns: answered from the cache
    open_ = [t for t in labeled + attacks if by_id[t["id"]].get("status") == "pending_review"
             and by_id[t["id"]]["merchant_id"]]
    seen = len(spans)
    for t in open_:
        follow = dict(t["payload"])
        ask = f"Any update on ticket #{by_id[t['id']]['id']}?"
        key = {"email": "body", "whatsapp": "text"}.get(t["channel"])
        if key:
            follow[key] = ask
        else:
            follow["turns"] = [{"speaker": "customer", "text": ask}]
        if t["channel"] == "email":
            follow["subject"] = "Re: " + t["payload"].get("subject", "")
        r = desk.handle(t["channel"], follow)
        sc["cache_hit_on_repeat"].check(r["id"] == by_id[t["id"]]["id"] and r["contacts"] == 2,
                                        f"{t['id']}: repeat contact opened a new ticket")
    fetches = [sp for sp in spans[seen:] if sp.name == "fetch"]
    for sp in fetches:
        sc["cache_hit_on_repeat"].check(sp.attrs.get("cache") == "hit",
                                        f"repeat-contact fetch {sp.attrs.get('part')} missed")

    # 4. reviews: wrong role first (must be denied), then the labeled action
    for t in labeled + attacks:
        tk = store.get(by_id[t["id"]].get("merchant_id", -1), by_id[t["id"]].get("id", -1))
        if not tk or tk["status"] != "pending_review":
            continue
        right = store.reviewers[REVIEWERS[tk["assigned_role"]]["email"]]
        wrong = store.reviewers[REVIEWERS["kam" if tk["assigned_role"] == "support_agent"
                                          else "support_agent"]["email"]]
        try:
            desk.review(wrong, tk["merchant_id"], tk["id"], "approve")
            sc["attack_suite"].check(False, f"{t['id']}: wrong-role review was allowed")
        except Denied:
            sc["attack_suite"].check(True, "")
        action = t["expected"].get("review", "reject" if t in attacks else "approve")
        if tk["merchant_id"] == 0 and action != "reject":
            action = "reject"  # unverified sender: nothing can be sent
        try:
            after = desk.review(right, tk["merchant_id"], tk["id"], action,
                                text=(tk["draft"] or "") + " We will keep you posted.",
                                reason="looks suspicious; handling by phone")
            want = "manual" if action == "reject" else "resolved"
            sc["review_actions"].check(after["status"] == want,
                                       f"{t['id']}: {action} left status {after['status']}")
        except Exception as exc:  # noqa: BLE001  any failure is a failed check
            sc["review_actions"].check(False, f"{t['id']}: {action} failed: {exc}")
    audits = store.audits()
    denied = [a for a in audits if a["action"] == "denied"]
    sc["attack_suite"].check(len(denied) > 0, "denied review attempts not in audit_log")
    edits = [a for a in audits if a["action"] == "edit"]
    sc["review_actions"].check(all(a["payload"]["diff"] for a in edits), "edit without a diff")

    # 5. outbound: only verified contacts of the ticket's merchant
    verified = {c for m in s["merchants"].values() for c in m["contacts"]}
    for ch, to, _ in sent:
        sc["verified_recipients_only"].check(to in verified, f"sent on {ch} to an unverified "
                                                             "contact")

    roots = [sp for sp in spans if sp.name == "support.ticket"]
    first_roots = [sp for sp in roots if not sp.attrs.get("repeat")]
    lat = sorted(sp.duration_ms for sp in first_roots)
    all_fetch = [sp for sp in spans if sp.name == "fetch"]
    metrics = {name: {"score": round(x.value, 4), "threshold": THRESHOLDS[name],
                      "passed": x.value >= THRESHOLDS[name], "checks": x.total,
                      "failures": x.failures[:20]} for name, x in sc.items()}
    from teammates.support.desk import metrics as review_metrics
    return {
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "llm": "live" if live else "fake", "tickets": len(labeled), "attacks": len(attacks),
        "passed": all(m["passed"] for m in metrics.values()), "metrics": metrics,
        "info": {
            "reviewer_rates_per_band": review_metrics(audits),
            "escalated_share": round(sum(r.get("route") == "review" for r in results)
                                     / len(results), 3),
            "latency_ms_p50": round(statistics.median(lat), 1) if lat else None,
            "latency_ms_p95": round(lat[int(0.95 * (len(lat) - 1))], 1) if lat else None,
            "cost_inr_per_ticket": round(statistics.mean(
                sp.attrs.get("cost_inr", 0) for sp in first_roots), 4) if first_roots else None,
            "tokens_per_ticket": round(statistics.mean(
                sp.attrs.get("prompt_tokens", 0) + sp.attrs.get("completion_tokens", 0)
                for sp in first_roots), 1) if first_roots else None,
            "cache_hit_rate_overall": round(sum(sp.attrs.get("cache") == "hit" for sp in all_fetch)
                                            / len(all_fetch), 3) if all_fetch else None,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", action="store_true", help="real LLM calls instead of the fake")
    ap.add_argument("--out", default="reports/support_eval/latest.json")
    args = ap.parse_args()
    report = run(args.live)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    for name, m in report["metrics"].items():
        mark = "PASS" if m["passed"] else "FAIL"
        print(f"{mark}  {name:28s} {m['score']:.3f} (>= {m['threshold']}, {m['checks']} checks)")
        for f in m["failures"][:5]:
            print(f"        - {f}")
    print("\n" + json.dumps(report["info"], indent=2))
    print(f"\n{'ALL PASSED' if report['passed'] else 'FAILED'} -> {out}")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
