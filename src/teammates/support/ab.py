"""LLM A/B test: candidate models on the same fixed tickets (the 36 labeled seed tickets).

    python -m teammates.support.ab                              # models from config.toml
    python -m teammates.support.ab --models sarvam-m,sarvam-105b-conversations

Runs the whole Desk per model (same tools, data, guard, risk engine) and compares latency
(LLM calls), cost and tokens per ticket, and quality: classification accuracy, FAQ decision
accuracy, share of LLM drafts that passed the grounding guard (vs the template fallback), and
escalation recall. Needs SARVAM_API_KEY; costs real tokens.
"""

import argparse
import statistics
from typing import Any

from teammates.fse import tracing
from teammates.support import risk
from teammates.support.evals import build
from teammates.support.tools import seed


def compare(models: list[str], client: Any) -> list[dict[str, Any]]:
    tickets = seed()["tickets"]
    rows = []
    for model in models:
        spans: list[tracing.Span] = []
        tracing.set_sink(spans.append)
        try:
            desk, *_ = build(client, model=model)
            results = desk.handle_many([(t["channel"], t["payload"]) for t in tickets])
        finally:
            tracing.set_sink(None)
        pairs = list(zip(tickets, results, strict=True))
        ok = [(t, r) for t, r in pairs if "error" not in r]
        calls = sorted(s.attrs.get("latency_ms", 0.0) for s in spans if s.name.startswith("llm."))
        risky = [(t, r) for t, r in ok if t["expected"]["risky"]]
        rows.append({
            "model": model, "tickets": len(tickets), "errors": len(pairs) - len(ok),
            "llm_ms_p50": round(statistics.median(calls), 1) if calls else None,
            "llm_ms_p95": round(calls[int(0.95 * (len(calls) - 1))], 1) if calls else None,
            "cost_inr_per_ticket": round(sum(s.attrs.get("cost_inr", 0) for s in spans
                                             if s.name.startswith("llm.")) / len(tickets), 4),
            "tokens_per_ticket": round(sum(s.attrs.get("prompt_tokens", 0)
                                           + s.attrs.get("completion_tokens", 0)
                                           for s in spans if s.name.startswith("llm."))
                                       / len(tickets), 1),
            "type_accuracy": _share(r["type"] == t["expected"]["type"] for t, r in ok),
            "faq_accuracy": _share((r["score"] == 0) == t["expected"]["faq"] for t, r in ok),
            "llm_draft_kept": _share(r["risk_breakdown"]["grounding"]["source"] == "llm"
                                     for _, r in ok if r["score"] > 0),
            "escalation_recall": _share(r["route"] == "review" for _, r in risky),
        })
    return rows


def _share(xs: Any) -> float | None:
    xs = list(xs)
    return round(sum(xs) / len(xs), 3) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default=",".join(risk.config()["llm"]["ab_models"]))
    args = ap.parse_args()
    from teammates.runtime import llm
    print_table(compare([m.strip() for m in args.models.split(",") if m.strip()], llm()))


def print_table(rows: list[dict[str, Any]]) -> None:
    """One row per model, one column per metric (shared by the support and KAM A/B runs)."""
    cols = list(rows[0])
    print("  ".join(f"{c:>20}" for c in cols))
    for r in rows:
        print("  ".join(f"{r[c]!s:>20}" for c in cols))


if __name__ == "__main__":
    main()
