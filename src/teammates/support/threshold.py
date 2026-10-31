"""Escalation threshold from cost-sensitive decision theory (Elkan 2001).

Escalate when P(risky | score) >= p* = C_review / (C_review + C_miss). P(risky | score) is
calibrated on the labeled seed tickets as the share of risky tickets per score bin. Only tickets
the threshold actually decides are used: FAQ-answerable tickets (score 0, never escalated by
score) and tickets with an override (always escalated) are left out.

    python -m teammates.support.threshold     # prints the calibration table and the sweep
"""

from datetime import datetime, timedelta, timezone
from functools import cache
from typing import Any

from teammates.guard import InputGuard
from teammates.support import risk
from teammates.support.intake import normalize
from teammates.support.tools import MockTools, parse_signals, seed

SEED_NOW = datetime(2026, 10, 30, 18, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))


def p_star(cfg: dict[str, Any]) -> float:
    t = cfg["threshold"]
    return t["c_review"] / (t["c_review"] + t["c_miss"])


def reference_risk(t: dict[str, Any], tools: MockTools, cfg: dict[str, Any]) -> dict[str, Any]:
    """The score a seed ticket gets from its labeled type/amount/FAQ (no LLM involved)."""
    tk, e = normalize(t["channel"], t["payload"]), t["expected"]
    mid = tools.find_merchant(tk.contact)
    sig = None if e["faq"] or mid is None else parse_signals(tools.get_security_signals(mid))
    return risk.score(faq=e["faq"], type_=e["type"], amount=e["amount"], signals=sig,
                      injection=InputGuard.scan(tk.text).injection, confidence=1.0,
                      verified_sender=mid is not None, now=tools.now, cfg=cfg)


def seed_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """[{id, score, risky, faq, overridden}] for every labeled seed ticket."""
    s = seed()
    tools = MockTools(s["merchants"], SEED_NOW)
    rows = []
    for t in s["tickets"]:
        r = reference_risk(t, tools, cfg)
        rows.append({"id": t["id"], "score": r["score"], "risky": t["expected"]["risky"],
                     "faq": t["expected"]["faq"], "overridden": bool(r["overrides"])})
    return rows


def calibrate(rows: list[dict[str, Any]], width: float) -> list[dict[str, Any]]:
    """Share of risky tickets per score bin [lo, lo + width), over threshold-decided rows."""
    bins: dict[int, list[bool]] = {}
    for r in rows:
        if not r["faq"] and not r["overridden"]:
            bins.setdefault(int(r["score"] // width), []).append(r["risky"])
    return [{"lo": k * width, "hi": (k + 1) * width, "n": len(v), "risky": sum(v),
             "p": sum(v) / len(v)} for k, v in sorted(bins.items())]


def choose(bins: list[dict[str, Any]], p: float) -> float:
    """Lowest bin whose risky share reaches p*. ponytail: a non-monotone bin above it is
    escalated too (safe direction); fit isotonic regression once the seed set is large."""
    return next((b["lo"] for b in bins if b["p"] >= p), float("inf"))


def escalated(r: dict[str, Any], t: float) -> bool:
    return r["overridden"] or (not r["faq"] and r["score"] > 0 and r["score"] >= t)


def sweep(rows: list[dict[str, Any]], thresholds: list[float]) -> list[dict[str, Any]]:
    """Escalation recall (risky tickets sent to a human) vs share of all tickets escalated."""
    risky = [r for r in rows if r["risky"]]
    return [{"t": t, "escalated": sum(escalated(r, t) for r in rows) / len(rows),
             "recall": sum(escalated(r, t) for r in risky) / len(risky) if risky else 1.0}
            for t in thresholds]


@cache
def score_threshold() -> float:
    cfg = risk.config()
    try:
        return choose(calibrate(seed_rows(cfg), cfg["threshold"]["bin_width"]), p_star(cfg))
    except (OSError, ValueError, KeyError):
        return cfg["threshold"]["fallback_score"]


def main() -> None:
    cfg = risk.config()
    rows, ps = seed_rows(cfg), p_star(cfg)
    bins = calibrate(rows, cfg["threshold"]["bin_width"])
    chosen = choose(bins, ps)
    c = cfg["threshold"]
    print(f"p* = C_review / (C_review + C_miss) = {c['c_review']} / "
          f"({c['c_review']} + {c['c_miss']}) = {ps:.3f}")
    print(f"{len(rows)} labeled tickets: {sum(r['faq'] for r in rows)} FAQ, "
          f"{sum(r['overridden'] for r in rows)} with an override (both left out of the bins)\n")
    print("score bin     n  risky  P(risky)  escalate")
    for b in bins:
        print(f"[{b['lo']:>4.0f},{b['hi']:>4.0f})  {b['n']:>3}  {b['risky']:>5}  {b['p']:>8.2f}"
              f"  {'yes' if b['p'] >= ps else 'no'}")
    width = c["bin_width"]
    sw = sweep(rows, [i * width for i in range(int(100 // width) + 1)])
    best = next((s["t"] for s in sorted(sw, key=lambda s: s["escalated"])
                 if s["recall"] >= c["target_recall"]), None)
    print("\nthreshold  escalated  recall")
    for s in sw:
        mark = " <- Elkan" if s["t"] == chosen else ""
        mark += f" <- cheapest with recall >= {c['target_recall']}" if s["t"] == best else ""
        print(f"{s['t']:>9.0f}  {s['escalated']:>9.0%}  {s['recall']:>6.0%}{mark}")
    print(f"\nElkan threshold: score >= {chosen} (overrides bypass it)")


if __name__ == "__main__":
    main()
