"""Risk scoring engine (T22): Likelihood x Impact with 0-9 factor ratings, averaged per side.
Every rating, band and cutoff comes from config.toml. Runs on every ticket.

score(...) returns the full breakdown that is stored in ticket.risk_breakdown and shown to the
reviewer: score, band, every factor with its value and rating, overrides fired, reasons.
"""

import tomllib
from datetime import datetime, timedelta
from functools import cache
from importlib.resources import files
from typing import Any

FACTOR_LABELS = {
    "two_fa": "2FA", "sim_change": "SIM/device change", "failed_logins": "Failed logins (7 days)",
    "past_fraud": "Past fraud", "identity": "Identity verified", "amount": "Amount at stake",
    "type": "Ticket type",
}


@cache
def config() -> dict[str, Any]:
    return tomllib.loads(files("teammates.support").joinpath("config.toml").read_text("utf-8"))


def band(score: float, cfg: dict[str, Any]) -> str:
    return [name for lo, name in cfg["score"]["bands"] if score >= lo][-1]


def failed_logins_rating(attempts: list[datetime], now: datetime,
                         cfg: dict[str, Any]) -> tuple[int, int]:
    """(attempts in the rolling window, rating). Window: (now - N days, now]."""
    since = now - timedelta(days=cfg["likelihood"]["failed_logins_window_days"])
    n = sum(since < t <= now for t in attempts)
    return n, [r for at_least, r in cfg["likelihood"]["failed_logins"] if n >= at_least][-1]


def amount_rating(amount: float | None, cfg: dict[str, Any]) -> int:
    i = cfg["impact"]
    if not amount:
        return i["amount"]["none"]
    if amount < i["amount_low_below"]:
        return i["amount"]["low"]
    return i["amount"]["mid"] if amount <= i["amount_high_above"] else i["amount"]["high"]


def likelihood(sig: dict[str, Any], now: datetime, cfg: dict[str, Any]) -> dict[str, Any]:
    """sig: two_fa bool, sim_change_at datetime|None, failed_logins [datetime], blocked_now
    bool, past_fraud none|resolved|confirmed, identity yes|partial|no."""
    lk = cfg["likelihood"]
    recent_sim = (sig["sim_change_at"] is not None
                  and now - sig["sim_change_at"] <= timedelta(days=lk["sim_change_recent_days"]))
    n, logins = failed_logins_rating(sig["failed_logins"], now, cfg)
    if sig["blocked_now"]:
        logins = lk["blocked_now"]
    return {
        "two_fa": {"value": "on" if sig["two_fa"] else "off",
                   "rating": lk["two_fa"]["on" if sig["two_fa"] else "off"]},
        "sim_change": {"value": "recent" if recent_sim else "none",
                       "rating": lk["sim_change"]["recent" if recent_sim else "none"]},
        "failed_logins": {"value": "blocked now" if sig["blocked_now"] else n, "rating": logins},
        "past_fraud": {"value": sig["past_fraud"], "rating": lk["past_fraud"][sig["past_fraud"]]},
        "identity": {"value": sig["identity"], "rating": lk["identity"][sig["identity"]]},
    }


def score(*, faq: bool, type_: str, amount: float | None, signals: dict[str, Any] | None,
          injection: bool, confidence: float, verified_sender: bool, now: datetime,
          cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """faq: the FAQ RAG can answer it (generic + confident retrieval); then score is 0 and
    signals are not fetched (None). Overrides are checked on every ticket, FAQ or not."""
    cfg = cfg or config()
    imp = {"amount": {"value": amount, "rating": amount_rating(amount, cfg)},
           "type": {"value": "faq" if faq else type_,
                    "rating": cfg["impact"]["type"]["faq" if faq else type_]}}
    lik = likelihood(signals, now, cfg) if signals else {}
    reasons = []
    avg_l = sum(f["rating"] for f in lik.values()) / len(lik) if lik else 0.0
    if faq:
        avg_i, value = 0.0, 0.0
        reasons.append("answerable from the FAQ: impact 0, score 0")
    else:
        avg_i = sum(f["rating"] for f in imp.values()) / len(imp)
        if avg_l < cfg["likelihood"]["floor"]:
            avg_l = cfg["likelihood"]["floor"]
            reasons.append("likelihood floor applied (non-FAQ ticket)")
        if avg_i < cfg["impact"]["floor"]:
            avg_i = cfg["impact"]["floor"]
            reasons.append("impact floor applied (not answerable from the FAQ)")
        value = avg_l * avg_i / 81 * 100
    reasons += [f"{FACTOR_LABELS[k]}: {f['value']} -> {f['rating']}"
                for k, f in {**lik, **imp}.items() if f["rating"]]

    o, overrides = cfg["overrides"], []
    if type_ == "dispute" and (amount or 0) > o["dispute_amount_above"]:
        overrides.append(f"dispute/chargeback over {o['dispute_amount_above']}")
    if signals and signals["identity"] == "no" and type_ in o["money_moving_types"]:
        overrides.append("identity check failed on a money-moving ticket")
    if signals and signals["past_fraud"] == "confirmed":
        overrides.append("confirmed fraud flag")
    if injection:
        overrides.append("input guard: injection flagged")
    if signals and signals["blocked_now"] and o["override_blocked_now"]:
        overrides.append("blocked from account now")
    if confidence < o["min_classify_confidence"]:
        overrides.append(f"classifier confidence {confidence:.2f} below "
                         f"{o['min_classify_confidence']}")
    if not verified_sender:
        overrides.append("sender is not a verified contact")
    return {
        "score": round(value, 1), "band": band(value, cfg),
        "avg_likelihood": round(avg_l, 2), "avg_impact": round(avg_i, 2),
        "likelihood": lik, "impact": imp, "overrides": overrides, "reasons": reasons,
        "security_flag": bool(signals and signals["blocked_now"]),
    }
