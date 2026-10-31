import copy
from datetime import UTC, datetime, timedelta

import pytest

from teammates.support import risk

NOW = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)
CFG = risk.config()


def sig(**kw):
    return {"two_fa": True, "sim_change_at": None, "failed_logins": [], "blocked_now": False,
            "past_fraud": "none", "identity": "yes"} | kw


def score(faq=False, type_="status", amount=None, signals=None, injection=False,
          confidence=0.9, verified=True, cfg=CFG):
    return risk.score(faq=faq, type_=type_, amount=amount,
                      signals=None if faq else (signals or sig()), injection=injection,
                      confidence=confidence, verified_sender=verified, now=NOW, cfg=cfg)


def ago(**kw):
    return NOW - timedelta(**kw)


@pytest.mark.parametrize("n, rating", [(0, 0), (1, 3), (2, 3), (3, 6), (4, 6), (5, 8), (9, 8)])
def test_failed_login_rating_boundaries(n, rating):
    assert risk.failed_logins_rating([ago(hours=i + 1) for i in range(n)], NOW, CFG) == (n, rating)


def test_failed_logins_rolling_seven_days_not_calendar_week():
    attempts = [ago(days=7) + timedelta(seconds=1),  # inside: 6 days 23:59:59 ago
                ago(days=7),                          # exactly 7 days: outside
                ago(days=8), NOW + timedelta(minutes=1)]  # older / in the future: outside
    assert risk.failed_logins_rating(attempts, NOW, CFG) == (1, 3)
    # Six days back from a Monday is the previous calendar week: still counted.
    monday = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)
    three = [monday - timedelta(days=6)] * 3
    assert risk.failed_logins_rating(three, monday, CFG) == (3, 6)


def test_blocked_now_rates_9_and_raises_security_flag():
    r = score(signals=sig(failed_logins=[ago(hours=1)], blocked_now=True))
    assert r["likelihood"]["failed_logins"] == {"value": "blocked now", "rating": 9}
    assert r["security_flag"] and "blocked from account now" in r["overrides"]


@pytest.mark.parametrize("amount, rating", [(None, 0), (0, 0), (1, 2), (999.99, 2), (1000, 5),
                                            (5000, 5), (5000.01, 9), (50000, 9)])
def test_amount_rating_boundaries(amount, rating):
    assert risk.amount_rating(amount, CFG) == rating


def test_likelihood_factor_ratings():
    lik = score(signals=sig(two_fa=False, sim_change_at=ago(days=14), past_fraud="resolved",
                            identity="partial"))["likelihood"]
    assert {k: f["rating"] for k, f in lik.items()} == {
        "two_fa": 8, "sim_change": 7, "failed_logins": 0, "past_fraud": 5, "identity": 5}
    assert score(signals=sig(sim_change_at=ago(days=15)))["likelihood"]["sim_change"]["rating"] == 0


def test_score_formula_avg_likelihood_times_avg_impact():
    # L = (8 + 7 + 6 + 0 + 0) / 5 = 4.2; I = (amount 2000 -> 5, refund -> 4) / 2 = 4.5
    r = score(type_="refund", amount=2000, signals=sig(
        two_fa=False, sim_change_at=ago(days=1), failed_logins=[ago(hours=h) for h in (1, 2, 3)]))
    assert (r["avg_likelihood"], r["avg_impact"]) == (4.2, 4.5)
    assert r["score"] == round(4.2 * 4.5 / 81 * 100, 1) == 23.3
    assert r["band"] == "medium" and r["overrides"] == []


def test_faq_answerable_scores_exactly_zero():
    r = score(faq=True, type_="faq")
    assert r["score"] == 0 and r["avg_impact"] == 0 and r["likelihood"] == {}
    assert r["band"] == "low"


def test_non_faq_likelihood_floor_keeps_score_above_zero():
    r = score(type_="status")  # spotless account, no amount
    assert r["avg_likelihood"] == 1 and r["score"] == round(1 * 2 / 81 * 100, 1) > 0
    assert any("likelihood floor" in x for x in r["reasons"])


def test_unanswerable_faq_type_gets_impact_floor():
    r = score(type_="faq")  # classified as a question but the FAQ can't answer it
    assert r["avg_impact"] == 1 and r["score"] > 0


def test_override_dispute_over_5000():
    assert score(type_="dispute", amount=5000)["overrides"] == []
    assert score(type_="dispute", amount=5001)["overrides"] == ["dispute/chargeback over 5000"]
    assert score(type_="refund", amount=9000)["overrides"] == []


def test_override_failed_identity_only_on_money_moving_tickets():
    for t in ("refund", "dispute"):
        assert "identity check failed on a money-moving ticket" in score(
            type_=t, signals=sig(identity="no"))["overrides"]
    assert score(type_="status", signals=sig(identity="no"))["overrides"] == []
    assert score(type_="refund", signals=sig(identity="partial"))["overrides"] == []


def test_override_confirmed_fraud_not_resolved():
    assert score(signals=sig(past_fraud="confirmed"))["overrides"] == ["confirmed fraud flag"]
    assert score(signals=sig(past_fraud="resolved"))["overrides"] == []


def test_override_injection_even_on_faq():
    r = score(faq=True, type_="faq", injection=True)
    assert r["score"] == 0 and r["overrides"] == ["input guard: injection flagged"]


def test_override_blocked_now_follows_config_flag():
    cfg = copy.deepcopy(CFG)
    cfg["overrides"]["override_blocked_now"] = False
    r = score(signals=sig(blocked_now=True), cfg=cfg)
    assert r["overrides"] == [] and r["security_flag"]  # still flagged for the reviewer


def test_override_low_confidence_and_unverified_sender():
    assert score(confidence=0.59)["overrides"] == ["classifier confidence 0.59 below 0.6"]
    assert score(confidence=0.6)["overrides"] == []
    assert score(verified=False)["overrides"] == ["sender is not a verified contact"]


def test_bands_from_config():
    assert [risk.band(s, CFG) for s in (0, 9.9, 10, 24.9, 25, 100)] == [
        "low", "low", "medium", "medium", "high", "high"]


def test_breakdown_lists_every_factor_with_reasons():
    r = score(type_="dispute", amount=3200, signals=sig(two_fa=False))
    assert set(r["likelihood"]) == {"two_fa", "sim_change", "failed_logins", "past_fraud",
                                    "identity"}
    assert set(r["impact"]) == {"amount", "type"}
    assert "2FA: off -> 8" in r["reasons"] and "Ticket type: dispute -> 8" in r["reasons"]
