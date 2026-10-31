from datetime import datetime

import pytest

from teammates.support.intake import amount_in, normalize


def test_email():
    t = normalize("email", {"from": "Ravi <RaviStores@Example.com>", "subject": "Refund",
                            "body": "Refund ₹1,200 please\n> old quoted text ₹99,999\nthanks",
                            "date": "2026-10-30T10:00:00+05:30"})
    assert (t.channel, t.contact, t.amount) == ("email", "ravistores@example.com", 1200)
    assert "99,999" not in t.text and t.text.startswith("Refund\n")
    assert t.received_at == datetime.fromisoformat("2026-10-30T10:00:00+05:30")


def test_whatsapp():
    t = normalize("whatsapp", {"from": "+91 99000-00101", "text": "Rs 8500 refund",
                               "timestamp": "2026-10-30T10:00:00+05:30"})
    assert (t.contact, t.amount, t.merchant_id) == ("919900000101", 8500, None)


def test_call_keeps_only_customer_turns():
    t = normalize("call", {"caller": "919900000109", "started_at": "2026-10-30T10:00:00+05:30",
                           "turns": [{"speaker": "agent", "text": "Refund of ₹5,000 done?"},
                                     {"speaker": "customer", "text": "my 450 rupees refund"},
                                     "garbage"]})
    assert t.text == "my 450 rupees refund" and t.amount == 450


@pytest.mark.parametrize("text, amount", [
    ("₹1,200", 1200), ("Rs. 8,500.50 today", 8500.5), ("INR 999", 999), ("450 rupees", 450),
    ("500/-", 500), ("order 5521, no amount", None), ("rs5", 5), ("verse 10", None)])
def test_amount_in(text, amount):
    assert amount_in(text) == amount


@pytest.mark.parametrize("channel, payload", [
    ("fax", {"from": "x"}),
    ("email", {"from": "a@b.com", "body": "hi"}),             # no date
    ("email", {"from": "no address", "body": "hi", "date": "2026-10-30T10:00:00"}),
    ("whatsapp", {"from": "9199", "text": "   ", "timestamp": "2026-10-30T10:00:00"}),
    ("whatsapp", {"from": "abc", "text": "hi", "timestamp": "2026-10-30T10:00:00"}),
    ("whatsapp", {"from": "9199", "text": "hi", "timestamp": "yesterday"}),
    ("call", {"caller": "9199", "started_at": "2026-10-30T10:00:00", "turns": "x"}),
    ("email", ["not", "a", "dict"]),
])
def test_bad_payloads_raise_value_error(channel, payload):
    with pytest.raises(ValueError):
        normalize(channel, payload)
