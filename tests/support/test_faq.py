import re

import pytest

from teammates.support import risk
from teammates.support.faq import FAQS, TRANSLATIONS, TfidfRetriever, answer, faq_match

CFG = risk.config()["faq"]
R = TfidfRetriever()


@pytest.mark.parametrize("q, faq_id", [
    ("how long does settlement take", "settle-time"),
    ("change bank account for settlement", "settle-bank"),
    ("how do I refund a customer", "refund-how"),
    ("customer money debited but payment failed", "failed-debit"),
    ("documents needed for KYC", "kyc-docs"),
    ("soundbox is silent", "soundbox-silent"),
    ("login on my new phone", "device-new"),
])
def test_retrieves_the_right_faq(q, faq_id):
    (faq, conf), *_ = R.search(q)
    assert faq["id"] == faq_id and conf >= CFG["min_confidence"]


def test_answer_only_when_generic_and_confident():
    m = faq_match(R, "How long does settlement take?", generic=True, cfg=CFG)
    assert m["answerable"] and m["faq"]["id"] == "settle-time"
    # Same words, but the question is about this merchant's own money: not an FAQ answer.
    m = faq_match(R, "How long does settlement take?", generic=False, cfg=CFG)
    assert not m["answerable"] and m["faq"]["id"] == "settle-time"  # still usable as context


def test_low_confidence_is_not_answerable_and_not_context():
    m = faq_match(R, "my cousin's wedding photos", generic=True, cfg=CFG)
    assert not m["answerable"] and m["faq"] is None and m["confidence"] < CFG["context_min"]


def test_empty_query():
    assert not faq_match(R, "???", generic=True, cfg=CFG)["answerable"]


def test_any_retriever_behind_the_interface():
    class Fixed:
        def search(self, query, k=3):
            return [(FAQS[0], 0.99)]

    assert faq_match(Fixed(), "anything", generic=True, cfg=CFG)["faq"] is FAQS[0]
    assert faq_match(type("Empty", (), {"search": lambda s, q, k=3: []})(), "x", True,
                     CFG)["answerable"] is False


def test_faq_set_covers_required_topics():
    assert {f["topic"] for f in FAQS} == {"settlements", "refunds", "failed payments", "KYC",
                                          "devices"}


def test_every_faq_is_translated_and_keeps_the_english_numbers():
    for f in FAQS:
        assert set(TRANSLATIONS[f["id"]]) == {"hi", "ta", "kn"}, f["id"]
        for lang, text in TRANSLATIONS[f["id"]].items():
            digits = sorted(re.findall(r"\d+", text))
            assert digits == sorted(re.findall(r"\d+", f["a"])), (f["id"], lang)
            assert answer(f, lang) == text


def test_answer_falls_back_to_english():
    assert answer(FAQS[0], "en") == answer(FAQS[0], "mr") == FAQS[0]["a"]
