"""Shared guards (T9) for untrusted text going into an LLM and LLM text going out.

In:  InputGuard.scan normalises and bounds the text and flags prompt injection, SQL payloads,
     links and personal identifiers; wrap_untrusted fences it so the prompt marks it as data.
     The scan only flags: SQL is always parameterised and the LLM never writes SQL, so a
     payload is harmless to the database, but the flag sends the ticket to a human.
Out: text_problems rejects LLM text with PII, links or numbers that are not in its facts.
Logs: mask_pii before anything customer-written reaches a log or a trace.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
URL_RE = re.compile(r"https?://|www\.", re.I)
LONG_DIGITS_RE = re.compile(  # phone / account no. / Aadhaar (4-4-4) / phone (5-5)
    r"\d{10,}|\b\d{4}[\s-]\d{4}[\s-]\d{4}\b|\b\d{5}[\s-]\d{5}\b")
PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
NUMBER_RE = re.compile(r"\d[\d,]*")
INJECTION_RE = re.compile(
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|rules|instructions)"
    r"|disregard\s+(the\s+|all\s+)?(rules|instructions)|system\s+prompt|you\s+are\s+now"
    r"|pretend\s+(to\s+be|you)|\b(system|assistant)\s*:|<<<|>>>|```|</?\s*untrusted", re.I)
SQL_RE = re.compile(
    r"['\"`]\s*(or|and)\s+['\"]?\w+['\"]?\s*=\s*['\"]?\w+|;\s*(drop|delete|insert|update|alter"
    r"|truncate|create|grant)\b|\bunion\s+(all\s+)?select\b|\bselect\b.+\bfrom\b\s+\w+\s*(--|;)"
    r"|--\s*$|/\*.*\*/|\b(xp_cmdshell|pg_sleep|sleep\s*\(|information_schema)\b", re.I | re.M)
UNTRUSTED_TAG_RE = re.compile(r"</?\s*untrusted[^>]*>", re.I)
MAX_LEN = 4000


@dataclass
class Scan:
    text: str          # normalised and bounded; what the LLM may see (inside wrap_untrusted)
    flags: list[str]   # prompt_injection, sql_injection, link, pii, truncated

    @property
    def injection(self) -> bool:
        return "prompt_injection" in self.flags or "sql_injection" in self.flags


class InputGuard:
    @staticmethod
    def scan(text: str, max_len: int = MAX_LEN) -> Scan:
        clean = " ".join("".join(c for c in text if c.isprintable() or c.isspace()).split())
        flags = []
        if len(clean) > max_len:
            clean, flags = clean[:max_len], ["truncated"]
        if INJECTION_RE.search(clean):
            flags.append("prompt_injection")
        if SQL_RE.search(clean):
            flags.append("sql_injection")
        if URL_RE.search(clean):
            flags.append("link")
        if EMAIL_RE.search(clean) or LONG_DIGITS_RE.search(clean) or PAN_RE.search(clean):
            flags.append("pii")
        return Scan(clean, flags)


def wrap_untrusted(text: str, source: str) -> str:
    """Fence customer text as data. Tags inside the text are removed so it can't close the
    fence and continue as instructions. source is ours (the channel), never customer text."""
    return f'<untrusted source="{source}">\n{UNTRUSTED_TAG_RE.sub("", text)}\n</untrusted>'


def mask_pii(text: str) -> str:
    """Phones, account numbers, Aadhaar, PAN and emails -> [masked], for logs and traces."""
    for rx in (EMAIL_RE, LONG_DIGITS_RE, PAN_RE):
        text = rx.sub("[masked]", text)
    return text


def mask_email(email: str) -> str:
    user, _, domain = email.partition("@")
    return f"{user[:2]}***@{domain}"


def recipient_problems(email: str, *, verified: bool, role: str, want_role: str,
                       scope: list[Any], what: str, scope_name: str) -> list[str]:
    """Output guard for scheduled emails: a verified recipient of the right role, a valid
    address, and something assigned to them (the collections brief, the KAM digest)."""
    problems = []
    if not verified:
        problems.append("recipient email not verified")
    if role != want_role:
        problems.append(f"role {role} does not receive the {what}")
    if not EMAIL_RE.fullmatch(email):
        problems.append("recipient email invalid")
    if not scope:
        problems.append(f"no {scope_name} assigned")
    return problems


def scope_problems(values: Iterable[Any], allowed: Iterable[Any], what: str) -> list[str]:
    """Output guard: every row of an email belongs to the recipient's scope."""
    outside = sorted({str(v) for v in values} - {str(a) for a in allowed})
    return [f"data outside the {what}: {', '.join(outside)}"] if outside else []


def text_problems(text: str, facts: str) -> list[str]:
    """Output guard for LLM text: no PII or links, and no number that isn't in the facts it
    was given (small counts up to 10 are allowed)."""
    problems = []
    if LONG_DIGITS_RE.search(text):
        problems.append("phone/account-like number")
    if EMAIL_RE.search(text):
        problems.append("email address")
    if URL_RE.search(text):
        problems.append("link")
    known = {n.replace(",", "") for n in NUMBER_RE.findall(facts)} | {str(i) for i in range(11)}
    if invented := sorted({n.replace(",", "") for n in NUMBER_RE.findall(text)} - known):
        problems.append("numbers not in the data: " + ", ".join(invented[:5]))
    return problems
