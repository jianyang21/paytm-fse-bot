"""Intake: email, WhatsApp and call transcripts -> one ticket schema.

Every field is customer-controlled and untrusted; the resolver scans `text` with InputGuard
before any LLM sees it. merchant_id is filled by the resolver from the contact (directory
tool), never from anything the customer wrote.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from teammates.guard import EMAIL_RE

CHANNELS = ("email", "whatsapp", "call")
AMOUNT_RE = re.compile(
    r"(?:₹|\brs\.?|\binr)\s*(\d[\d,]*(?:\.\d{1,2})?)|(\d[\d,]*(?:\.\d{1,2})?)\s*(?:rupees|/-)",
    re.I)


@dataclass
class Ticket:
    channel: str            # email | whatsapp | call
    contact: str            # email address or phone digits: where a reply would go
    text: str
    amount: float | None    # first rupee amount in the text, if any
    received_at: datetime
    merchant_id: int | None = None


def amount_in(text: str) -> float | None:
    m = AMOUNT_RE.search(text)
    return float((m.group(1) or m.group(2)).replace(",", "")) if m else None


def _digits(s: Any) -> str:
    return "".join(c for c in str(s) if c.isdigit())


def _need(p: dict[str, Any], *keys: str) -> None:
    if not isinstance(p, dict) or any(not isinstance(p.get(k), str | list) for k in keys):
        raise ValueError(f"payload needs {', '.join(keys)}")


def normalize(channel: str, p: dict[str, Any]) -> Ticket:
    """Raises ValueError on an unknown channel or a payload missing its fields."""
    if channel == "email":
        _need(p, "from", "body", "date")
        m = EMAIL_RE.search(p["from"])
        if not m:
            raise ValueError("email sender has no address")
        # Quoted earlier messages (">" lines) are not part of this contact.
        body = "\n".join(ln for ln in p["body"].splitlines() if not ln.lstrip().startswith(">"))
        text, contact, at = f"{p.get('subject', '')}\n{body}".strip(), m.group().lower(), p["date"]
    elif channel == "whatsapp":
        _need(p, "from", "text", "timestamp")
        text, contact, at = p["text"], _digits(p["from"]), p["timestamp"]
    elif channel == "call":
        _need(p, "caller", "started_at", "turns")
        # Only the customer's words: the agent's lines are ours and need no answer.
        text = "\n".join(str(t.get("text", "")) for t in p["turns"]
                         if isinstance(t, dict) and t.get("speaker") == "customer")
        contact, at = _digits(p["caller"]), p["started_at"]
    else:
        raise ValueError(f"unknown channel {channel!r}; expected one of {CHANNELS}")
    if not text.strip() or not contact:
        raise ValueError("empty message or contact")
    return Ticket(channel, contact, text, amount_in(text), datetime.fromisoformat(at))
