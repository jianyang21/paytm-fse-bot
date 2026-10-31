"""FSE daily-plan WhatsApp bot: Vonage sandbox, native list messages, Sarvam-written text.

    python -m teammates.fse.bot           # webhook server (POST /webhook, /status) + IST scheduler
    python -m teammates.fse.bot morning   # send today's plan right now (or: evening)
"""

import logging
import os
import re
import sqlite3
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from functools import cache
from pathlib import Path
from typing import Any

import redis
import requests
from flask import Flask, request

from teammates.config import load_settings, read_env_file
from teammates.llm.adapters.sarvam import SarvamClient
from teammates.llm.client import ChatRequest, LLMClient, LLMError, Message
from teammates.llm.rate_limit import RedisSlidingWindowRateLimiter

log = logging.getLogger(__name__)
_FILE_ENV = read_env_file(Path(".env"))
IST = timezone(timedelta(hours=5, minutes=30))  # servers run on UTC
VONAGE_URL = "https://messages-sandbox.nexmo.com/v1/messages"
MODEL = "sarvam-105b-conversations"  # same as orchestrator planner
REMINDER = "🪪 Carry your Paytm ID card, Aadhaar and driving licence."
REASONS = {  # code -> list row title (WhatsApp max 24 chars)
    "unavailable": "Merchant not available",
    "closed": "Shop closed",
    "stock": "Device/stock unavailable",
    "not_interested": "Merchant not interested",
    "no_time": "Ran out of time",
    "rescheduled": "Rescheduled",
    "other": "Other (type it)",
}

# ponytail: dummy tasks seeded per day; swap for the real merchant/task feed.
DAILY_TASKS = [  # (merchant, kind, priority 1=highest, effort 1-3)
    ("Sri Medicals", "QR replacement", 2, 1),
    ("Ravi Stores", "Multi-Soundbox install x3", 1, 3),
    ("Green Leaf Cafe", "Soundbox service visit", 3, 1),
    ("Lakshmi Bakery", "New merchant onboarding", 1, 2),
    ("Anand Kirana", "KYC re-verification", 2, 2),
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS fse(id INTEGER PRIMARY KEY, name TEXT, phone TEXT UNIQUE, area TEXT);
CREATE TABLE IF NOT EXISTS task(id INTEGER PRIMARY KEY, fse_id INT, day TEXT, merchant TEXT,
    kind TEXT, priority INT, effort INT, done INT DEFAULT 0, reason TEXT,
    UNIQUE(fse_id, day, merchant));
CREATE TABLE IF NOT EXISTS feedback(id INTEGER PRIMARY KEY, fse_id INT, at TEXT, body TEXT);
"""


def env(key: str, default: str = "") -> str:
    return os.environ.get(key) or _FILE_ENV.get(key, default)


def digits(phone: str) -> str:
    return re.sub(r"\D", "", phone)


@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    con = sqlite3.connect(env("DB_PATH", "fse.db"))
    con.row_factory = sqlite3.Row
    try:
        con.executescript(SCHEMA)
        if phone := digits(env("FSE_PHONE")):
            con.execute(
                "INSERT OR IGNORE INTO fse(name, phone, area) VALUES (?, ?, ?)",
                ("Anbuselvan", phone, "Bellandur"),
            )
        yield con
        con.commit()
    finally:
        con.close()


def today_tasks(con: sqlite3.Connection, fse_id: int) -> list[sqlite3.Row]:
    """Today's tasks, hardest/highest priority first. Ranking is code, never the LLM."""
    day = datetime.now(IST).date().isoformat()
    con.executemany(
        "INSERT OR IGNORE INTO task(fse_id, day, merchant, kind, priority, effort)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        [(fse_id, day, *t) for t in DAILY_TASKS],
    )
    return con.execute(
        "SELECT * FROM task WHERE fse_id = ? AND day = ? ORDER BY priority, effort DESC, id",
        (fse_id, day),
    ).fetchall()


# --- LLM ---------------------------------------------------------------------------------


class _Unlimited:
    def acquire(self) -> float:
        return 0.0


@cache
def llm() -> LLMClient:
    s = load_settings()
    try:
        r = redis.Redis.from_url(s.redis_url.get_secret_value(), socket_connect_timeout=2)
        r.ping()
        limiter = RedisSlidingWindowRateLimiter(r)
    except redis.RedisError:
        # ponytail: one call per FSE per morning; fine unlimited until FSE count grows.
        log.warning("Redis unreachable; Sarvam calls are not rate limited")
        limiter = _Unlimited()
    return SarvamClient(s.sarvam_api_key, limiter, base_url=s.sarvam_base_url)


def morning_text(fse: sqlite3.Row, tasks: list[sqlite3.Row]) -> str:
    lines = "\n".join(f"{i}. {t['kind']} - {t['merchant']}" for i, t in enumerate(tasks, 1))
    first = tasks[0]
    text = (
        f"Good morning {fse['name']}! {len(tasks)} tasks today in {fse['area']}. "
        f"Start with {first['kind']} at {first['merchant']}."
    )
    try:
        text = llm().complete(ChatRequest(model=MODEL, temperature=0.3, messages=[
            Message(role="system", content=(
                "You write a short, warm WhatsApp good-morning message for a Paytm field sales "
                "executive. At most 3 sentences, plain text, no list, no markdown. Say how many "
                "tasks there are and name the first one as the one to start with.")),
            Message(role="user", content=(
                f"FSE: {fse['name']}, area {fse['area']}. Tasks in priority order:\n{lines}")),
        ])).content.strip()
    except (LLMError, ValueError) as exc:  # ValueError: SARVAM_API_KEY missing
        log.warning("LLM failed, using template: %s", exc)
    return f"{text[:800]}\n\n{REMINDER}"


# --- Vonage ------------------------------------------------------------------------------


def send(to: str, msg: dict[str, Any]) -> bool:
    payload = {"from": env("VONAGE_WHATSAPP_FROM", "14157386102"), "to": to,
               "channel": "whatsapp", **msg}
    auth = (env("VONAGE_API_KEY"), env("VONAGE_API_SECRET"))
    try:
        r = requests.post(VONAGE_URL, json=payload, auth=auth, timeout=15)
    except requests.RequestException as exc:
        log.error("vonage send failed: %s", exc)
        return False
    if not r.ok:
        log.error("vonage %s HTTP %s: %s", msg["message_type"], r.status_code, r.text[:300])
    return r.ok


def send_text(to: str, text: str) -> bool:
    return send(to, {"message_type": "text", "text": text[:4096]})


def send_list(to: str, body: str, button: str, section: str, rows: list[dict[str, str]]) -> bool:
    interactive = {"type": "list", "body": {"text": body[:1024]},
                   "action": {"button": button, "sections": [{"title": section, "rows": rows}]}}
    return send(to, {"message_type": "custom",
                     "custom": {"type": "interactive", "interactive": interactive}})


def ask_why(to: str, tasks: list[sqlite3.Row]) -> None:
    """Ask the reason for the next unfinished task; once none are left, ask for feedback."""
    t = next((t for t in tasks if not t["done"] and not t["reason"]), None)
    if t is None:
        send_text(to, "🙏 Thanks! Any merchant feedback from today? Text or voice note works.")
        return
    rows = [{"id": f"why:{t['id']}:{code}", "title": title} for code, title in REASONS.items()]
    # ponytail: no text fallback here; the sandbox renders lists (verified on phone).
    send_list(to, f"❓ {t['kind']} at {t['merchant']} wasn't done today. Why?",
              "Pick reason", "Reason", rows)


def send_checklist(to: str, body: str, tasks: list[sqlite3.Row]) -> None:
    """Native list of open tasks (tap = done); numbered text list if Vonage rejects it."""
    open_tasks = [t for t in tasks if not t["done"]]
    if not open_tasks:
        send_text(to, body)
        return
    rows = [{"id": f"done:{t['id']}", "title": t["merchant"][:24], "description": t["kind"][:72]}
            for t in open_tasks[:10]]  # WhatsApp list limit
    if send_list(to, body, "📋 View tasks", "Tap a task when done", rows):
        return
    numbered = "\n".join(f"{'✅' if t['done'] else '⬜'} {i}. {t['kind']} - {t['merchant']}"
                         for i, t in enumerate(tasks, 1))
    send_text(to, f"{body}\n\n{numbered}\n\nReply with a task number when it's done.")


# --- Jobs and inbound --------------------------------------------------------------------


def send_morning() -> None:
    with db() as con:
        plans = [(f, today_tasks(con, f["id"])) for f in con.execute("SELECT * FROM fse")]
    for fse, tasks in plans:
        send_checklist(fse["phone"], morning_text(fse, tasks), tasks)


def send_evening() -> None:
    with db() as con:
        plans = [(f, today_tasks(con, f["id"])) for f in con.execute("SELECT * FROM fse")]
    for fse, tasks in plans:
        ask_why(fse["phone"], tasks)


def handle_inbound(m: dict[str, Any]) -> None:
    sender = digits(str(m.get("from", "")))
    kind = m.get("message_type")
    reply = m.get("reply") or {}
    text = str(m.get("text") or "").strip()
    body = text or (m.get("audio") or {}).get("url") or reply.get("title")
    action, _, rest = str(reply.get("id", "")).partition(":")
    task_id, _, code = rest.partition(":")
    with db() as con:
        fse = con.execute("SELECT * FROM fse WHERE phone = ?", (sender,)).fetchone()
        if fse is None:  # public webhook: only known FSEs get answers
            log.warning("ignoring message from unknown number")
            return
        tasks = today_tasks(con, fse["id"])
        task = con.execute("SELECT * FROM task WHERE id = ? AND fse_id = ?",
                           (task_id, fse["id"])).fetchone()
        n = re.fullmatch(r"(?:done\s*)?(\d+)", text, re.I) if kind == "text" else None
        if n and 1 <= int(n[1]) <= len(tasks):
            action, task = "done", tasks[int(n[1]) - 1]
        pending = next((t for t in tasks if t["reason"] == "other"), None)

        if action == "done" and task:
            con.execute("UPDATE task SET done = 1 WHERE id = ?", (task["id"],))
            outcome = "done"
        elif action == "why" and task and code in REASONS:
            con.execute("UPDATE task SET reason = ? WHERE id = ?", (code, task["id"]))
            outcome = "other" if code == "other" else "why"
        elif kind != "reply" and body and pending:
            con.execute("UPDATE task SET reason = ? WHERE id = ?",
                        (f"other: {body}", pending["id"]))
            outcome = "why"
        else:
            if body:
                con.execute("INSERT INTO feedback(fse_id, at, body) VALUES (?, ?, ?)",
                            (fse["id"], datetime.now(IST).isoformat(timespec="seconds"), body))
            outcome = "feedback"
        tasks = today_tasks(con, fse["id"])

    if outcome == "feedback":
        send_text(sender, "📝 Noted, thanks!")
    elif outcome == "other":
        send_text(sender, f"✍️ Type why {task['merchant']} wasn't done.")
    elif outcome == "why":
        ask_why(sender, tasks)
    elif left := sum(not t["done"] for t in tasks):
        send_checklist(sender, f"✅ {task['merchant']} done. {left} left.", tasks)
    else:
        send_text(sender, f"🎉 All {len(tasks)} tasks done today. Great work!")


app = Flask(__name__)


@app.get("/")
def health() -> str:
    return "ok"


@app.post("/webhook")
def webhook() -> tuple[str, int]:
    try:
        handle_inbound(request.get_json(silent=True) or {})
    except Exception:  # always 200: a Vonage retry would replay the message
        log.exception("inbound failed")
    return "", 200


@app.post("/status")
def status() -> tuple[str, int]:
    d = request.get_json(silent=True) or {}
    log.info("status %s %s %s", d.get("message_uuid"), d.get("status"), d.get("error") or "")
    return "", 200


JOBS = {"08:30": send_morning, "18:30": send_evening}


def scheduler() -> None:
    last = ""
    while True:
        now = datetime.now(IST).strftime("%H:%M")
        if now != last and now in JOBS:
            try:
                JOBS[now]()
            except Exception:
                log.exception("job %s failed", now)
        last = now
        time.sleep(20)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if job := {"morning": send_morning, "evening": send_evening}.get(" ".join(sys.argv[1:])):
        job()
        return
    threading.Thread(target=scheduler, daemon=True).start()
    app.run(host="0.0.0.0", port=int(env("PORT", "5000")))


if __name__ == "__main__":
    main()
