"""FSE daily-plan WhatsApp bot: Vonage sandbox, native list messages, Sarvam-written text.

    python -m teammates.fse.bot           # webhook server (POST /webhook, /status) + IST scheduler
    python -m teammates.fse.bot morning   # send today's plan right now (or: evening)

Needs DATABASE_URL (Postgres). REDIS_URL is optional: without it every read goes to Postgres.
"""

import logging
import os
import re
import sys
import threading
import time
from datetime import date, datetime
from functools import cache
from pathlib import Path
from typing import Any

import redis
import requests
from flask import Flask, request

from teammates.config import load_settings, read_env_file
from teammates.fse.store import IST, REASONS, Store, ist_today
from teammates.llm.adapters.sarvam import SarvamClient
from teammates.llm.client import ChatRequest, LLMClient, LLMError, Message
from teammates.llm.rate_limit import RedisSlidingWindowRateLimiter

log = logging.getLogger(__name__)
_FILE_ENV = read_env_file(Path(".env"))
VONAGE_URL = "https://messages-sandbox.nexmo.com/v1/messages"
MODEL = "sarvam-105b-conversations"  # same as orchestrator planner
REMINDER = "🪪 Carry your Paytm ID card, Aadhaar and driving licence."

Task = dict[str, Any]


def env(key: str, default: str = "") -> str:
    return os.environ.get(key) or _FILE_ENV.get(key, default)


def digits(phone: str) -> str:
    return re.sub(r"\D", "", phone)


@cache
def redis_client() -> redis.Redis | None:
    url = env("REDIS_URL")
    if not url:
        return None
    return redis.Redis.from_url(url, socket_timeout=2, socket_connect_timeout=2,
                                health_check_interval=30)


@cache
def store() -> Store:
    s = Store(env("DATABASE_URL"), cache=redis_client())
    s.migrate()
    if phone := digits(env("FSE_PHONE")):
        s.seed_demo(phone)
    return s


# --- LLM ---------------------------------------------------------------------------------


class _Unlimited:
    def acquire(self) -> float:
        return 0.0


@cache
def llm() -> LLMClient:
    s = load_settings()
    r = redis_client()
    try:
        if r is None:
            raise redis.ConnectionError("REDIS_URL not set")
        r.ping()
        limiter = RedisSlidingWindowRateLimiter(r)
    except redis.RedisError:
        # ponytail: one call per FSE per morning; fine unlimited until FSE count grows.
        log.warning("Redis unreachable; Sarvam calls are not rate limited")
        limiter = _Unlimited()
    return SarvamClient(s.sarvam_api_key, limiter, base_url=s.sarvam_base_url)


def morning_text(fse: dict[str, Any], tasks: list[Task]) -> str:
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


def ask_why(to: str, tasks: list[Task]) -> None:
    """Ask the reason for the next unfinished task; once none are left, ask for feedback."""
    t = next((t for t in tasks if t["status"] == "open"), None)
    if t is None:
        send_text(to, "🙏 Thanks! Any merchant feedback from today? Text or voice note works.")
        return
    rows = [{"id": f"why:{t['day']}:{t['id']}:{code}", "title": title}
            for code, title in REASONS.items()]
    # ponytail: no text fallback here; the sandbox renders lists (verified on phone).
    send_list(to, f"❓ {t['kind']} at {t['merchant']} wasn't done today. Why?",
              "Pick reason", "Reason", rows)


def send_checklist(to: str, body: str, tasks: list[Task]) -> None:
    """Native list of open tasks (tap = done); numbered text list if Vonage rejects it."""
    open_tasks = [t for t in tasks if t["status"] != "done"]
    if not open_tasks:
        send_text(to, body)
        return
    rows = [{"id": f"done:{t['day']}:{t['id']}", "title": t["merchant"][:24],
             "description": t["kind"][:72]} for t in open_tasks[:10]]  # WhatsApp list limit
    if send_list(to, body, "📋 View tasks", "Tap a task when done", rows):
        return
    numbered = "\n".join(
        f"{'✅' if t['status'] == 'done' else '⬜'} {i}. {t['kind']} - {t['merchant']}"
        for i, t in enumerate(tasks, 1))
    send_text(to, f"{body}\n\n{numbered}\n\nReply with a task number when it's done.")


# --- Jobs and inbound --------------------------------------------------------------------


def send_morning() -> None:
    store().ensure_partitions()  # creates next month's partitions well before they're needed
    for fse in store().fses():
        tasks = store().plan(fse["id"])
        send_checklist(fse["phone"], morning_text(fse, tasks), tasks)


def send_evening() -> None:
    for fse in store().fses():
        ask_why(fse["phone"], store().plan(fse["id"]))


def handle_inbound(m: dict[str, Any]) -> None:
    sender = digits(str(m.get("from", "")))
    fse = store().fse_by_phone(sender)
    if fse is None:  # public webhook: only known FSEs get answers
        log.warning("ignoring message from unknown number")
        return
    s, fse_id = store(), fse["id"]
    kind = m.get("message_type")
    reply = m.get("reply") or {}
    text = str(m.get("text") or "").strip()
    audio = (m.get("audio") or {}).get("url")
    # Reply ids: done:<day>:<task> / why:<day>:<task>:<code>; the day picks the partition.
    action, day_s, task_s, code = ([*str(reply.get("id", "")).split(":"), "", "", ""])[:4]
    try:
        day, task_id = date.fromisoformat(day_s), int(task_s)
    except ValueError:
        action, day, task_id = "", ist_today(), 0
    tasks = s.plan(fse_id)
    n = re.fullmatch(r"(?:done\s*)?(\d+)", text, re.I) if kind == "text" else None
    if n and 1 <= int(n[1]) <= len(tasks):
        action, day, task_id = "done", ist_today(), tasks[int(n[1]) - 1]["id"]
    pending = next((t for t in tasks
                    if t["reason_code"] == "other" and t["reason_text"] is None), None)

    if action == "done" and s.mark_done(fse_id, day, task_id):
        tasks = s.plan(fse_id)
        merchant = next((t["merchant"] for t in tasks if t["id"] == task_id), "Task")
        if left := sum(t["status"] != "done" for t in tasks):
            send_checklist(sender, f"✅ {merchant} done. {left} left.", tasks)
        else:
            send_text(sender, f"🎉 All {len(tasks)} tasks done today. Great work!")
    elif action == "why" and code in REASONS and s.set_reason(fse_id, day, task_id, code):
        if code == "other":
            merchant = next((t["merchant"] for t in tasks if t["id"] == task_id), "this task")
            send_text(sender, f"✍️ Type why {merchant} wasn't done.")
        else:
            ask_why(sender, s.plan(fse_id))
    elif kind != "reply" and (text or audio) and pending:
        s.set_reason(fse_id, date.fromisoformat(pending["day"]), pending["id"], "other",
                     text or audio)
        ask_why(sender, s.plan(fse_id))
    else:
        s.add_feedback(fse_id, text or reply.get("title"), audio)
        send_text(sender, "📝 Noted, thanks!")


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
    store()  # fail fast: migrate + seed before taking traffic
    if job := {"morning": send_morning, "evening": send_evening}.get(" ".join(sys.argv[1:])):
        job()
        return
    threading.Thread(target=scheduler, daemon=True).start()
    app.run(host="0.0.0.0", port=int(env("PORT", "5000")))


if __name__ == "__main__":
    main()
