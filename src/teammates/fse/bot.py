"""FSE daily-plan WhatsApp bot: Vonage sandbox, native list messages, Sarvam-written text.

    python -m teammates.fse.bot           # webhook server (POST /webhook, /status) + IST scheduler
    python -m teammates.fse.bot morning   # debug only: run one job now (morning/reviews/evening/
                                          # health/collections_brief/collections_preview/
                                          # kam_digest/kam_preview)
    python -m teammates.fse.bot web       # server without the scheduler (never sends jobs)

Needs DATABASE_URL (Postgres). REDIS_URL is optional: without it every read goes to Postgres.
"""

import json
import logging
import re
import sys
import threading
import time
from datetime import date, datetime
from functools import cache
from typing import Any

import requests
from flask import Flask, request

from teammates.collections import app as collections
from teammates.fse import tracing
from teammates.fse.store import IST, REASONS, Store, ist_today
from teammates.fse.tracing import mask, span
from teammates.kam import app as kam
from teammates.kam.digest import config as kam_config
from teammates.llm.client import ChatRequest, LLMError, Message
from teammates.runtime import MODEL, env, llm, redis_client
from teammates.support import app as support

log = logging.getLogger(__name__)
VONAGE_URL = "https://messages-sandbox.nexmo.com/v1/messages"
REMINDER = "🪪 Carry your Paytm ID card, Aadhaar and driving licence."

Task = dict[str, Any]


def digits(phone: str) -> str:
    return re.sub(r"\D", "", phone)


@cache
def store() -> Store:
    s = Store(env("DATABASE_URL"), cache=redis_client())
    s.migrate()
    tracing.set_sink(s.record_span)
    if phone := digits(env("FSE_PHONE")):
        s.seed_demo(phone)
    return s


# --- LLM ---------------------------------------------------------------------------------


def morning_text(fse: dict[str, Any], tasks: list[Task]) -> str:
    lines = "\n".join(f"{i}. {t['kind']} - {t['merchant']}" for i, t in enumerate(tasks, 1))
    first = tasks[0]
    text = (
        f"Good morning {fse['name']}! {len(tasks)} tasks today in {fse['area']}. "
        f"Start with {first['kind']} at {first['merchant']}."
    )
    with span("llm.greeting", model=MODEL) as sp:
        try:
            text = llm().complete(ChatRequest(model=MODEL, temperature=0.3, messages=[
                Message(role="system", content=(
                    "You write a short, warm WhatsApp good-morning message for a Paytm field "
                    "sales executive. At most 3 sentences, plain text, no list, no markdown. Say "
                    "how many tasks there are and name the first one as the one to start with.")),
                Message(role="user", content=(
                    f"FSE: {fse['name']}, area {fse['area']}. Tasks in priority order:\n{lines}")),
            ])).content.strip()
        except (LLMError, ValueError) as exc:  # ValueError: SARVAM_API_KEY missing
            sp.attrs["fallback"] = "template"  # the FSE still gets the plan, from the template
            sp.fail(f"LLM failed, template used: {tracing.describe(exc)}")
    return f"{text[:800]}\n\n📍 First stop: {where(first)}\n\n{REMINDER}"


def where(t: Task) -> str:
    """Merchant, address and a Maps link (when the merchant has coordinates)."""
    out = f"{t['merchant']}, {t['address']}" if t.get("address") else t["merchant"]
    if t.get("lat") is not None and t.get("lng") is not None:
        out += f" https://maps.google.com/?q={t['lat']},{t['lng']}"
    return out


# --- Vonage ------------------------------------------------------------------------------


def send(to: str, msg: dict[str, Any]) -> bool:
    """Vonage send, traced and recorded in message_log (ok, HTTP status, error, message_uuid)."""
    kind = msg.get("custom", {}).get("interactive", {}).get("type") or msg["message_type"]
    with span("vonage.send", kind=kind) as sp:
        ok, status, error, uuid = deliver(to, msg)
        sp.attrs.update(http_status=status, message_uuid=uuid)
        if not ok:
            sp.fail(f"Vonage rejected {kind}: {error}")
        try:
            store().log_message("out", to, kind, message_text(msg), ok, http_status=status,
                                error=error, message_uuid=uuid)
        except Exception as exc:  # the log must never block a send
            log.warning("message_log write failed: %s", exc)
    return ok


def deliver(to: str, msg: dict[str, Any]) -> tuple[bool, int | None, str | None, str | None]:
    """POST to Vonage -> (ok, http_status, error, message_uuid)."""
    payload = {"from": env("VONAGE_WHATSAPP_FROM", "14157386102"), "to": to,
               "channel": "whatsapp", **msg}
    auth = (env("VONAGE_API_KEY"), env("VONAGE_API_SECRET"))
    try:
        r = requests.post(VONAGE_URL, json=payload, auth=auth, timeout=15)
    except requests.RequestException as exc:
        return False, None, tracing.describe(exc), None
    if not r.ok:
        return False, r.status_code, r.text[:500], None
    try:
        uuid = r.json().get("message_uuid")
    except ValueError:
        uuid = None
    return True, r.status_code, None, uuid


def message_text(msg: dict[str, Any]) -> str:
    if msg["message_type"] == "text":
        return msg["text"]
    inter = msg.get("custom", {}).get("interactive", {})
    rows = [r["title"] for sec in inter.get("action", {}).get("sections", [])
            for r in sec.get("rows", [])]
    return f"{inter.get('body', {}).get('text', '')}\n[{' | '.join(rows)}]"


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
             "description": " · ".join(filter(None, (t["kind"], t.get("address"))))[:72]}
            for t in open_tasks[:10]]  # WhatsApp list limit
    if send_list(to, body, "📋 View tasks", "Tap a task when done", rows):
        return
    numbered = "\n".join(
        f"{'✅' if t['status'] == 'done' else '⬜'} {i}. {t['kind']} - {t['merchant']}"
        for i, t in enumerate(tasks, 1))
    send_text(to, f"{body}\n\n{numbered}\n\nReply with a task number when it's done.")


# --- Jobs and inbound --------------------------------------------------------------------


def for_each_fse(fn: Any) -> None:
    """Run fn(fse) for every FSE in its own span. One FSE failing is recorded (with the root
    cause) and the job carries on with the rest instead of stopping."""
    fses = store().fses()
    if root := tracing.current():
        root.attrs["fses"] = len(fses)
    for fse in fses:
        try:
            with span("fse", fse_id=fse["id"]):
                fn(fse)
        except Exception:
            log.exception("FSE %s failed in %s", fse["id"], root.name if root else "job")


def send_morning() -> None:
    store().ensure_partitions()  # creates next month's partitions well before they're needed

    def one(fse: dict[str, Any]) -> None:
        tasks = store().plan(fse["id"])
        if not tasks:
            send_text(fse["phone"], f"Good morning {fse['name']}! No open projects in "
                                    f"{fse['area']} today.")
            return
        send_checklist(fse["phone"], morning_text(fse, tasks), tasks)

    for_each_fse(one)


def ask_review(to: str, fse_id: int, day: date) -> None:
    """Ask what the customer said for the next done task; thank once all are answered."""
    if r := store().request_review(fse_id, day):
        send_text(to, f"⭐ What did {r['merchant']} say about today's visit? "
                      "Reply with text or a voice note.")
    else:
        send_text(to, "🙏 Thanks, all customer reviews saved.")


def send_reviews() -> None:
    """6:30 pm: collect a customer review for each task done today, one at a time."""
    def one(fse: dict[str, Any]) -> None:
        if any(t["status"] == "done" for t in store().plan(fse["id"])):
            ask_review(fse["phone"], fse["id"], ist_today())

    for_each_fse(one)


def send_evening() -> None:
    """8 pm: close the review window, then ask why for each task not done."""
    def one(fse: dict[str, Any]) -> None:
        store().drop_unanswered_reviews(fse["id"], ist_today())
        if tasks := store().plan(fse["id"]):
            ask_why(fse["phone"], tasks)

    for_each_fse(one)


def send_health() -> None:
    """22:00: did every job run, how late, what failed and why. Alerts ALERT_PHONE if not ok."""
    report = store().health(ist_today(), [name for name, *_ in JOBS if name != "health"])
    if root := tracing.current():
        root.attrs.update(ok=report["ok"], problems=len(report["problems"]))
    log.info("health %s", json.dumps(report, default=str))
    if not report["ok"] and (to := digits(env("ALERT_PHONE"))):
        lines = "\n".join(f"• {p}" for p in report["problems"][:10])
        send_text(to, f"⚠️ FSE bot health {report['day']}: {len(report['problems'])} "
                      f"problem(s)\n{lines}")


def handle_inbound(m: dict[str, Any]) -> None:
    sender = digits(str(m.get("from", "")))
    kind = str(m.get("message_type") or "")
    with span("inbound", kind=kind) as sp:
        fse = store().fse_by_phone(sender)
        body = (m.get("text") or (m.get("reply") or {}).get("id")
                or (m.get("audio") or {}).get("url"))
        store().log_message("in", sender, kind, body, fse is not None,
                            fse_id=fse["id"] if fse else None)
        if fse is None:  # public webhook: only known FSEs get answers
            sp.attrs.update(rejected="unregistered sender", sender=mask(sender))
            log.warning("ignoring message from unknown number %s", mask(sender))
            return
        sp.fse_id = fse["id"]
        sp.attrs["action"] = _handle(m, sender, fse, kind)


def _handle(m: dict[str, Any], sender: str, fse: dict[str, Any], kind: str) -> str:
    """Route one message from a registered FSE; returns which branch handled it (traced)."""
    s, fse_id = store(), fse["id"]
    if kind == "location":  # FSE shared their location: the fse table is updated, cache dropped
        loc = m.get("location") or {}
        try:
            lat, lng = float(loc["lat"]), float(loc.get("long", loc.get("lng")))
            s.set_fse_location(fse_id, lat, lng, loc.get("address") or loc.get("name"))
        except (KeyError, TypeError, ValueError):
            send_text(sender, "⚠️ Couldn't read that location. Please share it again.")
            return "location:unreadable"
        send_text(sender, "📍 Location saved.")
        return "location"
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
        return "done"
    elif action == "why" and code in REASONS and s.set_reason(fse_id, day, task_id, code):
        if code == "other":
            merchant = next((t["merchant"] for t in tasks if t["id"] == task_id), "this task")
            send_text(sender, f"✍️ Type why {merchant} wasn't done.")
        else:
            ask_why(sender, s.plan(fse_id))
        return f"reason:{code}"
    elif kind != "reply" and (text or audio) and pending:
        s.set_reason(fse_id, date.fromisoformat(pending["day"]), pending["id"], "other",
                     text or audio)
        ask_why(sender, s.plan(fse_id))
        return "reason:other:text"
    elif kind != "reply" and (text or audio) and (
            review := s.pending_review(fse_id, ist_today())):
        s.save_review(fse_id, date.fromisoformat(review["day"]), review["id"],
                      text or None, audio)
        ask_review(sender, fse_id, ist_today())
        return "review"
    else:
        s.add_feedback(fse_id, text or reply.get("title"), audio)
        send_text(sender, "📝 Noted, thanks!")
        return "feedback"


app = Flask(__name__)
app.register_blueprint(collections.bp)  # collections manager inbox
app.register_blueprint(support.bp)  # support desk: review queue + intake
app.register_blueprint(kam.bp)  # KAM: today's digest view


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
    try:
        record_status(d)
    except Exception:
        log.exception("status callback failed")
    return "", 200


FAILED_DELIVERY = {"rejected", "undeliverable", "failed"}


def record_status(d: dict[str, Any]) -> None:
    """Vonage delivery callback -> message_log; a failed delivery becomes an error span in the
    trace that sent the message, so it shows up in v_root_causes with Vonage's reason."""
    status, uuid = str(d.get("status") or ""), str(d.get("message_uuid") or "")
    err = d.get("error") or {}
    error = (" - ".join(str(err.get(k)) for k in ("title", "detail") if err.get(k))
             if isinstance(err, dict) else str(err)) or None
    row = store().record_delivery(uuid, status, error) if uuid else None
    log.info("status %s %s %s", uuid, status, error or "")
    if status in FAILED_DELIVERY:
        with span("vonage.delivery", fse_id=row["fse_id"] if row else None,
                  trace_id=row["trace_id"] if row else None,
                  status=status, message_uuid=uuid, kind=row["kind"] if row else None) as sp:
            sp.fail(f"WhatsApp {status}: {error or 'no reason given'}")


# (name, window start, window end) in IST, or in JOB_TZ[name]. A job runs once per day, at
# the first tick inside its window: on time when the bot is awake, or as soon as it
# wakes/restarts (catch-up).
JOBS = [
    ("morning", "08:30", "12:00"),
    ("reviews", "18:30", "20:00"),
    ("evening", "20:00", "22:00"),
    ("collections_brief", "09:00", "12:00"),  # collections manager's portfolio email
    ("kam_digest", kam_config().time, kam_config().window_end),  # KAM's key-account email
    ("health", "22:00", "23:59"),  # did the day run properly; alerts ALERT_PHONE if not
]
JOB_TZ = {"kam_digest": kam_config().tz}  # jobs scheduled in their own configured timezone
_claimed: set[tuple[str, date]] = set()  # saves a DB round trip per tick once a job is taken


def job_fn(name: str) -> Any:
    return {"morning": send_morning, "reviews": send_reviews, "evening": send_evening,
            "collections_brief": collections.send_brief,
            "collections_preview": collections.preview_brief, "kam_digest": kam.send_digest,
            "kam_preview": kam.preview_digest, "health": send_health}[name]


def run_due_jobs(now: datetime) -> None:
    """Run every job whose window contains `now` and that nobody has run today.
    The claim is a row in job_run, so restarts, deploys and extra instances never double-send."""
    for name, start, end in JOBS:
        local = now.astimezone(JOB_TZ.get(name, IST))
        hhmm, day = local.strftime("%H:%M"), local.date()
        if not start <= hhmm < end or (name, day) in _claimed:
            continue
        if store().claim_job(name, day):
            log.info("job %s started", name)
            late = (local.hour * 60 + local.minute) - (int(start[:2]) * 60 + int(start[3:]))
            # ponytail: claimed before running, so a crash mid-job isn't retried that day;
            # per-FSE send tracking would make it resumable.
            try:
                with span(f"job.{name}", minutes_late=late, window=f"{start}-{end}"):
                    job_fn(name)()
            except Exception:
                log.exception("job %s failed", name)
        _claimed.add((name, day))


def scheduler() -> None:
    while True:
        try:
            run_due_jobs(datetime.now(IST))
        except Exception:  # e.g. DB unreachable: try again next tick
            log.exception("scheduler tick failed")
        time.sleep(20)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    store()  # fail fast: migrate + seed before taking traffic
    collections.loan_store()
    if sys.argv[1:] == ["web"]:  # dashboard + webhooks only, no scheduler: safe any time
        app.run(host="0.0.0.0", port=int(env("PORT", "5000")))
        return
    if args := sys.argv[1:]:  # manual override for debugging only; production runs on its own
        try:
            job_fn(args[0])()
        finally:  # unclosed pools make interpreter exit wait 5 s per pool thread
            store().pool.close()
            collections.loan_store().pool.close()
        return
    threading.Thread(target=scheduler, daemon=True).start()
    app.run(host="0.0.0.0", port=int(env("PORT", "5000")))


if __name__ == "__main__":
    main()
