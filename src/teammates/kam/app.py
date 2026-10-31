"""KAM digest inside the bot process: same Postgres store, Redis cache, LLM client, scheduler,
tracing and message_log as the support desk and collections teammates.

Job "kam_digest" (config.toml [schedule], default 09:00 Asia/Kolkata): KamDigest.send_all.
`python -m teammates.fse.bot kam_preview` writes the drafts to reports/ without sending.

Delivery: KAM_DELIVERY=outbox (default) writes each email to reports/outbox/ and logs it in
message_log (the mocked sender of this build); KAM_DELIVERY=resend sends it through Resend,
like the collections brief. The recipient is always the KAM's login (SUPPORT_KAM_EMAIL).

GET /kam/digest[?account=<merchant_id>]  today's digest for the logged-in KAM (basic auth,
support_reviewer logins). Only KAMs, only their own accounts; every allow/deny is audited.
"""

import logging
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any

from flask import Blueprint, Response, g, request

from teammates.collections.app import Delivery, deliver_email
from teammates.guard import mask_email
from teammates.kam.digest import KamDigest, config, render
from teammates.kam.tools import PgKamTools, seed
from teammates.runtime import env, llm
from teammates.support import app as support
from teammates.support.desk import Denied

log = logging.getLogger(__name__)
OUTBOX = Path("reports/outbox")


def outbox_deliver(to: str, subject: str, html: str) -> Delivery:
    """Mocked no-reply sender: the email as a file in reports/outbox/."""
    try:
        OUTBOX.mkdir(parents=True, exist_ok=True)
        name = f"{datetime.now(config().tz):%Y%m%d_%H%M%S}_{mask_email(to).replace('*', 'x')}"
        path = OUTBOX / f"{name}.html"
        path.write_text(f"<!-- To: {to} | Subject: {subject} -->\n{html}", encoding="utf-8")
    except OSError as exc:
        return False, None, f"outbox write failed: {exc}", None
    return True, None, None, path.name


def _demo_seed() -> dict[str, Any]:
    """Seed meetings for the demo KAM go to demo_kam() (SUPPORT_KAM_EMAIL)."""
    s = seed()
    return s | {"meetings": [m | {"kam_id": support.demo_kam()}
                             if m["kam_id"] == support.DEMO_KAM else m for m in s["meetings"]]}


@cache
def digest() -> KamDigest:
    d, cfg, now = support.desk(), config(), datetime.now(config().tz)
    tools = PgKamTools(d.store, support.merchants(), now, cfg.tz)
    tools.migrate()
    s = _demo_seed()
    tools.seed_demo(s["meetings"], s["feed"], cfg.sources, now)
    deliver = deliver_email if env("KAM_DELIVERY") == "resend" else outbox_deliver
    base = env("PUBLIC_URL") or env("RENDER_EXTERNAL_URL") or "http://localhost:5000"
    return KamDigest(store=d.store, tools=tools, llm=llm(), cache=d.cache, deliver=deliver,
                     cfg=cfg, base_url=base.rstrip("/"))


def send_digest() -> int:
    return digest().send_all()


def preview_digest() -> int:
    return digest().send_all(preview_dir=Path("reports"))


bp = Blueprint("kam", __name__, url_prefix="/kam")


@bp.before_request
def require_login() -> Any:
    a = request.authorization
    r = support.desk().store.authenticate(a.username or "", a.password or "") if a else None
    if r is None:
        return Response("Login with your email and password", 401,
                        {"WWW-Authenticate": 'Basic realm="kam"'})
    g.reviewer = r
    return None


@bp.get("/digest")
def view() -> Any:
    account = request.args.get("account", type=int)
    try:
        d = digest().view(g.reviewer, account)
    except Denied as exc:
        return f"Not allowed: {exc}", 403
    return Response(render(d, digest().cfg, digest().base_url))
