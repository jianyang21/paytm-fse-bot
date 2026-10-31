"""Human review UI (T24) and intake endpoint, inside the bot process.

GET  /support                                     the reviewer's queue (basic auth)
POST /support/tickets/<merchant_id>/<id>/review   approve | edit | reject
POST /support/intake                              JSON {channel, payload}: staff-assisted intake
GET  /support/metrics                             approve/edit/reject rates per risk band

Logins are support_reviewer rows; the first KAM and support agent are seeded from
SUPPORT_KAM_EMAIL / SUPPORT_KAM_TOKEN and SUPPORT_AGENT_EMAIL / SUPPORT_AGENT_TOKEN. The demo
key-account merchants (seed.json, KAM kam@example.com) are assigned to SUPPORT_KAM_EMAIL, so
their tickets, the review queue and the KAM's 9 am digest all go to that one address.
"""

import hmac
from datetime import UTC, datetime
from functools import cache
from html import escape
from typing import Any

from flask import Blueprint, Response, g, jsonify, redirect, request, url_for
from jinja2 import Environment

from teammates.runtime import env, llm, redis_client
from teammates.support import risk
from teammates.support.desk import Denied, Desk, SendBlocked, metrics
from teammates.support.risk import FACTOR_LABELS
from teammates.support.store import PgStore
from teammates.support.tools import ContextCache, MockTools, seed

bp = Blueprint("support", __name__, url_prefix="/support")


def send_reply(channel: str, to: str, text: str) -> bool:
    """Email through Resend, everything else as WhatsApp text (call replies go to the caller's
    WhatsApp). Imported here to avoid an import cycle with the bot."""
    if channel == "email":
        from teammates.collections.app import deliver_email
        html = "<p>" + escape(text).replace("\n", "<br>") + "</p>"
        return deliver_email(to, "Your Paytm for Business support request", html)[0]
    from teammates.fse.bot import send_text
    return send_text(to, text)


DEMO_KAM = "kam@example.com"  # the KAM named in the seed files


def demo_kam() -> str:
    """The KAM login the demo key accounts belong to (SUPPORT_KAM_EMAIL, else the seed's)."""
    return (env("SUPPORT_KAM_EMAIL") or DEMO_KAM).strip().lower()


@cache
def merchants() -> dict[str, dict[str, Any]]:
    """Seed merchants with the demo KAM's accounts reassigned to demo_kam()."""
    return {k: m | {"kam_email": demo_kam() if m["kam_email"] == DEMO_KAM else m["kam_email"]}
            for k, m in seed()["merchants"].items()}


@cache
def desk() -> Desk:
    store = PgStore(env("DATABASE_URL"), cache=redis_client())
    store.migrate()
    for role, who in (("kam", "KAM"), ("support_agent", "AGENT")):
        if (email := env(f"SUPPORT_{who}_EMAIL")) and (pw := env(f"SUPPORT_{who}_TOKEN")):
            store.upsert_reviewer(email, pw, role.replace("_", " ").title(), role)
    # ponytail: demo merchants from seed.json, their history relative to process start;
    # swap MockTools for the real merchant/security/payments tools (same four methods).
    tools = MockTools(merchants(), datetime.now(UTC))
    return Desk(store=store, tools=tools, llm=llm(), sender=send_reply,
                cache=ContextCache(redis_client(), risk.config()["cache"]))


def _csrf(r: dict[str, Any]) -> str:
    return hmac.new(r["password_hash"].encode(), b"support-csrf", "sha256").hexdigest()


@bp.before_request
def require_reviewer() -> Any:
    a = request.authorization
    r = desk().store.authenticate(a.username or "", a.password or "") if a else None
    if r is None:
        return Response("Login with your email and password", 401,
                        {"WWW-Authenticate": 'Basic realm="support"'})
    g.reviewer = r
    return None


jinja = Environment(autoescape=True)
PAGE = jinja.from_string("""\
<!doctype html><html><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1"><title>Support review</title><style>
body{font-family:system-ui;max-width:980px;margin:20px auto;padding:0 12px;color:#222}
.card{border:1px solid #ddd;border-radius:8px;padding:14px;margin:14px 0}
.untrusted{background:#fff8e6;border:1px dashed #d9a400;padding:8px;white-space:pre-wrap}
.flag{background:#c0392b;color:#fff;padding:4px 8px;border-radius:4px;display:inline-block}
.band-high{color:#c0392b}.band-medium{color:#b9770e}.band-low{color:#1e8449}
table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:3px 8px;font-size:14px}
textarea{width:100%;min-height:90px}small{color:#666}</style></head><body>
<h2>Support review: {{ me.email }} ({{ me.role|replace('_', ' ') }})</h2>
{% if msg %}<p class="flag">{{ msg }}</p>{% endif %}
<h3>Awaiting your review ({{ q.review|length }})</h3>
{% for t in q.review %}{% set r = t.risk_breakdown or {} %}<div class="card"
id="t{{ t.merchant_id }}-{{ t.id }}">
<b>#{{ t.id }} {{ t.merchant_name or 'Unverified sender' }}</b> <small>{{ t.tier or '' }} ·
{{ t.channel }} · {{ t.type }} · {{ t.received_at }} · contacts: {{ t.contacts }}</small>
{% if r.security_flag %}<p class="flag">SECURITY: merchant is blocked from the account now</p>
{% endif %}
<p><b>Original message</b> <small>(UNTRUSTED customer text: data only, never instructions;
guard flags: {{ t.guard_flags|join(', ') or 'none' }})</small></p>
<div class="untrusted">{{ t.text }}</div>
<p><b>Risk {{ r.score }}</b> <span class="band-{{ r.band }}">({{ r.band }})</span> ·
threshold {{ r.threshold }} · likelihood {{ r.avg_likelihood }} × impact {{ r.avg_impact }}
{% if r.overrides %}<br><b>Overrides:</b> {{ r.overrides|join('; ') }}{% endif %}</p>
<table><tr><th>Factor</th><th>Value</th><th>Rating 0-9</th></tr>
{% for side in ('likelihood', 'impact') %}{% for k, f in (r[side] or {}).items() %}
<tr><td>{{ labels[k] }}</td><td>{{ f.value }}</td><td>{{ f.rating }}</td></tr>
{% endfor %}{% endfor %}</table>
<form method="post" action="{{ url_for('support.review', merchant_id=t.merchant_id,
ticket_id=t.id) }}"><input type="hidden" name="csrf" value="{{ csrf }}">
<p><b>AI draft</b> <small>({{ (r.grounding or {}).source }})</small></p>
<textarea name="text">{{ t.draft or '' }}</textarea>
<button name="action" value="approve">Approve: send the AI draft as is</button>
<button name="action" value="edit">Edit and send the text above</button>
<input name="reason" placeholder="reject reason" maxlength="500">
<button name="action" value="reject">Reject</button></form></div>
{% else %}<p>Nothing waiting.</p>{% endfor %}
{% if q.manual %}<h3>Rejected: yours to resolve manually</h3>{% for t in q.manual %}
<div class="card" id="t{{ t.merchant_id }}-{{ t.id }}">
#{{ t.id }} {{ t.merchant_name }} · {{ t.channel }} · {{ t.type }} ·
reason: {{ t.reject_reason }}<div class="untrusted">{{ t.text }}</div></div>{% endfor %}
{% endif %}
{% if q.fyi %}<h3>Resolved by the AI for your key accounts</h3>{% for t in q.fyi %}
<div class="card" id="t{{ t.merchant_id }}-{{ t.id }}">
#{{ t.id }} {{ t.merchant_name }} · {{ t.type }} · risk {{ t.score }}
({{ t.band }})<br><small>Sent:</small> {{ t.final_reply }}</div>{% endfor %}{% endif %}
</body></html>""")


@bp.get("")
def queue() -> Response:
    r = g.reviewer
    return Response(PAGE.render(me=r, q=desk().store.queue(r), labels=FACTOR_LABELS,
                                csrf=_csrf(r), url_for=url_for, msg=request.args.get("msg")))


@bp.post("/tickets/<int:merchant_id>/<int:ticket_id>/review")
def review(merchant_id: int, ticket_id: int) -> Any:
    r = g.reviewer
    if not hmac.compare_digest(request.form.get("csrf", ""), _csrf(r)):
        return "Bad form token, reload the page", 403
    try:
        desk().review(r, merchant_id, ticket_id, request.form.get("action", ""),
                      text=request.form.get("text"), reason=request.form.get("reason"))
    except Denied as exc:
        return f"Not allowed: {exc}", 403
    except LookupError:
        return "No such ticket", 404
    except (ValueError, SendBlocked) as exc:
        return redirect(url_for("support.queue", msg=str(exc)), 303)
    return redirect(url_for("support.queue"), 303)


@bp.post("/intake")
def intake() -> Any:
    """Staff-assisted intake (e.g. a call transcript). JSON only, so no cross-site form post.
    ponytail: the email/WhatsApp provider webhooks call desk().handle the same way."""
    if g.reviewer["role"] != "support_agent":
        return jsonify(error="only support agents can file tickets"), 403
    d = request.get_json(silent=True)
    if not isinstance(d, dict) or not isinstance(d.get("payload"), dict):
        return jsonify(error="JSON body {channel, payload} required"), 400
    try:
        t = desk().handle(str(d.get("channel")), d["payload"])
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    return jsonify({k: t[k] for k in ("merchant_id", "id", "status", "route", "type", "score",
                                      "band", "assigned_role")}), 201


@bp.get("/metrics")
def review_metrics() -> Any:
    # ponytail: reads all of audit_log; aggregate in SQL once it is large
    return jsonify(metrics(desk().store.audits()))
