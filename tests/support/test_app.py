import base64

import pytest
from flask import Flask

from teammates.support import app as sup
from tests.support.conftest import SEED


def auth(email, pw="pw"):
    return {"Authorization": "Basic " + base64.b64encode(f"{email}:{pw}".encode()).decode()}


AGENT, KAM = auth("agent@example.com"), auth("kam@example.com")


@pytest.fixture
def client(env, monkeypatch):
    monkeypatch.setattr(sup, "desk", lambda: env.desk)
    app = Flask(__name__)
    app.register_blueprint(sup.bp)
    c = app.test_client()
    c.env = env
    return c


def csrf(env, who="agent@example.com"):
    return sup._csrf(env.store.reviewers[who])


def test_login_required(client):
    assert client.get("/support").status_code == 401
    assert client.get("/support", headers=auth("agent@example.com", "bad")).status_code == 401


def test_queue_shows_untrusted_message_escaped_with_risk_breakdown(client):
    t = client.env.desk.handle("whatsapp", SEED["T13"]["payload"] | {
        "text": "Refund of ₹450 pending <script>alert(1)</script>"})
    page = client.get("/support", headers=AGENT).get_data(as_text=True)
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page and "<script>alert" not in page
    assert "UNTRUSTED customer text" in page and f"#{t['id']} Iyer Tailoring" in page
    for s in ("Failed logins (7 days)", "Amount at stake", f"Risk {t['score']}", "AI draft",
              "Approve: send the AI draft as is", "Edit and send", "Reject"):
        assert s in page, s


def test_security_flag_and_overrides_visible(client):
    client.env.run("T17")
    page = client.get("/support", headers=AGENT).get_data(as_text=True)
    assert "SECURITY: merchant is blocked" in page and "blocked from account now" in page


def test_approve_from_the_form(client):
    t = client.env.run("T13")
    url = f"/support/tickets/110/{t['id']}/review"
    assert client.post(url, headers=AGENT, data={"action": "approve"}).status_code == 403
    r = client.post(url, headers=AGENT, data={"action": "approve", "csrf": csrf(client.env)})
    assert r.status_code == 303 and client.env.store.get(110, t["id"])["status"] == "resolved"


def test_edit_and_reject_from_the_form(client):
    a, b = client.env.run("T13"), client.env.run("T15")
    c = csrf(client.env)
    client.post(f"/support/tickets/110/{a['id']}/review", headers=AGENT,
                data={"action": "edit", "text": "Done.", "csrf": c})
    assert client.env.store.get(110, a["id"])["final_reply"] == "Done."
    r = client.post(f"/support/tickets/103/{b['id']}/review", headers=AGENT,
                    data={"action": "reject", "csrf": c})
    assert r.status_code == 303 and "msg=" in r.location  # no reason: shown, nothing changed
    client.post(f"/support/tickets/103/{b['id']}/review", headers=AGENT,
                data={"action": "reject", "reason": "duplicate", "csrf": c})
    assert client.env.store.get(103, b["id"])["status"] == "manual"


def test_wrong_role_gets_403(client):
    t = client.env.run("T14")  # KAM ticket
    r = client.post(f"/support/tickets/102/{t['id']}/review", headers=AGENT,
                    data={"action": "approve", "csrf": csrf(client.env)})
    assert r.status_code == 403 and "Not allowed" in r.get_data(as_text=True)
    assert client.post("/support/tickets/102/999/review", headers=KAM, data={
        "action": "approve", "csrf": csrf(client.env, "kam@example.com")}).status_code == 404


def test_kam_sees_fyi_for_ai_resolved_key_accounts(client):
    client.env.run("T05")
    page = client.get("/support", headers=KAM).get_data(as_text=True)
    assert "Resolved by the AI for your key accounts" in page and "Ravi Stores" in page


def test_intake_endpoint(client):
    body = {"channel": "call", "payload": SEED["T04"]["payload"]}
    assert client.post("/support/intake", headers=KAM, json=body).status_code == 403
    assert client.post("/support/intake", headers=AGENT, data="x").status_code == 400
    assert client.post("/support/intake", headers=AGENT,
                       json={"channel": "fax", "payload": {}}).status_code == 400
    r = client.post("/support/intake", headers=AGENT, json=body)
    assert r.status_code == 201 and r.json["status"] == "resolved" and r.json["score"] == 0


def test_metrics_endpoint(client):
    t = client.env.run("T13")
    client.env.desk.review(client.env.agent, 110, t["id"], "approve")
    assert client.get("/support/metrics", headers=AGENT).json == {
        t["band"]: {"n": 1, "approve_rate": 1.0, "edit_rate": 0.0, "reject_rate": 0.0}}


def test_send_reply_routes_by_channel(monkeypatch):
    from teammates.collections import app as coll
    from teammates.fse import bot

    got = []
    monkeypatch.setattr(coll, "deliver_email",
                        lambda to, subj, html: got.append((to, html)) or (True, 200, None, "id"))
    monkeypatch.setattr(bot, "send_text", lambda to, text: got.append((to, text)) or True)
    assert sup.send_reply("email", "a@x.com", "Hi <b>\nthere")
    assert sup.send_reply("whatsapp", "9199", "Hi")
    assert got == [("a@x.com", "<p>Hi &lt;b&gt;<br>there</p>"), ("9199", "Hi")]
