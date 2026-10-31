import base64
from types import SimpleNamespace

import pytest
from flask import Flask

from teammates.kam import app as kam_app
from teammates.support import app as support_app


def auth(email, pw="pw"):
    return {"Authorization": "Basic " + base64.b64encode(f"{email}:{pw}".encode()).decode()}


@pytest.fixture
def client(env, monkeypatch):
    monkeypatch.setattr(kam_app, "digest", lambda: env.digest)
    monkeypatch.setattr(support_app, "desk", lambda: SimpleNamespace(store=env.store))
    app = Flask(__name__)
    app.register_blueprint(kam_app.bp)
    return app.test_client()


def test_login_required(client):
    assert client.get("/kam/digest").status_code == 401
    assert client.get("/kam/digest", headers=auth("kam@example.com", "wrong")).status_code == 401


def test_kam_sees_own_digest(client):
    r = client.get("/kam/digest", headers=auth("kam@example.com"))
    assert r.status_code == 200 and b"Ravi Stores" in r.data and b"Imran Electronics" not in r.data
    r = client.get("/kam/digest?account=104", headers=auth("kam@example.com"))
    assert r.status_code == 200 and b"Lakshmi Bakery" in r.data and b"Ravi Stores" not in r.data


def test_other_kams_account_and_wrong_role_denied(client, env):
    assert client.get("/kam/digest?account=201",
                      headers=auth("kam@example.com")).status_code == 403
    assert client.get("/kam/digest", headers=auth("agent@example.com")).status_code == 403
    assert [a["action"] for a in env.store.audits() if a["ticket_id"] == 0] == ["denied",
                                                                                 "denied"]


def test_outbox_sender_writes_the_email(tmp_path, monkeypatch):
    monkeypatch.setattr(kam_app, "OUTBOX", tmp_path)
    ok, _, err, name = kam_app.outbox_deliver("kam@example.com", "Subj", "<p>hi</p>")
    assert ok and err is None
    body = (tmp_path / name).read_text(encoding="utf-8")
    assert "Subject: Subj" in body and "<p>hi</p>" in body
    assert "kam@example.com" not in name  # file names are masked


def test_demo_kam_email_is_configurable(monkeypatch):
    monkeypatch.setenv("SUPPORT_KAM_EMAIL", "Real.KAM@Company.in")
    support_app.merchants.cache_clear()
    try:
        assert support_app.demo_kam() == "real.kam@company.in"
        keys = {m["kam_email"] for m in support_app.merchants().values() if m["tier"] == "key"}
        assert keys == {"real.kam@company.in"}
        assert {m["kam_id"] for m in kam_app._demo_seed()["meetings"]} == {
            "real.kam@company.in", "kam2@example.com"}
    finally:
        support_app.merchants.cache_clear()
