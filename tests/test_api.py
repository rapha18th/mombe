"""Hub API guards: approvals need the approver code, and the limiter caps bursts."""

from types import SimpleNamespace

from mombe.server import api


def req(headers):
    return SimpleNamespace(headers=headers, client=SimpleNamespace(host="10.0.0.1"))


def test_hub_key_alone_cannot_approve(monkeypatch):
    monkeypatch.setattr(api, "APPROVER_KEY", "approve-abc123")
    assert not api._approver(req({"x-mombe-key": api.KEY}))
    assert not api._approver(req({"x-mombe-approver": "wrong"}))
    assert api._approver(req({"x-mombe-approver": "approve-abc123"}))


def test_no_approver_code_configured_refuses_everyone(monkeypatch):
    monkeypatch.setattr(api, "APPROVER_KEY", "")
    assert not api._approver(req({"x-mombe-approver": ""}))


def test_limiter_caps_a_burst_per_client():
    lim = api._Limiter()
    assert all(lim.allow(("create", "a"), 3, 60) for _ in range(3))
    assert not lim.allow(("create", "a"), 3, 60)
    assert lim.allow(("create", "b"), 3, 60)


def test_client_is_the_first_forwarded_address():
    assert api._client(req({"x-forwarded-for": "41.1.2.3, 130.176.0.1"})) == "41.1.2.3"
    assert api._client(req({})) == "10.0.0.1"


def test_officer_passcodes_open_the_app_and_stay_revocable(monkeypatch):
    monkeypatch.setattr(api, "OFFICER_KEYS", ["judge-1"])
    assert api._auth(req({"x-mombe-key": "judge-1"}))
    assert api._auth(req({"x-mombe-key": api.KEY}))
    monkeypatch.setattr(api, "OFFICER_KEYS", [])
    assert not api._auth(req({"x-mombe-key": "judge-1"}))
