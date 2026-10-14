"""Session store and registry over the hub's REST API, for the deployed agent."""

from __future__ import annotations

import httpx

from mombe.server.store import Capture, Session


class _Hub:
    def __init__(self, base: str, key: str):
        self.http = httpx.Client(base_url=base.rstrip("/"), headers={"x-mombe-key": key}, timeout=30)

    def get(self, path: str, **kw):
        r = self.http.get(path, **kw)
        r.raise_for_status()
        return r.json()

    def send(self, method: str, path: str, body):
        r = self.http.request(method, path, json=body)
        r.raise_for_status()
        return r.json()


class RemoteStore(_Hub):
    root = "remote"

    def load(self, sid: str) -> Session:
        d = self.get(f"/api/sessions/{sid}")
        d["captures"] = [Capture(**c) for c in d["captures"]]
        return Session(**d)

    def save(self, s: Session) -> None:
        from dataclasses import asdict
        self.send("PUT", f"/api/sessions/{s.id}", asdict(s))


class RemoteRegistry(_Hub):
    def get(self, animal_id: str):  # noqa: D401 - registry lookup
        return super().get(f"/api/animals/{animal_id}")["record"]

    def put(self, animal_id: str, record: dict) -> None:
        self.send("PUT", f"/api/animals/{animal_id}", record)

    def request_approval(self, session_id: str, kind: str, summary: dict) -> dict:
        return self.send("POST", "/api/approvals", {"session_id": session_id, "kind": kind, "summary": summary})

    def decide(self, approval_id: str, status: str, by: str, note: str = "") -> dict:
        return self.send("POST", f"/api/approvals/{approval_id}", {"status": status, "by": by, "note": note})

    def approval(self, approval_id: str) -> dict | None:
        return _Hub.get(self, f"/api/approvals/{approval_id}")


class HubTraceSink(_Hub):
    def __call__(self, sid: str, event: dict) -> None:
        self.send("POST", f"/api/sessions/{sid}/trace", event)
