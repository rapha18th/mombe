"""Session hub REST routes, mounted on the vision server.

Phone app:   create a session, upload captures, read instructions and outcome.
Approvers:   list pending approvals, read the herd valuation and prices, decide.
Agent:       read and write the session (ledger, instructions), approvals and
             the decision trace, so the controller on AgentCore Runtime runs
             against the same state the vision tools read.

Every request carries X-Mombe-Key. CloudFront adds it for browser traffic;
the agent reads it from its environment. A decision on an approval also needs
X-Mombe-Approver, a code only approvers hold: the key the phone app and the
agent carry can request an approval and never grant one. Session creation and
uploads are rate limited per client, and registry reads land in an audit log.
"""

from __future__ import annotations

import hmac
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response

KEY = os.environ.get("MOMBE_KEY", "dev-key")
APPROVER_KEY = os.environ.get("MOMBE_APPROVER_KEY", "")
LIMITS = {"create": (30, 3600), "upload": (240, 3600)}     # requests per window (s) per client
_on_event = None   # set by mcp_app: callable(session_id, event) that wakes the agent


def _auth(req: Request) -> bool:
    return hmac.compare_digest(req.headers.get("x-mombe-key", ""), KEY)


def _approver(req: Request) -> bool:
    code = req.headers.get("x-mombe-approver", "")
    return bool(APPROVER_KEY) and hmac.compare_digest(code, APPROVER_KEY)


def _client(req: Request) -> str:
    fwd = req.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (req.client.host if req.client else "unknown")


class _Limiter:
    """Sliding window per (route, client)."""

    def __init__(self):
        self.hits: dict[tuple, deque] = defaultdict(deque)

    def allow(self, key: tuple, n: int, window: float) -> bool:
        now, q = time.monotonic(), self.hits[key]
        while q and now - q[0] > window:
            q.popleft()
        if len(q) >= n:
            return False
        q.append(now)
        return True


_limiter = _Limiter()


def _deny() -> JSONResponse:
    return JSONResponse({"error": "unauthorised"}, status_code=401)


def register(mcp, tools, on_event=None) -> None:
    global _on_event
    _on_event = on_event
    store, registry = tools.store, tools.registry
    trace_dir = Path(store.root) / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    def _mark_running(sid: str) -> None:
        s = store.load(sid)
        s.status = "running"
        store.save(s)

    audit = Path(store.root) / "audit.jsonl"

    def _audit(req: Request, **event) -> None:
        with audit.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "client": _client(req), **event}) + "
")

    def route(path, methods, limit=None):
        def deco(fn):
            async def wrapped(req: Request):
                if not _auth(req):
                    return _deny()
                if limit and req.method == "POST" and not _limiter.allow((limit, _client(req)), *LIMITS[limit]):
                    return JSONResponse({"error": "too many requests"}, status_code=429)
                try:
                    return await fn(req)
                except FileNotFoundError:
                    return JSONResponse({"error": "not found"}, status_code=404)
            mcp.custom_route(path, methods=methods)(wrapped)
            return fn
        return deco

    # -- sessions -----------------------------------------------------------
    @route("/api/sessions", ["POST"], limit="create")
    async def create(req: Request):
        b = await req.json()
        s = store.create(b["task"], claimed_owner=b.get("claimed_owner"), policy_animal=b.get("policy_animal"),
                         loan_threshold_usd=b.get("loan_threshold_usd"), officer=b.get("officer", "officer-demo"))
        return JSONResponse(asdict(s))

    @route("/api/sessions/{sid}", ["GET", "PUT"])
    async def session(req: Request):
        sid = req.path_params["sid"]
        if req.method == "PUT":
            from mombe.server.store import Capture, Session
            d = await req.json()
            d["captures"] = [Capture(**c) for c in d["captures"]]
            incoming, current = Session(**d), store.load(sid)
            # Evidence that arrived while the agent was running is never dropped.
            if len(current.captures) > len(incoming.captures):
                incoming.captures = current.captures
                incoming.status = "running"
            if len(current.instructions) > len(incoming.instructions):
                incoming.instructions = current.instructions
            store.save(incoming)
            return JSONResponse({"ok": True})
        return JSONResponse(asdict(store.load(sid)))

    @route("/api/sessions/{sid}/captures", ["POST"], limit="upload")
    async def upload(req: Request):
        sid = req.path_params["sid"]
        form = await req.form()
        muzzle, body = [], []
        for key, val in form.multi_items():
            if not hasattr(val, "read"):
                continue
            img = cv2.imdecode(np.frombuffer(await val.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            (muzzle if key.startswith("muzzle") else body).append(img)
        meta = json.loads(form.get("meta", "{}"))
        c = store.add_capture(sid, muzzle=muzzle, body=body, meta=meta)
        if body and hasattr(tools.body, "start"):
            # measuring a walk-around takes minutes: start now, wake the agent when it is done
            tools.body.start(c, on_done=(lambda: _on_event(sid, "body_ready")) if _on_event else None)
        _mark_running(sid)
        if _on_event:
            _on_event(sid, "capture")
        return JSONResponse({"capture": c.index, "muzzle": len(muzzle), "body": len(body)})

    # -- trace ----------------------------------------------------------------
    @route("/api/sessions/{sid}/trace", ["GET", "POST"])
    async def trace(req: Request):
        p = trace_dir / f"{req.path_params['sid']}.jsonl"
        if req.method == "POST":
            events = await req.json()
            with p.open("a", encoding="utf-8") as f:
                for e in events if isinstance(events, list) else [events]:
                    f.write(json.dumps(e) + "\n")
            return JSONResponse({"ok": True})
        lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
        return JSONResponse([json.loads(x) for x in lines])

    @route("/api/sessions/{sid}/artifacts/{name}", ["GET"])
    async def artifact(req: Request):
        p = store.artifact_path(req.path_params["sid"], req.path_params["name"])
        return FileResponse(p) if p.exists() else Response(status_code=404)

    # -- registry and approvals --------------------------------------------
    @route("/api/animals/{aid}", ["GET", "PUT"])
    async def animal(req: Request):
        aid = req.path_params["aid"]
        if req.method == "PUT":
            registry.put(aid, await req.json())
        _audit(req, registry=req.method.lower(), animal_id=aid)
        return JSONResponse({"animal_id": aid, "record": registry.get(aid)})

    # -- lender view: herd totals and the prices behind them ----------------
    @route("/api/herd", ["GET"])
    async def herd(req: Request):
        qp = req.query_params
        ids = [x for x in qp.get("ids", "").split(",") if x]
        return JSONResponse(tools.value_herd(owner=qp.get("owner") or None, animal_ids=ids or None,
                                             session_id=qp.get("session") or None))

    @route("/api/prices", ["GET"])
    async def prices(req: Request):
        return JSONResponse(tools.price_book())

    @route("/api/approvals", ["GET", "POST"])
    async def approvals(req: Request):
        if req.method == "POST":
            b = await req.json()
            return JSONResponse(registry.request_approval(b["session_id"], b["kind"], b.get("summary", {})))
        status = req.query_params.get("status")
        items = [a for a in registry.approvals.values() if not status or a["status"] == status]
        return JSONResponse(sorted(items, key=lambda a: a["created"], reverse=True))

    @route("/api/approvals/{aid}", ["GET", "POST"])
    async def approval(req: Request):
        aid = req.path_params["aid"]
        if req.method == "POST":
            if not _approver(req):
                return JSONResponse({"error": "approver sign-in required"}, status_code=403)
            b = await req.json()
            by = req.headers.get("x-mombe-approver-name") or b.get("by", "approver")
            a = registry.decide(aid, b["status"], by=by, note=b.get("note", ""))
            _audit(req, approval=aid, decision=b["status"], by=by)
            _mark_running(a["session_id"])
            if _on_event:
                _on_event(a["session_id"], "approval")
            return JSONResponse(a)
        return JSONResponse(registry.approval(aid))
