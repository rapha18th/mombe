"""The MCP route answers to the hub key alone."""

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from mombe.server.guard import McpKeyGuard


async def _ok(_):
    return JSONResponse({"ok": True})


def _client(key="hub-key"):
    app = Starlette(routes=[Route("/mcp", _ok, methods=["POST"]), Route("/health", _ok)])
    return TestClient(McpKeyGuard(app, key))


def test_mcp_needs_the_hub_key():
    c = _client()
    assert c.post("/mcp", json={}).status_code == 401
    assert c.post("/mcp", json={}, headers={"x-mombe-key": "officer-pass"}).status_code == 401
    assert c.post("/mcp", json={}, headers={"x-mombe-key": "hub-key"}).status_code == 200


def test_subpaths_are_guarded_and_other_routes_pass():
    c = _client()
    assert c.post("/mcp/", json={}).status_code == 401
    assert c.get("/health").status_code == 200


def test_empty_key_refuses_everyone():
    c = _client(key="")
    assert c.post("/mcp", json={}, headers={"x-mombe-key": ""}).status_code == 401
