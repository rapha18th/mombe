"""Hub key check for the MCP route.

The REST routes in api.py check X-Mombe-Key themselves. The MCP SDK mounts
/mcp, so this ASGI wrapper checks it. /mcp answers to the hub key alone, the
key the agent on AgentCore Runtime carries. Officer passcodes open the REST
routes the phone app uses; the tools stay behind the agent and its policy.
"""

from __future__ import annotations

import hmac

from starlette.responses import JSONResponse


def _header(scope, name: bytes) -> bytes:
    for k, v in scope.get("headers", []):
        if k.lower() == name:
            return v
    return b""


class McpKeyGuard:
    """Wrap an ASGI app; refuse requests under `prefix` that lack the hub key."""

    def __init__(self, app, key: str, prefix: str = "/mcp"):
        self.app = app
        self.key = key.encode()
        self.prefix = prefix.rstrip("/")

    def guarded(self, path: str) -> bool:
        return path == self.prefix or path.startswith(self.prefix + "/")

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and self.guarded(scope.get("path", "")):
            given = _header(scope, b"x-mombe-key")
            if not (self.key and hmac.compare_digest(given, self.key)):
                await JSONResponse({"error": "unauthorised"}, status_code=401)(scope, receive, send)
                return
        await self.app(scope, receive, send)
