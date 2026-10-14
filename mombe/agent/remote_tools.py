"""VisionTools over MCP: the agent's view of vision-mcp on Graviton.

Same method names and return shapes as mombe.server.tools.VisionTools, so the
Controller runs unchanged whether the tools are in-process (simulator, tests)
or on the COOL instance across the VPC (deployed).
"""

from __future__ import annotations

import json
import time
import uuid

from mcp.client.streamable_http import streamable_http_client
from strands.tools.mcp import MCPClient


class RemoteVisionTools:
    def __init__(self, url: str, store=None):
        self.url = url
        self.store = store                     # session metadata (S3/DynamoDB-backed in the cloud)
        self.client = MCPClient(lambda: streamable_http_client(url))
        self.client.start()
        self.timings: list[dict] = []

    def close(self):
        self.client.stop(None, None, None)

    def _call(self, name: str, **args) -> dict:
        t = time.perf_counter()
        r = self.client.call_tool_sync(uuid.uuid4().hex, name, args)
        rtt = round((time.perf_counter() - t) * 1000, 1)
        out = r.get("structuredContent") if isinstance(r, dict) else None
        if not out:
            content = r["content"] if isinstance(r, dict) else r.content
            text = next((c["text"] for c in content if "text" in c), "{}")
            out = json.loads(text)
        out = out.get("result", out) if set(out) == {"result"} else out
        if r.get("status") == "error":
            out = {"error": out}
        self.timings.append({"tool": name, "ms_round_trip": rtt, "ms_server": out.get("elapsed_ms")})
        out["round_trip_ms"] = rtt
        return out

    def assess_capture(self, session_id: str) -> dict:
        return self._call("assess_capture", session_id=session_id)

    def identify_muzzle(self, session_id: str) -> dict:
        return self._call("identify_muzzle", session_id=session_id)

    def lookup_animal(self, animal_id: str) -> dict:
        return self._call("lookup_animal", animal_id=animal_id)

    def check_continuity(self, session_id: str) -> dict:
        return self._call("check_continuity", session_id=session_id)

    def calibrate_scale(self, session_id: str) -> dict:
        return self._call("calibrate_scale", session_id=session_id)

    def measure_body(self, session_id: str) -> dict:
        return self._call("measure_body", session_id=session_id)

    def estimate_weight(self, session_id: str) -> dict:
        return self._call("estimate_weight", session_id=session_id)

    def value_animal(self, session_id: str, weight: dict) -> dict:
        return self._call("value_animal", session_id=session_id, kg=weight["kg"], lo=weight["lo"], hi=weight["hi"])
