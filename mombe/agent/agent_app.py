"""AgentCore Runtime entrypoint.

The hub invokes the agent on three events for a session: start, a new capture
upload, and an approval decision. Each invocation resumes the session's ledger,
lets the driver act until the policy parks the session (waiting for the officer
or a human) or finishes it, then persists the ledger.

Payload: {"session_id": "...", "event": "start|capture|approval", "driver": "claude|autopilot"}
"""

from __future__ import annotations

import os

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from mombe.agent.remote_state import HubTraceSink, RemoteRegistry, RemoteStore
from mombe.agent.remote_tools import RemoteVisionTools
from mombe.agent.runtime import Controller, autopilot

HUB = os.environ.get("MOMBE_HUB", "http://127.0.0.1:8080")
KEY = os.environ.get("MOMBE_KEY", "dev-key")

app = BedrockAgentCoreApp()


def handle(payload: dict) -> dict:
    sid = payload["session_id"]
    store, registry = RemoteStore(HUB, KEY), RemoteRegistry(HUB, KEY)
    tools = RemoteVisionTools(f"{HUB}/mcp", store=store, key=KEY)
    try:
        ctrl = Controller(sid, tools, registry, trace_sink=HubTraceSink(HUB, KEY))
        if payload.get("driver", "claude") == "autopilot":
            res = autopilot(ctrl)
        else:
            from mombe.agent.agent import run_session
            res = run_session(ctrl)
        ctrl.persist()
        return {"session_id": sid, "event": payload.get("event"), "outcome": ctrl.state.outcome,
                "required_next": ctrl.state.required, **{k: v for k, v in res.items() if k in ("final_message", "usage")}}
    finally:
        tools.close()


@app.entrypoint
def invoke(payload, context=None):
    return handle(payload)


if __name__ == "__main__":
    app.run()
