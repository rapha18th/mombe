"""vision-mcp: OpenCV 5 tools over MCP (streamable HTTP).

Runs on EC2 Graviton4 from the COOL AMI. The agent on AgentCore Runtime reaches
it over the VPC. Every tool reads the session's captures from the store and
returns JSON evidence plus the time it took on this host, which feeds the COOL
benchmark and the trace.

    python -m mombe.server.mcp_app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import cv2
import uvicorn
from mcp.server.mcpserver import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse

from mombe.server import api
from mombe.server.store import LocalRegistry, LocalStore
from mombe.server.tools import VisionTools
from mombe.vision.muzzle import Gallery, MuzzleEngine

ROOT = Path(os.environ.get("MOMBE_HOME", Path(__file__).resolve().parents[2]))


def build_tools() -> VisionTools:
    store = LocalStore(Path(os.environ.get("MOMBE_STORE", ROOT / "var" / "store")))
    registry = LocalRegistry(Path(os.environ.get("MOMBE_REGISTRY", ROOT / "var" / "registry")))
    gpath = Path(os.environ.get("MOMBE_GALLERY", ROOT / "var" / "gallery.npz"))
    gallery = Gallery.load(gpath) if gpath.exists() else Gallery()
    th_path = ROOT / "eval" / "out" / "thresholds.json"
    th = json.loads(th_path.read_text()) if th_path.exists() else {}
    engine = MuzzleEngine(**{k: v for k, v in th.items() if k in ("t_match", "t_new")})
    body = None
    if os.environ.get("MOMBE_BODY", "sim") == "sim":
        from mombe.sim.field import SimBodyProvider
        body = SimBodyProvider()
    return VisionTools(store, registry, gallery, engine=engine, body=body)


tools = build_tools()
mcp = MCPServer("mombe-vision", instructions="OpenCV 5 cattle vision tools: quality, identity, body, weight, value.")


@mcp.tool()
def assess_capture(session_id: str) -> dict:
    """Sharpness, glare, exposure and nose size for the latest capture, with officer instructions."""
    return tools.assess_capture(session_id)


@mcp.tool()
def identify_muzzle(session_id: str) -> dict:
    """Noseprint identity: DINOv2 shortlist, ALIKED + LightGlue, MAGSAC-verified inliers and a decision band."""
    return tools.identify_muzzle(session_id)


@mcp.tool()
def lookup_animal(animal_id: str) -> dict:
    """Registry record for an animal."""
    return tools.lookup_animal(animal_id)


@mcp.tool()
def check_continuity(session_id: str) -> dict:
    """Optical-flow chain from the nose close-up to the body walk-around."""
    return tools.check_continuity(session_id)


@mcp.tool()
def calibrate_scale(session_id: str) -> dict:
    """ChArUco-based metric scale and its relative error."""
    return tools.calibrate_scale(session_id)


@mcp.tool()
def measure_body(session_id: str) -> dict:
    """Torso coverage, heart girth and body length."""
    return tools.measure_body(session_id)


@mcp.tool()
def estimate_weight(session_id: str) -> dict:
    """Live weight with a 90% interval."""
    return tools.estimate_weight(session_id)


@mcp.tool()
def value_animal(session_id: str, kg: float, lo: float, hi: float) -> dict:
    """Value range from a weight interval and the configured price per kg."""
    return tools.value_animal(session_id, {"kg": kg, "lo": lo, "hi": hi})


def _wake_agent(session_id: str, event: str) -> None:
    """Invoke the agent for this session: AgentCore Runtime when deployed, in-process locally."""
    import threading
    arn = os.environ.get("MOMBE_AGENT_ARN")
    driver = os.environ.get("MOMBE_DRIVER", "claude")
    payload = {"session_id": session_id, "event": event, "driver": driver}

    def run():
        if arn:
            import boto3
            boto3.client("bedrock-agentcore", region_name=os.environ.get("AWS_REGION", "eu-west-1")).invoke_agent_runtime(
                agentRuntimeArn=arn, runtimeSessionId=f"mombe-session-{session_id}".ljust(33, "0"),
                payload=json.dumps(payload).encode())
        elif os.environ.get("MOMBE_LOCAL_AGENT"):
            from mombe.agent.agent_app import handle
            handle(payload)
    threading.Thread(target=run, daemon=True).start()


api.register(mcp, tools, on_event=_wake_agent)


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse({"ok": True, "opencv": cv2.__version__, "machine": platform.machine(),
                         "threads": cv2.getNumThreads(), "gallery_images": len(tools.gallery.features),
                         "cpu_features": cv2.getCPUFeaturesLine() if hasattr(cv2, "getCPUFeaturesLine") else None})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    app = mcp.streamable_http_app(stateless_http=True, json_response=True, host=a.host)
    from starlette.staticfiles import StaticFiles
    app.mount("/app", StaticFiles(directory=str(Path(__file__).resolve().parents[2] / "web"), html=True), name="app")
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
