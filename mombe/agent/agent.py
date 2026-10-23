"""The Mombe field agent: Claude on Amazon Bedrock, driven through Strands.

The model sees every OpenCV 5 result together with the evidence policy's view
(gates, the action the evidence requires next, the actions it allows). It
plans the observations, writes the officer's instructions in English and
Shona, writes the approval summaries, and reports. Actions the evidence does
not allow are refused by the controller and the refusal is traced.

Each tool call also stamps OpenTelemetry attributes on its span, so AgentCore
Observability shows, per step, whether a vision result changed the next action.
"""

from __future__ import annotations

import json
import os

from strands import Agent, tool
from strands.models import BedrockModel

from mombe.agent.runtime import Controller

try:
    from opentelemetry import trace as otel
except ImportError:  # local runs without OTEL
    otel = None

MODEL_ID = os.environ.get("MOMBE_MODEL_ID", "eu.anthropic.claude-sonnet-4-6")
REGION = os.environ.get("AWS_REGION", "eu-west-1")

SYSTEM = """You are Mombe, a field agent that identifies, measures and values cattle for a lender and an insurer in Zimbabwe.
An officer stands next to the animal with a phone. You see the animal only through vision tools built on OpenCV 5.

How you work:
- Every tool result carries `policy`: gates with their measured values, `required_next` (the action the evidence demands),
  `suggested_observation` (the next measurement to take) and `allowed_actions`.
- When `required_next` is set, take that action. Fill its arguments from `required_args`; you may improve the officer
  message, keeping it short and concrete, and add the Shona version after a ' / '. Write plain sentences
  without dashes.
- When nothing is required, take `suggested_observation` or another observation that fills an unmeasured gate.
- Never state an identity, weight or value that a tool did not report. Report intervals, never single numbers alone.
- State a value as the P10 value with the price's source and date and the expiry date. A person approves every loan.
- For request_approval, write a two-sentence note for the human: what the images show and what remains uncertain.
- A failed liveness check means the nose may be a print, a screen or a resent photo. Ask politely for the live
  animal; a second failure goes to a fraud review under the policy.
- After request_capture: if the result carries `new_capture`, the officer has already captured again, so continue
  with `suggested_observation`. If it says the session waits for the next upload, or a body tool answers
  that the walk-around is still processing, call `finish`: the hub wakes you when there is more.
- Keep going until the policy reports done. Stop with `finish` when the policy says done. Your final message to the officer: outcome, animal id, weight and value
  intervals when measured, and what happens next.
"""


def _stamp(name: str, res: dict) -> None:
    if otel is None:
        return
    span = otel.get_current_span()
    pol = res.get("policy", {})
    span.set_attribute("mombe.tool", name)
    span.set_attribute("mombe.required_next", str(pol.get("required_next")))
    span.set_attribute("mombe.allowed_actions", ",".join(pol.get("allowed_actions", [])))
    for g in pol.get("gates", []):
        span.set_attribute(f"mombe.gate.{g['gate']}", str(g["passed"]))
    if "error" in res:
        span.set_attribute("mombe.refused", res["error"][:200])


def build_agent(ctrl: Controller, model_id: str = MODEL_ID, region: str = REGION) -> Agent:
    sid = ctrl.sid

    def call(name: str, args: dict | None = None) -> str:
        res = ctrl.execute(name, args or {}, actor="claude")
        _stamp(name, res)
        return json.dumps(res, default=str)

    @tool
    def assess_capture() -> str:
        """Measure sharpness, glare, exposure and nose size in the latest capture."""
        return call("assess_capture")

    @tool
    def check_liveness() -> str:
        """Check the nose is live: depth across the burst, no screen grid, no replayed enrolled photo."""
        return call("check_liveness")

    @tool
    def identify_muzzle() -> str:
        """Identify the animal from its noseprint: DINOv2 shortlist, ALIKED + LightGlue matches, MAGSAC verification."""
        return call("identify_muzzle")

    @tool
    def lookup_animal(animal_id: str) -> str:
        """Read an animal's registry record."""
        return call("lookup_animal", {"animal_id": animal_id})

    @tool
    def check_continuity() -> str:
        """Check the video runs unbroken from the nose to the body (chain of custody)."""
        return call("check_continuity")

    @tool
    def calibrate_scale() -> str:
        """Fix metric scale from the printed ChArUco board seen in the walk-around; also reports registered keyframes."""
        return call("calibrate_scale")

    @tool
    def measure_body() -> str:
        """Measure orbit coverage, the seen arc of the girth loop, heart girth, body length and withers height."""
        return call("measure_body")

    @tool
    def estimate_weight() -> str:
        """Estimate live weight with a 90% interval."""
        return call("estimate_weight")

    @tool
    def value_animal() -> str:
        """Value the animal: P10 value at the floor price, market value, the price's source and date, and the expiry."""
        return call("value_animal")

    @tool
    def request_capture(code: str, message: str, gate: str = "") -> str:
        """Send the officer one concrete re-capture instruction. Use the code from policy.required_args."""
        return call("request_capture", {"code": code, "message": message, "gate": gate})

    @tool
    def request_approval(kind: str, note: str) -> str:
        """Ask a named human to approve: kind is credit_offer, claim_payout or adjudicate."""
        return call("request_approval", {"kind": kind, "note": note})

    @tool
    def write_registry(kind: str, approval_id: str = "") -> str:
        """Write the record. Credit and claim writes need an approved approval id."""
        return call("write_registry", {"kind": kind, "approval_id": approval_id})

    @tool
    def flag_fraud(reason: str) -> str:
        """Flag a duplicate, an owner mismatch, a presentation attack or a wrong animal for investigation."""
        return call("flag_fraud", {"reason": reason})

    @tool
    def finish(outcome: str, summary: str) -> str:
        """End the session once the policy reports done."""
        return call("finish", {"outcome": outcome, "summary": summary})

    model = BedrockModel(model_id=model_id, region_name=region, temperature=0.2, max_tokens=2000, streaming=False)
    return Agent(
        model=model,
        system_prompt=SYSTEM,
        tools=[assess_capture, check_liveness, identify_muzzle, lookup_animal, check_continuity, calibrate_scale,
               measure_body, estimate_weight, value_animal, request_capture, request_approval, write_registry,
               flag_fraud, finish],
        trace_attributes={"mombe.session": sid, "mombe.task": ctrl.L.task.value},
        callback_handler=None,
    )


def run_session(ctrl: Controller, **kw) -> dict:
    agent = build_agent(ctrl, **kw)
    task = ctrl.L.task.value
    extra = []
    if ctrl.L.claimed_owner:
        extra.append(f"claimed owner: {ctrl.L.claimed_owner}")
    if ctrl.L.policy_animal:
        extra.append(f"insured animal: {ctrl.L.policy_animal}")
    if ctrl.L.loan_threshold_usd:
        extra.append(f"requested loan: ${ctrl.L.loan_threshold_usd:.0f}")
    prompt = (f"New session {ctrl.sid}. Task: {task}. {'; '.join(extra)}. "
              f"Initial policy: {json.dumps(ctrl.policy_view())}. Begin.")
    result = agent(prompt)
    # The policy, not the model, decides when the session is over: nudge the
    # model while evidence is still being gathered.
    for _ in range(4):
        if ctrl.state.done:
            break
        result = agent("The policy reports the session is still open: "
                       f"{json.dumps(ctrl.policy_view())}. Continue.")
    usage = getattr(result, "metrics", None)
    return {"outcome": ctrl.state.outcome, "final_message": str(result),
            "usage": usage.accumulated_usage if usage else None}
