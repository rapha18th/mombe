"""Evidence policy: what the vision results allow the agent to do next.

The language model plans, explains and talks to the officer. It never decides
on its own that the evidence is good enough. After every tool call the policy
re-reads the evidence ledger, evaluates a fixed set of gates, and returns:

  * the gates and their measured values,
  * the actions the evidence now allows,
  * the action the evidence requires next (if any).

The agent's tool layer refuses any action outside the allowed set, so the
visual evidence changes what the system does next by construction. Every
evaluation is written to the decision trace.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

MAX_RECAPTURES_PER_GATE = 2
LIVENESS_RECAPTURES = 1         # one more live look, then a presentation attack is escalated

LIVE_INSTRUCTION = {"code": "LIVE_RECAPTURE",
                    "message": "Photograph the animal's own nose, then take the second photo a hand's width to the side."}
LIVENESS_REASONS = {
    "parallax": "the burst shows no depth: a flat print or screen explains it",
    "replay": "the photo matches an enrolled image pixel for pixel: a resent file",
    "moire": "a screen pixel grid shows in the spectrum",
    "glint": "the wet glints of a live nose are missing: paper holds highlights below its white",
    "range": "the light range is too narrow for a live nose: ink and paper bound a print",
}


class Task(str, Enum):
    ENROL = "enrol"        # new animal into the registry
    VERIFY = "verify"      # who is this animal, and who owns it
    VALUE = "value"        # identity + weight + value, for a loan
    CLAIM = "claim"        # insurance claim: carcass muzzle against the policy animal


# Which observation tool fills each gate.
GATE_TOOL = {"capture_quality": "assess_capture", "liveness": "check_liveness", "identity": "identify_muzzle",
             "ownership": "identify_muzzle", "chain_of_custody": "check_continuity", "metric_scale": "calibrate_scale",
             "body_coverage": "measure_body", "weight_interval": "estimate_weight"}

# Actions the agent can take. Tools map onto these names.
OBSERVE = {"assess_capture", "check_liveness", "identify_muzzle", "check_continuity", "calibrate_scale",
           "measure_body", "estimate_weight", "value_animal", "lookup_animal"}
ACT = {"request_capture", "request_approval", "write_registry", "flag_fraud", "finish"}


@dataclass
class Gate:
    name: str
    passed: bool | None            # None: evidence not collected yet
    value: object = None
    threshold: object = None
    reason: str = ""
    instruction: dict | None = None  # what to ask the officer when it fails


@dataclass
class PolicyState:
    task: str
    gates: list[Gate]
    allowed: list[str]
    required: str | None
    required_args: dict = field(default_factory=dict)
    done: bool = False
    outcome: str | None = None
    next_observation: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gates"] = [asdict(g) for g in self.gates]
        return d


@dataclass
class Ledger:
    """Everything the tools have reported in this session, newest last."""
    task: Task
    claimed_owner: str | None = None
    policy_animal: str | None = None          # CLAIM: the insured animal id
    loan_threshold_usd: float | None = None   # VALUE: amount the officer wants to lend
    ltv: float = 0.5
    quality: dict | None = None
    liveness: dict | None = None
    identity: dict | None = None
    registry: dict | None = None              # lookup result for the identified animal
    continuity: dict | None = None
    scale: dict | None = None
    body: dict | None = None
    weight: dict | None = None
    value: dict | None = None
    approval: dict | None = None
    written: bool = False
    fraud_flagged: bool = False
    recaptures: dict = field(default_factory=dict)  # gate name -> count
    reviews: dict = field(default_factory=dict)     # gate name -> human adjudication record
    awaiting_capture: int | None = None             # latest capture index when a re-capture was requested
    awaiting_processing: int | None = None          # capture whose walk-around the hub is still measuring

    def recapture_budget_left(self, gate: str) -> bool:
        return self.recaptures.get(gate, 0) < MAX_RECAPTURES_PER_GATE

    def to_dict(self) -> dict:
        d = asdict(self)
        d["task"] = self.task.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Ledger":
        return cls(**{**d, "task": Task(d["task"])})


def _quality_gate(L: Ledger) -> Gate:
    if L.quality is None:
        return Gate("capture_quality", None, reason="no capture assessed")
    issues = L.quality.get("issues", [])
    instr = (L.quality.get("instructions") or [None])[0]
    return Gate("capture_quality", not issues, issues, "no issues",
                "muzzle frames usable" if not issues else f"capture issues: {', '.join(issues)}", instr)


def _liveness_gate(L: Ledger) -> Gate:
    """Print, screen and replay attacks: the nose must be live before anything rests on it."""
    if L.liveness is None:
        return Gate("liveness", None, reason="liveness not checked")
    lv = L.liveness
    value = {"score": lv.get("score"), "failed_cue": lv.get("failed_cue"), "capture": lv.get("capture")}
    th = f"> {lv.get('threshold')}"
    same = (L.identity or {}).get("capture", lv.get("capture"))
    if same != lv.get("capture"):
        return Gate("liveness", False, value, th, "liveness was measured on another capture", LIVE_INSTRUCTION)
    if lv.get("live"):
        return Gate("liveness", True, value, th, "live nose: depth across the burst, no screen grid, no replayed file")
    why = LIVENESS_REASONS.get(lv.get("failed_cue"), "liveness evidence fell short")
    return Gate("liveness", False, value, th, f"presentation attack suspected: {why}", LIVE_INSTRUCTION)


def _identity_gate(L: Ledger) -> Gate:
    if L.identity is None:
        return Gate("identity", None, reason="identity not run")
    d = L.identity["decision"]
    th = L.identity.get("thresholds", {})
    value = {"decision": d, "score": L.identity["score"], "margin": L.identity["margin"],
             "animal_id": L.identity.get("animal_id")}
    if d == "new" and L.task != Task.ENROL and L.recaptures.get("identity", 0) < 1:
        return Gate("identity", False, value, th,
                    f"no registered animal reached {th.get('t_new')} verified inliers; one more look before calling it unregistered",
                    {"code": "STRAIGHT_ON", "message": "Photograph the nose straight on, filling the frame."})
    if d == "ambiguous":
        return Gate("identity", False, value, th,
                    f"verified inliers {L.identity['score']} sit between new ({th.get('t_new')}) and match ({th.get('t_match')})",
                    {"code": "SECOND_ANGLE", "message": "Take the nose again from slightly to the left, filling the frame."})
    return Gate("identity", True, value, th, f"identity decision: {d}")


def _ownership_gate(L: Ledger, identity: Gate) -> Gate:
    """Duplicate enrolment, claimed-owner mismatch, or wrong animal on a claim."""
    if identity.passed is not True:
        return Gate("ownership", None, reason="needs a firm identity first")
    d, aid = L.identity["decision"], L.identity.get("animal_id")
    if L.task == Task.ENROL and d == "match":
        owner = (L.registry or {}).get("owner")
        return Gate("ownership", False, {"registered_to": owner, "animal_id": aid}, "absent from registry",
                    f"this muzzle is already registered as {aid} to {owner}")
    if L.task in (Task.VERIFY, Task.VALUE) and d == "new":
        return Gate("ownership", False, {"animal_id": None}, "registered animal",
                    "animal is absent from the registry; enrol it before valuing")
    if L.task in (Task.VERIFY, Task.VALUE) and L.claimed_owner and L.registry:
        owner = L.registry.get("owner")
        if owner != L.claimed_owner:
            return Gate("ownership", False, {"registered_to": owner, "claimed_by": L.claimed_owner},
                        "claimed owner == registered owner", "claimed owner differs from the registered owner")
    if L.task == Task.CLAIM:
        if d != "match" or aid != L.policy_animal:
            return Gate("ownership", False, {"identified": aid, "insured": L.policy_animal}, "insured animal",
                        "carcass muzzle does not match the insured animal")
    return Gate("ownership", True, {"animal_id": aid}, reason="identity consistent with the registry")


def _continuity_gate(L: Ledger) -> Gate:
    if L.continuity is None:
        return Gate("chain_of_custody", None, reason="continuity not checked")
    ok = L.continuity.get("continuous", False)
    return Gate("chain_of_custody", ok, L.continuity.get("breaks"), "zero breaks",
                "one unbroken video from muzzle to body" if ok else "the video breaks between the muzzle and the body",
                None if ok else {"code": "ONE_TAKE", "message": "Record again in one take: nose first, then walk around the animal without stopping."})


def _scale_gate(L: Ledger) -> Gate:
    if L.scale is None:
        return Gate("metric_scale", None, reason="scale not calibrated")
    rel = L.scale.get("scale_rel_err", 1.0)
    ok = L.scale.get("board_frames", 0) >= 6 and rel <= 0.01
    return Gate("metric_scale", ok, {"rel_err": rel, "board_frames": L.scale.get("board_frames")}, "<= 1% from >= 6 frames",
                "board fixes metric scale" if ok else "scale too uncertain; volume error triples it",
                None if ok else {"code": "BOARD_CLOSER", "message": "Hold the printed board flat against the shoulder and keep it in view."})


def _coverage_gate(L: Ledger) -> Gate:
    if L.body is None:
        return Gate("body_coverage", None, reason="body not measured")
    cov = L.body.get("coverage", 0.0)
    ok = cov >= 0.7
    side = L.body.get("missing_side", "left")
    return Gate("body_coverage", ok, cov, ">= 0.70 of the torso",
                "torso seen from enough angles" if ok else f"torso coverage {cov:.2f}",
                None if ok else {"code": "SHOW_SIDE", "message": f"Walk slowly along the animal's {side} side."})


def _weight_gate(L: Ledger) -> Gate:
    if L.weight is None:
        return Gate("weight_interval", None, reason="weight not estimated")
    lo, hi, kg = L.weight["lo"], L.weight["hi"], L.weight["kg"]
    width = (hi - lo) / kg
    ok = width <= 0.12
    reason = f"90% interval {lo:.0f} to {hi:.0f} kg ({width:.0%} wide)"
    # Value of information: a second pass is worth it only when it can change
    # the lending decision, i.e. the value interval straddles the requested amount.
    if L.value and L.loan_threshold_usd:
        v_lo, v_hi = L.value["lo"] * L.ltv, L.value["hi"] * L.ltv
        if v_lo < L.loan_threshold_usd <= v_hi:
            if L.recapture_budget_left("weight_interval"):
                ok = False
                reason += f"; lendable range ${v_lo:.0f} to ${v_hi:.0f} straddles the ${L.loan_threshold_usd:.0f} request"
            else:
                reason += f"; still straddles after re-captures, so the offer is capped at the lower bound ${v_lo:.0f}"
                return Gate("weight_interval", True, {"kg": kg, "lo": lo, "hi": hi, "offer_cap_usd": round(v_lo)},
                            "<= 12% wide and decision-stable", reason)
    return Gate("weight_interval", ok, {"kg": kg, "lo": lo, "hi": hi}, "<= 12% wide and decision-stable", reason,
                None if ok else {"code": "SECOND_PASS", "message": "One more slow walk-around, keeping the board in view."})


def evaluate(L: Ledger) -> PolicyState:
    ident = _identity_gate(L)
    gates = [_quality_gate(L), _liveness_gate(L), ident, _ownership_gate(L, ident)]
    if L.task == Task.VALUE:
        gates += [_continuity_gate(L), _scale_gate(L), _coverage_gate(L), _weight_gate(L)]

    allowed = set(OBSERVE)
    state = PolicyState(L.task.value, gates, [], None)

    def need(action: str, **args) -> PolicyState:
        state.required, state.required_args = action, args
        state.allowed = sorted(allowed | {action})
        return state

    def finish(outcome: str) -> PolicyState:
        state.done, state.outcome = True, outcome
        state.required = "finish"
        state.required_args = {"outcome": outcome}
        state.allowed = sorted(allowed | {"finish"})
        return state

    by = {g.name: g for g in gates}

    # In the field a re-capture request parks the session until the officer uploads.
    if L.awaiting_capture is not None:
        return finish("awaiting_capture")
    if L.awaiting_processing is not None:
        return finish("awaiting_processing")

    # Fraud and ownership conflicts outrank everything after a firm identity.
    own = by["ownership"]
    if own.passed is False:
        if L.task == Task.ENROL or L.task == Task.CLAIM or "claimed_by" in (own.value or {}):
            if not L.fraud_flagged:
                return need("flag_fraud", reason=own.reason, evidence=own.value)
            if L.approval is None:
                return need("request_approval", kind="adjudicate", reason=own.reason)
            return finish("escalated")
        return finish("not_registered")

    # A failed liveness check on usable frames outranks the remaining gates: no
    # identity, registry write or approval rests on a nose that may be a print,
    # a screen or a replayed file. One live re-capture, then a fraud review.
    # A reviewer cannot wave it through inside the session.
    live = by["liveness"]
    usable = by["capture_quality"].passed or (L.reviews.get("capture_quality") or {}).get("status") == "approved"
    if live.passed is False and usable:
        if L.recaptures.get("liveness", 0) < LIVENESS_RECAPTURES:
            return need("request_capture", gate="liveness", **live.instruction)
        if not L.fraud_flagged:
            return need("flag_fraud", reason=live.reason, evidence=live.value)
        if L.approval is None:
            return need("request_approval", kind="adjudicate", reason=live.reason)
        return finish("escalated")

    # Walk the gates in order. The first unmet gate with an officer instruction
    # becomes the next required action while the re-capture budget lasts.
    for g in gates:
        if g.passed is None:
            break
        if g.passed is False and g.instruction:
            review = L.reviews.get(g.name)
            if review and review.get("status") == "approved":
                g.passed, g.reason = True, g.reason + f" (accepted by {review.get('decided_by', 'reviewer')})"
                continue
            if review and review.get("status") == "pending":
                return finish("awaiting_review")
            if review:
                return finish("declined")
            if L.recapture_budget_left(g.name):
                return need("request_capture", gate=g.name, **g.instruction)
            return need("request_approval", kind="adjudicate", gate=g.name,
                        reason=f"{g.name}: {g.reason} after {MAX_RECAPTURES_PER_GATE} re-captures")

    pending = [g for g in gates if g.passed is None]
    if pending:
        state.allowed = sorted(allowed | {"request_capture"})
        state.next_observation = GATE_TOOL[pending[0].name]
        return state

    # All gates pass. Task-specific completion.
    if L.task == Task.ENROL:
        if not L.written:
            allowed.add("write_registry")
            return need("write_registry", kind="enrolment")
        return finish("enrolled")
    if L.task == Task.VERIFY:
        return finish("verified")
    if L.task == Task.VALUE and L.value is None:
        state.allowed = sorted(allowed)
        state.next_observation = "value_animal"
        return state
    if L.task in (Task.VALUE, Task.CLAIM):
        kind = "credit_offer" if L.task == Task.VALUE else "claim_payout"
        if L.approval is None:
            cap = (by.get("weight_interval").value or {}).get("offer_cap_usd") if "weight_interval" in by else None
            return need("request_approval", kind=kind, **({"offer_cap_usd": cap} if cap else {}))
        if L.approval.get("status") == "approved" and not L.written:
            allowed.add("write_registry")
            return need("write_registry", kind=kind, approval_id=L.approval.get("id"))
        if L.approval.get("status") == "pending":
            return finish("awaiting_approval")
        return finish("approved" if L.approval.get("status") == "approved" else "declined")
    return state
