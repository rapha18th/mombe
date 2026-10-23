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

import math
from dataclasses import asdict, dataclass, field
from enum import Enum

from mombe.agent.valuation import is_current, today

MAX_RECAPTURES_PER_GATE = 2

# -- body measurement evidence: each floor with the data it came from ------------------------
# Registration health reads two signals. The 14 synthetic sweep takes (stock and COOL, 7 animals,
# input S, truth G) are the data; eval/body_gates.py prints both.
# 1. Registered fraction of the walk-around's keyframes. Every sweep take registered 48 of 48,
#    COOL sweep_21 included, so this floor cannot separate it and the sweep sets no lower bound.
#    The floor leaves room for the nose opening (up to a tenth of the keyframes) and the
#    continuity check's tolerance of one dropped frame in ten after it.
REGISTERED_FRAC_FLOOR = 0.80
# 2. Length agreement: the fitted trunk length over the direct cut, two readings off one surface.
#    A reconstruction that holds together gives the same animal twice; one that failed to close
#    stretches the fitted trunk. 13 good takes read 0.98 to 1.13; COOL sweep_21 (girth -11.2%,
#    length +25.8%) reads 1.34. The band refuses it and keeps the rest. Small N: one bad take.
LENGTH_AGREEMENT = (0.80, 1.25)
# Fraction of 30-degree bearing sectors around the animal with a registered camera.
ORBIT_COVERAGE_FLOOR = 0.70
# Observed arc fraction: share of the 5-degree bins around the girth loop with surface in them.
# Sweep takes (input S, truth G): 13 good takes read 0.639 to 0.861 (girth error -2% to +7%);
# COOL sweep_21 read 0.542 (girth -11.2%, length +25.8%). 0.60 refuses it and keeps the rest.
LOOP_COVERAGE_FLOOR = 0.60

# Plausibility envelope: proportions a measured body must keep. The ratios are scale-free, so
# they catch what a scale check cannot: a girth slice on the neck, a bloated surface, a bent orbit.
# Bounds hold every reference below with a margin; they are wide on purpose and catch gross
# failures. kg per cubic metre of withers height stands in for weight per height: weight grows
# with the cube of size, so kg per cm alone has no fixed range from calves to oxen. Evidence and
# percentiles: eval/body_gates.py.
#   A  Acme AI B4 side keypoints, 1,944 Bangladeshi cattle, labels 69-307 kg (CC BY 4.0, data/acme),
#      P0.5 to P99.5: shoulder-to-pin length / height 0.90-1.27; girth proxy (ellipse through chest
#      depth and rear width) / height 1.12-1.53, / length 1.08-1.42. Weight per height cubed on 300
#      of them spans 0.56-1.72 times its median; 93-285 kg/m^3 at the sticker size (11.5 cm) that
#      makes Schaeffer's formula return the labels
#   K  Lukuyu et al. 2016, SpringerPlus 5:63, Table 2: 452 Kenyan crossbred dairy cows and heifers on
#      a scale; group means: girth / withers 1.30-1.37, length / withers 0.93-0.99, girth / length
#      1.33-1.42, 149-198 kg per withers m^3
#   H  Bene et al. 2007, Arch. Tierz. 50:363, Tables 2-3: 18 Hereford cows, 517 kg (SD 91), withers
#      130.3 cm, length 132.6 cm: length / withers 1.02, 234 kg per withers m^3
#   S  synthetic sweep animals (truth G): girth / length 1.08-1.27; COOL sweep_21 measured 0.895
ENVELOPE = {
    #  ratio                lo     hi     sources
    "girth_withers":      (1.05, 1.75),  # A, K
    "length_withers":     (0.80, 1.40),  # A, K, H
    "girth_length":       (1.00, 1.75),  # A, K, S
    "kg_per_withers_m3":  (85., 400.),   # A, K, H; a Hereford cow 3 SD heavy at mean withers reads 357
}


class Task(str, Enum):
    ENROL = "enrol"        # new animal into the registry
    VERIFY = "verify"      # who is this animal, and who owns it
    VALUE = "value"        # identity + weight + value, for a loan
    CLAIM = "claim"        # insurance claim: carcass muzzle against the policy animal


# Which observation tool fills each gate.
GATE_TOOL = {"capture_quality": "assess_capture", "identity": "identify_muzzle", "ownership": "identify_muzzle",
             "chain_of_custody": "check_continuity", "registration_health": "calibrate_scale",
             "metric_scale": "calibrate_scale", "body_coverage": "measure_body",
             "plausibility": "estimate_weight",      # pending only once the body is measured
             "weight_interval": "estimate_weight"}

# Actions the agent can take. Tools map onto these names.
OBSERVE = {"assess_capture", "identify_muzzle", "check_continuity", "calibrate_scale",
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


def _registration_gate(L: Ledger) -> Gate:
    """A reconstruction that lost part of the orbit, or failed to close it, measures part of an animal."""
    if L.scale is None:
        return Gate("registration_health", None, reason="reconstruction not run")
    reg, n = L.scale.get("registered_frames") or 0, L.scale.get("frames") or 0
    frac = reg / n if n else 0.0
    lo, hi = LENGTH_AGREEMENT
    th = f">= {REGISTERED_FRAC_FLOOR:.2f} of keyframes registered; fitted length / direct cut {lo:g} to {hi:g}"
    value = {"registered": reg, "frames": n, "fraction": round(frac, 3)}
    instr = {"code": "REORBIT", "message": "Walk all the way around the animal again at a steady pace, keeping the whole body in frame."}
    if frac < REGISTERED_FRAC_FLOOR:
        return Gate("registration_health", False, value, th, f"{reg} of {n} keyframes registered into one reconstruction", instr)
    ln, cut = (L.body or {}).get("length_cm"), (L.body or {}).get("length_cut_cm")
    if ln and cut:                      # checked once the body is measured
        value["length_agreement"] = round(ln / cut, 3)
        if not lo <= ln / cut <= hi:
            return Gate("registration_health", False, value, th,
                        f"the reconstruction disagrees with itself: fitted length {ln:.0f} cm against a direct cut of {cut:.0f} cm", instr)
    return Gate("registration_health", True, value, th, f"{reg} of {n} keyframes registered into one reconstruction")


def _scale_gate(L: Ledger) -> Gate:
    if L.scale is None:
        return Gate("metric_scale", None, reason="scale not calibrated")
    rel = L.scale.get("scale_rel_err", 1.0)      # the calibrated claim; the raw jackknife rides beside it
    raw = L.scale.get("scale_rel_err_raw")
    ok = L.scale.get("board_frames", 0) >= 6 and rel <= 0.01
    reason = "board fixes metric scale" if ok else "scale too uncertain; volume error triples it"
    if raw is not None and raw <= 0.01 < rel:
        reason += f" (raw claim {raw:.2%} reads {rel:.2%} once calibrated)"
    return Gate("metric_scale", ok, {"rel_err": rel, "raw_rel_err": raw, "board_frames": L.scale.get("board_frames")},
                "<= 1% (calibrated) from >= 6 frames", reason,
                None if ok else {"code": "BOARD_CLOSER", "message": "Hold the printed board flat against the shoulder and keep it in view."})


# What fills an unseen arc of the girth loop.
ARC_PASS = {
    "belly": {"code": "LOW_CHEST", "message": "Crouch at the left elbow and film under the chest for 10 seconds, board in view."},
    "top": {"code": "HIGH_PASS", "message": "Hold the phone above the back and walk slowly along the animal once more."},
}


def _coverage_gate(L: Ledger) -> Gate:
    if L.body is None:
        return Gate("body_coverage", None, reason="body not measured")
    cov = L.body.get("coverage", 0.0)
    loop = L.body.get("girth_loop_coverage") or 0.0
    gap, arc = L.body.get("girth_gap_deg"), L.body.get("girth_gap_arc")
    value = {"orbit": cov, "girth_loop": loop, "gap_deg": gap, "gap_arc": arc}
    th = f">= {ORBIT_COVERAGE_FLOOR:.2f} of the orbit and >= {LOOP_COVERAGE_FLOOR:.2f} of the girth loop"
    if cov < ORBIT_COVERAGE_FLOOR:
        side = L.body.get("missing_side") or "left"
        return Gate("body_coverage", False, value, th, f"orbit coverage {cov:.2f}",
                    {"code": "SHOW_SIDE", "message": f"Walk slowly along the animal's {side} side."})
    if loop < LOOP_COVERAGE_FLOOR:
        where = f"; the largest unseen arc ({gap:.0f} deg) is the {arc}" if gap is not None and arc else ""
        side = arc if arc in ("left", "right") else (L.body.get("missing_side") or "left")
        instr = ARC_PASS.get(arc) or {"code": "SHOW_SIDE", "message": f"Walk slowly along the animal's {side} side, board in view."}
        return Gate("body_coverage", False, value, th, f"girth loop seen {loop:.2f}{where}", instr)
    return Gate("body_coverage", True, value, th, "torso and girth loop seen from enough angles")


def _plausibility_gate(L: Ledger) -> Gate:
    """Body proportions inside the envelope of real cattle, or the weight is refused."""
    if L.body is None:
        return Gate("plausibility", None, reason="body not measured")
    b = L.body

    def num(k):
        v = b.get(k)
        return float(v) if isinstance(v, (int, float)) and math.isfinite(v) and v > 0 else None
    g, ln, w = num("girth_cm"), num("length_cm"), num("withers_cm")
    r = {}
    if g and w:
        r["girth_withers"] = g / w
    if ln and w:
        r["length_withers"] = ln / w
    if g and ln:
        r["girth_length"] = g / ln
    kg = (L.weight or {}).get("kg")
    if kg and w:
        r["kg_per_withers_m3"] = kg / (w / 100) ** 3
    value = {k: round(v, 3) for k, v in r.items()}
    instr = {"code": "RECHECK_BODY", "message": "The measurements fall outside cattle proportions. Walk around once more, slowly, with the board flat on the shoulder."}
    if "girth_length" not in r:
        return Gate("plausibility", False, value, "girth and length measured", "girth or length missing", instr)
    out = [k for k, v in r.items() if not ENVELOPE[k][0] <= v <= ENVELOPE[k][1]]
    if out:
        why = ", ".join(f"{k} {r[k]:.2f} outside {ENVELOPE[k][0]:g} to {ENVELOPE[k][1]:g}" for k in out)
        return Gate("plausibility", False, value, "inside the cattle envelope", f"weight refused: {why}", instr)
    if L.weight is None:
        return Gate("plausibility", None, value, reason="proportions inside the envelope; weight not estimated")
    return Gate("plausibility", True, value, "inside the cattle envelope", "body proportions and weight per height plausible")


def _weight_gate(L: Ledger) -> Gate:
    if L.weight is None:
        return Gate("weight_interval", None, reason="weight not estimated")
    if not L.weight.get("kg"):          # the body pipeline answered without a weight
        return Gate("weight_interval", False, None, "<= 12% wide and decision-stable",
                    f"no weight: {L.weight.get('reason') or 'measurement failed'}",
                    {"code": "SECOND_PASS", "message": "One more slow walk-around, keeping the board in view."})
    lo, hi, kg = L.weight["lo"], L.weight["hi"], L.weight["kg"]
    width = (hi - lo) / kg
    ok = width <= 0.12
    reason = f"90% interval {lo:.0f} to {hi:.0f} kg ({width:.0%} wide)"
    # Value of information: a second pass is worth it only when it can change
    # the lending decision, i.e. the value interval straddles the requested amount.
    # Loans are sized on the P10 value at the floor price (mombe.agent.valuation).
    if L.value and L.loan_threshold_usd:
        v_lo, v_hi = L.value.get("p10_usd", L.value["lo"]) * L.ltv, L.value["hi"] * L.ltv
        if L.loan_threshold_usd > v_hi:
            # beyond what even the top of the interval supports: no re-capture can reach it
            reason += f"; the ${L.loan_threshold_usd:.0f} request exceeds the lendable range, so the offer is capped at P10 ${v_lo:.0f}"
            return Gate("weight_interval", ok, {"kg": kg, "lo": lo, "hi": hi, "offer_cap_usd": round(v_lo)},
                        "<= 12% wide and decision-stable", reason,
                        None if ok else {"code": "SECOND_PASS", "message": "One more slow walk-around, keeping the board in view."})
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
    gates = [_quality_gate(L), ident, _ownership_gate(L, ident)]
    if L.task == Task.VALUE:
        gates += [_continuity_gate(L), _registration_gate(L), _scale_gate(L), _coverage_gate(L),
                  _plausibility_gate(L), _weight_gate(L)]

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
    # A lapsed valuation is valued again before anyone is asked to lend on it.
    # One that lapses on the day it is made rests on stale price quotes: the price book needs new quotes.
    if L.task == Task.VALUE and (L.value is None or (L.approval is None and not is_current(L.value))):
        if L.value is not None and L.value.get("valued_on") == today().isoformat():
            return finish("price_stale")
        state.allowed = sorted(allowed)
        state.next_observation = "value_animal"
        return state
    if L.task in (Task.VALUE, Task.CLAIM):
        kind = "credit_offer" if L.task == Task.VALUE else "claim_payout"
        if L.approval is None:
            cap = (by.get("weight_interval").value or {}).get("offer_cap_usd") if "weight_interval" in by else None
            if L.task == Task.VALUE and L.loan_threshold_usd:
                # whatever path the evidence took, a request above the P10 lendable reaches the person capped
                lendable = round(L.value.get("p10_usd", L.value["lo"]) * L.ltv)
                if L.loan_threshold_usd > lendable:
                    cap = min(cap or lendable, lendable)
            return need("request_approval", kind=kind, **({"offer_cap_usd": cap} if cap else {}))
        if L.approval.get("status") == "approved" and not L.written:
            allowed.add("write_registry")
            return need("write_registry", kind=kind, approval_id=L.approval.get("id"))
        if L.approval.get("status") == "pending":
            return finish("awaiting_approval")
        return finish("approved" if L.approval.get("status") == "approved" else "declined")
    return state
