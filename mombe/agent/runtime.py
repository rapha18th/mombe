"""Session controller, decision trace and the three drivers.

Every action, whoever chooses it, goes through Controller.execute:

  1. the policy checks the action is allowed by the evidence so far;
  2. the tool runs (OpenCV 5 vision tools, or an action on the registry);
  3. the ledger absorbs the result;
  4. the policy re-evaluates, and the trace records whether the new evidence
     changed the next required action.

Drivers:
  * LLM agent (Claude on Bedrock via Strands), see agent.py;
  * Autopilot: follows the policy deterministically, the closed-loop reference;
  * OpenLoop: one pass, no re-captures, argmax identity; the baseline the
    closed loop must beat.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from typing import TYPE_CHECKING

from mombe.agent.policy import ACT, OBSERVE, Ledger, Task, evaluate

if TYPE_CHECKING:  # the deployed agent ships without OpenCV; these are type hints only
    from mombe.server.store import LocalRegistry
    from mombe.server.tools import VisionTools

CAPTURE_EVIDENCE = ("quality", "liveness", "identity", "registry", "continuity", "scale", "body", "weight", "value")



BODY_TOOLS = {"check_continuity": "continuity", "calibrate_scale": "scale",
              "measure_body": "body", "estimate_weight": "weight"}

class Trace:
    def __init__(self, path: Path | None = None, sink=None, session_id: str | None = None):
        self.events: list[dict] = []
        self.path = path
        self.sink, self.sid = sink, session_id
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")

    def add(self, **ev) -> dict:
        ev["t"] = round(time.time(), 3)
        ev["step"] = len(self.events)
        self.events.append(ev)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ev, default=str) + "\n")
        if self.sink:
            self.sink(self.sid, json.loads(json.dumps(ev, default=str)))
        return ev


def _summary(name: str, out: dict) -> dict:
    """The fields of a tool result that matter for decisions, for the trace."""
    keys = {
        "assess_capture": ("capture", "issues", "usable_frames", "best_frame"),
        "check_liveness": ("capture", "live", "score", "threshold", "failed_cue"),
        "identify_muzzle": ("decision", "animal_id", "score", "margin", "evidence_image"),
        "check_continuity": ("continuous", "breaks"),
        "calibrate_scale": ("board_frames", "scale_rel_err"),
        "measure_body": ("coverage", "girth_cm", "length_cm"),
        "estimate_weight": ("kg", "lo", "hi"),
        "value_animal": ("usd", "lo", "hi", "price_per_kg_usd"),
    }.get(name)
    if not keys:
        return out
    return {k: out.get(k) for k in keys if k in out}


class Controller:
    def __init__(self, session_id: str, tools: "VisionTools", registry: "LocalRegistry",
                 sim=None, trace_path: Path | None = None, trace_sink=None):
        self.sid = session_id
        self.tools, self.registry, self.sim = tools, registry, sim
        s = tools.store.load(session_id)
        if s.ledger:
            self.L = Ledger.from_dict(s.ledger)
        else:
            self.L = Ledger(Task(s.task), claimed_owner=s.claimed_owner, policy_animal=s.policy_animal,
                            loan_threshold_usd=s.loan_threshold_usd)
        self.trace = Trace(trace_path, trace_sink, session_id)
        resumed = self._resume(s)
        self.state = evaluate(self.L)
        self.trace.add(kind="resume" if resumed else "start", task=s.task, session=session_id,
                       resumed=resumed, policy=self.state.to_dict())

    def _resume(self, s) -> dict | None:
        """Absorb what happened while the session was parked."""
        L, what = self.L, {}
        if L.awaiting_processing is not None:
            # woken by the hub (or a new upload): ask again; a cached result returns at once
            what["processing_check"] = L.awaiting_processing
            L.awaiting_processing = None
        if L.awaiting_capture is not None and s.latest and s.latest.index > L.awaiting_capture:
            what["new_capture"] = s.latest.index
            L.awaiting_capture = None
        for gate, rv in list(L.reviews.items()):
            cur = self.registry.approval(rv["id"]) if rv.get("status") == "pending" else None
            if cur and cur["status"] != "pending":
                L.reviews[gate] = cur
                what[f"review_{gate}"] = cur["status"]
        if L.approval and L.approval.get("status") == "pending":
            cur = self.registry.approval(L.approval["id"])
            if cur and cur["status"] != "pending":
                L.approval = cur
                what["approval"] = cur["status"]
        return what or None

    # -- the single gate every action passes -------------------------------
    def execute(self, action: str, args: dict | None = None, actor: str = "agent") -> dict:
        args = args or {}
        before = self.state
        if action not in OBSERVE | ACT:
            return self._refuse(action, args, actor, f"unknown action {action}")
        if action in ACT and action not in before.allowed:
            return self._refuse(action, args, actor,
                                f"blocked by evidence policy; required next: {before.required} {before.required_args}")
        # The evidence fixes what kind of approval or write is due; a driver cannot swap it,
        # e.g. turn a fraud review into a credit approval.
        want = before.required_args.get("kind") if action == before.required else None
        if action in ("request_approval", "write_registry") and want and args.get("kind") and args["kind"] != want:
            return self._refuse(action, args, actor, f"blocked by evidence policy: {action} kind must be {want}")

        out = self._run(action, args)
        self.state = evaluate(self.L)
        changed = (before.required, before.required_args) != (self.state.required, self.state.required_args)
        self.trace.add(kind="action" if action in ACT else "observation", actor=actor, name=action, args=args,
                       result=_summary(action, out), required_before=before.required,
                       required_after=self.state.required, required_args_after=self.state.required_args,
                       changed_next_action=changed, gates={g.name: g.passed for g in self.state.gates},
                       elapsed_ms=out.get("elapsed_ms"))
        return {"result": out, "policy": self.policy_view()}

    def persist(self) -> None:
        s = self.tools.store.load(self.sid)
        s.ledger = self.L.to_dict()
        out = self.state.outcome or ""
        s.status = "waiting" if out.startswith("awaiting") else ("done" if self.state.done else "open")
        self.tools.store.save(s)

    def policy_view(self) -> dict:
        s = self.state
        return {"required_next": s.required, "required_args": s.required_args, "suggested_observation": s.next_observation,
                "allowed_actions": s.allowed, "done": s.done, "outcome": s.outcome,
                "gates": [{"gate": g.name, "passed": g.passed, "reason": g.reason} for g in s.gates]}

    def _refuse(self, action, args, actor, why) -> dict:
        self.trace.add(kind="refused", actor=actor, name=action, args=args, reason=why)
        return {"error": why, "policy": self.policy_view()}

    def _run(self, action: str, args: dict) -> dict:
        L, t, sid = self.L, self.tools, self.sid
        if action == "assess_capture":
            out = t.assess_capture(sid)
            L.quality = out
        elif action == "check_liveness":
            out = t.check_liveness(sid)
            L.liveness = out
        elif action == "identify_muzzle":
            out = t.identify_muzzle(sid)
            L.identity = out
            L.registry = self.registry.get(out["animal_id"]) if out.get("animal_id") else None
        elif action == "lookup_animal":
            out = t.lookup_animal(args["animal_id"])
            L.registry = out["record"]
        elif action in BODY_TOOLS:
            out = getattr(t, action)(sid)
            if out.get("pending"):
                # the hub is still measuring the walk-around: park until it wakes us
                L.awaiting_processing = out.get("capture", -1)
            else:
                setattr(L, BODY_TOOLS[action], out)
        elif action == "value_animal":
            if not L.weight:
                return {"error": "estimate_weight first"}
            out = t.value_animal(sid, L.weight); L.value = out
        elif action == "request_capture":
            gate = args.get("gate") or self.state.required_args.get("gate") or "capture_quality"
            L.recaptures[gate] = L.recaptures.get(gate, 0) + 1
            instr = {k: args.get(k) or self.state.required_args.get(k) for k in ("code", "message")}
            s = t.store.load(sid)
            s.instructions.append({**instr, "gate": gate, "t": time.time()})
            t.store.save(s)
            for k in CAPTURE_EVIDENCE:
                setattr(L, k, None)
            s.ledger = L.to_dict()
            t.store.save(s)
            if self.sim:
                c = self.sim.on_request_capture(sid, instr)
                out = {"instruction": instr, "new_capture": c.index, "mode": "simulated officer"}
            else:
                L.awaiting_capture = s.latest.index if s.latest else -1
                out = {"instruction": instr, "status": "sent to officer; session waits for the next upload"}
        elif action == "request_approval":
            kind = args.get("kind") or self.state.required_args.get("kind")
            summary = {"task": L.task.value, "identity": _summary("identify_muzzle", L.identity or {}),
                       "weight": L.weight, "value": L.value, "reason": args.get("reason"),
                       "agent_note": args.get("note")}
            gate = args.get("gate") or self.state.required_args.get("gate")
            if args.get("offer_cap_usd") or self.state.required_args.get("offer_cap_usd"):
                summary["offer_cap_usd"] = args.get("offer_cap_usd") or self.state.required_args.get("offer_cap_usd")
            ap = self.registry.request_approval(sid, kind, {**summary, "gate": gate})
            if self.sim:
                ap = self.registry.decide(ap["id"], self.sim.approver(sid, ap), by="simulated-officer")
            if kind == "adjudicate" and gate:
                L.reviews[gate] = ap
            else:
                L.approval = ap
            out = {"approval": ap}
        elif action == "write_registry":
            kind = args.get("kind") or self.state.required_args.get("kind")
            if kind in ("credit_offer", "claim_payout"):
                ap = self.registry.approval(args.get("approval_id") or (L.approval or {}).get("id"))
                if not ap or ap["status"] != "approved":
                    return {"error": "write refused: no approved approval on record"}
            animal_id = (L.identity or {}).get("animal_id") or f"MB-{sid}"
            # A write records only what its action owns: ownership is set at
            # enrolment and never by a loan or a claim.
            rec = {"last_session": sid, f"last_{kind}": sid}
            if kind == "enrolment":
                rec["owner"] = L.claimed_owner
            if L.weight:
                rec.update(weight_kg=L.weight["kg"], weight_interval=[L.weight["lo"], L.weight["hi"]])
            if L.value:
                rec.update(value_usd=L.value["usd"], value_interval=[L.value["lo"], L.value["hi"]])
            self.registry.put(animal_id, rec)
            L.written = True
            out = {"written": animal_id, "kind": kind}
            if kind == "enrolment" and hasattr(t, "enrol_muzzle"):
                # the new noseprint joins the gallery, so a second enrolment of this animal is caught
                out["gallery"] = t.enrol_muzzle(sid, animal_id)
        elif action == "flag_fraud":
            L.fraud_flagged = True
            out = {"flagged": True, "reason": args.get("reason") or self.state.required_args.get("reason")}
        elif action == "finish":
            out = {"finished": True, "outcome": self.state.outcome or args.get("outcome")}
        else:
            out = {"error": f"unhandled {action}"}
        return out

    @property
    def done(self) -> bool:
        return self.state.done


# -- deterministic drivers -------------------------------------------------
def autopilot(ctrl: Controller, max_steps: int = 40) -> dict:
    for _ in range(max_steps):
        s = ctrl.state
        if s.done:
            ctrl.execute("finish", {"outcome": s.outcome}, actor="autopilot")
            break
        if s.required:
            ctrl.execute(s.required, dict(s.required_args), actor="autopilot")
        elif s.next_observation:
            ctrl.execute(s.next_observation, {}, actor="autopilot")
        else:
            break
    return {"outcome": ctrl.state.outcome, "identity": ctrl.L.identity, "steps": len(ctrl.trace.events)}


def open_loop(ctrl: Controller) -> dict:
    """Single pass. Ignores quality issues, takes the top candidate, never re-captures."""
    L = ctrl.L
    for obs in ("assess_capture", "identify_muzzle"):
        ctrl.execute(obs, actor="open_loop")
    ident = L.identity or {}
    top = (ident.get("shortlist") or [{}])[0].get("animal_id")
    decided = ident.get("animal_id") or (top if ident.get("decision") == "ambiguous" else None)
    if L.task == Task.VALUE:
        for obs in ("check_continuity", "calibrate_scale", "measure_body", "estimate_weight", "value_animal"):
            ctrl.execute(obs, actor="open_loop")
    return {"outcome": "open_loop", "decided_animal": decided, "decision": ident.get("decision"),
            "steps": len(ctrl.trace.events)}
