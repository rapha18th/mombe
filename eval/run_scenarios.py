"""Scenario suite: does closing the loop change outcomes?

Each scenario is one field session on a held-out test animal. Drivers:
  open_loop  one pass, argmax identity, no re-captures (baseline)
  autopilot  follows the evidence policy deterministically (closed loop)
  claude     Claude Sonnet 5 on Bedrock via Strands, same policy and tools

Usage: python eval/run_scenarios.py --drivers open_loop autopilot [--per-kind 6] [--kinds ...]
Writes eval/out/<run>/results.jsonl, traces/, and summary.json.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.dataset import build_gallery, field_paths, make_split  # noqa: E402
from mombe.agent.runtime import Controller, autopilot, open_loop  # noqa: E402
from mombe.server.store import LocalRegistry, LocalStore  # noqa: E402
from mombe.server.tools import VisionTools  # noqa: E402
from mombe.sim.field import FieldSim, Scenario, SimBodyProvider  # noqa: E402
from mombe.vision.muzzle import MuzzleEngine  # noqa: E402

OWNERS = ["Moyo", "Ncube", "Dube", "Sibanda", "Chikore", "Mutasa", "Nyathi", "Banda", "Zulu", "Mlambo"]
KINDS = ["verify_clean", "verify_glare", "verify_blur", "verify_partial", "enrol_new", "enrol_duplicate",
         "owner_mismatch", "claim_true", "claim_wrong", "value_clean", "value_board_far", "value_splice",
         "value_low_coverage", "value_straddle"]


def owner_of(a: str) -> str:
    return OWNERS[int(a.split("_")[-1]) % len(OWNERS)]


def make_scenarios(split, per_kind: int, kinds: list[str], seed: int = 11) -> list[Scenario]:
    rng = random.Random(seed)
    test, unreg = list(split.test), list(split.unregistered)
    out = []
    for kind in kinds:
        for n in range(per_kind):
            a = rng.choice(test)
            sid = f"{kind}-{n:02d}"
            kg = rng.uniform(300, 560)
            if kind == "verify_clean":
                sc = Scenario(sid, "verify", a, "verified", claimed_owner=owner_of(a))
            elif kind in ("verify_glare", "verify_blur", "verify_partial"):
                sc = Scenario(sid, "verify", a, "verified", defects=[kind.split("_")[1]], claimed_owner=owner_of(a))
            elif kind == "enrol_new":
                sc = Scenario(sid, "enrol", rng.choice(unreg), "enrolled", claimed_owner="Moyo")
            elif kind == "enrol_duplicate":
                sc = Scenario(sid, "enrol", a, "escalated", claimed_owner="Stranger", approve=False,
                              note="registered animal presented as new under another owner")
            elif kind == "owner_mismatch":
                sc = Scenario(sid, "verify", a, "escalated", claimed_owner="Stranger", approve=False)
            elif kind == "claim_true":
                sc = Scenario(sid, "claim", a, "approved", policy_animal=a)
            elif kind == "claim_wrong":
                other = rng.choice([x for x in test if x != a])
                sc = Scenario(sid, "claim", a, "escalated", policy_animal=other, approve=False)
            else:
                body = {"true_kg": kg}
                fix = {}
                loan = round(kg * 2.0 * 0.5 * 0.6)                       # comfortably lendable
                if kind == "value_board_far":
                    body.update(board_frames=3, scale_rel_err=0.035); fix = {"board_frames": 16, "scale_rel_err": 0.004}
                elif kind == "value_splice":
                    body.update(continuous=False); fix = {"continuous": True}
                elif kind == "value_low_coverage":
                    body.update(coverage=0.45, missing_side=rng.choice(["left", "right"])); fix = {"coverage": 0.86}
                elif kind == "value_straddle":
                    loan = round(kg * 2.0 * 0.5 * 1.01)                  # sits inside the lendable interval
                    body.update(board_frames=8, scale_rel_err=0.008)
                sc = Scenario(sid, "value", a, "approved", claimed_owner=owner_of(a), loan_threshold_usd=loan,
                              body=body, body_fix=fix)
            out.append(sc)
    return out


def judge(sc: Scenario, driver: str, res: dict, ctrl: Controller) -> dict:
    L = ctrl.L
    ident = L.identity or {}
    if driver == "open_loop":
        decided = res.get("decided_animal")
        if sc.task in ("verify", "value"):
            correct = decided == sc.animal and not (sc.expected == "escalated")
        elif sc.task == "enrol":
            correct = res.get("decision") == "new" and sc.expected == "enrolled"
        else:  # claim: open loop pays whenever it finds any match
            correct = (decided == sc.policy_animal) == (sc.expected == "approved")
        outcome = "open_loop"
    else:
        outcome = ctrl.state.outcome
        correct = outcome == sc.expected
        if correct and sc.task in ("verify", "value", "claim") and sc.expected in ("verified", "approved"):
            correct = ident.get("animal_id") == sc.animal
    lent_info = {}
    if sc.task == "value":
        # A loan decision is correct when the money lent stays within the
        # animal's true lendable value and the weighed body is provably the
        # identified animal.
        price, ltv = ctrl.tools.price, L.ltv
        true_lendable = sc.body["true_kg"] * price * ltv
        custody = (L.continuity or {}).get("continuous", True)
        if driver == "open_loop":
            kg = (L.weight or {}).get("kg", 0)
            lent = sc.loan_threshold_usd if kg * price * ltv >= sc.loan_threshold_usd else 0
            id_ok = res.get("decided_animal") == sc.animal
        else:
            ap = (L.approval or {}).get("summary", {})
            lent = 0 if ctrl.state.outcome != "approved" else min(sc.loan_threshold_usd, ap.get("offer_cap_usd") or sc.loan_threshold_usd)
            id_ok = ident.get("animal_id") == sc.animal
        correct = bool(id_ok and custody and 0 < lent <= true_lendable * 1.001)
        lent_info = {"lent_usd": round(lent), "true_lendable_usd": round(true_lendable)}
    r = {**lent_info, "scenario": sc.id, "kind": sc.id.rsplit("-", 1)[0], "driver": driver, "expected": sc.expected,
         "outcome": outcome, "correct": bool(correct), "recaptures": sum(L.recaptures.values()),
         "steps": len(ctrl.trace.events), "identity": {k: ident.get(k) for k in ("decision", "animal_id", "score", "margin")},
         "true_animal": sc.animal, "refusals": sum(1 for e in ctrl.trace.events if e["kind"] == "refused"),
         "loop_changes": sum(1 for e in ctrl.trace.events if e.get("changed_next_action")),
         "tool_ms": sum(e.get("elapsed_ms") or 0 for e in ctrl.trace.events)}
    if sc.task == "value" and L.weight:
        tk = sc.body["true_kg"]
        r["weight"] = {"true": round(tk, 1), **{k: L.weight[k] for k in ("kg", "lo", "hi")},
                       "covered": L.weight["lo"] <= tk <= L.weight["hi"],
                       "ape": round(abs(L.weight["kg"] - tk) / tk, 4)}
    if "usage" in res:
        r["usage"] = res["usage"]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drivers", nargs="+", default=["open_loop", "autopilot"])
    ap.add_argument("--per-kind", type=int, default=4)
    ap.add_argument("--kinds", nargs="*", default=KINDS)
    ap.add_argument("--run", default=time.strftime("%m%d-%H%M"))
    a = ap.parse_args()

    out = ROOT / "eval" / "out" / a.run
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    split = make_split()
    gallery = build_gallery(split)
    th_path = ROOT / "eval" / "out" / "thresholds.json"
    th = json.loads(th_path.read_text()) if th_path.exists() else {}
    engine = MuzzleEngine(**{k: v for k, v in th.items() if k in ("t_match", "t_new")})
    scenarios = make_scenarios(split, a.per_kind, a.kinds)
    fields = field_paths(split)

    results = []
    for driver in a.drivers:
        root = out / driver
        store, reg = LocalStore(root / "store"), LocalRegistry(root / "registry")
        for an in split.registered:
            reg.put(an, {"owner": owner_of(an), "enrolled": "seed"})
        tools = VisionTools(store, reg, gallery, engine=engine, body=SimBodyProvider())
        sim = FieldSim(store, fields, seed=3)
        for sc in scenarios:
            sim.cursor.clear()
            sid = sim.start(sc)
            ctrl = Controller(sid, tools, reg, sim=sim, trace_path=root / "traces" / f"{sid}.jsonl")
            t = time.time()
            if driver == "open_loop":
                res = open_loop(ctrl)
            elif driver == "autopilot":
                res = autopilot(ctrl)
            else:
                from mombe.agent.agent import run_session
                res = run_session(ctrl)
            r = judge(sc, driver, res, ctrl)
            r["seconds"] = round(time.time() - t, 1)
            results.append(r)
            with (out / "results.jsonl").open("a") as f:
                f.write(json.dumps(r, default=str) + "\n")
            print(f"{driver:10s} {sc.id:22s} {str(r['outcome']):18s} correct={r['correct']} recaptures={r['recaptures']} {r['seconds']}s", flush=True)

    summ = defaultdict(lambda: defaultdict(lambda: {"n": 0, "correct": 0, "recaptures": 0}))
    for r in results:
        for key in (r["kind"], "ALL"):
            s = summ[r["driver"]][key]
            s["n"] += 1; s["correct"] += r["correct"]; s["recaptures"] += r["recaptures"]
    summary = {d: {k: {**v, "accuracy": round(v["correct"] / v["n"], 3)} for k, v in kinds.items()} for d, kinds in summ.items()}
    for d in a.drivers:
        w = [r["weight"] for r in results if r["driver"] == d and "weight" in r]
        if w:
            summary[d]["weight"] = {"n": len(w), "interval_coverage": round(sum(x["covered"] for x in w) / len(w), 3),
                                    "mape": round(sum(x["ape"] for x in w) / len(w), 4)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({d: summary[d]["ALL"] for d in summary}, indent=1))


if __name__ == "__main__":
    main()
