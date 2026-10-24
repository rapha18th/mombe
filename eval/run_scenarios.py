"""Scenario suite: does closing the loop change outcomes?

Each scenario is one field session on a held-out test animal (enrol_double runs
a first enrolment, then the scored second one). Attack kinds present SYNTHETIC
print, screen or replay frames on every capture. Drivers:
  open_loop  one pass, argmax identity, no re-captures (baseline)
  autopilot  follows the evidence policy deterministically (closed loop)
  claude     Claude on Bedrock via Strands (MOMBE_MODEL_ID, default Sonnet 4.6), same policy and tools

Usage: python eval/run_scenarios.py --drivers open_loop autopilot [--per-kind 6] [--kinds ...]
Writes eval/out/<run>/results.jsonl, traces/, and summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Valuations in the suite are made on a fixed day, so results reproduce as the price quotes age.
os.environ.setdefault("MOMBE_TODAY", "2026-10-21")

from eval.dataset import IMGS, build_gallery, field_paths, folder, make_split  # noqa: E402
from mombe.agent.runtime import Controller, autopilot, open_loop  # noqa: E402
from mombe.agent.valuation import Prices  # noqa: E402
from mombe.server.store import LocalRegistry, LocalStore  # noqa: E402
from mombe.server.tools import VisionTools  # noqa: E402
from mombe.sim.field import FieldSim, Scenario, SimBodyProvider  # noqa: E402
from mombe.vision.muzzle import MuzzleEngine  # noqa: E402

OWNERS = ["Moyo", "Ncube", "Dube", "Sibanda", "Chikore", "Mutasa", "Nyathi", "Banda", "Zulu", "Mlambo"]
KINDS = ["verify_clean", "verify_glare", "verify_blur", "verify_partial", "enrol_new", "enrol_duplicate",
         "owner_mismatch", "claim_true", "claim_wrong", "value_clean", "value_board_far", "value_splice",
         "value_low_coverage", "value_straddle",
         # body measurement gates (synthetic error model, input S): each bad take reads the animal light,
         # so a loop that trusts it refuses credit the animal supports
         "value_poor_registration", "value_belly_unseen", "value_implausible", "value_scale_overclaim",
         "value_far_side_missing",
         # presentation attacks (SYNTHETIC frames) and double enrolment
         "verify_print_attack", "verify_screen_attack", "verify_replay", "enrol_double"]
BODY_GATE_KINDS = ["value_poor_registration", "value_belly_unseen", "value_implausible", "value_scale_overclaim",
                   "value_far_side_missing"]
FLOOR = Prices().low["usd_per_kg"]     # loans are sized at the floor price, as the policy sizes them


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
            elif kind in ("verify_print_attack", "verify_screen_attack", "verify_replay"):
                # a fraudster holds up the registered animal's nose (print, screen) or resends its
                # enrolled photo, claiming it as the owner would; the frames are SYNTHETIC
                sc = Scenario(sid, "verify", a, "escalated", claimed_owner=owner_of(a), approve=False,
                              attack=kind.split("_")[1], note="synthetic presentation attack")
            elif kind == "enrol_double":
                # the same unregistered animal enrolled to Moyo, then presented again under another owner
                sc = Scenario(sid, "enrol", rng.choice(unreg), "escalated", claimed_owner="Stranger", approve=False,
                              prior_owner="Moyo", note="double enrolment to borrow twice")
            else:
                body = {"true_kg": kg}
                fix = {}
                loan = round(kg * FLOOR * 0.5 * 0.6)                     # comfortably lendable
                if kind == "value_board_far":
                    body.update(board_frames=3, scale_rel_err=0.035); fix = {"board_frames": 16, "scale_rel_err": 0.004}
                elif kind == "value_splice":
                    body.update(continuous=False); fix = {"continuous": True}
                elif kind == "value_low_coverage":
                    body.update(coverage=0.45, missing_side=rng.choice(["left", "right"])); fix = {"coverage": 0.86}
                elif kind == "value_straddle":
                    loan = round(kg * FLOOR * 0.5 * 1.01)                # sits inside the lendable interval
                    body.update(board_frames=8, scale_rel_err=0.008)
                elif kind in BODY_GATE_KINDS:
                    loan = round(kg * FLOOR * 0.5 * (0.95 if kind == "value_scale_overclaim" else 0.9))
                    if kind == "value_poor_registration" and n % 2 == 0:  # half the orbit registered: the trunk reads short
                        body.update(registered_frac=0.55, weight_bias=-0.35); fix = {"registered_frac": 1.0, "weight_bias": 0.0}
                    elif kind == "value_poor_registration":               # all frames registered, orbit never closed (COOL sweep_21)
                        body.update(length_cut_ratio=0.746, weight_bias=-0.35)
                        fix = {"length_cut_ratio": 0.95, "weight_bias": 0.0}
                    elif kind == "value_belly_unseen":                    # the ellipse closes inside the unseen brisket
                        body.update(loop_coverage=0.48, gap_arc="belly", girth_scale=0.9)
                        fix = {"loop_coverage": 0.8, "girth_scale": 1.0}
                    elif kind == "value_implausible":                     # the girth slice landed on the neck
                        body.update(girth_scale=0.72); fix = {"girth_scale": 1.0}
                    elif kind == "value_scale_overclaim":                 # raw claim inside the 1% gate, true error -5%
                        body.update(raw_claim=0.009, scale_true_err=-0.05); fix = {"raw_claim": 0.003, "scale_true_err": 0.0}
                    elif kind == "value_far_side_missing":                # cameras went round; the far flank stayed out of the surface
                        side = rng.choice(["left", "right"])
                        body.update(coverage=0.75, loop_coverage=0.5, gap_arc=side, girth_scale=0.86)
                        fix = {"coverage": 0.86, "loop_coverage": 0.82, "girth_scale": 1.0}
                sc = Scenario(sid, "value", a, "approved", claimed_owner=owner_of(a), loan_threshold_usd=loan,
                              body=body, body_fix=fix)
            out.append(sc)
    return out


def score_session(sc: Scenario, driver: str, res: dict, ctrl: Controller) -> dict:
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
    if sc.attack or sc.prior_owner:
        r["fraud_accepted"] = fraud_accepted(sc, driver, res, outcome)
    return r


def fraud_accepted(sc: Scenario, driver: str, res: dict, outcome) -> bool:
    """Did the presentation attack or the second enrolment get through?"""
    if driver == "open_loop":
        if sc.task == "enrol":
            return res.get("decision") == "new"       # open loop enrols whatever it fails to recognise
        return res.get("decided_animal") == sc.animal
    return outcome in ("verified", "enrolled", "approved")


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

    enrolled = {a: [str(IMGS / folder(im) / f"{im}.jpg") for im in split.gallery_imgs[a]] for a in split.registered}
    n_seed = len(gallery.features)

    def drive(driver: str, ctrl: Controller) -> dict:
        if driver == "open_loop":
            return open_loop(ctrl)
        if driver == "autopilot":
            return autopilot(ctrl)
        from mombe.agent.agent import run_session
        try:
            return run_session(ctrl)
        except Exception as e:      # a model that breaks mid-session leaves it open: scored as a miss
            return {"error": f"{type(e).__name__}: {str(e)[:300]}"}

    results = []
    for driver in a.drivers:
        root = out / driver
        store, reg = LocalStore(root / "store"), LocalRegistry(root / "registry")
        for an in split.registered:
            reg.put(an, {"owner": owner_of(an), "enrolled": "seed"})
        tools = VisionTools(store, reg, gallery, engine=engine, body=SimBodyProvider())
        sim = FieldSim(store, fields, seed=3, enrolled_images=enrolled)
        for sc in scenarios:
            sim.cursor.clear()
            t = time.time()
            prior = None
            if sc.prior_owner:
                # first enrolment of the same animal, by the same driver; the scored session comes second
                first = Scenario(f"{sc.id}-first", "enrol", sc.animal, "enrolled", claimed_owner=sc.prior_owner)
                pid = sim.start(first)
                pctrl = Controller(pid, tools, reg, sim=sim, trace_path=root / "traces" / f"{pid}.jsonl")
                drive(driver, pctrl)
                prior = "open_loop" if driver == "open_loop" else (pctrl.state.outcome or "unfinished")
            sid = sim.start(sc)
            ctrl = Controller(sid, tools, reg, sim=sim, trace_path=root / "traces" / f"{sid}.jsonl")
            res = drive(driver, ctrl)
            r = score_session(sc, driver, res, ctrl)
            if prior is not None:
                r["prior_outcome"] = prior
                if driver != "open_loop" and prior != "enrolled":
                    r["correct"] = False          # the double enrolment never set up: the case proves nothing
            if sc.attack:
                r["synthetic_attack"] = sc.attack
            if "error" in res:
                r["error"] = res["error"]
            r["seconds"] = round(time.time() - t, 1)
            gallery.truncate(n_seed)              # enrolments made in this scenario leave with it
            results.append(r)
            with (out / "results.jsonl").open("a") as f:
                f.write(json.dumps(r, default=str) + "\n")
            print(f"{driver:10s} {sc.id:22s} {str(r['outcome']):18s} correct={r['correct']} recaptures={r['recaptures']} {r['seconds']}s", flush=True)

    summ = defaultdict(lambda: defaultdict(lambda: {"n": 0, "correct": 0, "recaptures": 0}))
    for r in results:
        for key in (r["kind"], "ALL"):
            s = summ[r["driver"]][key]
            s["n"] += 1; s["correct"] += r["correct"]; s["recaptures"] += r["recaptures"]
            if "fraud_accepted" in r:
                s["fraud_accepted"] = s.get("fraud_accepted", 0) + r["fraud_accepted"]
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
