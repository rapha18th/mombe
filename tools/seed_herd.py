"""Seed a replayed herd for the lender view, and optionally one loan awaiting approval.

The herd comes from eval/out/value_herd_demo.json (eval/value_herds.py --demo 20):
real Acme animals with out-of-fold weights and 90% intervals, labelled replayed.
Each animal goes into the registry under one owner.

  python tools/seed_herd.py --hub https://<cloudfront> --owner Moyo     # live hub over REST, key from MOMBE_KEY
  python tools/seed_herd.py --local var/lender-demo --owner Moyo --loan 300 --approval

With --approval (local only) the Controller values the last animal of the herd and
asks a person to approve a loan on it, so the approver screen shows a real request.
Its identity fields are a demonstration fixture and say so.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEMO = ROOT / "eval" / "out" / "value_herd_demo.json"
SOURCE = "replayed Acme B4 animal (CC BY 4.0); out-of-fold weight"


def record(a: dict, owner: str) -> dict:
    return {"owner": owner, "weight_kg": a["kg"], "weight_interval": [a["lo"], a["hi"]], "source": SOURCE}


def seed_hub(url: str, key: str, owner: str, herd: list[dict]) -> None:
    import httpx
    with httpx.Client(base_url=url.rstrip("/"), headers={"x-mombe-key": key}, timeout=30) as http:
        for a in herd:
            http.put(f"/api/animals/{a['animal_id']}", json=record(a, owner)).raise_for_status()
        print(json.dumps(http.get("/api/herd", params={"owner": owner}).json(), indent=1)[:1500])


def seed_local(root: Path, owner: str, herd: list[dict], loan: float | None, approval: bool) -> None:
    from mombe.agent.runtime import Controller, autopilot
    from mombe.server.store import LocalRegistry, LocalStore
    from mombe.server.tools import VisionTools

    store, reg = LocalStore(root / "store"), LocalRegistry(root / "registry")
    keep = herd[:-1] if approval else herd
    for a in keep:
        reg.put(a["animal_id"], record(a, owner))
    if not approval:
        return
    a = herd[-1]
    reg.put(a["animal_id"], {"owner": owner, "source": SOURCE})
    s = store.create("value", claimed_owner=owner, loan_threshold_usd=loan, officer="demo")
    s.ledger = {
        "task": "value", "claimed_owner": owner, "loan_threshold_usd": loan, "ltv": 0.5,
        "quality": {"issues": [], "instructions": [], "fixture": True},
        "identity": {"decision": "match", "animal_id": a["animal_id"], "score": None, "margin": None,
                     "thresholds": {}, "fixture": "demonstration identity, no noseprint run"},
        "registry": reg.get(a["animal_id"]),
        "continuity": {"continuous": True, "breaks": 0, "fixture": True},
        "scale": {"board_frames": 12, "scale_rel_err": 0.004, "fixture": True},
        "body": {"coverage": 0.85, "fixture": True},
        "weight": {"kg": a["kg"], "lo": a["lo"], "hi": a["hi"], "interval": "90% split conformal, Acme B4"},
        # two-photo intervals run wider than the 12% gate: a reviewer accepted this one, and the P10 absorbs the width
        "reviews": {"weight_interval": {"id": "fixture", "status": "approved", "decided_by": "demo reviewer"}},
    }
    store.save(s)
    tools = VisionTools(store, reg, gallery=None, engine=object())
    ctrl = Controller(s.id, tools, reg)
    res = autopilot(ctrl)
    ctrl.persist()
    print(json.dumps({"session": s.id, "outcome": res["outcome"], "approval": (ctrl.L.approval or {}).get("id"),
                      "required": ctrl.state.required, "next": ctrl.state.next_observation}, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", default="Moyo")
    ap.add_argument("--hub")
    ap.add_argument("--local", type=Path)
    ap.add_argument("--loan", type=float, default=None)
    ap.add_argument("--approval", action="store_true")
    a = ap.parse_args()
    herd = json.loads(DEMO.read_text())["animals"]
    if a.hub:
        seed_hub(a.hub, os.environ.get("MOMBE_KEY", "dev-key"), a.owner, herd)
    elif a.local:
        seed_local(a.local, a.owner, herd, a.loan, a.approval)
    else:
        ap.error("give --hub or --local")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
