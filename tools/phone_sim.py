"""Scripted phone against the hub: the deployed loop, end to end.

Creates a session, uploads a degraded nose, waits for the agent's instruction,
answers it with a clean capture, and prints the outcome and the trace.

Usage: python tools/phone_sim.py --hub http://127.0.0.1:8088 --animal cattle_0400 [--defect glare]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mombe.sim.field import DEFECTS  # noqa: E402


def jpg(img) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", default="http://127.0.0.1:8088")
    ap.add_argument("--key", default="dev-key")
    ap.add_argument("--animal", required=True)
    ap.add_argument("--owner", default=None)
    ap.add_argument("--defect", default="glare")
    ap.add_argument("--task", default="verify")
    a = ap.parse_args()
    http = httpx.Client(base_url=a.hub, headers={"x-mombe-key": a.key}, timeout=120)
    imgs = sorted((ROOT / "data" / "muzzle" / a.animal).glob("*.jpg"))
    rng = random.Random(1)

    s = http.post("/api/sessions", json={"task": a.task, "claimed_owner": a.owner}).json()
    sid = s["id"]
    print("session", sid)

    def upload(paths, defect=None):
        files = []
        for i, p in enumerate(paths):
            im = cv2.imread(str(p))
            if defect:
                im = DEFECTS[defect](im, rng)
            files.append((f"muzzle_{i}", (f"m{i}.jpg", jpg(im), "image/jpeg")))
        return http.post(f"/api/sessions/{sid}/captures", files=files, data={"meta": json.dumps({"defect": defect})}).json()

    def settle(after_events: int) -> dict:
        """Wait until the agent has handled the latest upload and parked or finished."""
        for _ in range(180):
            time.sleep(2)
            ev = http.get(f"/api/sessions/{sid}/trace").json()
            s = http.get(f"/api/sessions/{sid}").json()
            if len(ev) > after_events and s["status"] in ("done", "waiting") and (s.get("ledger") or {}):
                last = ev[-1]
                if last["kind"] != "start" and last["kind"] != "resume":
                    return s
        return s

    print("upload 1", upload(imgs[-2:], a.defect))
    seen = 0
    for _ in range(3):
        n_before = len(http.get(f"/api/sessions/{sid}/trace").json())
        s = settle(n_before)
        if len(s["instructions"]) > seen:
            ins = s["instructions"][-1]
            seen = len(s["instructions"])
            print("officer sees:", ins["code"], "-", ins["message"])
            print("upload", seen + 1, upload(imgs[-4:-2]))
            continue
        break
    trace = http.get(f"/api/sessions/{sid}/trace").json()
    print("status", s["status"], "| identity", {k: ((s.get("ledger") or {}).get("identity") or {}).get(k) for k in ("decision", "animal_id", "score")})
    for e in trace:
        if e["kind"] in ("observation", "action", "refused", "start", "resume"):
            print(f"  {e['step']:>2} {e['kind']:<11} {str(e.get('name') or ''):<16} -> next {e.get('required_after')} changed={e.get('changed_next_action')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
