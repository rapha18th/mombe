"""A value session against the live hub with a real body walk-around.

Uploads the nose of a registered animal and a walk-around (frames from a
folder, e.g. a synthetic walk), lets the deployed agent run, and follows the
session through parking while the hub measures, the wake-up, and the result.

Usage: python tools/live_value.py --walk build/synth/a3c [--animal cattle_4833] [--loan 150]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", required=True)
    ap.add_argument("--animal", default="cattle_4833")
    ap.add_argument("--loan", type=float, default=150.0)
    ap.add_argument("--minutes", type=float, default=20)
    a = ap.parse_args()
    st = json.loads((ROOT / "infra" / "state.json").read_text())
    http = httpx.Client(base_url=f"https://{st['cdn_domain']}", headers={"x-mombe-key": st["hub_key"]}, timeout=120)
    owner = (http.get(f"/api/animals/{a.animal}").json().get("record") or {}).get("owner")
    sid = http.post("/api/sessions", json={"task": "value", "claimed_owner": owner,
                                           "loan_threshold_usd": a.loan}).json()["id"]
    print("session", sid, "| animal", a.animal, "| owner on record", owner, flush=True)
    nose = sorted((ROOT / "data" / "muzzle" / a.animal).glob("*.jpg"))[-2:]
    walk = sorted(Path(a.walk).glob("frame_*.jpg"))
    files = [(f"muzzle_{i}", (f"m{i}.jpg", p.read_bytes(), "image/jpeg")) for i, p in enumerate(nose)]
    files += [(f"body_{i}", (f"b{i}.jpg", p.read_bytes(), "image/jpeg")) for i, p in enumerate(walk)]
    t0 = time.time()
    r = http.post(f"/api/sessions/{sid}/captures", files=files, data={"meta": json.dumps({"source": "live_value"})})
    print("uploaded", r.json(), f"in {time.time() - t0:.0f}s", flush=True)
    seen, last_status = 0, None
    while time.time() - t0 < a.minutes * 60:
        time.sleep(10)
        s = http.get(f"/api/sessions/{sid}").json()
        ev = http.get(f"/api/sessions/{sid}/trace").json()
        for e in ev[seen:]:
            if e["kind"] in ("observation", "action", "refused", "start", "resume"):
                res = e.get("result") or {}
                brief = {k: res[k] for k in ("decision", "animal_id", "pending", "scale_rel_err", "board_frames", "continuous",
                                             "girth_cm", "length_cm", "kg", "lo", "hi", "low_usd", "high_usd") if k in res}
                print(f"{time.time() - t0:5.0f}s {e['kind']:<11} {str(e.get('name') or ''):<17} {json.dumps(brief)} -> {e.get('required_after')}",
                      flush=True)
        seen = len(ev)
        if s["status"] != last_status:
            print(f"{time.time() - t0:5.0f}s status {s['status']} | outcome {((s.get('ledger') or {}) and s['status'])}", flush=True)
            last_status = s["status"]
        if s["status"] == "done":
            break
    s = http.get(f"/api/sessions/{sid}").json()
    L = s.get("ledger") or {}
    print("final status", s["status"], "| instructions", [i.get("code") for i in s["instructions"]])
    for k in ("continuity", "scale", "body", "weight", "value"):
        print(k, json.dumps(L.get(k), default=str)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
