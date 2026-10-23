"""Is the dense stage deterministic? One reconstruction, dense fusion repeated.

Runs Reconstructor.run once on a take, then DenseFusion.run_stereo three
times at the default thread count and once on one OpenCV thread, and reports
the point count, a hash of the points, and the girth each time.

Usage: python dense_repeat.py --take DIR --out OUT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("MOMBE_CODE", str(Path(__file__).resolve().parents[2])))
import cv2  # noqa: E402

from mombe.body.dense import DenseFusion  # noqa: E402
from mombe.body.measure import measure  # noqa: E402
from mombe.body.recon import Reconstructor  # noqa: E402


def h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    take = Path(a.take)
    truth = json.loads((take / "truth.json").read_text())
    frames = [cv2.imread(str(p)) for p in sorted(take.glob("frame_*.jpg"))]
    rec = Reconstructor().run(frames)
    dn = DenseFusion()
    runs = []
    T = cv2.getNumThreads()
    for threads in (T, T, T, 1, 1):
        cv2.setNumThreads(threads)
        r = dn.run_stereo(rec, frames)
        m = measure(r.points, rec.boards)
        # per-stage hashes inside dense: stereo depth per frame
        runs.append({"threads": threads, "points": int(len(r.points)), "points_hash": h(r.points),
                     "sorted_points_hash": h(np.round(r.points[np.lexsort(r.points.T)], 6)) if len(r.points) else None,
                     "girth_m": m.get("heart_girth_m"), "coverage": m.get("girth", {}).get("coverage"),
                     "girth_err": round(m["heart_girth_m"] / truth["heart_girth_m"] - 1, 4) if m.get("ok") else None,
                     "valid_hash": h(np.array([[f["frame"], f["valid_raw"], f["valid"]] for f in r.frames]))})
        print(json.dumps(runs[-1]), flush=True)
    cv2.setNumThreads(T)
    # depth maps from a single pair, repeated: is StereoSGBM itself repeatable?
    W, H = rec.size
    s = 1920 / W
    Kf = rec.K.copy() * s
    Kf[2, 2] = 1
    fw, fh = 1920, round(H * s)
    reg = sorted(rec.R)
    small = {f: cv2.resize(frames[f], (fw, fh), interpolation=cv2.INTER_AREA) for f in reg[:3]}
    dh = []
    for threads in (T, T, 1):
        cv2.setNumThreads(threads)
        dh.append(h(DenseFusion.stereo_depth(rec, reg[0], reg[2], small[reg[0]], small[reg[2]], Kf, (fw, fh))))
    cv2.setNumThreads(T)
    out = {"opencv": cv2.__version__, "take": take.name, "runs": runs, "stereo_depth_hashes": dh,
           "deterministic_default_threads": len({r["points_hash"] for r in runs[:3]}) == 1,
           "deterministic_one_thread": len({r["points_hash"] for r in runs[3:]}) == 1}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
