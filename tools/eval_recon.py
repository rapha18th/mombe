"""Score a reconstruction of a synthetic walk-around against its ground truth.

Reports registration, bundle-adjustment error, focal error, the scale error the
pipeline claims (jackknife) next to the scale error it actually has (camera
centres aligned to the truth by a similarity), and camera position error.

Usage: python tools/eval_recon.py --name a3
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mombe.body.recon import Reconstructor, umeyama  # noqa: E402


def load(name: str):
    d = ROOT / "build" / "synth" / name
    truth = json.loads((d / "truth.json").read_text())
    truth["name"] = name
    frames = [cv2.imread(str(p)) for p in sorted(d.glob("frame_*.jpg"))]
    return d, truth, frames


def score(rec, truth) -> dict:
    fs = sorted(rec.R)
    est = np.array([rec.centre(f) for f in fs])
    tru = np.array([-np.array(truth["R"][f]).T @ np.array(truth["t"][f]) for f in fs])
    s, R, t = umeyama(est, tru)
    aligned = (s * (R @ est.T)).T + t
    pos_err = np.linalg.norm(aligned - tru, axis=1)
    # Scale where it matters: sparse points on the animal against the same tracks
    # triangulated with the true cameras (a trimmed similarity, so silhouette
    # points that no single surface owns do not decide it).
    from mombe.body.recon import triangulate_multi
    K = np.array(truth["K"])
    P = {f: K @ np.hstack([np.array(truth["R"][f]), np.array(truth["t"][f])[:, None]]) for f in range(len(truth["R"]))}
    est_p, tru_p = [], []
    for n, tid in enumerate(rec.point_track):
        obs = [(f, k) for f, k in rec.tracks[tid] if f in rec.R]
        Xt = triangulate_multi([P[f] for f, _ in obs], [rec.feats_kpts[f][k] for f, k in obs])
        if abs(Xt[0]) < 1.5 and abs(Xt[1]) < 0.6 and Xt[2] > 0.1:
            est_p.append(rec.points[n]); tru_p.append(Xt)
    est_p, tru_p = np.array(est_p), np.array(tru_p)
    keep = np.ones(len(est_p), bool)
    for _ in range(3):
        sa, Ra, ta = umeyama(est_p[keep], tru_p[keep])
        res_a = np.linalg.norm((sa * (Ra @ est_p.T)).T + ta - tru_p, axis=1)
        keep = res_a < 3 * np.median(res_a)
    return {"animal_scale_err": round(sa - 1, 5), "animal_points": len(est_p),
            "animal_point_err_cm": round(float(np.median(res_a[keep])) * 100, 2),
            "orbit_scale_err": round(s - 1, 5), "claimed_scale_rel_err": round(rec.scale_rel_err, 5),
            "camera_pos_err_cm": {"median": round(float(np.median(pos_err)) * 100, 2), "max": round(float(pos_err.max()) * 100, 2)},
            "focal_err": round(rec.K[0, 0] / truth["K"][0][0] - 1, 5)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    d, truth, frames = load(a.name)
    t0 = time.time()
    rec = Reconstructor().run(frames, cache=d)
    rec.feats_kpts = [f.kpts for f in rec.feats]
    out = {"name": a.name, "seconds": round(time.time() - t0, 1),
           **{k: v for k, v in rec.stats.items() if k != "pairs"}, "scale": rec.scale_detail,
           "ground": {k: v for k, v in rec.ground.items() if k in ("found", "inliers", "camera_height_m")},
           **score(rec, truth)}
    with open(d / "recon.pkl", "wb") as f:
        rec.feats = []
        pickle.dump(rec, f)
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
