"""Scale-claim coverage per take under float-noise perturbations, on one build.

For every take, Reconstructor.run is repeated with the calibrated focal
multiplied by (1 + j * EPS), j = 0..N-1. j = 0 is the unperturbed pipeline.
EPS = 1e-10 is the size of the COOL vs stock difference in calibrateCamera
(5.9e-11 relative on sweep_21), so the spread over j is the pipeline's own
sensitivity to float noise on that build. One JSON line per run.

Usage: python ensemble.py --takes takes/sweep_21 ... --n 6 --out OUT.jsonl [--dense-first 1] [--eps 1e-10]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("MOMBE_CODE", str(Path(__file__).resolve().parents[2])))
import cv2  # noqa: E402

from mombe.body.recon import Reconstructor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--takes", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--eps", type=float, default=1e-10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dense-first", type=int, default=1, help="run dense + girth for the first k perturbations")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tools.eval_recon import score
    dense = None
    for take in a.takes:
        take = Path(take)
        truth = json.loads((take / "truth.json").read_text())
        frames = [cv2.imread(str(p)) for p in sorted(take.glob("frame_*.jpg"))]
        for j in range(a.n):
            t0 = time.time()
            R = Reconstructor()
            R.seed = a.seed
            orig = R.calibrate
            eps = j * a.eps

            def calibrate(obs, size, K0=None, _orig=orig, _eps=eps):
                cal = _orig(obs, size, K0)
                if _eps:
                    K = cal["K"].copy()
                    K[0, 0] *= 1 + _eps
                    K[1, 1] *= 1 + _eps
                    cal = {**cal, "K": K}
                return cal
            R.calibrate = calibrate
            row = {"take": take.name, "j": j, "eps": eps, "seed": a.seed, "opencv": cv2.__version__, "frames": len(frames),
                   "true_girth_m": truth["heart_girth_m"]}
            try:
                rec = R.run(frames)
                rec.feats_kpts = rec.feats_kpts or []
                sc = score(rec, truth)
                det = rec.scale_detail
                pp = det.get("per_placement", [])
                row.update(registered=len(rec.R), registered_frac=round(len(rec.R) / len(frames), 4),
                           points=int(len(rec.points)), ba_rms_px=rec.stats.get("ba_rms_px"),
                           claim=round(rec.scale_rel_err, 5), true_scale_err=sc["animal_scale_err"],
                           orbit_scale_err=sc["orbit_scale_err"], focal_err=sc["focal_err"],
                           covered=bool(rec.scale_rel_err >= abs(sc["animal_scale_err"])),
                           placements=det.get("placements"), between=det.get("between_placements_rel"),
                           per_placement=[{k: p[k] for k in ("frames", "metres_per_unit", "jackknife_rel_err", "fit_rms_mm")} for p in pp],
                           board_frames=det.get("board_frames"), unregistered=rec.stats.get("unregistered", []),
                           camera_pos_err_cm=sc["camera_pos_err_cm"], recon_s=round(time.time() - t0, 1))
                if j < a.dense_first:
                    from mombe.body.dense import DenseFusion
                    from mombe.body.measure import measure
                    dense = dense or DenseFusion()
                    dn = dense.run_stereo(rec, frames)
                    m = measure(dn.points, rec.boards)
                    if m.get("ok"):
                        row.update(girth_m=m["heart_girth_m"], girth_err=round(m["heart_girth_m"] / truth["heart_girth_m"] - 1, 4),
                                   loop_coverage=m["girth"]["coverage"], length_err=round(m["body_length_m"] / truth["body_length_m"] - 1, 4))
                    else:
                        row["measure_failed"] = m.get("reason")
            except Exception as e:  # a failed walk is a result too
                row["error"] = f"{type(e).__name__}: {e}"
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: row.get(k) for k in ("take", "j", "registered", "claim", "true_scale_err", "covered", "girth_err", "error")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
