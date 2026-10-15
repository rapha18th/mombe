"""Body accuracy over many synthetic animals and reconstruction seeds.

Each animal differs in size and build; each walk is reconstructed under
several RANSAC seeds, so the table shows both the error and how much it
moves when nothing but sampling changes. One JSON line per run.

Usage: python tools/sweep_body.py [--animals 5] [--seeds 7 11] [--out eval/out/sweep_body.jsonl]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ANIMALS = [  # (render seed, size scale): small heifer to large ox
    (21, 0.85), (22, 0.95), (23, 1.0), (24, 1.08), (25, 1.15), (26, 0.9), (27, 1.12),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--animals", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 11])
    ap.add_argument("--out", default="eval/out/sweep_body.jsonl")
    a = ap.parse_args()
    from mombe.body.dense import DenseFusion
    from mombe.body.measure import measure, schaeffer_kg
    from mombe.body.recon import Reconstructor
    from tools.eval_recon import load, score

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    dense = DenseFusion()
    for rs, scale in ANIMALS[:a.animals]:
        name = f"sweep_{rs}"
        if not (ROOT / "build" / "synth" / name / "truth.json").exists():
            subprocess.run([sys.executable, str(ROOT / "tools" / "synth_walk.py"), "--name", name, "--seed", str(rs),
                            "--scale", str(scale)], check=True)
        d, truth, frames = load(name)
        for seed in a.seeds:
            t0 = time.time()
            R = Reconstructor()
            R.seed = seed
            row = {"animal": name, "scale": scale, "seed": seed, "true_girth_m": round(truth["heart_girth_m"], 4),
                   "true_length_m": round(truth["body_length_m"], 4)}
            try:
                rec = R.run(frames, cache=d)
                t1 = time.time()
                dn = dense.run_stereo(rec, frames)
                m = measure(dn.points, rec.boards)
                rec.feats_kpts = rec.feats_kpts or []
                sc = score(rec, truth)
                row.update(recon_s=round(t1 - t0, 1), dense_s=dn.seconds, registered=len(rec.R),
                           claimed_scale_err=round(rec.scale_rel_err, 5), animal_scale_err=sc["animal_scale_err"])
                if m.get("ok"):
                    row.update(girth_m=m["heart_girth_m"], length_m=m["body_length_m"],
                               girth_err=round(m["heart_girth_m"] / truth["heart_girth_m"] - 1, 4),
                               loop_coverage=m["girth"]["coverage"],
                               kg=round(schaeffer_kg(m["heart_girth_m"], m["body_length_m"]), 1),
                               kg_from_truth=round(schaeffer_kg(truth["heart_girth_m"], truth["body_length_m"]), 1))
                else:
                    row["measure_failed"] = m.get("reason")
            except Exception as e:  # a failed walk is a result too
                row["error"] = f"{type(e).__name__}: {e}"
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    errs = np.array([r["girth_err"] for r in rows if "girth_err" in r])
    if len(errs):
        print(json.dumps({"runs": len(rows), "measured": len(errs), "girth_err_mean": round(float(errs.mean()), 4),
                          "girth_err_sd": round(float(errs.std(ddof=1)), 4) if len(errs) > 1 else None,
                          "girth_abs_err_median": round(float(np.median(np.abs(errs))), 4)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
