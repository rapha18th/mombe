"""identify() decisions on fixed probes, for decision agreement between builds.

Uses the hub's own gallery (var/gallery.npz, built once and shipped to both
hubs; its hash goes in the header) and one shipped config (identify_config.json:
the live COOL settings, verify_top 2) with the p0-frozen thresholds t_match 90,
t_new 20. Each probe is extracted on this build and identified.
One JSON line per probe: decision, chosen animal, score, margin, shortlist
order, and a hash of the probe's embedding and keypoints.

Usage: python identify_probes.py --probes data/probes.json --root data --out OUT.jsonl [--config identify_config.json] [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

CODE = Path(os.environ.get("MOMBE_CODE", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(CODE))
import cv2  # noqa: E402

from mombe.vision.muzzle import Gallery, MuzzleEngine, PipelineConfig, identify  # noqa: E402


def h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    # one shipped config for every build (identify_config.json: the live COOL settings), never a hub's own file
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent / "identify_config.json"))
    ap.add_argument("--t-match", type=int, default=90)        # p0-frozen thresholds
    ap.add_argument("--t-new", type=int, default=20)
    ap.add_argument("--gallery", default=str(CODE / "var" / "gallery.npz"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cfg = PipelineConfig.from_json(a.config)
    th = {"t_match": a.t_match, "t_new": a.t_new}
    eng = MuzzleEngine(t_match=th["t_match"], t_new=th["t_new"], config=cfg)
    gp = Path(a.gallery)
    g = Gallery.load(gp)
    probes = json.loads(Path(a.probes).read_text())
    if a.limit:
        probes = probes[: a.limit]
    head = {"opencv": cv2.__version__, "cv2_file": cv2.__file__, "numpy": np.__version__, "threads": cv2.getNumThreads(),
            "config": cfg.__dict__, "thresholds": th, "gallery_images": len(g.image_ids),
            "gallery_hash": h(g.embeddings), "gallery_file_sha": hashlib.sha256(gp.read_bytes()).hexdigest()[:16]}
    with open(a.out, "w") as f:
        f.write(json.dumps({"header": head}) + "\n")
    t0 = time.time()
    for n, p in enumerate(probes):
        img = cv2.imread(str(Path(a.root) / p["path"]))
        q = eng.extract(img)
        r = identify(eng, g, [q])
        row = {"path": p["path"], "expected": p["expected"], "decision": r.decision, "animal_id": r.animal_id,
               "score": r.score, "margin": r.margin, "shortlist": [s["animal_id"] for s in r.shortlist],
               "cosines": [s["cosine"] for s in r.shortlist], "inliers": {s["animal_id"]: s["inliers"] for s in r.shortlist if s["inliers"]},
               "emb_hash": h(q.embedding), "kpts_hash": h(q.kpts), "n_kpts": int(len(q.kpts)), "desc_hash": h(q.desc)}
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        if n % 50 == 0:
            print(n, round(time.time() - t0), "s", flush=True)
    print("done", len(probes), round(time.time() - t0), "s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
