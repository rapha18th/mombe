"""Extract OpenCV 5 muzzle features for every image in the dataset.

Writes one .npz per image under data/features/<animal>/<image>.npz holding
ALIKED keypoints and descriptors, the working size, and the DINOv2 embedding.
Runs in parallel; each worker owns its own MuzzleEngine.

Usage: python tools/build_features.py [--workers 6] [--limit N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "muzzle"
OUT = ROOT / "data" / "features"

_engine = None


def _init():
    global _engine
    import cv2
    from mombe.vision.muzzle import MuzzleEngine
    _engine = MuzzleEngine()


def _one(path: str) -> tuple[str, float]:
    import cv2
    import numpy as np
    p = Path(path)
    out = OUT / p.parent.name / (p.stem + ".npz")
    if out.exists():
        return path, 0.0
    t = time.time()
    img = cv2.imread(path)
    f = _engine.extract(img)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, kpts=f.kpts, desc=f.desc, size=np.array(f.size), emb=f.embedding)
    return path, time.time() - t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per-animal", type=int, default=0, help="evenly spaced images per animal (0 = all)")
    a = ap.parse_args()
    paths = sorted(str(p) for p in DATA.glob("*/*.jpg"))
    if a.per_animal:
        by: dict[str, list[str]] = {}
        for p in paths:
            by.setdefault(Path(p).parent.name, []).append(p)
        paths = []
        for imgs in by.values():
            n = min(a.per_animal, len(imgs))
            paths += [imgs[round(i * (len(imgs) - 1) / max(n - 1, 1))] for i in range(n)]
    if a.limit:
        paths = paths[: a.limit]
    t0, done = time.time(), 0
    if a.workers == 1:
        _init()
        results = map(_one, paths)
    else:
        ex = ProcessPoolExecutor(a.workers, initializer=_init)
        results = ex.map(_one, paths, chunksize=4)
    if True:
        for _, dt in results:
            done += 1
            if done % 200 == 0 or done == len(paths):
                el = time.time() - t0
                print(f"{done}/{len(paths)}  {el:.0f}s elapsed  eta {el / done * (len(paths) - done):.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
