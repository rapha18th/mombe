"""Synthetic check for measure_v2: the seven sweep animals as true-depth point clouds.

Each animal comes from tools/synth_walk.animal_for(seed, scale), the same
builder the body sweep used (eval/out/sweep_body3.jsonl lists the seven).
The renderer's true depth from a 48-frame handheld walk-around (reduced
resolution) is back-projected to world points and voxel-downsampled at 1 cm,
like the Ruchay preprocessing. The belly underside stays unseen, as on a
phone walk-around. Truth: Animal.heart_girth_m() and Animal.body_length_m().

Usage: python eval/synth_clouds_v2.py            (writes build/synth/v2_clouds/<name>.npz)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ANIMALS = [(21, 0.85), (22, 0.95), (23, 1.0), (24, 1.08), (25, 1.15), (26, 0.9), (27, 1.12)]
OUT = ROOT / "build" / "synth" / "v2_clouds"
SIZE = (480, 270)


def cloud_for(seed: int, scale: float, frames: int = 48, radius: float = 2.8) -> tuple[np.ndarray, dict]:
    from mombe.sim.render import Scene, walk_around
    from tools.synth_walk import animal_for
    animal = animal_for(seed, scale)
    scene = Scene(animal=animal)
    cams = walk_around(frames, radius=radius * scale ** 0.5, seed=seed, start_deg=45.0, size=SIZE)
    pts = []
    for c in cams:
        r = scene.render(c, board=None)
        z, obj = r["depth"], r["obj"]
        w, h = c.size
        u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        m = (z > 0) & (obj >= 2) | ((obj == 0) & (z > 0))
        x = (u - c.K[0, 2]) / c.K[0, 0] * z
        y = (v - c.K[1, 2]) / c.K[1, 1] * z
        Pc = np.stack([x[m], y[m], z[m]], 1)
        Pw = (Pc - c.t) @ c.R                     # camera -> world: R^T (Pc - t)
        keep = np.hypot(Pw[:, 0] - 0.1, Pw[:, 1]) < 2.2 * scale
        pts.append(Pw[keep])
    P = np.vstack(pts)
    q = np.floor(P / 0.01).astype(np.int64)
    _, idx, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv).astype(float)
    P = np.stack([np.bincount(inv, P[:, k]) for k in range(3)], 1) / cnt[:, None]
    truth = {"seed": seed, "scale": scale, "heart_girth_m": animal.heart_girth_m(), "body_length_m": animal.body_length_m(),
             "girth_x": animal.girth_x(), "legs_x": list(animal.legs_x), "barrel": list(animal.barrel),
             "barrel_z": animal.barrel_z, "front": "+x"}
    return P, truth


def load_all() -> list[tuple[str, np.ndarray, dict]]:
    out = []
    for seed, scale in ANIMALS:
        name = f"sweep_{seed}"
        f = OUT / f"{name}.npz"
        if not f.exists():
            OUT.mkdir(parents=True, exist_ok=True)
            P, truth = cloud_for(seed, scale)
            np.savez_compressed(f, xyz=P.astype(np.float32), truth=json.dumps(truth))
        z = np.load(f)
        out.append((name, z["xyz"].astype(np.float64), json.loads(str(z["truth"]))))
    return out


if __name__ == "__main__":
    for name, P, t in load_all():
        print(name, len(P), round(t["heart_girth_m"], 4), round(t["body_length_m"], 4))
