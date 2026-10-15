"""Render a synthetic walk-around to disk with its ground truth.

Writes build/synth/<name>/frame_XX.jpg, depth_XX.npy (float16 metres) and
truth.json (K, poses, heart girth, body length, animal geometry).

Usage: python tools/synth_walk.py --name a3 --frames 48 [--seed 3] [--scale 1.0]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mombe.sim.render import Animal, Scene, walk_around  # noqa: E402


def animal_for(seed: int, scale: float) -> Animal:
    """Animals of different size and build: the barrel scales, legs and head follow."""
    rng = np.random.default_rng(seed)
    a = Animal(seed=seed)
    k = scale * (1 + 0.04 * rng.standard_normal())
    wide = 1 + 0.08 * rng.standard_normal()                  # body condition: wider or leaner barrel
    a.barrel = (a.barrel[0] * k, a.barrel[1] * k * wide, a.barrel[2] * k)
    a.barrel_z *= k
    a.legs_x = (a.legs_x[0] * k, a.legs_x[1] * k)
    a.neck_c = tuple(np.array(a.neck_c) * k)
    a.head_c = tuple(np.array(a.head_c) * k)
    return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--radius", type=float, default=2.8)
    a = ap.parse_args()
    out = ROOT / "build" / "synth" / a.name
    out.mkdir(parents=True, exist_ok=True)
    animal = animal_for(a.seed, a.scale)
    scene = Scene(animal=animal)
    cams = walk_around(a.frames, radius=a.radius * a.scale ** 0.5, seed=a.seed, start_deg=45.0)
    for i, c in enumerate(cams):
        # The protocol: board on the left shoulder for the first half, the right for the second.
        r = scene.render(c, board="left" if i < len(cams) // 2 else "right")
        cv2.imwrite(str(out / f"frame_{i:02d}.jpg"), r["image"], [cv2.IMWRITE_JPEG_QUALITY, 92])
        np.save(out / f"depth_{i:02d}.npy", r["depth"].astype(np.float16))
    truth = {"K": cams[0].K.tolist(), "R": [c.R.tolist() for c in cams], "t": [c.t.tolist() for c in cams],
             "heart_girth_m": animal.heart_girth_m(), "body_length_m": animal.body_length_m(),
             "girth_x": animal.girth_x(), "board_switch": len(cams) // 2, "animal": asdict(animal)}
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    print(f"RENDERED {len(cams)} frames to {out}; girth {truth['heart_girth_m']:.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
