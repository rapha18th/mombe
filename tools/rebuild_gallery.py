"""Re-extract the hub's gallery under the deployed PipelineConfig (pipeline.json).

Descriptors depend on preprocessing and the networks' engines and precision,
so a configuration that changes extraction needs a gallery extracted the same
way. Writes var/gallery.npz in place, keeping animal and image ids.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    import cv2
    from mombe.vision.muzzle import Gallery, MuzzleEngine, PipelineConfig

    eng = MuzzleEngine(config=PipelineConfig.from_json(ROOT / "pipeline.json"))
    meta = json.loads((ROOT / "var" / "gallery.json").read_text())
    g, t = Gallery(), time.time()
    for an, im in zip(meta["animal_ids"], meta["image_ids"]):
        img = cv2.imread(str(ROOT / "data" / "muzzle" / "_".join(im.split("_")[:2]) / f"{im}.jpg"))
        g.add(an, im, eng.extract(img))
    g.save(ROOT / "var" / "gallery.npz")
    print(f"REBUILT {len(g.features)} images in {time.time() - t:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
