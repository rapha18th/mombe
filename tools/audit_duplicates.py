"""Audit the public muzzle dataset for one animal filed under two IDs.

Every animal's first image is verified against the three other animals
DINOv2 ranks closest. Genuine impostors verify at about 5 to 20 inliers; a
pair above DUP_INLIERS is the same nose under two labels. Writes
eval/out/duplicates.json with the pairs and their scores.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.dataset import FEAT, cached, load_feat  # noqa: E402
from mombe.vision.muzzle import Gallery, MuzzleEngine  # noqa: E402

DUP_INLIERS = 60


def main() -> int:
    animals = sorted(p.name for p in FEAT.iterdir() if p.is_dir() and len(cached(p.name)) >= 6)
    g = Gallery()
    for a in animals:
        for im in cached(a)[::2]:
            g.add(a, im, load_feat(a, im))
    eng = MuzzleEngine()
    pairs, scores = {}, []
    for n, a in enumerate(animals):
        im = cached(a)[1]
        q = load_feat(a, im)
        for b, cos, idxs in [s for s in g.shortlist(q.embedding, k=4) if s[0] != a][:3]:
            inl = eng.verify(q, g.features[idxs[0]])[1]
            scores.append(inl)
            if inl >= DUP_INLIERS:
                key = tuple(sorted((a, b)))
                pairs[key] = max(pairs.get(key, 0), inl)
        if n % 25 == 0:
            print(f"{n}/{len(animals)} pairs so far {len(pairs)}", flush=True)
    out = {"threshold": DUP_INLIERS, "animals": len(animals), "comparisons": len(scores),
           "duplicates": [{"a": a, "b": b, "inliers": v} for (a, b), v in sorted(pairs.items(), key=lambda kv: -kv[1])]}
    (ROOT / "eval" / "out").mkdir(parents=True, exist_ok=True)
    (ROOT / "eval" / "out" / "duplicates.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
