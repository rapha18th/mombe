"""Calibrate the identity bands on dev animals, then measure them on test animals.

Genuine score: a field image verified against its own animal's closest gallery
image. Impostor score: the same field image verified against the hardest other
animals, the ones DINOv2 ranks closest. Hard impostors are what a registry of
similar-looking cattle produces, so thresholds are set on them.

Writes eval/out/thresholds.json and eval/out/calibration.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.dataset import build_gallery, load_feat, make_split  # noqa: E402
from mombe.vision.muzzle import MuzzleEngine  # noqa: E402

OUT = ROOT / "eval" / "out"


def scores(engine, gallery, split, animals, per_animal=2, hard=3):
    gen, imp, rank1, rows = [], [], 0, 0
    for a in animals:
        for im in split.field_imgs[a][:per_animal]:
            q = load_feat(a, im)
            short = gallery.shortlist(q.embedding, k=hard + 1)
            rows += 1
            rank1 += short[0][0] == a
            own = [s for s in short if s[0] == a]
            if own:
                i = own[0][2][0]
            else:  # own animal outside the shortlist: verify its closest image anyway
                idx = [j for j, x in enumerate(gallery.animal_ids) if x == a]
                i = max(idx, key=lambda j: float(gallery.features[j].embedding @ q.embedding))
            gen.append(engine.verify(q, gallery.features[i])[1])
            for b, _, idxs in [s for s in short if s[0] != a][:hard]:
                imp.append(engine.verify(q, gallery.features[idxs[0]])[1])
    return np.array(gen), np.array(imp), rank1 / max(rows, 1)


def main():
    t0 = time.time()
    split = make_split()
    gallery = build_gallery(split)
    eng = MuzzleEngine()
    g, i, r1 = scores(eng, gallery, split, split.dev)
    imp_max, imp_995 = int(i.max()), float(np.percentile(i, 99.5))
    t_new = int(np.ceil(imp_995))
    t_match = max(int(imp_max) + 4, int(np.percentile(g, 5)))  # beyond every dev impostor
    th = {"t_new": t_new, "t_match": t_match}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "thresholds.json").write_text(json.dumps(th, indent=1))

    eng.t_new, eng.t_match = t_new, t_match
    gt, it, r1t = scores(eng, gallery, split, split.test)
    rep = {
        "dev": {"genuine_n": len(g), "impostor_n": len(i), "dinov2_rank1": round(r1, 4),
                "genuine_pct": {p: float(np.percentile(g, p)) for p in (1, 5, 25, 50)},
                "impostor_pct": {p: float(np.percentile(i, p)) for p in (50, 95, 99, 99.5, 100)}},
        "thresholds": th,
        "test": {"genuine_n": len(gt), "impostor_n": len(it), "dinov2_rank1": round(r1t, 4),
                 "false_match_rate": float((it >= t_match).mean()),
                 "false_non_match_rate": float((gt < t_match).mean()),
                 "genuine_in_ambiguous_band": float(((gt > t_new) & (gt < t_match)).mean()),
                 "genuine_called_new": float((gt <= t_new).mean()),
                 "impostor_in_ambiguous_band": float(((it > t_new) & (it < t_match)).mean())},
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / "calibration.json").write_text(json.dumps(rep, indent=1))
    np.savez(OUT / "scores.npz", dev_gen=g, dev_imp=i, test_gen=gt, test_imp=it)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
