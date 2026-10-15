"""One tuning trial of the identity workload, run on a hub instance.

Given a PipelineConfig as JSON, builds the engine, re-extracts a fixed mini
gallery (40 animals, one image each) under that configuration, identifies a
second image of each animal, and reports:

  accuracy         share of queries matched to their own animal at the live thresholds
  genuine_p10      10th percentile of verified inliers for true pairs
  impostor_max     strongest verified impostor (hardest DINOv2 neighbours)
  separation       genuine_p10 / impostor_max
  latency          median ms: extract, embed, verify, identify end to end

Prints one JSON line. Errors print {"error": ...} so the tuner can steer.

Usage: python tools/trial.py --config '{"aliked_input": 512}'
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

N_ANIMALS = 40


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="{}")
    a = ap.parse_args()
    try:
        import cv2
        import numpy as np
        from mombe.vision.muzzle import Gallery, MuzzleEngine, PipelineConfig, identify

        cfg = PipelineConfig(**json.loads(a.config))
        th = json.loads((ROOT / "eval" / "out" / "thresholds.json").read_text())
        t0 = time.perf_counter()
        eng = MuzzleEngine(config=cfg, t_match=th["t_match"], t_new=th["t_new"])
        load_s = time.perf_counter() - t0

        meta = json.loads((ROOT / "var" / "gallery.json").read_text())
        by: dict[str, list[str]] = {}
        for an, im in zip(meta["animal_ids"], meta["image_ids"]):
            by.setdefault(an, []).append(im)
        animals = [an for an in sorted(by) if len(by[an]) >= 2][:N_ANIMALS]
        path = lambda im: str(ROOT / "data" / "muzzle" / "_".join(im.split("_")[:2]) / f"{im}.jpg")

        g, ext_ms = Gallery(), []
        for an in animals:
            img = cv2.imread(path(by[an][0]))
            t = time.perf_counter()
            g.add(an, by[an][0], eng.extract(img))
            ext_ms.append((time.perf_counter() - t) * 1000)

        correct, gen, imp, id_ms, ver_ms, imp_pairs = 0, [], [], [], [], []
        for an in animals:
            img = cv2.imread(path(by[an][1]))
            t = time.perf_counter()
            q = eng.extract(img)
            res = identify(eng, g, [q])
            id_ms.append((time.perf_counter() - t) * 1000)
            correct += res.decision == "match" and res.animal_id == an
            own = g.animal_ids.index(an)
            t = time.perf_counter()
            gen.append(eng.verify(q, g.features[own])[1])
            ver_ms.append((time.perf_counter() - t) * 1000)
            for b, _, idxs in [s for s in g.shortlist(q.embedding, k=3) if s[0] != an][:2]:
                v = eng.verify(q, g.features[idxs[0]])[1]
                imp.append(v)
                imp_pairs.append((v, an, b))

        gen_p10 = float(np.percentile(gen, 10))
        imp_max = max(imp) if imp else 0
        out = {
            "config": cfg.__dict__, "opencv": cv2.__version__,
            "accuracy": round(correct / len(animals), 4), "n": len(animals),
            "genuine_p10": round(gen_p10, 1), "genuine_median": float(np.median(gen)),
            "impostor_max": int(imp_max),
            "top_impostors": [{"inliers": v, "query": q_, "candidate": c_} for v, q_, c_ in sorted(imp_pairs, reverse=True)[:3]], "separation": round(gen_p10 / max(imp_max, 1), 2),
            "latency_ms": {"extract": round(statistics.median(ext_ms), 1), "verify": round(statistics.median(ver_ms), 1),
                           "identify_end_to_end": round(statistics.median(id_ms), 1)},
            "load_s": round(load_s, 2), "thresholds": th,
        }
    except Exception as e:  # the tuner reads failures as evidence
        out = {"error": f"{type(e).__name__}: {str(e)[:400]}", "trace": traceback.format_exc()[-600:], "config": json.loads(a.config)}
    print("TRIAL_RESULT " + json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
