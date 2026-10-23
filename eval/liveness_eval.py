"""Liveness against SYNTHETIC presentation attacks, scored with ISO/IEC 30107-3 metrics.

Input: the manifest from tools/synth_attacks.py (real live bursts plus rendered
print, screen and replay attacks). Steps:

1. cues: ALIKED features per frame (cached beside each rendered frame), then
   the five liveness cues of mombe.vision.liveness against the registry gallery
   (the pixel replay test reads the enrolled photos from data/muzzle).
2. calibrate: dev identities only. Live dev bursts give each cue's reference
   distribution; the threshold on the fused score is set from live dev bursts
   alone (leave-one-out tail probabilities) to a target dev BPCER. No attack
   sample touches calibration.
3. report: held-out identities only. APCER per attack type and the maximum
   over types, BPCER, BPCER20 (BPCER where the worst-type APCER reaches 5%),
   each with N and a Wilson 95% interval; per-cue catch rates; breakdowns by
   attack parameters.

Labels: attacks are input S (synthetic), truth G (attack by construction).
Live bursts are input R (real dataset photos), truth G (live by construction).

Usage: python eval/liveness_eval.py [--manifest build/attacks/manifest.jsonl] [--workers 4] [--target-bpcer 0.05]
Writes eval/out/liveness/{cues.jsonl,summary.json} and mombe/vision/liveness_calibration.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mombe.vision import liveness  # noqa: E402

OUT = ROOT / "eval" / "out" / "liveness"
ATTACKS = ("print", "screen", "replay")

_engine = _gallery = None


def _init(threads: int):
    global _engine, _gallery
    import cv2
    cv2.setNumThreads(threads)
    from eval.dataset import build_gallery, make_split
    from mombe.vision.muzzle import MuzzleEngine
    _engine = MuzzleEngine()
    _gallery = build_gallery(make_split())


def _enrolled_image(animal_id: str, image_id: str):
    import cv2
    from eval.dataset import IMGS, folder
    p = IMGS / folder(image_id) / f"{image_id}.jpg"
    return cv2.imread(str(p)) if p.exists() else None


def _features(path: str):
    import cv2
    from mombe.server.tools import _cached_features
    from mombe.vision.muzzle import Features
    f = _cached_features(path)
    if f is not None:
        return f
    npz = Path(path).with_suffix(".npz")
    if npz.exists():
        z = np.load(npz)
        return Features(z["kpts"], z["desc"], tuple(int(x) for x in z["size"]), z["emb"])
    f = _engine.extract(cv2.imread(path))
    tmp = npz.with_suffix(f".{os.getpid()}.tmp")
    with tmp.open("wb") as fh:                 # write then rename: a reader never sees half a file
        np.savez_compressed(fh, kpts=f.kpts, desc=f.desc, size=np.array(f.size), emb=f.embedding)
    os.replace(tmp, npz)
    return f


def _path(p: str) -> str:
    """Manifest paths are relative to the repo root."""
    return p if Path(p).is_absolute() else str(ROOT / p)


def _cues(s: dict) -> dict:
    """Cue values for one sample; a sample that fails twice is kept as an error row and left out of the metrics."""
    err = None
    for _ in range(2):
        try:
            return _cues_once(s)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:200]}"
    return {k: s[k] for k in ("id", "identity", "group", "kind", "registered")} | {"error": err, "ms": 0}


def _cues_once(s: dict) -> dict:
    import cv2
    t = time.perf_counter()
    paths = [_path(p) for p in s["frames"]]
    frames = [cv2.imread(p) for p in paths]
    feats = [_features(p) for p in paths]
    vals = liveness.cue_values(frames, feats, _gallery, _enrolled_image)
    return {k: s[k] for k in ("id", "identity", "group", "kind", "registered")} | \
        {"meta": s.get("meta", {}), "values": vals, "ms": round((time.perf_counter() - t) * 1000)}


def subsample(samples: list[dict], per_cell: int) -> list[dict]:
    """At most `per_cell` attacks of each type per group, spread evenly over identities. Every live burst stays."""
    if not per_cell:
        return samples
    cells = defaultdict(list)
    for s in samples:
        cells[(s["group"], s["kind"])].append(s)
    keep = set()
    for (_, kind), ss in cells.items():
        if kind == "bona_fide" or len(ss) <= per_cell:
            keep.update(s["id"] for s in ss)
        else:
            keep.update(ss[i]["id"] for i in np.linspace(0, len(ss) - 1, per_cell).round().astype(int))
    return [s for s in samples if s["id"] in keep]


# -- statistics -----------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def rate(k: int, n: int) -> dict:
    return {"rate": round(k / n, 4) if n else None, "k": int(k), "n": int(n), "wilson95": wilson(k, n)}


def calibrate(dev_live: list[dict], target: float, fusion: str = "fisher", cues=liveness.CUES) -> dict:
    cues = list(cues)
    live = {c: sorted(round(r["values"][c]["score"], 4) for r in dev_live
                      if r["values"][c].get("available", True) and "score" in r["values"][c]) for c in cues}
    calib = {"cues": cues, "fusion": fusion, "live": live}
    # leave-one-out fused scores of the live dev bursts set the threshold
    loo = []
    for r in dev_live:
        own = {c: list(calib["live"][c]) for c in cues}
        for c in cues:
            v = r["values"][c]
            if v.get("available", True) and "score" in v and round(v["score"], 4) in own[c]:
                own[c].remove(round(v["score"], 4))
        loo.append(liveness.fuse(r["values"], {**calib, "live": own})[0])
    loo = np.sort(np.array(loo))
    k = int(math.floor(target * len(loo)))
    # largest threshold that rejects at most `target` of live dev bursts (live when score > threshold)
    thr = float(loo[k - 1]) if k >= 1 else 0.0
    calib.update(threshold=round(thr, 6), target_dev_bpcer=target, n_dev_live=len(dev_live),
                 dev_bpcer_loo=round(float((loo <= thr).mean()), 4),
                 note="threshold from live dev bursts only (leave-one-out tail probabilities); no attack data")
    return calib


def scored(rows: list[dict], calib: dict) -> list[dict]:
    out = []
    for r in rows:
        s, worst, ps = liveness.fuse(r["values"], calib)
        out.append({**r, "fused": s, "worst": worst, "p": ps})
    return out


def metrics(rows: list[dict], thr: float) -> dict:
    live = [r for r in rows if r["kind"] == "bona_fide"]
    rep = {"bpcer": rate(sum(r["fused"] <= thr for r in live), len(live)), "apcer": {}}
    for k in ATTACKS:
        att = [r for r in rows if r["kind"] == k]
        if att:
            rep["apcer"][k] = rate(sum(r["fused"] > thr for r in att), len(att))
    worst = max(rep["apcer"], key=lambda k: rep["apcer"][k]["rate"]) if rep["apcer"] else None
    rep["apcer_max"] = {"type": worst, **rep["apcer"][worst]} if worst else None
    att_all = [r for r in rows if r["kind"] in ATTACKS]
    rep["apcer_pooled"] = rate(sum(r["fused"] > thr for r in att_all), len(att_all))
    return rep


def bpcer_at_apcer(rows: list[dict], apcer: float = 0.05, kinds=ATTACKS) -> dict:
    """BPCER at the strictest-needed threshold where the worst listed attack type has APCER <= apcer."""
    live = np.array([r["fused"] for r in rows if r["kind"] == "bona_fide"])
    cands = np.unique(np.concatenate([[r["fused"] for r in rows], [0.0, 1.0]]))
    for t in cands:                      # ascending: the loosest threshold that meets the APCER bound
        ok = True
        for k in kinds:
            att = np.array([r["fused"] for r in rows if r["kind"] == k])
            if len(att) and (att > t).mean() > apcer:
                ok = False
                break
        if ok:
            k_rej = int((live <= t).sum())
            return {"threshold": round(float(t), 6), **rate(k_rej, len(live))}
    return {"threshold": 1.0, **rate(len(live), len(live))}


def cue_catch(rows: list[dict], thr: float) -> dict:
    """Rejections per sample type, credited to the most suspicious cue of each rejected sample."""
    out = {}
    for k in ("bona_fide",) + ATTACKS:
        sub = [r for r in rows if r["kind"] == k]
        out[k] = {c: rate(sum(r["fused"] <= thr and r["worst"] == c for r in sub), len(sub)) for c in liveness.CUES}
    return out


def breakdown(rows: list[dict], thr: float) -> dict:
    groups = defaultdict(list)
    for r in rows:
        m = r.get("meta") or {}
        if r["kind"] == "print":
            groups[f"print/{m.get('paper')}"].append(r)
            groups["print/curled" if m.get("curl_cm", 0) > 0 else "print/flat"].append(r)
            groups["print/sheen" if m.get("sheen", 0) > 0 else "print/no_sheen"].append(r)
        elif r["kind"] == "screen":
            p = m.get("camera_px_per_screen_px", 0)
            groups["screen/px_per_screen_px<1.5" if p < 1.5 else "screen/px_per_screen_px>=1.5"].append(r)
            groups["screen/glare" if m.get("glare", 0) > 0 else "screen/no_glare"].append(r)
        elif r["kind"] == "replay":
            groups[f"replay/{m.get('mode')}"].append(r)
        elif r["kind"] == "bona_fide":
            groups["bona_fide/registered" if r["registered"] else "bona_fide/unregistered"].append(r)
    res = {}
    for g, sub in sorted(groups.items()):
        if g.startswith("bona_fide"):
            res[g] = {"bpcer": rate(sum(r["fused"] <= thr for r in sub), len(sub))}
        else:
            res[g] = {"apcer": rate(sum(r["fused"] > thr for r in sub), len(sub))}
    return res


def cue_stats(rows: list[dict]) -> dict:
    out = {}
    for k in ("bona_fide",) + ATTACKS:
        sub = [r for r in rows if r["kind"] == k]
        out[k] = {}
        for c in liveness.CUES:
            v = [r["values"][c]["score"] for r in sub if r["values"][c].get("available", True) and "score" in r["values"][c]]
            out[k][c] = {"n": len(v), "available": round(len(v) / max(1, len(sub)), 3),
                         **({f"p{q}": round(float(np.percentile(v, q)), 4) for q in (1, 5, 50, 95, 99)} if v else {})}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "build" / "attacks" / "manifest.jsonl"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--target-bpcer", type=float, default=0.05)
    ap.add_argument("--per-cell", type=int, default=0, help="at most N attacks of each type per group (0: all)")
    ap.add_argument("--reuse", action="store_true", help="score eval/out/liveness/cues.jsonl as it stands")
    ap.add_argument("--resume", action="store_true", help="keep finished rows of cues.jsonl, compute the rest")
    ap.add_argument("--no-write-calibration", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cues_path = OUT / "cues.jsonl"
    manifest = [json.loads(x) for x in Path(a.manifest).read_text().splitlines() if x.strip()]
    samples = subsample(manifest, a.per_cell)

    def saved() -> list[dict]:
        out = []
        for x in cues_path.read_text().splitlines() if cues_path.exists() else []:
            try:
                out.append(json.loads(x))
            except json.JSONDecodeError:      # a line cut off by an interrupted run
                pass
        return out

    t0 = time.time()
    if a.reuse:
        rows = saved()
    else:
        want = {s["id"] for s in samples}
        rows = [r for r in saved() if r["id"] in want and "error" not in r] if a.resume else []
        have = {r["id"] for r in rows}
        todo = [s for s in samples if s["id"] not in have]
        if a.workers == 1:
            _init(a.threads)
            it = map(_cues, todo)
        else:
            it = ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.threads,)).map(_cues, todo, chunksize=4)
        with cues_path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
            for i, r in enumerate(it, 1):
                rows.append(r)
                f.write(json.dumps(r, default=str) + "\n")
                f.flush()
                if i % 100 == 0 or i == len(todo):
                    print(f"{i}/{len(todo)} cues  {time.time() - t0:.0f}s", flush=True)

    errors = [r["id"] for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    dev = [r for r in rows if r["group"] == "dev"]
    held = [r for r in rows if r["group"] == "heldout"]
    dev_live = [r for r in dev if r["kind"] == "bona_fide"]

    # Design choice on dev identities only: the fusion rule with the lower worst-type dev APCER.
    design = {}
    for fusion in ("fisher", "min"):
        c = calibrate(dev_live, a.target_bpcer, fusion)
        design[fusion] = metrics(scored(dev, c), c["threshold"])["apcer_max"]
    fusion = min(design, key=lambda f: (design[f]["rate"], f != "fisher"))
    calib = calibrate(dev_live, a.target_bpcer, fusion)
    calib["source"] = {"manifest": Path(a.manifest).name, "seed": samples[0].get("seed"),
                       "dev_identities": len({r["identity"] for r in dev}),
                       "fusion_choice_dev_apcer_max": {f: d["rate"] for f, d in design.items()}}
    if not a.no_write_calibration:
        tmp = liveness.CALIBRATION.with_suffix(".tmp")
        tmp.write_text(json.dumps(calib, indent=1))
        os.replace(tmp, liveness.CALIBRATION)
    thr = calib["threshold"]
    dev_s, held_s = scored(dev, calib), scored(held, calib)

    # Each cue's share: drop one cue, or keep one, recalibrate on dev, measure held-out.
    ablation = {}
    for name, cues in [(f"without_{c}", [x for x in liveness.CUES if x != c]) for c in liveness.CUES] + \
                      [(f"only_{c}", [c]) for c in liveness.CUES]:
        c = calibrate(dev_live, a.target_bpcer, fusion, cues)
        m = metrics(scored(held, c), c["threshold"])
        ablation[name] = {"bpcer": m["bpcer"]["rate"], **{f"apcer_{k}": v["rate"] for k, v in m["apcer"].items()}}

    summary = {
        "label": "SYNTHETIC presentation attacks (rendered from real photos); live bursts are real dataset photos",
        "evidence": {"apcer": {"input": "S", "truth": "G"}, "bpcer": {"input": "R", "truth": "G"}},
        "standard": "ISO/IEC 30107-3 metrics: APCER per PAI species, APCER_max, BPCER, BPCER20",
        "split": {"dev_identities": len({r["identity"] for r in dev}),
                  "heldout_identities": len({r["identity"] for r in held}),
                  "identity_disjoint": not ({r["identity"] for r in dev} & {r["identity"] for r in held}),
                  "sampling": {"manifest_samples": len(manifest), "evaluated": len(rows), "per_cell": a.per_cell,
                               "rule": "every live burst; at most per_cell attacks per type and group, evenly over identities",
                               "errors": errors},
                  "counts": {g: {k: sum(1 for r in rr if r["kind"] == k) for k in ("bona_fide",) + ATTACKS}
                             for g, rr in (("dev", dev), ("heldout", held))}},
        "operating_point": {"threshold": thr, "fusion": fusion,
                            "rule": ("live when the product of cue tail probabilities > threshold" if fusion == "fisher"
                                     else "live when the smallest cue tail probability > threshold"),
                            "chosen_on": "live dev bursts only (leave-one-out)", "target_dev_bpcer": a.target_bpcer,
                            "dev_bpcer_loo": calib["dev_bpcer_loo"],
                            "fusion_chosen_on": "dev identities: lower worst-type APCER",
                            "fusion_dev_apcer_max": {f: d["rate"] for f, d in design.items()}},
        "heldout": {**metrics(held_s, thr), "bpcer20": bpcer_at_apcer(held_s),
                    "bpcer20_per_type": {k: bpcer_at_apcer(held_s, kinds=(k,)) for k in ATTACKS},
                    "cue_catch": cue_catch(held_s, thr), "breakdown": breakdown(held_s, thr),
                    "ablation_recalibrated_on_dev": ablation},
        "dev": {**metrics(dev_s, thr), "bpcer20": bpcer_at_apcer(dev_s)},
        "cue_distributions_heldout": cue_stats(held),
        "timing_ms": {"median": float(np.median([r["ms"] for r in rows])),
                      "p95": float(np.percentile([r["ms"] for r in rows], 95))},
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    h = summary["heldout"]
    print(json.dumps({"threshold": thr, "bpcer": h["bpcer"], "apcer": h["apcer"], "apcer_max": h["apcer_max"],
                      "bpcer20": h["bpcer20"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
