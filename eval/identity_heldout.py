"""Held-out identity: the BC et al. 2026 leakage-controlled open-set protocol, run on Mombe.

The earlier identity figures (eval/calibrate.py) are development numbers: thresholds
were set and measured on one fixed split, and gallery and probe images interleave
inside one camera burst. This script replaces them with a held-out protocol rebuilt
from BC, Chaudhary and Nepal, arXiv 2608.28663 (Aug 2026), on the same Zenodo set:

* duplicate identities are found and removed first (stages screen, confirm, reconcile);
* 5 random identity-disjoint partitions, each 124 known / 50 validation-unknown /
  75 test-unknown identities (scaled if the duplicate count differs from theirs);
* known animals enrol their first 3 frames in capture order; probes come from later
  frames of the same animal; unknown animals are never enrolled;
* thresholds are set on validation (validation probes of known animals plus the
  validation-unknown pool) and transferred unchanged to test.

Stages (python eval/identity_heldout.py <stage> ...):
  dups       duplicate audit: exhaustive classical screen over every identity pair (first and
             last frame each; ALIKED mutual nearest neighbours + ratio test + MAGSAC), then screen
             hits and the earlier audit pairs re-checked 3 x 3 with Mombe's verifier (LightGlue + MAGSAC)
  reconcile  camera-frame provenance, decision per pair, duplicate groups, removal list, BC et al. comparison
  plan       drop duplicate identities, capture-order enrolment, 5 partitions
  pairs      verification scores for one shard of probes (local, hub or Kaggle)
  kaggle     write self-contained Kaggle kernels, one per shard
  report     DIR@FAR, rank-k, recall@20, three-way decisions, latency, cost inputs
  bench      per-stage latency on this machine
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
import math
import os
import pickle
import platform
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

OUT = ROOT / "eval" / "out" / "identity_heldout"
IMGS = ROOT / "data" / "muzzle"
FEAT = ROOT / "data" / "features"
MODELS = ROOT / "models"

NAME = re.compile(r"^(cattle_\d{4})_(DSCF|DJI_)(\d{4})$")
N_ENROL = 3            # enrolment frames per known animal: the first frames in capture order
P_MAX = 8              # probe frames per animal, evenly spaced over the later frames
POOLS = (124, 50, 75)  # BC et al. pools for 249 identities
N_PARTITIONS = 5
PARTITION_SEED = 2608  # partition p uses random.Random(PARTITION_SEED + p)
SHORTLIST = 5          # animals verified per probe; identify() escalates to 5 before saying new
VERIFY_TOP = 2         # deployed identify() verifies the top 2 first (tuner run tune-1015)
BURST_GAP = 50         # frames: a larger gap inside one animal starts a new burst
T_DUP = 90             # LightGlue median at or above: duplicate (the frozen t_match)
T_REVIEW = 46          # LightGlue median at or above (below T_DUP): image review decides
FROZEN = {"t_new": 20, "t_match": 90}  # eval/out/thresholds.json, set on the development split
BC = {  # BC, Chaudhary, Nepal 2026, arXiv 2608.28663: figures from the paper's tables and text
    "duplicates_removed": {"identities": 19, "images": 280}, "after_removal": {"identities": 249, "images": 4643},
    "oracle_dir_far_1e-1_1e-2_1e-3": {"hybrid_cnn_vit": [98.31, 96.37, 93.59], "megadescriptor_l": [99.29, 98.10, 96.07]},
    "transferred_1pct": {"hybrid_cnn_vit": {"detection": 96.5, "realised_far": 1.03},
                         "megadescriptor_l": {"detection": 98.2, "realised_far": 2.44}},
    "closed_set_rank1": {"hybrid_cnn_vit": "99.53 +/- 0.42", "megadescriptor_l": "99.86 +/- 0.13"},
    "split_within_identity": "image level, 50% train / 25% gallery / 25% probe; gallery prototype = mean embedding",
    "duplicate_list": "github.com/Dharmendra016/Muzzlenet, HTTP 404 on 2026-10-20 and 2026-10-21",
}


# --------------------------------------------------------------------------- dataset

def parse(image_id: str):
    m = NAME.match(image_id)
    return (m.group(1), m.group(2), int(m.group(3))) if m else None


def image_index(img_root: Path) -> tuple[dict[str, list[str]], dict[str, str], int]:
    """Every <root>/**/cattle_XXXX/cattle_XXXX_<cam><frame>.jpg: identity -> image ids,
    image id -> path, and the count of rejected files."""
    ids: dict[str, list[str]] = collections.defaultdict(list)
    paths: dict[str, str] = {}
    rejected = 0
    for p in Path(img_root).rglob("*.jpg"):
        q = parse(p.stem)
        if q is None or p.parent.name != q[0]:
            rejected += 1
            continue
        if p.stem in paths:
            continue
        ids[q[0]].append(p.stem)
        paths[p.stem] = str(p)
    return dict(ids), paths, rejected


def capture_order(images: list[str]) -> list[str]:
    """One animal's frames in capture order. The 4-digit frame counter wraps at 9999."""
    by_cam = collections.defaultdict(list)
    for im in images:
        by_cam[parse(im)[1]].append(parse(im)[2])
    wrap = {c: max(v) - min(v) > 5000 for c, v in by_cam.items()}

    def k(im):
        _, cam, n = parse(im)
        return (0 if cam == "DSCF" else 1, n + 10000 if wrap[cam] and n < 5000 else n)
    return sorted(images, key=k)


def frame_pos(order: list[str]) -> dict[str, tuple[int, float]]:
    """(burst index, unwrapped frame number) per image, for frames already in capture order."""
    out, burst, prev = {}, 0, None
    for im in order:
        _, cam, n = parse(im)
        if prev is not None:
            pcam, pn = prev
            if cam == pcam and n < pn - 5000:
                n += 10000
            if cam != pcam or n - pn > BURST_GAP:
                burst += 1
        out[im] = (burst, float(n))
        prev = (cam, n)
    return out


def folder(image_id: str) -> str:
    return "_".join(image_id.split("_")[:2])


# --------------------------------------------------------------------------- features and workers

def load_cached(image_id: str):
    from mombe.vision.muzzle import Features
    z = np.load(FEAT / folder(image_id) / f"{image_id}.npz")
    return Features(z["kpts"], z["desc"], tuple(int(x) for x in z["size"]), z["emb"])


def as_tuple(f):
    return (f.kpts, f.desc, f.size, f.embedding)


_W: dict = {}  # per-process worker state


def _init_worker(models: str, feats_file: str | None, threads: int):
    import cv2
    from mombe.vision.muzzle import MuzzleEngine, PipelineConfig
    cv2.setNumThreads(threads)
    _W["engine"] = MuzzleEngine(Path(models), config=PipelineConfig(threads=threads))
    _W["feats"] = {}
    if feats_file:
        with open(feats_file, "rb") as f:
            _W["feats"] = pickle.load(f)


def _extract_one(item):
    import cv2
    image_id, path = item
    t = time.perf_counter()
    img = cv2.imread(path)
    if img is None:
        return image_id, None, 0.0
    return image_id, as_tuple(_W["engine"].extract(img)), time.perf_counter() - t


def _verify_task(item):
    from mombe.vision.muzzle import Features
    q, targets = item
    F = _W["feats"]
    fq = Features(*F[q])
    out = []
    for g in targets:
        t = time.perf_counter()
        m, inl, _, _ = _W["engine"].verify(fq, Features(*F[g]))
        out.append((q, g, int(m), int(inl), round(time.perf_counter() - t, 3)))
    return out


def pool_ctx():
    import multiprocessing as mp
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"   # inherited by spawned workers before they import numpy
    return mp.get_context("spawn")


def host_info() -> dict:
    import cv2
    return {"machine": platform.machine(), "system": platform.system(), "processor": platform.processor(),
            "cpus": os.cpu_count(), "python": platform.python_version(), "opencv": cv2.__version__}


# --------------------------------------------------------------------------- screen

def _screen_init(sel_file: str):
    import cv2
    cv2.setNumThreads(1)
    with open(sel_file, "rb") as f:
        _W["k"], _W["d"], _W["own"] = pickle.load(f)


def mnn_magsac(ka, da, kb, db, S=None) -> int:
    """Classical screen: mutual nearest neighbours, ratio 0.9, MAGSAC++ homography inliers."""
    import cv2
    if len(ka) < 8 or len(kb) < 8:
        return 0
    if S is None:
        S = da @ db.T
    i12 = S.argmax(1)
    i21 = S.argmax(0)
    m = np.nonzero(i21[i12] == np.arange(len(i12)))[0]
    if len(m) < 8:
        return 0
    top2 = np.partition(S[m], -2, axis=1)[:, -2:]
    d1 = np.sqrt(np.maximum(2 - 2 * top2[:, 1], 0))
    d2 = np.sqrt(np.maximum(2 - 2 * top2[:, 0], 0))
    keep = m[d1 < 0.9 * d2]
    if len(keep) < 8:
        return 0
    H, mask = cv2.findHomography(ka[keep], kb[i12[keep]], cv2.USAC_MAGSAC, 4.0)
    return 0 if mask is None else int(mask.sum())


def _screen_task(i):
    K, D, own = _W["k"], _W["d"], _W["own"]
    js = [j for j in range(i + 1, len(D)) if own[j] != own[i]]
    out = []
    for s in range(0, len(js), 48):
        blk = js[s:s + 48]
        big = D[i] @ np.concatenate([D[j] for j in blk]).T
        off = 0
        for j in blk:
            n = len(D[j])
            out.append((i, j, mnn_magsac(K[i], D[i], K[j], D[j], big[:, off:off + n])))
            off += n
    return out


def extract_many(items: list[tuple[str, str]], models: str, workers: int, threads: int, feats: dict, label: str) -> list[float]:
    """Extract features for (image id, path) items in a worker pool; fills feats in place."""
    if not items:
        return []
    t0, dts = time.time(), []
    with pool_ctx().Pool(workers, initializer=_init_worker, initargs=(models, None, threads)) as pool:
        for n, (x, f, dt) in enumerate(pool.imap_unordered(_extract_one, items, chunksize=2)):
            if f is not None:
                feats[x] = f
                dts.append(dt)
            if n % 100 == 0:
                print(f"{label} extract {n}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    return dts


def stage_dups(a):
    """Duplicate-identity audit in one run: exhaustive classical screen, then LightGlue confirmation.

    Frames per identity: first, middle and last in capture order over every image of the animal.
    Screen: first and last frame, all 4 combinations for every identity pair.
    Confirm: every pair with screen max >= a.screen_cut (top a.max_confirm), the 2026-10-14 audit
    pairs and any --extra pairs, verified 3 x 3 with LightGlue + MAGSAC (Mombe's verifier)."""
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx, paths, rejected = image_index(Path(a.images))
    ids = sorted(idx)
    print(f"images {len(paths)} rejected {rejected} identities {len(ids)}", flush=True)
    pick = {}
    for an in ids:
        o = capture_order(idx[an])
        pick[an] = [o[0], o[len(o) // 2], o[-1]]
    feats: dict = {}
    todo = []
    for an in ids:
        for im in pick[an]:
            if a.use_cache and (FEAT / folder(im) / f"{im}.npz").exists():
                feats[im] = as_tuple(load_cached(im))
            else:
                todo.append((im, paths[im]))
    t0 = time.time()
    dts = extract_many(todo, a.models, a.workers, a.threads, feats, "dups")
    t_ext = time.time() - t0
    sel = [im for an in ids for im in (pick[an][0], pick[an][2])]
    K = [feats[im][0] for im in sel]
    D = [feats[im][1] for im in sel]
    own = [folder(im) for im in sel]
    tmp = Path(a.work) / "_screen_sel.pkl"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        pickle.dump((K, D, own), f)
    t1, res = time.time(), []
    order = list(range(len(sel)))
    order = order[0::2] + order[1::2]
    with pool_ctx().Pool(a.workers, initializer=_screen_init, initargs=(str(tmp),)) as pool:
        for n, r in enumerate(pool.imap_unordered(_screen_task, order, chunksize=2)):
            res.extend(r)
            if n % 50 == 0:
                print(f"screen anchors {n}/{len(order)} pairs {len(res)} {time.time() - t1:.0f}s", flush=True)
    t_screen = time.time() - t1
    tmp.unlink(missing_ok=True)
    best: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for i, j, s in res:
        best[tuple(sorted((own[i], own[j])))].append(s)
    rows = [{"a": x, "b": y, "max": max(v), "min": min(v), "scores": v} for (x, y), v in best.items()]
    rows.sort(key=lambda r: -r["max"])
    allmax = np.array([r["max"] for r in rows])
    screen = {
        "method": "ALIKED mutual nearest neighbours, ratio 0.9, MAGSAC++ 4 px; first and last frame in capture order per identity, all 4 combinations per identity pair",
        "identities": len(ids), "images": len(sel), "image_pairs": len(res), "identity_pairs": len(rows),
        "wall_s": round(t_screen, 1), "workers": a.workers, "host": host_info(),
        "identity_pair_max_percentiles": {str(p): float(np.percentile(allmax, p)) for p in (50, 90, 99, 99.9, 99.99)},
        "histogram_max": {f"{lo}-{hi - 1}": int(((allmax >= lo) & (allmax < hi)).sum())
                          for lo, hi in [(0, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 60), (60, 100), (100, 10000)]},
        "top": rows[:300],
    }
    (out_dir / "dup_screen.json").write_text(json.dumps(screen, indent=1))
    print(json.dumps({k: v for k, v in screen.items() if k != "top"}, indent=1), flush=True)
    # confirmation with Mombe's verifier
    cand: dict[tuple[str, str], dict] = {}
    for r in rows[:a.max_confirm]:
        if r["max"] >= a.screen_cut:
            cand[(r["a"], r["b"])] = {"why": ["screen"], "screen": r}
    smap = {(r["a"], r["b"]): r for r in rows}
    for d in a.prior_pairs:
        x, y, s = d.split(",")
        k = tuple(sorted((x, y)))
        cand.setdefault(k, {"why": [], "screen": smap.get(k)})
        cand[k]["why"].append("audit_1014")
        cand[k]["audit_1014"] = int(s)
    for p in a.extra:
        k = tuple(sorted(p.split(",")))
        cand.setdefault(k, {"why": [], "screen": smap.get(k)})["why"].append("extra")
    ff = Path(a.work) / "_confirm_feats.pkl"
    with open(ff, "wb") as f:
        pickle.dump({im: feats[im] for an in ids for im in pick[an]}, f)
    jobs = [(qx, pick[y]) for (x, y) in cand for qx in pick[x]]
    t2, conf = time.time(), []
    with pool_ctx().Pool(a.workers, initializer=_init_worker, initargs=(a.models, str(ff), a.threads)) as pool:
        for r in pool.imap_unordered(_verify_task, jobs):
            conf.extend(r)
            if len(conf) % 90 == 0:
                print(f"confirm {len(conf)}/{len(jobs) * 3} pairs {time.time() - t2:.0f}s", flush=True)
    t_conf = time.time() - t2
    ff.unlink(missing_ok=True)
    per: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for q, g, m, inl, dt in conf:
        per[tuple(sorted((folder(q), folder(g))))].append(inl)
    crow = []
    for (x, y), v in cand.items():
        s = per[(x, y)]
        sc = v.get("screen") or {}
        crow.append({"a": x, "b": y, "why": v["why"], "images_a": len(idx[x]), "images_b": len(idx[y]),
                     "screen_max": sc.get("max"), "screen_min": sc.get("min"), "audit_1014_inliers": v.get("audit_1014"),
                     "lightglue_3x3": sorted(s), "lightglue_median": float(np.median(s)),
                     "lightglue_min": int(min(s)), "lightglue_max": int(max(s)),
                     "frames_a": pick[x], "frames_b": pick[y]})
    crow.sort(key=lambda r: -r["lightglue_median"])
    confirm = {"lightglue_pairs": len(conf), "wall_s": round(t_conf, 1), "workers": a.workers,
               "threads_per_worker": a.threads, "extract_wall_s": round(t_ext, 1), "images_extracted": len(dts),
               "extract_s_per_image_median_in_worker": float(np.median(dts)) if dts else None,
               "lightglue_s_per_pair_median_in_worker": float(np.median([r[4] for r in conf])) if conf else None,
               "screen_cut": a.screen_cut, "max_confirm": a.max_confirm, "host": host_info(), "rows": crow}
    (out_dir / "dup_confirm.json").write_text(json.dumps(confirm, indent=1))
    for r in crow:
        print(r["a"], r["b"], r["why"], "screen", r["screen_max"], "LG", r["lightglue_3x3"], flush=True)


def frame_sharing_pairs(idx: dict[str, list[str]]) -> dict[tuple[str, str], list[tuple[str, int]]]:
    """Identity pairs whose folders hold the same camera frame number (BC et al. flag duplicates
    'by camera-frame provenance'). Frame counters repeat across sessions and one telephoto frame
    can hold two animals, so a shared number is a flag only."""
    by = collections.defaultdict(set)
    for an, ims in idx.items():
        for im in ims:
            by[parse(im)[1:]].add(an)
    out = collections.defaultdict(list)
    for k, s in by.items():
        s = sorted(s)
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                out[(s[i], s[j])].append(k)
    return {k: sorted(v) for k, v in out.items()}


def stage_prov(a):
    """Provenance check: for every identity pair sharing camera frame numbers, verify the two crops
    of up to 3 shared frames (first, middle, last shared number) with Mombe's verifier. Crops of one
    animal from one photo match with hundreds of inliers; two animals in one photo do not."""
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx, paths, rejected = image_index(Path(a.images))
    fs = frame_sharing_pairs(idx)
    print(f"images {len(paths)} rejected {rejected} identities {len(idx)} frame-sharing pairs {len(fs)}", flush=True)
    jobs, feats, todo = [], {}, set()
    for (x, y), keys in sorted(fs.items()):
        pick = sorted({keys[0], keys[len(keys) // 2], keys[-1]})
        for cam, n in pick:
            ia, ib = f"{x}_{cam}{n:04d}", f"{y}_{cam}{n:04d}"
            jobs.append((ia, [ib]))
            todo.update((ia, ib))
    t0 = time.time()
    extract_many(sorted((i, paths[i]) for i in todo), a.models, a.workers, a.threads, feats, "prov")
    t_ext = time.time() - t0
    ff = Path(a.work) / "_prov_feats.pkl"
    ff.parent.mkdir(parents=True, exist_ok=True)
    with open(ff, "wb") as f:
        pickle.dump(feats, f)
    t1, res = time.time(), []
    with pool_ctx().Pool(a.workers, initializer=_init_worker, initargs=(a.models, str(ff), a.threads)) as pool:
        for r in pool.imap_unordered(_verify_task, jobs):
            res.extend(r)
    t_ver = time.time() - t1
    ff.unlink(missing_ok=True)
    import cv2
    per = collections.defaultdict(list)
    for q, g, m, inl, dt in res:
        sa = cv2.imread(paths[q], cv2.IMREAD_REDUCED_GRAYSCALE_2).shape
        sb = cv2.imread(paths[g], cv2.IMREAD_REDUCED_GRAYSCALE_2).shape
        per[(folder(q), folder(g))].append({"frame": q.split("_")[-1], "inliers": inl, "half_size_a": sa, "half_size_b": sb})
    rows = []
    for (x, y), keys in fs.items():
        v = per[(x, y)]
        s = [r["inliers"] for r in v]
        rows.append({"a": x, "b": y, "images_a": len(idx[x]), "images_b": len(idx[y]), "shared_frames": len(keys),
                     "checked": v, "inliers_median": float(np.median(s)) if s else None, "inliers_max": max(s) if s else None})
    rows.sort(key=lambda r: -(r["inliers_median"] or 0))
    out = {"method": "same camera frame number in two identity folders; LightGlue+MAGSAC inliers between the two crops of up to 3 shared frames",
           "pairs": len(rows), "lightglue_pairs": len(res), "extract_wall_s": round(t_ext, 1), "verify_wall_s": round(t_ver, 1),
           "workers": a.workers, "host": host_info(), "rows": rows}
    (out_dir / "prov_check.json").write_text(json.dumps(out, indent=1))
    for r in rows:
        print(r["a"], r["b"], r["shared_frames"], [c["inliers"] for c in r["checked"]], flush=True)


# --------------------------------------------------------------------------- confirm + reconcile

def provenance(a: str, b: str, idx: dict[str, list[str]], paths: dict[str, str]) -> dict:
    """Camera-frame provenance for an identity pair: shared frame numbers, near-identical
    frames among them (64x48 grey thumbnails, mean abs diff < 3), and DSCF range gap."""
    import cv2
    fa = {parse(x)[1:]: x for x in idx[a]}
    fb = {parse(x)[1:]: x for x in idx[b]}
    shared = sorted(set(fa) & set(fb))
    near = 0
    for k in shared:
        ta = cv2.imread(paths[fa[k]], cv2.IMREAD_REDUCED_GRAYSCALE_8)
        tb = cv2.imread(paths[fb[k]], cv2.IMREAD_REDUCED_GRAYSCALE_8)
        if ta is None or tb is None or ta.shape != tb.shape:
            continue
        ta = cv2.resize(ta, (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
        tb = cv2.resize(tb, (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
        near += float(np.abs(ta - tb).mean()) < 3.0

    def rng(x):
        n = [parse(i)[2] for i in x if parse(i)[1] == "DSCF"]
        return [min(n), max(n)] if n else None
    ra, rb = rng(idx[a]), rng(idx[b])
    gap = max(ra[0], rb[0]) - min(ra[1], rb[1]) if ra and rb else None
    return {"shared_frame_numbers": len(shared), "near_identical_frames": int(near),
            "dscf_range_a": ra, "dscf_range_b": rb, "range_gap_frames": gap}


def stage_reconcile(a):
    """Merge the verifier audit (dups) and the camera-frame provenance check (prov) into one decision
    per identity pair, then duplicate groups and the removal list.

    Rules: LightGlue 3x3 median >= T_DUP is a duplicate; a same-frame crop median >= T_DUP is a
    duplicate (one animal, one photo); a median in [T_REVIEW, T_DUP), or a 3x3 max >= T_DUP with a
    low median (one folder may hold two animals), goes to image review; the rest are distinct.
    --review "a,b=duplicate|probable_duplicate|mixed|distinct" records the image-review verdict."""
    conf = json.loads((Path(a.dups) / "dup_confirm.json").read_text())
    screen = json.loads((Path(a.dups) / "dup_screen.json").read_text())
    pv = Path(a.prov) / "prov_check.json"
    prov = json.loads(pv.read_text()) if pv.exists() else {"rows": [], "pairs": 0}
    review = dict(x.split("=", 1) for x in a.review)
    notes = dict(x.split("=", 1) for x in a.note)
    idx, paths, _ = image_index(IMGS)
    pairs: dict[tuple[str, str], dict] = {}
    for r in conf["rows"]:
        pairs[(r["a"], r["b"])] = {"a": r["a"], "b": r["b"], "found_by": list(r["why"]), "screen_max": r["screen_max"],
                                   "lightglue_3x3": r["lightglue_3x3"], "lightglue_median": r["lightglue_median"],
                                   "lightglue_max": r["lightglue_max"]}
    for r in prov["rows"]:
        k = (r["a"], r["b"])
        p = pairs.setdefault(k, {"a": r["a"], "b": r["b"], "found_by": [], "screen_max": None,
                                 "lightglue_3x3": None, "lightglue_median": None, "lightglue_max": None})
        p["found_by"].append("frame_provenance")
        p["same_frame_inliers"] = [c["inliers"] for c in r["checked"]]
        p["same_frame_median"] = r["inliers_median"]
    smap = {(r["a"], r["b"]): r for r in screen["top"]}
    rows = []
    for (x, y), p in pairs.items():
        if p["screen_max"] is None and (x, y) in smap:
            p["screen_max"] = smap[(x, y)]["max"]
        k = f"{x},{y}"
        med, mx, sf = p["lightglue_median"], p["lightglue_max"], p.get("same_frame_median")
        if sf is not None and sf >= T_DUP:
            d, basis = "duplicate", f"same-frame crops match (median {sf:g} inliers): one animal in one photo"
        elif med is not None and med >= T_DUP:
            d, basis = "duplicate", f"LightGlue 3x3 median {med:g} >= {T_DUP}"
        elif med is not None and (med >= T_REVIEW or mx >= T_DUP):
            d = review.get(k, "unreviewed")
            why = f"median {med:g} in [{T_REVIEW}, {T_DUP})" if med >= T_REVIEW else f"max {mx} >= {T_DUP} with median {med:g}"
            basis = f"LightGlue {why}; image review: {d}"
        else:
            d = "distinct"
            basis = (f"same-frame crops differ (median {sf:g}): two animals in one photo or a reused frame number"
                     if sf is not None else f"LightGlue 3x3 median {med:g} < {T_REVIEW}")
        if k in review and d == "distinct":
            d, basis = review[k], basis + f"; image review: {review[k]}"
        pr = provenance(x, y, idx, paths)
        rows.append({**p, "images_a": len(idx[x]), "images_b": len(idx[y]), "dscf_range_a": pr["dscf_range_a"],
                     "dscf_range_b": pr["dscf_range_b"], "shared_frame_numbers": pr["shared_frame_numbers"],
                     "range_gap_frames": pr["range_gap_frames"], "decision": d, "basis": basis, "note": notes.get(k, "")})
    if any(r["decision"] == "unreviewed" for r in rows):
        print("WARNING: unreviewed pairs:", [f"{r['a']},{r['b']}" for r in rows if r["decision"] == "unreviewed"])
    REMOVE = ("duplicate", "probable_duplicate", "mixed")
    parent: dict[str, str] = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x
    for r in rows:
        if r["decision"] in REMOVE:
            ra, rb = find(r["a"]), find(r["b"])
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    groups = collections.defaultdict(set)
    for r in rows:
        if r["decision"] in REMOVE:
            groups[find(r["a"])].update((r["a"], r["b"]))
    groups = sorted(sorted(g) for g in groups.values())
    removed = sorted(x for g in groups for x in g)
    mixed = sorted({x for r in rows if r["decision"] == "mixed" for x in (r["a"], r["b"])})
    keep_one = [sorted(g, key=lambda x: -len(idx[x]))[1:] for g in groups]
    n_img = lambda xs: int(sum(len(idx[x]) for x in xs))  # noqa: E731
    total_ids, total_imgs = len(idx), n_img(idx)
    order = {"duplicate": 0, "probable_duplicate": 1, "mixed": 2, "unreviewed": 3, "distinct": 4}
    rows.sort(key=lambda r: (order.get(r["decision"], 3), -max(r.get("same_frame_median") or 0, r.get("lightglue_median") or 0)))
    prov_dups = [r for r in rows if "frame_provenance" in r["found_by"] and (r.get("same_frame_median") or 0) >= T_DUP]
    out = {
        "rule": {"duplicate": f"LightGlue+MAGSAC median over 3x3 frames (first, middle, last) >= {T_DUP}, or same-frame crops with median >= {T_DUP}",
                 "review": f"3x3 median in [{T_REVIEW}, {T_DUP}), or 3x3 max >= {T_DUP} with a low median: decided by viewing the images and the drawn inliers",
                 "distinct": f"3x3 median < {T_REVIEW} and max < {T_DUP}, or same-frame crops that do not match",
                 "screen_cut": conf["screen_cut"]},
        "screen_summary": {k: screen[k] for k in ("method", "identities", "images", "image_pairs", "identity_pairs",
                                                  "wall_s", "identity_pair_max_percentiles", "histogram_max")},
        "verifier_confirm_summary": {"pairs": len(conf["rows"]), "lightglue_pairs": conf["lightglue_pairs"], "wall_s": conf["wall_s"],
                                     "lightglue_s_per_pair_median_in_worker": conf["lightglue_s_per_pair_median_in_worker"],
                                     "host": conf["host"]},
        "provenance_summary": {"frame_sharing_pairs": prov["pairs"], "same_frame_crops_match": len(prov_dups),
                               "same_animal_also_found_by_verifier_screen": sum("screen" in r["found_by"] for r in prov_dups)},
        "rows": rows, "groups": groups, "mixed_folders_involved": mixed,
        "policy": "remove every identity in a duplicate group: a doubled or mixed folder leaves no trustworthy label, and only 3 frames per folder were checked",
        "removed_identities": removed, "removed_images": n_img(removed),
        "after_removal": {"identities": total_ids - len(removed), "images": total_imgs - n_img(removed)},
        "alt_keep_largest_folder": {"removed_identities": len(sum(keep_one, [])), "removed_images": n_img(sum(keep_one, [])),
                                    "caveat": "keeps one folder per group; wrong for groups with a mixed folder"},
        "dataset": {"identities": total_ids, "images": total_imgs},
        "bc_et_al": BC["duplicates_removed"] | BC["after_removal"] | {"list": BC["duplicate_list"],
                                                                      "method": "flagged by camera-frame provenance, confirmed by consensus visual review (paper text)"},
    }
    (OUT / "duplicates_reconciled.json").write_text(json.dumps(out, indent=1))
    lines = ["| Pair | Images | Found by | Screen max | LightGlue 3x3 median (min to max) | Same-frame crops | DSCF ranges (gap) | Decision | Basis |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["decision"] == "distinct" and r.get("lightglue_median") is None and (r.get("same_frame_median") or 0) < T_REVIEW:
            continue  # frame-sharing pairs with two animals: summarised below the table
        lg = (f"{r['lightglue_median']:g} ({min(r['lightglue_3x3'])} to {max(r['lightglue_3x3'])})" if r["lightglue_3x3"] else "")
        sf = ", ".join(str(v) for v in r.get("same_frame_inliers", [])) if r.get("same_frame_inliers") else ""
        rg = f"{r['dscf_range_a']} / {r['dscf_range_b']} ({r['range_gap_frames']})"
        lines.append(f"| {r['a']} / {r['b']} | {r['images_a']} / {r['images_b']} | {', '.join(r['found_by'])} | {r['screen_max'] if r['screen_max'] is not None else ''} | "
                     f"{lg} | {sf} | {rg} | {r['decision']} | {r['basis']}{'; ' + r['note'] if r['note'] else ''} |")
    rest = [r for r in rows if r["decision"] == "distinct" and r.get("lightglue_median") is None and (r.get("same_frame_median") or 0) < T_REVIEW]
    if rest:
        mxs = [r.get("same_frame_median") or 0 for r in rest]
        lines += ["", f"{len(rest)} more identity pairs share camera frame numbers but their same-frame crops do not match "
                      f"(median inliers {min(mxs):g} to {max(mxs):g}): two animals in one photo, or a frame counter reused across sessions. Decision: distinct."]
    (OUT / "duplicates_reconciled.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({k: out[k] for k in ("groups", "removed_images", "after_removal", "alt_keep_largest_folder", "provenance_summary")}, indent=1))


# --------------------------------------------------------------------------- plan

def stage_plan(a):
    rec = json.loads((OUT / "duplicates_reconciled.json").read_text())
    remove = set(rec["removed_identities"])
    idx, _, rejected = image_index(IMGS)
    ids = sorted(x for x in idx if x not in remove)
    n = len(ids)
    k = round(n * POOLS[0] / sum(POOLS))
    v = round(n * POOLS[1] / sum(POOLS))
    plan = {"protocol": "BC et al. 2026 (arXiv 2608.28663) leakage-controlled open set, Mombe adaptation",
            "n_enrol": N_ENROL, "p_max": P_MAX, "pools": [k, v, n - k - v], "partition_seed": PARTITION_SEED,
            "removed": sorted(remove), "identities": {}, "partitions": [],
            "images_total": sum(len(idx[x]) for x in ids), "rejected_files": rejected}
    for an in ids:
        order = capture_order(idx[an])
        pos = frame_pos(order)
        E, rest = order[:N_ENROL], order[N_ENROL:]
        pick = sorted({int(round(x)) for x in np.linspace(0, len(rest) - 1, P_MAX)}) if len(rest) > P_MAX else range(len(rest))
        P = [rest[i] for i in pick]
        eb = {pos[e][0] for e in E}
        info = {}
        for p_ in P:
            b, fr = pos[p_]
            near = [abs(fr - pos[e][1]) for e in E if pos[e][0] == b]
            info[p_] = {"burst": b, "cross_burst": b not in eb, "frames_from_enrol": float(min(near)) if near else None}
        plan["identities"][an] = {"n_images": len(order), "n_bursts": len({x[0] for x in pos.values()}),
                                  "enrol": E, "probes": P, "val_probes": P[0::2], "test_probes": P[1::2],
                                  "probe_info": info}
    for p in range(N_PARTITIONS):
        s = ids[:]
        random.Random(PARTITION_SEED + p).shuffle(s)
        plan["partitions"].append({"seed": PARTITION_SEED + p, "known": sorted(s[:k]),
                                   "val_unknown": sorted(s[k:k + v]), "test_unknown": sorted(s[k + v:])})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plan.json").write_text(json.dumps(plan, indent=1))
    nq = sum(len(x["probes"]) for x in plan["identities"].values())
    cb = sum(i["cross_burst"] for x in plan["identities"].values() for i in x["probe_info"].values())
    print(f"identities {n} images {plan['images_total']} pools {plan['pools']} probes {nq} (cross-burst {cb}) enrolment images {n * N_ENROL}")


def shortlist_tables(plan: dict, emb: dict[str, np.ndarray], queries: list[str]):
    """Per probe: best cosine per identity (N,) and the index of its closest enrolment frame (N,)."""
    ids = sorted(plan["identities"])
    E = np.stack([np.stack([emb[e] for e in plan["identities"][x]["enrol"]]) for x in ids])  # N x 3 x d
    best, arg = {}, {}
    for q in queries:
        sims = E @ emb[q]
        best[q], arg[q] = sims.max(1), sims.argmax(1)
    return ids, best, arg


def partition_masks(plan: dict, ids: list[str]) -> list[np.ndarray]:
    pos = {x: i for i, x in enumerate(ids)}
    out = []
    for part in plan["partitions"]:
        m = np.zeros(len(ids), bool)
        m[[pos[x] for x in part["known"]]] = True
        out.append(m)
    return out


def needed_pairs(plan: dict, emb: dict[str, np.ndarray], queries: list[str]) -> dict[str, list[str]]:
    """Per probe, the enrolment frames it is verified against in any partition: the closest
    enrolment frame of each of the SHORTLIST animals DINOv2 ranks highest in that partition's known pool."""
    ids, best, arg = shortlist_tables(plan, emb, queries)
    masks = partition_masks(plan, ids)
    out = {}
    for q in queries:
        need = set()
        for m in masks:
            s = np.where(m, best[q], -np.inf)
            for i in np.argsort(-s, kind="stable")[:SHORTLIST]:
                need.add(plan["identities"][ids[i]]["enrol"][arg[q][i]])
        out[q] = sorted(need)
    return out


# --------------------------------------------------------------------------- pairs

def mem_note() -> str:
    """' mem avail <MB>' from /proc/meminfo on Linux, empty elsewhere."""
    try:
        with open("/proc/meminfo") as f:
            kv = {line.split(":")[0]: int(line.split()[1]) for line in f}
        return f" mem avail {kv['MemAvailable'] // 1024} MB of {kv['MemTotal'] // 1024}"
    except (OSError, KeyError, ValueError, IndexError):
        return ""


def run_verify(jobs, ctx, workers: int, initargs: tuple, ckpt: Path, n_pairs: int, stall_s: float = 1800,
               task=None, init=None, recycle: int = 10):
    """Verify (probe, targets) jobs in worker processes and survive a worker that dies.

    multiprocessing.Pool replaces a dead worker but loses its task, so imap never returns: three
    Kaggle shards hung that way on 2026-10-22. Here a dead or stalled worker breaks the executor, the
    in-flight probes are re-queued, and a probe caught in flight twice is re-run one pair at a time
    in a single-worker pool, so a pair that kills its worker is isolated and recorded as crashed.
    Workers are replaced every `recycle` probes each, which bounds memory growth inside one process.
    Finished probes are appended to ckpt (JSON lines) and skipped on a rerun."""
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from concurrent.futures.process import BrokenProcessPool
    task, init = task or _verify_task, init or _init_worker
    done: dict[str, list] = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            try:
                q, r = json.loads(line)
                done.setdefault(q, []).extend(tuple(x) for x in r)
            except ValueError:
                pass
    todo = collections.deque(j for j in jobs if j[0] not in done)
    print(f"verify: {len(done)} probes from checkpoint, {len(todo)} to run", flush=True)
    suspect: collections.Counter = collections.Counter()
    solo: collections.deque = collections.deque()
    crash_log: list[dict] = []
    t0, n_done = time.time(), sum(len(v) for v in done.values())
    fh = open(ckpt, "a")

    def record(q, r):
        nonlocal n_done
        done.setdefault(q, []).extend(r)
        n_done += len(r)
        fh.write(json.dumps([q, r]) + "\n")
        fh.flush()

    def kill(ex):
        for p in list(getattr(ex, "_processes", {}).values()):
            try:
                p.terminate()
            except Exception:
                pass
        ex.shutdown(wait=False, cancel_futures=True)

    last_print = 0.0
    while todo or solo:
        single = not todo
        if single:   # isolate one pair of a suspect probe per task
            q, targets = solo.popleft()
            left = [g for g in targets if (q, g) not in {(x[0], x[1]) for x in done.get(q, [])}]
            batch = collections.deque((q, [g]) for g in left)
        else:        # fresh workers every `recycle` probes per worker bound any per-process memory growth
            batch = collections.deque(todo.popleft() for _ in range(min(len(todo), max(1, recycle) * workers)))
        ex = ProcessPoolExecutor(1 if single else workers, mp_context=ctx, initializer=init, initargs=initargs)
        inflight: dict = {}
        broken = False
        try:
            while batch or inflight:
                while batch and len(inflight) < (1 if single else workers):
                    j = batch.popleft()
                    inflight[ex.submit(task, j)] = j
                fin, _ = wait(list(inflight), timeout=stall_s, return_when=FIRST_COMPLETED)
                if not fin:
                    print(f"verify: no result for {stall_s:.0f}s, restarting workers", flush=True)
                    broken = True
                    break
                for f in fin:
                    j = inflight.pop(f)
                    try:
                        record(j[0], f.result())
                    except BrokenProcessPool:
                        inflight[f] = j
                        broken = True
                if broken:
                    break
                if time.time() - last_print > 120:
                    last_print = time.time()
                    el = time.time() - t0
                    print(f"verify probes {len(done)}/{len(jobs)} pairs {n_done}/{n_pairs} {el:.0f}s{mem_note()}", flush=True)
        finally:
            if broken:
                kill(ex)
            else:
                ex.shutdown(wait=True)
        if not broken:
            continue
        lost = list(inflight.values())
        crash_log.append({"t_s": round(time.time() - t0), "in_flight": [j[0] for j in lost], "single_pair_mode": single})
        print(f"verify: worker died or stalled with {len(lost)} jobs in flight (single={single})", flush=True)
        if single:
            q, g = lost[0][0], lost[0][1][0]
            crash_log[-1]["crashed_pair"] = [q, g]
            rest = [j for j in batch if j not in lost]   # remaining single pairs of this probe
            if rest:
                solo.appendleft((q, [jj[1][0] for jj in rest]))
            continue
        crash_log[-1]["mem"] = mem_note().strip()
        for j in reversed(batch):   # queued in this round but never submitted
            todo.appendleft(j)
        for j in lost:
            suspect[j[0]] += 1
            (solo if suspect[j[0]] >= 2 else todo).appendleft(j)
    fh.close()
    uq = list(dict.fromkeys(q for q, _ in jobs))   # a probe may head several jobs; its results are pooled under it
    res = [tuple(x) for q in uq for x in done.get(q, [])]
    print(f"verify done: pairs {len(res)}/{n_pairs}, worker crashes {len(crash_log)}, {time.time() - t0:.0f}s", flush=True)
    return res, crash_log


def stage_pairs(a):
    """Verification scores for shard a.shard of a.shards. Features come from the local
    cache when asked (--use-cache) and are extracted from the images otherwise."""
    plan = json.loads(Path(a.plan).read_text())
    out_dir, work = Path(a.out), Path(a.work)
    out_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    idx, paths, rejected = image_index(Path(a.images))
    print(f"images found {len(paths)} rejected {rejected} identities {len(idx)}", flush=True)
    ids = sorted(plan["identities"])
    enrol = [e for x in ids for e in plan["identities"][x]["enrol"]]
    queries = sorted(q for x in ids for q in plan["identities"][x]["probes"])
    mine = queries[a.shard::a.shards]
    if a.only:
        mine = [q for q in mine if folder(q) in set(a.only)]
    if a.limit:
        mine = mine[:a.limit]
    want = enrol + mine
    cached = lambda x: a.use_cache and (FEAT / folder(x) / f"{x}.npz").exists()  # noqa: E731
    missing = [x for x in want if x not in paths and not cached(x)]
    print(f"shard {a.shard}/{a.shards}: probes {len(mine)} of {len(queries)}, enrolment images {len(enrol)}, "
          f"files missing {len(missing)}", flush=True)
    feats, t_ext, todo = {}, [], []
    for x in want:
        if cached(x):
            feats[x] = as_tuple(load_cached(x))
        elif x in paths:
            todo.append((x, paths[x]))
    t0 = time.time()
    ctx = pool_ctx()
    if todo:
        with ctx.Pool(a.workers, initializer=_init_worker, initargs=(a.models, None, a.threads)) as pool:
            for n, (x, f, dt) in enumerate(pool.imap_unordered(_extract_one, todo, chunksize=2)):
                if f is not None:
                    feats[x] = f
                    t_ext.append(dt)
                if n % 100 == 0:
                    print(f"extract {n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    t_extract = time.time() - t0
    emb = {x: f[3] for x, f in feats.items()}
    if any(e not in feats for e in enrol):
        print(f"FATAL: {sum(e not in feats for e in enrol)} enrolment images without features", flush=True)
        raise SystemExit(2)
    absent = [q for q in mine if q not in feats]
    mine = [q for q in mine if q in feats]
    need = needed_pairs(plan, emb, mine)
    n_pairs = sum(len(v) for v in need.values())
    print(f"pairs to verify {n_pairs} ({n_pairs / max(len(mine), 1):.2f} per probe){mem_note()}", flush=True)
    ff = work / f"feats_{a.shard}.pkl"
    with open(ff, "wb") as f:
        pickle.dump(feats, f)
    t1 = time.time()
    tag = a.tag or f"shard{a.shard}"
    jobs = [(q, need[q]) for q in mine]
    res, crash_log = run_verify(jobs, ctx, a.workers, (a.models, str(ff), a.threads),
                                out_dir / f"pairs_{tag}.partial.jsonl", n_pairs, a.stall_s,
                                recycle=a.recycle)
    t_verify = time.time() - t1
    np.savez_compressed(out_dir / f"emb_{tag}.npz", ids=np.array(list(emb)), emb=np.stack(list(emb.values())))
    meta = {"shard": a.shard, "shards": a.shards, "tag": tag, "probes": len(mine), "probes_without_features": absent,
            "pairs_attempted": n_pairs, "pairs_done": len(res), "zero_match_pairs": int(sum(r[2] == 0 for r in res)),
            "worker_crashes": crash_log,
            "extract_wall_s": round(t_extract, 1), "verify_wall_s": round(t_verify, 1),
            "workers": a.workers, "threads_per_worker": a.threads, "features_from_cache": a.use_cache,
            "extract_per_image_s_median_in_worker": float(np.median(t_ext)) if t_ext else None,
            "verify_per_pair_s_median_in_worker": float(np.median([r[4] for r in res])) if res else None,
            "effective_verify_s_per_pair": round(t_verify / max(len(res), 1), 3),
            "host": host_info()}
    (out_dir / f"pairs_{tag}.json").write_text(json.dumps({"meta": meta, "pairs": [r[:4] for r in res]}))
    print(json.dumps(meta, indent=1), flush=True)
    ff.unlink(missing_ok=True)


# --------------------------------------------------------------------------- bench

def stage_bench(a):
    import cv2
    from mombe.vision.muzzle import Features, Gallery, MuzzleEngine, preprocess
    eng = MuzzleEngine(MODELS)
    idx, paths, _ = image_index(IMGS)
    ids = sorted(idx)[:: max(1, len(idx) // a.n)][:a.n]
    imgs = [paths[capture_order(idx[x])[0]] for x in ids]
    eng.extract(cv2.imread(imgs[0]))
    t_pre, t_al, t_di, t_ex, feats = [], [], [], [], []
    for p in imgs:
        bgr = cv2.imread(p)
        t = time.perf_counter(); img = preprocess(bgr); t_pre.append(time.perf_counter() - t)
        t = time.perf_counter(); eng.aliked.detectAndCompute(img, None); t_al.append(time.perf_counter() - t)
        t = time.perf_counter(); eng.embed(img); t_di.append(time.perf_counter() - t)
        t = time.perf_counter(); feats.append(eng.extract(bgr)); t_ex.append(time.perf_counter() - t)
    t_v = []
    for i in range(a.n):
        t = time.perf_counter(); eng.verify(feats[i], feats[(i + 1) % len(feats)]); t_v.append(time.perf_counter() - t)
    g = Gallery()
    rng = np.random.default_rng(0)
    for i in range(POOLS[0] * N_ENROL):
        v = rng.normal(size=768).astype(np.float32)
        g.add(f"a{i // N_ENROL}", f"i{i}", Features(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32), (1, 1), v / np.linalg.norm(v)))
    t_s = []
    for i in range(50):
        t = time.perf_counter(); g.shortlist(feats[i % len(feats)].embedding, k=8); t_s.append(time.perf_counter() - t)

    def st(x):
        x = np.array(x) * 1000
        return {"median_ms": round(float(np.median(x)), 1), "p90_ms": round(float(np.percentile(x, 90)), 1), "n": len(x)}
    scale = {}
    for n_an in (1000, 10000):   # registry size in animals, 3 enrolment images each
        G = Gallery()
        V = rng.normal(size=(n_an * N_ENROL, 768)).astype(np.float32)
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        z2, z128 = np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
        for i in range(len(V)):
            G.add(f"a{i // N_ENROL}", f"i{i}", Features(z2, z128, (1, 1), V[i]))
        G.shortlist(feats[0].embedding, k=8)
        ts = []
        for i in range(10):
            t = time.perf_counter(); G.shortlist(feats[i % len(feats)].embedding, k=8); ts.append(time.perf_counter() - t)
        scale[f"shortlist_{n_an}_animals"] = st(ts)
    out = {"host": host_info(), "label": a.label, "note": a.note, "cv2_threads": cv2.getNumThreads(),
           "preprocess": st(t_pre), "aliked": st(t_al), "dinov2": st(t_di), "extract_total": st(t_ex),
           "lightglue_magsac_verify": st(t_v), f"shortlist_{POOLS[0] * N_ENROL}_images": st(t_s),
           "shortlist_scaling_gallery_shortlist_python": scale}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"bench_{a.label}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


# --------------------------------------------------------------------------- kaggle

KERNEL = r'''# Mombe held-out identity: {label}. Self-contained: sources and plan inline.
import base64, glob, gzip, hashlib, json, os, subprocess, sys, time, urllib.request, zipfile
from pathlib import Path

if __name__ == "__main__":
    T0 = time.time()
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "opencv-python", "opencv-python-headless",
                    "opencv-contrib-python", "opencv-contrib-python-headless"], check=False)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "opencv-contrib-python-headless==5.0.0.93"])
    if r.returncode != 0:
        raise SystemExit("pip install failed")
    SRC = Path("/tmp/mombe_src")
    for rel, b64 in {sources}.items():
        p = SRC / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(gzip.decompress(base64.b64decode(b64)))
    MODELS = Path("/tmp/models"); MODELS.mkdir(exist_ok=True)
    for url, sha1, name in {models}:
        dst = MODELS / name
        for attempt in range(4):
            if not dst.exists():
                try:
                    urllib.request.urlretrieve(url, dst)
                except Exception as e:
                    print("download retry", name, e, flush=True); time.sleep(10); continue
            if hashlib.sha1(dst.read_bytes()).hexdigest() == sha1:
                break
            print("sha1 mismatch", name, flush=True); dst.unlink()
        else:
            raise SystemExit("model download failed: " + name)
    hits = glob.glob("/kaggle/input/**/cattle_*/cattle_*.jpg", recursive=True)
    roots = sorted({{str(Path(p).parent.parent) for p in hits}})
    print("input jpgs", len(hits), "roots", roots[:4], flush=True)
    data_root = None
    for r_ in roots:
        n = len(glob.glob(r_ + "/cattle_*/cattle_*.jpg"))
        print("root", r_, n, flush=True)
        if n >= 4900:
            data_root = r_
    if data_root is None:
        print("attached dataset absent: downloading Zenodo 6324361", flush=True)
        z = Path("/tmp/muzzle.zip")
        urllib.request.urlretrieve("https://zenodo.org/api/records/6324361/files/BeefCattle_Muzzle_database.zip/content", z)
        zipfile.ZipFile(z).extractall("/tmp/muzzle")
        data_root = "/tmp/muzzle"
    print("data root", data_root, "setup s", round(time.time() - T0), flush=True)
    sys.path.insert(0, str(SRC))
    from eval import identity_heldout as ih
    sub = {{"@DATA": data_root, "@MODELS": str(MODELS), "@SRC": str(SRC), "@CPUS": str(os.cpu_count()), "@PLAN": str(SRC / "plan.json")}}
    argv = [sub.get(x, x) for x in {argv}]
    print("argv", argv, flush=True)
    ih.main(argv)
    print("total s", round(time.time() - T0), flush=True)
'''


def write_kernel(d: Path, slug: str, user: str, label: str, argv: list[str], with_plan: bool):
    sys.path.insert(0, str(ROOT / "tools"))
    from fetch_models import MODELS as REG
    models = [list(REG[k]) for k in ("aliked", "aliked_lightglue", "dinov2_small")]
    src = {
        "mombe/__init__.py": (ROOT / "mombe" / "__init__.py").read_bytes(),
        "mombe/vision/__init__.py": (ROOT / "mombe" / "vision" / "__init__.py").read_bytes(),
        "mombe/vision/muzzle.py": (ROOT / "mombe" / "vision" / "muzzle.py").read_bytes(),
        "eval/__init__.py": b"",
        "eval/identity_heldout.py": Path(__file__).read_bytes(),
    }
    if with_plan:
        src["plan.json"] = (OUT / "plan.json").read_bytes()
    import gzip
    enc = {k: base64.b64encode(gzip.compress(v, 9)).decode() for k, v in src.items()}
    d.mkdir(parents=True, exist_ok=True)
    (d / "script.py").write_text(KERNEL.format(label=label, sources=repr(enc), models=repr(models), argv=repr(argv)))
    meta = {"id": f"{user}/{slug}", "title": slug, "code_file": "script.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": ["parthibanmarimuthu/beefcattle-muzzle"],
            "competition_sources": [], "kernel_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))


def stage_kaggle(a):
    common = ["--images", "@DATA", "--models", "@MODELS", "--out", "/kaggle/working", "--work", "/tmp/work",
              "--workers", "@CPUS", "--threads", "1"]
    if a.kind == "prov":
        write_kernel(OUT / "kaggle" / "prov", "mombe-identity-prov", a.user, "camera-frame provenance check",
                     ["prov", *common], False)
        print(f"wrote {OUT / 'kaggle' / 'prov'}")
        return
    if a.kind == "dups":
        prior = json.loads((ROOT / "eval" / "out" / "duplicates.json").read_text())["duplicates"]
        argv = ["dups", *common, "--screen-cut", str(a.screen_cut), "--max-confirm", str(a.max_confirm),
                "--prior-pairs", *[f"{d['a']},{d['b']},{d['inliers']}" for d in prior], "--extra", *a.extra]
        write_kernel(OUT / "kaggle" / "dups", "mombe-identity-dups", a.user, "duplicate audit", argv, False)
        print(f"wrote {OUT / 'kaggle' / 'dups'}")
        return
    todo = a.only_shards if a.only_shards else list(range(a.shards))
    for s in todo:
        argv = ["pairs", *common, "--plan", "@PLAN", "--shard", str(s), "--shards", str(a.shards)]
        if a.shards != 8:   # shard s of 16 is half of shard s % 8 of 8; the tag keeps the file names apart
            argv += ["--tag", f"shard{s}of{a.shards}"]
        write_kernel(OUT / "kaggle" / f"{a.prefix}{s}", f"mombe-identity-heldout-{a.prefix}{s}", a.user,
                     f"pairs shard {s} of {a.shards}", argv, True)
    print(f"wrote {len(todo)} kernels under {OUT / 'kaggle'}: {[f'{a.prefix}{s}' for s in todo]}")


# --------------------------------------------------------------------------- report

def decide(best: int, second: int, t_new: float, t_match: float) -> str:
    """Same rule as MuzzleEngine.decide."""
    if best >= t_match and best - second >= t_new:
        return "match"
    if best <= t_new:
        return "new"
    return "ambiguous"


def tau_for(unknown: np.ndarray, far: float) -> float:
    """Accept-if-score-greater-than threshold c with FAR(c) <= far on these unknown scores."""
    u = np.sort(np.asarray(unknown, float))[::-1]
    k = int(math.floor(far * len(u)))
    return float(u[k]) if k < len(u) else -np.inf


def dir_far(kscore, kcorrect, uscore, c) -> tuple[float, float]:
    kscore, uscore = np.asarray(kscore, float), np.asarray(uscore, float)
    return float(np.mean(np.asarray(kcorrect) & (kscore > c))), float(np.mean(uscore > c))


def auroc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    from scipy.stats import rankdata
    r = rankdata(allv)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def simulate_identify(cands: list[tuple[str, int | None]], t_new, t_match) -> tuple[str, str | None, int, int, int]:
    """Replays identify()'s control flow on precomputed scores. cands: the 5 shortlisted animals in
    DINOv2 order with verified inliers. Returns decision, top animal, score, margin, verifications.
    The extra-frame step for ambiguous cases (identify() re-verifies the other enrolment frames)
    has no precomputed scores; those cases keep their first-pass scores."""
    per = {b: (s or 0) for b, s in cands[:VERIFY_TOP]}

    def top2(d):
        v = sorted(d.values(), reverse=True) + [0, 0]
        return v[0], v[1]
    if decide(*top2(per), t_new, t_match) == "new":
        for b, s in cands[VERIFY_TOP:SHORTLIST]:
            per[b] = s or 0
    ranked = sorted(per.items(), key=lambda kv: kv[1], reverse=True)
    top_id, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0
    return decide(best, second, t_new, t_match), top_id, best, best - second, len(per)


def calibrate(val_unknown_scores: np.ndarray) -> dict:
    """eval/calibrate.py rule, applied to validation only: t_new at the 99.5th percentile of
    unknown scores, t_match at twice the strongest unknown score (at least +20)."""
    u = np.asarray(val_unknown_scores, float)
    mx = float(u.max())
    return {"t_new": int(math.ceil(np.percentile(u, 99.5))), "t_match": int(max(2 * mx, mx + 20))}


def _first_k(sc: np.ndarray, correct: np.ndarray, k: int):
    """Best score, second score and whether the best is the true animal, over the first k candidates
    (ties keep DINOv2 order, as identify() does)."""
    s = sc[:, :k]
    j = np.argmax(s, 1)
    best = s[np.arange(len(s)), j]
    second = np.sort(s, 1)[:, -2] if k > 1 else np.zeros(len(s))
    return best, second, correct[np.arange(len(s)), j]


def tune_on_validation(vk: list[dict], vu: list[dict], grid_new=range(0, 121), grid_match=range(10, 401)) -> dict:
    """Thresholds chosen on validation probes only, known and validation-unknown together.
    Objective, per validation probe: +1 for a correct automatic decision (known: correct match;
    unknown: new), 0 for ambiguous (Mombe asks for a second angle), -1 for a known animal called new
    (a double registration), subject to zero false matches on validation. Ties: the median t_match of
    the optimal set, then the median t_new at that t_match. Vectorised identify() replay."""
    rows = vk + vu
    sc = np.array([[c[1] or 0 for c in r["cands"][:SHORTLIST]] for r in rows], float)
    cor = np.array([[c[0] == r["id"] for c in r["cands"][:SHORTLIST]] for r in rows], bool)
    is_k = np.array([True] * len(vk) + [False] * len(vu))
    b2, s2, c2 = _first_k(sc, cor, VERIFY_TOP)
    b5, s5, c5 = _first_k(sc, cor, SHORTLIST)
    tn = np.array(list(grid_new), float)[:, None, None]
    tm = np.array(list(grid_match), float)[None, :, None]
    esc = (b2[None, :] <= tn[:, 0])                                    # T_new x N
    best = np.where(esc, b5, b2)[:, None, :]
    second = np.where(esc, s5, s2)[:, None, :]
    corr = np.where(esc, c5, c2)[:, None, :]
    match = (best >= tm) & (best - second >= tn)
    new = (~match) & (best <= tn)
    good = (np.where(is_k, match & corr, new).astype(float) - (is_k & new)).mean(2)
    fm = (match & ~(corr & is_k)).sum(2)
    obj = np.where((fm == 0) & (tn[:, :, 0] <= tm[:, :, 0]), good, -1.0)
    opt = np.argwhere(obj == obj.max())
    gm = np.array(list(grid_match))[opt[:, 1]]
    t_match = int(sorted(gm)[len(gm) // 2])
    sel = opt[gm == t_match]
    gn = sorted(np.array(list(grid_new))[sel[:, 0]])
    return {"t_new": int(gn[len(gn) // 2]), "t_match": int(t_match), "val_objective": float(obj.max()),
            "optimal_set_size": int(len(opt)), "grid": f"t_new {min(grid_new)}..{max(grid_new)}, t_match {min(grid_match)}..{max(grid_match)}"}


def latency_cost(dec_means: dict, metas: list[dict] | None = None) -> dict:
    """Latency per probe = extract + shortlist + verifications x verify, with verifications per probe
    measured in this protocol and stage times from several sources, each labelled."""
    srcs = {}
    pc = OUT / "bench_this_pc.json"
    if pc.exists():
        b = json.loads(pc.read_text())
        srcs["this_pc"] = {"label": f"measured on this PC ({b['host']['processor']}, {b['host']['cpus']} logical CPUs, "
                                    f"{b['host']['system']}, OpenCV {b['host']['opencv']}, {b['cv2_threads']} threads)"
                                    + (f"; {b['note']}" if b.get("note") else ""),
                           "extract_ms": b["extract_total"]["median_ms"], "verify_ms": b["lightglue_magsac_verify"]["median_ms"],
                           "shortlist_ms": next(v["median_ms"] for k, v in b.items() if k.startswith("shortlist_")),
                           "usd_per_hour": None, "shortlist_scaling": b.get("shortlist_scaling_gallery_shortlist_python")}
    ex = [m["extract_per_image_s_median_in_worker"] for m in metas or [] if m.get("extract_per_image_s_median_in_worker")]
    vf = [m["verify_per_pair_s_median_in_worker"] for m in metas or [] if m.get("verify_per_pair_s_median_in_worker")]
    if ex and vf:
        h = metas[0]["host"]
        srcs["kaggle_worker"] = {"label": f"measured in this run's Kaggle shards: one OpenCV thread per worker, {metas[0]['workers']} workers "
                                          f"sharing {h['cpus']} vCPUs ({h['machine']} {h['system']}, OpenCV {h['opencv']}); median over shards",
                                 "extract_ms": round(1000 * float(np.median(ex)), 1), "verify_ms": round(1000 * float(np.median(vf)), 1),
                                 "shortlist_ms": 1.0, "usd_per_hour": None}
    for name, f, price in (("cool_hub_projection", "cool.json", 0.34112 + 0.02), ("stock_hub_projection", "stock.json", 0.34112)):
        p = ROOT / "eval" / "out" / "bench" / f
        if p.exists():
            b = json.loads(p.read_text())["pipeline"]
            srcs[name] = {"label": f"projection: c8g.2xlarge stage medians from eval/out/bench/{f} (2026-10-15) times verifications per probe measured here",
                          "extract_ms": b["preprocess_clahe_resize"]["median_ms"] + b["aliked_extract"]["median_ms"] + b["dinov2_embed"]["median_ms"],
                          "verify_ms": b["lightglue_magsac_verify"]["median_ms"], "shortlist_ms": 1.0, "usd_per_hour": price}
    out = {"verifications_per_probe": dec_means, "sources": {}}
    for name, s in srcs.items():
        r = {"label": s["label"], "stage_ms": {k: s[k] for k in ("extract_ms", "verify_ms", "shortlist_ms")}}
        if s.get("shortlist_scaling"):
            r["shortlist_scaling"] = s["shortlist_scaling"]
        for role, nv in dec_means.items():
            ms_ = s["extract_ms"] + s["shortlist_ms"] + nv * s["verify_ms"]
            r[f"latency_ms_{role}"] = round(ms_, 1)
            if s["usd_per_hour"]:
                r[f"usd_per_1000_{role}"] = round(ms_ / 1000 / 3600 * 1000 * s["usd_per_hour"], 4)
        enrol_ms = N_ENROL * s["extract_ms"] + s["shortlist_ms"] + dec_means["unknown_probe"] * s["verify_ms"]
        r["enrolment_ms_per_animal"] = round(enrol_ms, 1)
        if s["usd_per_hour"]:
            r["usd_per_1000_enrolments"] = round(enrol_ms / 1000 / 3600 * 1000 * s["usd_per_hour"], 4)
        out["sources"][name] = r
    out["assumptions"] = ("one request at a time per instance (the OpenCV DNN engine uses every core per request); "
                          "on-demand eu-west-1 price; enrolment = 3 frames extracted plus one de-duplication search "
                          "costed like an unknown probe; storage and network excluded")
    return out


def stage_report(a):
    plan = json.loads(Path(a.plan).read_text())
    S, emb, metas = {}, {}, []
    for f in sorted(Path(a.pairs).glob("pairs_*.json")):
        d = json.loads(f.read_text())
        metas.append(d["meta"])
        for q, g, m, inl in d["pairs"]:
            S[(q, g)] = inl
    for f in sorted(Path(a.pairs).glob("emb_*.npz")):
        z = np.load(f)
        for i, e in zip(z["ids"], z["emb"]):
            emb[str(i)] = e
    ids = sorted(plan["identities"])
    probes_all = [q for x in ids for q in plan["identities"][x]["probes"]]
    probes = [q for q in probes_all if q in emb]
    ids_, best, arg = shortlist_tables(plan, emb, probes)
    masks = partition_masks(plan, ids)
    pos = {x: i for i, x in enumerate(ids)}
    info = {q: plan["identities"][folder(q)]["probe_info"][q] for q in probes}
    valset = {q for x in ids for q in plan["identities"][x]["val_probes"]}
    rng = np.random.default_rng(0)

    parts, missing_total = [], 0
    for p, (part, m) in enumerate(zip(plan["partitions"], masks)):
        role_of = {}
        for x in part["known"]:
            role_of[x] = "known"
        for x in part["val_unknown"]:
            role_of[x] = "val_unknown"
        for x in part["test_unknown"]:
            role_of[x] = "test_unknown"
        rows = []
        for q in probes:
            a_ = folder(q)
            role = role_of[a_]
            if role == "known":
                role = "val_known" if q in valset else "test_known"
            s = np.where(m, best[q], -np.inf)
            order = np.argsort(-s, kind="stable")
            rank = int(np.nonzero(order == pos[a_])[0][0]) + 1 if m[pos[a_]] else None
            cands, miss = [], 0
            for i in order[:SHORTLIST]:
                g = plan["identities"][ids[i]]["enrol"][arg[q][i]]
                sc = S.get((q, g))
                miss += sc is None
                cands.append((ids[i], sc))
            sv = [c[1] or 0 for c in cands]
            j = int(np.argmax(sv))
            rows.append({"q": q, "id": a_, "role": role, "rank": rank, "cos1": float(s[order[0]]),
                         "cos_top": ids[order[0]], "s5": int(sv[j]), "id5": cands[j][0], "cands": cands, "missing": miss,
                         "cross_burst": info[q]["cross_burst"]})
        missing_total += sum(r["missing"] for r in rows)
        R = lambda role: [r for r in rows if r["role"] == role]  # noqa: E731
        vk, tk, vu, tu = R("val_known"), R("test_known"), R("val_unknown"), R("test_unknown")
        res = {"partition": p, "seed": part["seed"],
               "n": {"known_ids": len(part["known"]), "val_unknown_ids": len(part["val_unknown"]),
                     "test_unknown_ids": len(part["test_unknown"]), "val_known_probes": len(vk),
                     "test_known_probes": len(tk), "val_unknown_probes": len(vu), "test_unknown_probes": len(tu),
                     "enrolment_images": len(part["known"]) * N_ENROL, "missing_pair_scores": sum(r["missing"] for r in rows)}}
        # closed set (test known probes)
        rk = np.array([r["rank"] for r in tk])
        res["closed_set"] = {"dinov2_rank1": float(np.mean(rk == 1)), "dinov2_rank5": float(np.mean(rk <= 5)),
                             "pipeline_rank1": float(np.mean([r["id5"] == r["id"] for r in tk])),
                             "pipeline_rank5": float(np.mean(rk <= 5)),
                             **{f"dinov2_recall@{k}": float(np.mean(rk <= k)) for k in (1, 2, 5, 8, 20)}}
        # top-20 shortlist recall at enrolment: every known probe (val + test) searched against the registry
        rka = np.array([r["rank"] for r in vk + tk])
        res["enrolment_top20_recall"] = {"recall": float(np.mean(rka <= 20)), "n": len(rka), "misses": int(np.sum(rka > 20))}
        # open set: DIR@FAR, score = best verified inliers over the 5 shortlisted animals
        ks, kc = np.array([r["s5"] for r in tk]), np.array([r["id5"] == r["id"] for r in tk])
        us, vus = np.array([r["s5"] for r in tu]), np.array([r["s5"] for r in vu])
        ksc, kcc = np.array([r["cos1"] for r in tk]), np.array([r["rank"] == 1 for r in tk])
        usc, vusc = np.array([r["cos1"] for r in tu]), np.array([r["cos1"] for r in vu])
        res["open_set"] = {}
        for name, (KS, KC, US, VUS) in {"pipeline": (ks, kc, us, vus), "dinov2_only": (ksc, kcc, usc, vusc)}.items():
            o = {"auroc_known_vs_unknown": auroc(KS, US)}
            for far in (0.01, 0.001):
                c_or = tau_for(US, far)
                d_or, f_or = dir_far(KS, KC, US, c_or)
                c_tr = tau_for(VUS, far)
                d_tr, f_tr = dir_far(KS, KC, US, c_tr)
                o[f"far_{far:g}"] = {
                    "oracle": {"threshold_gt": c_or, "dir": d_or, "far": f_or},
                    "transferred": {"threshold_gt": c_tr, "dir": d_tr, "realised_far": f_tr,
                                    "val_far": dir_far(KS, KC, VUS, c_tr)[1]},
                    "resolution": {"val_unknown_probes": len(VUS), "test_unknown_probes": len(US),
                                   "below_resolution": bool(far * len(VUS) < 1 or far * len(US) < 1)}}
            res["open_set"][name] = o
        # identity-clustered bootstrap on test (pipeline score), transferred 1% and oracle 1%
        kid = np.array([r["id"] for r in tk]); uid = np.array([r["id"] for r in tu])
        kg = {x: np.nonzero(kid == x)[0] for x in np.unique(kid)}
        ug = {x: np.nonzero(uid == x)[0] for x in np.unique(uid)}
        c_tr = res["open_set"]["pipeline"]["far_0.01"]["transferred"]["threshold_gt"]
        bt, bf, bo = [], [], []
        kk, uu = list(kg), list(ug)
        for _ in range(a.boot):
            ki = np.concatenate([kg[x] for x in rng.choice(kk, len(kk))])
            ui = np.concatenate([ug[x] for x in rng.choice(uu, len(uu))])
            d_, f_ = dir_far(ks[ki], kc[ki], us[ui], c_tr)
            bt.append(d_); bf.append(f_)
            bo.append(dir_far(ks[ki], kc[ki], us[ui], tau_for(us[ui], 0.01))[0])
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
        res["bootstrap_identity_clustered"] = {"resamples": a.boot, "transferred_1pct_dir_ci95": ci(bt),
                                               "transferred_1pct_far_ci95": ci(bf), "oracle_1pct_dir_ci95": ci(bo)}
        # three-way decisions on test
        th_cal = calibrate(vus)
        th_tun = tune_on_validation(vk, vu)
        res["thresholds"] = {"calibrated_on_validation": th_cal, "tuned_on_validation": th_tun, "frozen_dev": FROZEN}
        res["decisions"] = {}
        for tname, th in (("calibrated", th_cal), ("val_tuned", th_tun), ("frozen_dev", FROZEN)):
            dd = {}
            for group, rr in (("val_known", vk), ("test_known", tk), ("val_unknown", vu), ("test_unknown", tu)):
                cnt, nver = collections.Counter(), []
                for r in rr:
                    dec, top, sc, mg, nv = simulate_identify(r["cands"], th["t_new"], th["t_match"])
                    nver.append(nv)
                    if group in ("val_known", "test_known"):  # "test_unknown" also ends with "known"
                        cnt[{"match": "correct_match" if top == r["id"] else "false_match",
                             "new": "false_new", "ambiguous": "ambiguous_second_angle"}[dec]] += 1
                        if dec == "ambiguous":
                            cnt["ambiguous_top_correct"] += top == r["id"]
                    else:
                        cnt[{"match": "false_match", "new": "correct_new", "ambiguous": "ambiguous_second_angle"}[dec]] += 1
                n_ = max(len(rr), 1)
                dd[group] = {"n": len(rr), **{k: cnt[k] for k in sorted(cnt)},
                             **{f"{k}_rate": cnt[k] / n_ for k in sorted(cnt)},
                             "mean_verifications": float(np.mean(nver)) if nver else None}
            res["decisions"][tname] = dd
        # error anatomy, calibrated thresholds, test known probes
        an = collections.Counter()
        for r in tk:
            dec, top, sc, mg, nv = simulate_identify(r["cands"], th_cal["t_new"], th_cal["t_match"])
            if dec == "match" and top == r["id"]:
                continue
            key = {"new": "false_new", "ambiguous": "ambiguous", "match": "false_match"}[dec]
            an[key] += 1
            an[f"{key}_true_outside_top5"] += r["rank"] > SHORTLIST
            an[f"{key}_cross_burst"] += r["cross_burst"]
            an[f"{key}_true_score_le_t_new"] += next((c[1] or 0 for c in r["cands"] if c[0] == r["id"]), 0) <= th_cal["t_new"]
        res["error_anatomy_calibrated_test_known"] = {"n_test_known": len(tk), "n_cross_burst": sum(r["cross_burst"] for r in tk), **dict(an)}
        # cross-burst probes (enrolment and probe in different camera bursts)
        cb = [r for r in tk if r["cross_burst"]]
        res["cross_burst_test_known"] = {"n": len(cb), "ids": len({r["id"] for r in cb}),
                                         "pipeline_rank1": float(np.mean([r["id5"] == r["id"] for r in cb])) if cb else None,
                                         "dinov2_rank1": float(np.mean([r["rank"] == 1 for r in cb])) if cb else None,
                                         "dir_transferred_1pct": float(np.mean([(r["id5"] == r["id"]) and r["s5"] > c_tr for r in cb])) if cb else None}
        # strongest test-unknown scores (residual duplicate check)
        res["top_unknown_scores"] = sorted(({"q": r["q"], "s5": r["s5"], "matched": r["id5"]} for r in tu + vu),
                                           key=lambda r: -r["s5"])[:8]
        parts.append(res)
        print(f"partition {p}: rank1 {res['closed_set']['pipeline_rank1']:.4f} "
              f"DIR@1% oracle {res['open_set']['pipeline']['far_0.01']['oracle']['dir']:.4f} "
              f"transferred {res['open_set']['pipeline']['far_0.01']['transferred']['dir']:.4f} "
              f"at FAR {res['open_set']['pipeline']['far_0.01']['transferred']['realised_far']:.4f}", flush=True)

    def ms(get):
        v = [get(r) for r in parts]
        v = [x for x in v if x is not None]
        return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "per_partition": v}
    P = "pipeline"
    summary = {
        "track": "1e identity held out", "generated": time.strftime("%Y-%m-%d %H:%M"),
        "evidence_label": {"input": "R (real captures, Zenodo 6324361)", "truth": "L (curator identity labels, duplicate-audited)"},
        "protocol": {"source": "BC, Chaudhary, Nepal 2026, arXiv 2608.28663", "partitions": N_PARTITIONS,
                     "pools": plan["pools"], "partition_seed": PARTITION_SEED, "enrolment_frames_per_animal": N_ENROL,
                     "enrolment_rule": "first 3 frames in capture order (camera frame number, wrap-aware)",
                     "probe_rule": f"up to {P_MAX} frames evenly spaced over the remaining frames; alternate frames go to validation and test",
                     "unknown_probes": "the same probe frames of animals in the unknown pools",
                     "score": f"best LightGlue+MAGSAC inliers over the {SHORTLIST} animals DINOv2 shortlists from the known pool; each verified against its closest enrolment frame",
                     "far_definition": "share of unknown probes whose best score exceeds the threshold",
                     "dir_definition": "share of known probes whose top-ranked animal is correct and whose score exceeds the threshold",
                     "threshold_transfer": "threshold set on validation-unknown probes, applied unchanged to test",
                     "decision_thresholds": "eval/calibrate.py rule on validation-unknown scores (t_new = ceil p99.5, t_match = max(2 max, max + 20)); frozen dev thresholds 20/90 shown beside",
                     "identify_replay": f"verify top {VERIFY_TOP}, escalate to {SHORTLIST} before a new verdict; the extra-frame pass for ambiguous cases is outside the replay"},
        "data": {"identities": len(ids), "images": plan["images_total"], "probes": len(probes_all),
                 "probes_with_features": len(probes), "removed_identities": plan["removed"],
                 "pair_scores": len(S), "missing_pair_scores_all_partitions": missing_total,
                 "shards": metas},
        "mean_sd": {
            "dir_far_1pct_oracle": ms(lambda r: r["open_set"][P]["far_0.01"]["oracle"]["dir"]),
            "dir_far_0.1pct_oracle": ms(lambda r: r["open_set"][P]["far_0.001"]["oracle"]["dir"]),
            "dir_far_1pct_transferred": ms(lambda r: r["open_set"][P]["far_0.01"]["transferred"]["dir"]),
            "realised_far_1pct_transferred": ms(lambda r: r["open_set"][P]["far_0.01"]["transferred"]["realised_far"]),
            "dir_far_0.1pct_transferred": ms(lambda r: r["open_set"][P]["far_0.001"]["transferred"]["dir"]),
            "realised_far_0.1pct_transferred": ms(lambda r: r["open_set"][P]["far_0.001"]["transferred"]["realised_far"]),
            "auroc_known_vs_unknown": ms(lambda r: r["open_set"][P]["auroc_known_vs_unknown"]),
            "pipeline_rank1": ms(lambda r: r["closed_set"]["pipeline_rank1"]),
            "pipeline_rank5": ms(lambda r: r["closed_set"]["pipeline_rank5"]),
            "dinov2_rank1": ms(lambda r: r["closed_set"]["dinov2_rank1"]),
            "dinov2_only_dir_far_1pct_oracle": ms(lambda r: r["open_set"]["dinov2_only"]["far_0.01"]["oracle"]["dir"]),
            "dinov2_only_dir_far_1pct_transferred": ms(lambda r: r["open_set"]["dinov2_only"]["far_0.01"]["transferred"]["dir"]),
            "enrolment_top20_recall": ms(lambda r: r["enrolment_top20_recall"]["recall"]),
            "cross_burst_pipeline_rank1": ms(lambda r: r["cross_burst_test_known"]["pipeline_rank1"]),
            "cross_burst_dir_transferred_1pct": ms(lambda r: r["cross_burst_test_known"]["dir_transferred_1pct"]),
            **{f"{t}_{g}_{k}": ms(lambda r, t=t, g=g, k=k: r["decisions"][t][g].get(k, 0.0))
               for t in ("calibrated", "val_tuned", "frozen_dev")
               for g, ks_ in (("test_known", ("correct_match_rate", "false_match_rate", "false_new_rate", "ambiguous_second_angle_rate", "mean_verifications")),
                              ("test_unknown", ("correct_new_rate", "false_match_rate", "ambiguous_second_angle_rate", "mean_verifications")))
               for k in ks_},
        },
        "partitions": parts,
        "bc_et_al_for_context": BC,
    }
    hk = {"dir_far_1pct_transferred": "DIR@FAR 1%, threshold from validation-unknown, test pools",
          "realised_far_1pct_transferred": "test FAR at that threshold",
          "dir_far_0.1pct_transferred": "DIR@FAR 0.1%, threshold from validation-unknown (below resolution)",
          "realised_far_0.1pct_transferred": "test FAR at that threshold",
          "dir_far_1pct_oracle": "DIR@FAR 1%, oracle threshold on test", "dir_far_0.1pct_oracle": "DIR@FAR 0.1%, oracle threshold on test",
          "pipeline_rank1": "closed-set rank-1, test known probes", "pipeline_rank5": "closed-set rank-5, test known probes",
          "enrolment_top20_recall": "top-20 DINOv2 shortlist recall, known probes",
          "calibrated_test_known_correct_match_rate": "three-way, known probe: correct match",
          "calibrated_test_known_false_match_rate": "three-way, known probe: false match (wrong animal)",
          "calibrated_test_known_false_new_rate": "three-way, known probe: false new",
          "calibrated_test_known_ambiguous_second_angle_rate": "three-way, known probe: ambiguous, second angle",
          "calibrated_test_unknown_correct_new_rate": "three-way, unknown probe: correct new",
          "calibrated_test_unknown_false_match_rate": "three-way, unknown probe: false match",
          "calibrated_test_unknown_ambiguous_second_angle_rate": "three-way, unknown probe: ambiguous, second angle"}
    summary["headline"] = {"note": "fractions; mean and sample sd over the partitions; thresholds calibrated on validation only "
                                   "(eval/calibrate.py rule on validation-unknown scores); evidence input R, truth L",
                           "partitions_completed": len(parts), **{k: {"label": v, **summary["mean_sd"][k]} for k, v in hk.items()}}
    summary = {"headline": summary.pop("headline"), **summary}
    # residual label-noise check: strongest verified scores between different identities (any probe role)
    imp = collections.defaultdict(list)
    for (q, g), v in S.items():
        if folder(q) != folder(g):
            imp[tuple(sorted((folder(q), folder(g))))].append(int(v))
    iv = np.array([x for v in imp.values() for x in v])
    summary["residual_impostor_pairs"] = {
        "impostor_pair_scores": int(len(iv)), "identity_pairs": len(imp),
        "scores_ge_review_46": int(np.sum(iv >= T_REVIEW)), "scores_ge_90": int(np.sum(iv >= T_DUP)),
        "percentiles": {p: float(np.percentile(iv, p)) for p in (50, 99, 99.9)} if len(iv) else {},
        "top_identity_pairs": sorted(({"a": k[0], "b": k[1], "max": max(v), "n_scores": len(v),
                                       "n_ge_46": sum(x >= T_REVIEW for x in v)} for k, v in imp.items()),
                                     key=lambda r: -r["max"])[:15]}
    rp = OUT / "duplicates_reconciled.json"
    if rp.exists():
        rec = json.loads(rp.read_text())
        summary["duplicates"] = {k: rec[k] for k in ("groups", "removed_identities", "removed_images", "after_removal",
                                                     "alt_keep_largest_folder", "provenance_summary", "mixed_folders_involved",
                                                     "policy", "bc_et_al")}
        summary["duplicates"]["n_groups"] = len(rec["groups"])
        summary["duplicates"]["n_removed_identities"] = len(rec["removed_identities"])
        summary["duplicates"]["decisions"] = dict(collections.Counter(r["decision"] for r in rec["rows"]))
        summary["duplicates"]["table"] = "eval/out/identity_heldout/duplicates_reconciled.md"
    nv = {"known_probe": summary["mean_sd"]["calibrated_test_known_mean_verifications"]["mean"],
          "unknown_probe": summary["mean_sd"]["calibrated_test_unknown_mean_verifications"]["mean"]}
    summary["latency_cost"] = latency_cost(nv, metas)
    Path(a.summary).write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary["mean_sd"], indent=1, default=float)[:4000])


# --------------------------------------------------------------------------- residual

def stage_residual(a):
    """Re-check the strongest impostor pairs left after duplicate removal with the reconciliation rule:
    LightGlue+MAGSAC over 3 x 3 frames (first, middle, last in capture order), cached features."""
    s = json.loads(Path(a.summary).read_text())
    top = [r for r in s["residual_impostor_pairs"]["top_identity_pairs"] if r["max"] >= a.min_max]
    idx, paths, _ = image_index(IMGS)
    pick = lambda an: (lambda o: [o[0], o[len(o) // 2], o[-1]])(capture_order(idx[an]))  # noqa: E731
    jobs, feats, extracted = [], {}, 0
    eng = None
    for r in top:
        A, B = pick(r["a"]), pick(r["b"])
        for x in A + B:
            if x in feats:
                continue
            if (FEAT / folder(x) / f"{x}.npz").exists():
                feats[x] = as_tuple(load_cached(x))
            else:   # the cache holds about half the images; extract the rest here
                import cv2
                from mombe.vision.muzzle import MuzzleEngine
                eng = eng or MuzzleEngine(MODELS)
                feats[x] = as_tuple(eng.extract(cv2.imread(paths[x])))
                extracted += 1
        jobs += [(x, B) for x in A]
    del eng
    print(f"residual: {len(top)} identity pairs, {len(feats)} images ({extracted} extracted), {len(jobs) * 3} verifications", flush=True)
    work = OUT / "_work"
    work.mkdir(exist_ok=True)
    ff = work / "feats_residual.pkl"
    with open(ff, "wb") as f:
        pickle.dump(feats, f)
    t0 = time.time()
    res, _ = run_verify(jobs, pool_ctx(), a.workers, (str(MODELS), str(ff), 1), work / "residual.partial.jsonl", len(jobs) * 3)
    ff.unlink(missing_ok=True)
    (work / "residual.partial.jsonl").unlink(missing_ok=True)
    rows = []
    for r in top:
        v = sorted(x[3] for x in res if {folder(x[0]), folder(x[1])} == {r["a"], r["b"]})
        med = float(np.median(v))
        dec = "duplicate" if med >= T_DUP else ("review" if med >= T_REVIEW or max(v) >= T_DUP else "distinct")
        rows.append({**r, "lightglue_3x3": v, "median_3x3": med, "decision_by_rule": dec})
        print(f"{r['a']} / {r['b']}: protocol max {r['max']}, 3x3 {v} median {med:.0f} -> {dec}", flush=True)
    out = {"rule": f"median >= {T_DUP} duplicate; median in [{T_REVIEW}, {T_DUP}) or max >= {T_DUP}: review; else distinct",
           "pairs": rows, "verifications": len(res), "wall_s": round(time.time() - t0, 1), "workers": a.workers,
           "host": host_info(), "features": f"data/features cache, {extracted} images extracted on this host (ALIKED keypoints, descriptors, DINOv2 embedding)",
           "use": "sensitivity check only: the headline keeps the duplicate list fixed before any test score was seen"}
    (OUT / "residual_check.json").write_text(json.dumps(out, indent=1))


# --------------------------------------------------------------------------- readme

def stage_readme(a):
    """README.md for eval/out/identity_heldout, rendered from summary.json so every number matches it."""
    s = json.loads(Path(a.summary).read_text())
    M, P0, D = s["mean_sd"], s["partitions"], s.get("duplicates", {})
    pct = lambda k: f"{100 * M[k]['mean']:.2f} ± {100 * M[k]['sd']:.2f}"  # noqa: E731
    per = lambda k: ", ".join(f"{100 * v:.1f}" for v in M[k]["per_partition"])  # noqa: E731
    n0 = P0[0]["n"]
    nk = [p["n"]["test_known_probes"] for p in P0]
    nu = [p["n"]["test_unknown_probes"] for p in P0]
    nvu = [p["n"]["val_unknown_probes"] for p in P0]
    th = [p["thresholds"]["calibrated_on_validation"] for p in P0]
    shards = s["data"]["shards"]
    att = sum(m["pairs_attempted"] for m in shards)
    done = sum(m["pairs_done"] for m in shards)
    lc = s["latency_cost"]
    L = []
    w = L.append
    w("# Held-out identity (track 1e)\n")
    w(f"Generated {s['generated']} from `summary.json` by `eval/identity_heldout.py readme`. "
      "These numbers replace the development identity figures (test top-1 98.6%, FMR 0.36%, FNMR 1.4%), "
      "whose thresholds were set and measured on one split with gallery and probe frames from one camera burst.\n")
    w(f"Evidence label for every number below: input **R** (real captures, Zenodo 6324361, {s['data']['identities']} identities "
      f"and {s['data']['images']:,} images after duplicate removal); truth **L** (curator identity labels, duplicate-audited).\n")
    w("## Headline: test pools, mean ± sd over 5 identity-disjoint partitions\n")
    w("| Metric | Mombe | Per partition (%) |")
    w("|---|---|---|")
    rows = [
        ("DIR@FAR 1%, threshold transferred from validation", "dir_far_1pct_transferred"),
        ("Realised test FAR at that threshold", "realised_far_1pct_transferred"),
        ("DIR@FAR 1%, oracle threshold set on test", "dir_far_1pct_oracle"),
        ("DIR@FAR 0.1%, threshold transferred from validation", "dir_far_0.1pct_transferred"),
        ("Realised test FAR at that threshold", "realised_far_0.1pct_transferred"),
        ("DIR@FAR 0.1%, oracle threshold set on test", "dir_far_0.1pct_oracle"),
        ("Closed-set rank-1 (DINOv2 shortlist, then LightGlue+MAGSAC rerank of top 5)", "pipeline_rank1"),
        ("Closed-set rank-5 (true animal among the 5 verified)", "pipeline_rank5"),
        ("Closed-set rank-1, DINOv2 cosine alone", "dinov2_rank1"),
        ("Top-20 shortlist recall at enrolment (known probes, validation and test)", "enrolment_top20_recall"),
        ("AUROC, known versus unknown probes (best inliers)", "auroc_known_vs_unknown"),
    ]
    for name, k in rows:
        w(f"| {name} | {pct(k)}% | {per(k)} |")
    w("")
    bt = [p["bootstrap_identity_clustered"] for p in P0]
    w("Identity-clustered bootstrap, 95% intervals on test at the transferred 1% threshold, partitions 0 to 4: DIR "
      + ", ".join(f"{100 * b['transferred_1pct_dir_ci95'][0]:.1f} to {100 * b['transferred_1pct_dir_ci95'][1]:.1f}%" for b in bt)
      + "; realised FAR " + ", ".join(f"{100 * b['transferred_1pct_far_ci95'][0]:.1f} to {100 * b['transferred_1pct_far_ci95'][1]:.1f}%" for b in bt) + ".\n")
    w(f"Per partition, test has {min(nk)} to {max(nk)} known probes and {min(nu)} to {max(nu)} unknown probes; "
      f"validation has {min(nvu)} to {max(nvu)} unknown probes. "
      + ("FAR 0.1% needs at least 1,000 unknown probes to resolve one false accept, so both 0.1% rows sit below resolution: "
         "the transferred threshold is the highest validation-unknown score and the oracle threshold is the highest test-unknown score.\n"
         if max(nvu) < 1000 and max(nu) < 1000 else "\n"))
    w("Three-way decisions on test (identify() replay: verify the top 2, escalate to 5 before a new verdict):\n")
    w("| Thresholds | Known: correct match | Known: false match | Known: false new | Known: ambiguous, second angle | Unknown: correct new | Unknown: false match | Unknown: ambiguous |")
    w("|---|---|---|---|---|---|---|---|")
    for t, lab in (("calibrated", "Calibrated on validation (t_new / t_match per partition: " + ", ".join(f"{x['t_new']}/{x['t_match']}" for x in th) + ")"),
                   ("frozen_dev", "Frozen development 20/90")):
        w(f"| {lab} | {pct(f'{t}_test_known_correct_match_rate')}% | {pct(f'{t}_test_known_false_match_rate')}% | "
          f"{pct(f'{t}_test_known_false_new_rate')}% | {pct(f'{t}_test_known_ambiguous_second_angle_rate')}% | "
          f"{pct(f'{t}_test_unknown_correct_new_rate')}% | {pct(f'{t}_test_unknown_false_match_rate')}% | "
          f"{pct(f'{t}_test_unknown_ambiguous_second_angle_rate')}% |")
    w("")
    w("Per partition, calibrated thresholds (test pools; counts in `summary.json`, key `decisions`):\n")
    w("| Partition | t_new / t_match | Known: correct match | Known: false match | Known: false new | Known: ambiguous | Unknown: correct new | Unknown: false match | Unknown: ambiguous |")
    w("|---|---|---|---|---|---|---|---|---|")
    for p_, t_ in zip(P0, th):
        k_, u_ = p_["decisions"]["calibrated"]["test_known"], p_["decisions"]["calibrated"]["test_unknown"]
        f_ = lambda d, k: f"{100 * d.get(k, 0.0):.1f}%"  # noqa: E731
        w(f"| {p_['partition']} | {t_['t_new']} / {t_['t_match']} | {f_(k_, 'correct_match_rate')} | {f_(k_, 'false_match_rate')} | "
          f"{f_(k_, 'false_new_rate')} | {f_(k_, 'ambiguous_second_angle_rate')} | {f_(u_, 'correct_new_rate')} | "
          f"{f_(u_, 'false_match_rate')} | {f_(u_, 'ambiguous_second_angle_rate')} |")
    w("")
    if all("error_anatomy_calibrated_test_known" in p for p in P0):
        E = collections.Counter()
        for p in P0:
            E.update({k: v for k, v in p["error_anatomy_calibrated_test_known"].items()})
        fn, am = E["false_new"], E["ambiguous"]
        if fn or am:
            w(f"Where the known-probe misses come from, summed over the 5 partitions (calibrated thresholds): {fn} false new verdicts. "
              f"In {E['false_new_true_outside_top5']} of them the true animal sits outside the DINOv2 top 5, so the verifier sees only other animals; "
              f"{E['false_new_cross_burst']} are cross-burst probes, which make up {E['n_cross_burst']} of {E['n_test_known']} known probes. "
              f"{am} ambiguous verdicts send the farmer for a second angle; "
              + ("the true animal is in the top 5 in every one of them.\n" if E["ambiguous_true_outside_top5"] == 0
                 else f"in {am - E['ambiguous_true_outside_top5']} of them the true animal is in the top 5.\n"))
    tt = [p["thresholds"]["tuned_on_validation"] for p in P0]
    w(f"A second validation-only setting tunes both thresholds on validation known and validation-unknown probes together "
      f"(zero validation false matches, then the most correct automatic decisions; t_new / t_match per partition: "
      + ", ".join(f"{x['t_new']}/{x['t_match']}" for x in tt) + "). On test it gives known correct match "
      f"{pct('val_tuned_test_known_correct_match_rate')}%, known false match {pct('val_tuned_test_known_false_match_rate')}%, "
      f"known false new {pct('val_tuned_test_known_false_new_rate')}%, known ambiguous {pct('val_tuned_test_known_ambiguous_second_angle_rate')}%, "
      f"unknown correct new {pct('val_tuned_test_unknown_correct_new_rate')}%, unknown false match {pct('val_tuned_test_unknown_false_match_rate')}%.\n")
    cb = M.get("cross_burst_pipeline_rank1", {})
    if cb.get("per_partition"):
        ncb = [p["cross_burst_test_known"]["n"] for p in P0]
        w(f"Cross-burst test probes (enrolment and probe in different camera bursts, {min(ncb)} to {max(ncb)} per partition): "
          f"rank-1 {pct('cross_burst_pipeline_rank1')}%, DIR@FAR 1% transferred {pct('cross_burst_dir_transferred_1pct')}%. "
          "These few probes are the closest thing to a second session in this set.\n")
    bc = s["bc_et_al_for_context"]
    w("### BC et al. 2026 on the same images, for context\n")
    w(f"Oracle DIR@FAR 1% / 0.1%: hybrid CNN-ViT {bc['oracle_dir_far_1e-1_1e-2_1e-3']['hybrid_cnn_vit'][1]} / "
      f"{bc['oracle_dir_far_1e-1_1e-2_1e-3']['hybrid_cnn_vit'][2]}, MegaDescriptor-L {bc['oracle_dir_far_1e-1_1e-2_1e-3']['megadescriptor_l'][1]} / "
      f"{bc['oracle_dir_far_1e-1_1e-2_1e-3']['megadescriptor_l'][2]}. Transferred 1%: {bc['transferred_1pct']['hybrid_cnn_vit']['detection']} at realised FAR "
      f"{bc['transferred_1pct']['hybrid_cnn_vit']['realised_far']}% (hybrid) and {bc['transferred_1pct']['megadescriptor_l']['detection']} at "
      f"{bc['transferred_1pct']['megadescriptor_l']['realised_far']}% (MegaDescriptor-L). Closed-set rank-1 {bc['closed_set_rank1']['hybrid_cnn_vit']} "
      f"and {bc['closed_set_rank1']['megadescriptor_l']}. Their models train on 50% of each known animal's images and use 25% as a mean-embedding "
      "gallery. Mombe is training-free and enrols 3 frames per animal. The pool ratios, partition count and metric definitions match. "
      "Their duplicate list and split files sit in a repository that returns HTTP 404, so our partitions are our own. Read the comparison as like-for-like in protocol and "
      "approximate in data.\n")
    if D:
        rec = json.loads((OUT / "duplicates_reconciled.json").read_text())
        sfm = [v for r in rec["rows"] if (r.get("same_frame_median") or 0) >= T_DUP for v in r["same_frame_inliers"]]
        sfd = [v for r in rec["rows"] if r.get("same_frame_inliers") and (r.get("same_frame_median") or 0) < T_DUP for v in r["same_frame_inliers"]]
        rem = [r for r in rec["rows"] if r["decision"] in ("duplicate", "probable_duplicate", "mixed")]
        near = sum(r["range_gap_frames"] is not None and r["range_gap_frames"] <= 35 for r in rem)
        w("## Duplicate reconciliation\n")
        g = D["groups"]
        w(f"Full table: `duplicates_reconciled.md` (machine-readable: `duplicates_reconciled.json`). "
          f"We found **{len(g)} duplicate groups holding {D['n_removed_identities']} identity folders ({D['removed_images']} images)**. "
          f"Removing every folder in a group leaves {D['after_removal']['identities']} identities and {D['after_removal']['images']:,} images. "
          f"BC et al. removed {BC['duplicates_removed']['identities']} identities ({BC['duplicates_removed']['images']} images), leaving "
          f"{D['bc_et_al']['identities']} and {D['bc_et_al']['images']:,}. Their pools of 124 / 50 / 75 identities scale here to "
          f"{s['protocol']['pools'][0]} / {s['protocol']['pools'][1]} / {s['protocol']['pools'][2]} at the same ratios.\n")
        w("How the list was built:\n")
        w("1. Exhaustive screen: every one of the 35,778 identity pairs, first and last frame each (4 image pairs per identity pair), "
          "ALIKED mutual nearest neighbours with ratio test and MAGSAC++ inliers. 143,112 image pairs on Kaggle.")
        w("2. Verifier confirmation: every pair with a screen score of 20 or more, the 12 pairs of the 2026-10-14 audit and cattle_2200 / cattle_3819, "
          "verified 3 x 3 (first, middle, last frame) with Mombe's own LightGlue + MAGSAC verifier.")
        w(f"3. Camera-frame provenance, the signal BC et al. name: {D['provenance_summary']['frame_sharing_pairs']} identity pairs hold the same camera frame number. "
          f"Verifying the two crops of up to 3 shared frames splits them cleanly: {D['provenance_summary']['same_frame_crops_match']} pairs match with {min(sfm)} to {max(sfm)} inliers "
          f"(one animal cropped twice from one photo), and the rest stay at {min(sfd)} to {max(sfd)} (two animals in one telephoto frame, or a frame counter reused across sessions). "
          f"All {D['provenance_summary']['same_frame_crops_match']} were already in the verifier list.")
        w("4. Image review for the borderline pairs, with drawn inliers.\n")
        w("Rule: a 3 x 3 median of 90 inliers or more (the frozen t_match) is a duplicate; a median of 46 to 89, or a high maximum with a low median, "
          "goes to image review. Decisions: " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted(D["decisions"].items())) + ".\n")
        w("Groups removed:\n")
        for grp in g:
            w(f"- {' + '.join(grp)}")
        w("")
        w("cattle_2200 / cattle_3819 is resolved as a duplicate. All 9 frame combinations give 60 to 82 inliers. The cattle_2200 crops show only the right half of "
          "the muzzle, which caps the count, and the drawn inliers cover the shared right-nostril region. The two bursts sit 32 frames apart. "
          "Folder cattle_5407 holds two animals: most frames are cattle_6011 and one is the cattle_5408 / cattle_6012 animal. "
          f"{near} of the {len(rem)} removed pairs sit in adjacent bursts (frame ranges overlapping or at most 35 frames apart): the same animal photographed again and given a new number.\n")
        w(f"Why our count differs from 19: their list sits behind a 404, so the comparison is by count. Their method flags camera-frame provenance and confirms by eye. "
          f"Frame provenance alone finds {D['provenance_summary']['same_frame_crops_match']} of our pairs; the others hold distinct frame numbers and the visual screen finds them. "
          f"Keeping the largest folder of each group would remove {D['alt_keep_largest_folder']['removed_identities']} identities ({D['alt_keep_largest_folder']['removed_images']} images), "
          "close to their 19. We remove every folder in a group, because only 3 frames per folder were checked and one folder is mixed.\n")
    ri = s.get("residual_impostor_pairs", {})
    if ri:
        w("## Residual impostor check\n")
        w(f"After removal, {ri['impostor_pair_scores']:,} verified scores compare different identities across {ri['identity_pairs']:,} identity pairs. "
          f"{ri['scores_ge_review_46']} reach the review band (46 inliers or more) and {ri['scores_ge_90']} reach 90. "
          "Strongest pairs: " + ", ".join(f"{r['a']} / {r['b']} ({r['max']})" for r in ri["top_identity_pairs"][:5]) + ".")
        rc = OUT / "residual_check.json"
        if rc.exists():
            R_ = json.loads(rc.read_text())
            w("The same 3 x 3 rule used for the duplicate list re-checks every pair with a protocol maximum of 45 or more: "
              + "; ".join(f"{r['a']} / {r['b']} median {r['median_3x3']:.0f}, {r['decision_by_rule']}" for r in R_["pairs"])
              + ". This is a sensitivity check. The headline keeps the duplicate list fixed as it stood before any protocol score existed.")
        plan = json.loads((OUT / "plan.json").read_text())
        drv = collections.defaultdict(list)
        for p_, t_ in zip(P0, th):
            top_ = p_["top_unknown_scores"][0] if p_["top_unknown_scores"] else None
            if top_ and folder(top_["q"]) in plan["partitions"][p_["partition"]]["val_unknown"] and t_["t_match"] == int(max(2 * top_["s5"], top_["s5"] + 20)):
                drv[(top_["q"], top_["s5"], top_["matched"])].append(p_["partition"])
        for (q_, s_, m_), ps in drv.items():
            if len(ps) > 1:
                w(f"The calibrated t_match follows the strongest validation-unknown score. One probe, {q_}, scores {s_} inliers against {m_} "
                  f"and sets t_match to {int(max(2 * s_, s_ + 20))} in partitions {', '.join(map(str, ps[:-1]))} and {ps[-1]}, which turns some correct matches into second-angle requests there.")
        w("")
    w("## Protocol\n")
    pr = s["protocol"]
    w(f"- Source: {pr['source']}. Partitions: {pr['partitions']}, seeds {pr['partition_seed']} to {pr['partition_seed'] + pr['partitions'] - 1}.")
    w(f"- Pools per partition: {pr['pools'][0]} known, {pr['pools'][1]} validation-unknown, {pr['pools'][2]} test-unknown identities. "
      "BC et al. use 124 / 50 / 75 for 249 identities; ours scale the same ratios to the identities left after our duplicate removal.")
    w(f"- Enrolment: {pr['enrolment_rule']}; {n0['enrolment_images']} enrolment images per partition. Only known animals are enrolled.")
    w(f"- Probes: {pr['probe_rule']}. Unknown animals contribute the same probe frames.")
    w(f"- Score: {pr['score']}.")
    w(f"- DIR: {pr['dir_definition']}. FAR: {pr['far_definition']}. Threshold rule: accept when the score is strictly above the threshold; "
      "the threshold is the smallest one whose FAR on the reference unknown set is at most the target.")
    w(f"- Threshold transfer: {pr['threshold_transfer']}. Decision thresholds: {pr['decision_thresholds']}. The rule reads only validation-unknown scores; validation known probes report validation decision rates.")
    w(f"- identify() replay: {pr['identify_replay']}.")
    w("- Identity-clustered bootstrap (1,000 resamples per partition) on test for the transferred and oracle 1% figures: `summary.json`, key `bootstrap_identity_clustered`.\n")
    w("## Compute and coverage\n")
    w(f"- Probes: {s['data']['probes_with_features']} of {s['data']['probes']} planned have features. Pair scores: {done:,} of {att:,} attempted "
      f"across {len(shards)} Kaggle shards; missing pair scores over all partitions: {s['data']['missing_pair_scores_all_partitions']}.")
    hosts = {f"{m['host']['machine']} {m['host']['system']}, {m['host']['cpus']} vCPUs, OpenCV {m['host']['opencv']}" for m in shards}
    w(f"- Where: Kaggle CPU kernels ({'; '.join(sorted(hosts))}), features extracted from the original JPEGs in each kernel. "
      f"Verify wall time per shard: {min(m['verify_wall_s'] for m in shards) / 60:.0f} to {max(m['verify_wall_s'] for m in shards) / 60:.0f} min.")
    reran = [m for m in shards if "worker_crashes" in m]
    if reran:
        nc = sum(len(m["worker_crashes"]) for m in reran)
        cp = [c["crashed_pair"] for m in reran for c in m["worker_crashes"] if c.get("crashed_pair")]
        w(f"- Shards 0, 4 and 5 of 8 ran again on 2026-10-23 ({', '.join(m['tag'] for m in reran)}). On 2026-10-22 one verifier worker died in each first run, "
          "and multiprocessing.Pool waited forever for the lost task. The rerun uses a recovering executor: it re-queues in-flight probes "
          "after a worker death and isolates a pair that kills its worker twice. "
          + ("Shard 0 ran as two halves (shards 0 and 8 of 16) after its first rerun ended in a Kaggle ERROR with an empty log. "
             if any("of16" in m["tag"] for m in reran) else "")
          + f"Worker deaths in the reruns: {nc}" + (f"; pairs recorded as crashed and scored as missing: {'; '.join(f'{x[0]} against {x[1]}' for x in cp)}." if cp else "; every pair scored."))
        if cp:
            w(f"- The cause is a rare descriptor pair, {len(cp)} in all. On this PC each of these LightGlue calls raises cv2.error (\"Unknown C++ exception\"), "
              "which verify() catches and scores as 0 inliers. On Kaggle Linux the same call kills the process. The report scores a crashed pair as 0, "
              "the same outcome verify() gives on Windows. The hubs run Linux, so the identify service needs the same process isolation.")
    vpc = lc["sources"].get("this_pc", {}).get("stage_ms", {}).get("verify_ms")
    w("- This PC ran the duplicate image review, the plan, the report and the latency bench. A full local run was out of budget: LightGlue took "
      + (f"{vpc / 1000:.1f} s" if vpc else "several seconds") + f" per pair here under shared load, and the protocol needs {att:,} pairs.")
    w("- `tools/hub_jobs/identity_heldout.sh` runs the same plan on the Graviton4 COOL hub once AWS credentials are back.\n")
    w("## Latency and cost inputs\n")
    w(f"Verifications per probe measured in this protocol (calibrated thresholds): known {lc['verifications_per_probe']['known_probe']:.2f}, "
      f"unknown {lc['verifications_per_probe']['unknown_probe']:.2f}. Latency per probe = extract + shortlist + verifications x verify.\n")
    w("| Source | Extract ms | Verify ms | Known probe ms | Unknown probe ms | USD per 1,000 known | USD per 1,000 unknown |")
    w("|---|---|---|---|---|---|---|")
    for name, r in lc["sources"].items():
        st_ = r["stage_ms"]
        w(f"| {name}: {r['label']} | {st_['extract_ms']:g} | {st_['verify_ms']:g} | {r['latency_ms_known_probe']:,.0f} | {r['latency_ms_unknown_probe']:,.0f} | "
          f"{r.get('usd_per_1000_known_probe', 'n/a')} | {r.get('usd_per_1000_unknown_probe', 'n/a')} |")
    w("")
    sc = lc["sources"].get("this_pc", {}).get("shortlist_scaling")
    if sc:
        w("Shortlist cost against registry size on this PC (Gallery.shortlist, a Python loop over every enrolment image): "
          + ", ".join(f"{k.split('_')[1]} animals {v['median_ms']} ms" for k, v in sc.items())
          + ". It stays small next to one verification.\n")
    w(f"Assumptions: {lc['assumptions']}. The hub rows are projections from the 2026-10-15 hub bench; the hub job re-measures them on this plan.\n")
    w("## Limitations\n")
    w("- One session. 256 of the 268 identities sit in one camera burst, so most genuine probes are frames seconds from enrolment. "
      + (f"The few cross-burst probes already score lower (rank-1 {pct('cross_burst_pipeline_rank1')}%). " if cb.get("per_partition") else "")
      + "Cross-session field captures need their own test before any field claim.")
    w("- Telephoto feedyard images of one breed group. Phone close-ups and Zimbabwean breeds need their own test.")
    w("- FAR 0.1% is below the resolution of these pools.")
    w("- Duplicate removal checks 3 frames per folder. A mixed folder that this misses stays in as label noise and counts against Mombe.\n")
    w("## Reproduce\n")
    w("```")
    w("python eval/identity_heldout.py kaggle --kind dups      # then push; pull dup_*.json into dups/")
    w("python eval/identity_heldout.py kaggle --kind prov      # then push; pull prov_check.json into prov/")
    w("python eval/identity_heldout.py reconcile --review ... --note ...   # verdicts as recorded in duplicates_reconciled.json")
    w("python eval/identity_heldout.py plan")
    w("python eval/identity_heldout.py kaggle --kind pairs --shards 8      # push s0..s7; pull pairs_*.json and emb_*.npz into pairs/")
    w("python eval/identity_heldout.py kaggle --kind pairs --shards 8 --only-shards 0 4 5 --prefix r   # reruns r4 and r5 (r0 ended in a Kaggle ERROR)")
    w("python eval/identity_heldout.py kaggle --kind pairs --shards 16 --only-shards 0 8 --prefix h   # shard 0 of 8 as two halves, h0 and h8")
    w("python eval/identity_heldout.py residual --workers 2   # sensitivity: 3 x 3 re-check of the strongest impostor pairs")
    w("python eval/identity_heldout.py report && python eval/identity_heldout.py readme")
    w("```")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")


# --------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("dups")
    s.add_argument("--images", default=str(IMGS)); s.add_argument("--models", default=str(MODELS))
    s.add_argument("--out", default=str(OUT / "dups")); s.add_argument("--work", default=str(OUT / "_work"))
    s.add_argument("--workers", type=int, default=6); s.add_argument("--threads", type=int, default=1)
    s.add_argument("--use-cache", action="store_true")
    s.add_argument("--screen-cut", type=int, default=20); s.add_argument("--max-confirm", type=int, default=150)
    s.add_argument("--prior-pairs", nargs="*", default=[]); s.add_argument("--extra", nargs="*", default=[])
    s = sub.add_parser("prov")
    s.add_argument("--images", default=str(IMGS)); s.add_argument("--models", default=str(MODELS))
    s.add_argument("--out", default=str(OUT / "prov")); s.add_argument("--work", default=str(OUT / "_work"))
    s.add_argument("--workers", type=int, default=6); s.add_argument("--threads", type=int, default=1)
    s = sub.add_parser("reconcile"); s.add_argument("--dups", default=str(OUT / "dups"))
    s.add_argument("--prov", default=str(OUT / "prov"))
    s.add_argument("--review", nargs="*", default=[]); s.add_argument("--note", nargs="*", default=[])
    sub.add_parser("plan")
    s = sub.add_parser("pairs")
    s.add_argument("--plan", default=str(OUT / "plan.json")); s.add_argument("--images", default=str(IMGS))
    s.add_argument("--models", default=str(MODELS)); s.add_argument("--out", default=str(OUT / "pairs"))
    s.add_argument("--work", default=str(OUT / "_work")); s.add_argument("--shard", type=int, default=0)
    s.add_argument("--shards", type=int, default=1); s.add_argument("--workers", type=int, default=6)
    s.add_argument("--threads", type=int, default=1); s.add_argument("--limit", type=int, default=0)
    s.add_argument("--only", nargs="*", default=[]); s.add_argument("--tag", default="")
    s.add_argument("--use-cache", action="store_true"); s.add_argument("--stall-s", type=float, default=1800)
    s.add_argument("--recycle", type=int, default=10)
    s = sub.add_parser("kaggle"); s.add_argument("--kind", choices=["dups", "prov", "pairs"], default="pairs")
    s.add_argument("--shards", type=int, default=8); s.add_argument("--user", default="thetraveller")
    s.add_argument("--screen-cut", type=int, default=20); s.add_argument("--max-confirm", type=int, default=150)
    s.add_argument("--extra", nargs="*", default=["cattle_2200,cattle_3819"])
    s.add_argument("--only-shards", type=int, nargs="*", default=[]); s.add_argument("--prefix", default="s")
    s = sub.add_parser("report"); s.add_argument("--pairs", default=str(OUT / "pairs")); s.add_argument("--boot", type=int, default=1000)
    s.add_argument("--plan", default=str(OUT / "plan.json")); s.add_argument("--summary", default=str(OUT / "summary.json"))
    s = sub.add_parser("bench"); s.add_argument("--n", type=int, default=12); s.add_argument("--label", default="this_pc")
    s.add_argument("--note", default="")
    s = sub.add_parser("readme"); s.add_argument("--summary", default=str(OUT / "summary.json"))
    s.add_argument("--out", default=str(OUT / "README.md"))
    s = sub.add_parser("residual"); s.add_argument("--summary", default=str(OUT / "summary.json"))
    s.add_argument("--min-max", type=int, default=45); s.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    {"dups": stage_dups, "prov": stage_prov, "reconcile": stage_reconcile, "plan": stage_plan,
     "pairs": stage_pairs, "kaggle": stage_kaggle, "report": stage_report, "bench": stage_bench, "readme": stage_readme,
     "residual": stage_residual}[a.stage](a)


if __name__ == "__main__":
    main()
