"""Grouped weight evaluation on the Acme AI / bhalo / mPower cattle (Bangladesh, CC BY 4.0).

The question: does the two-photo weight model of eval/acme_weight.py transfer to a
capture batch it never saw? Random 10-fold CV inside batch B4 gave 13.3% MAPE.
Random folds share each batch's camera habits, sticker, herd and labelling crew
between training and test. Here each model trains on the other batches and is
tested on the held-out one (leave one batch out), beside null baselines and
beside the same models under random folds, so the gap between protocols shows.

Labels are tape-formula weights (truth label L), so a model built on girth and
length partly re-derives the label formula. The grouped numbers measure transfer
across batches against that label.

Two keypoint protocols, because the batches were annotated differently:
  full    9 side points (withers, pin bone, point of shoulder, girth tops and
          bottoms, height) + 4 rear points. B3 and B4 carry it. The models of
          eval/acme_weight.py: L shoulder to pin bone, D chest depth, H height,
          W rear width through the shared height, G ellipse through D and W.
  common  the 6 side points every batch shares (withers, pin bone, front and
          rear girth top and bottom), side photo only. B2 (its "Side" subset),
          B3 and B4 carry it. Lw withers to pin bone, Df front girth depth,
          Dr rear girth depth.
B2's "Side_2" subset and every B2 rear file use numbered keypoints ("1".."27")
with no published map; they are dropped and counted, never guessed.

Methods (all on log weight):
  constant        training median weight
  sex             training median weight per sex
  sticker_only    log w = a + b log(sticker major axis in pixels); no body pixels
  physics_b1      log w = a + log(V), V = G^2 L (full) or Df^2 Lw (common);
                  exponent fixed at 1, one constant (it absorbs the unpublished
                  sticker size and the tape formula)
  physics         log w = a + b log(V), as eval/acme_weight.py
  fitted          least squares on the log lengths and sex, as eval/acme_weight.py
  fitted_noscale  the fitted model on raw pixel lengths (sticker scale removed)

Intervals: split conformal on absolute log residuals. Each model is fitted on 75%
of the training animals; the other 25%, drawn from the training batches only,
calibrate the interval. The quantile is the ceil((n+1)(1-alpha))-th order
statistic. Mondrian intervals calibrate separately per band of PREDICTED weight,
the only band known at prediction time; coverage is reported by true and by
predicted band.

Pre-registered before any grouped run (2026-10-21): bands under 150 kg, 150 to
250 kg, over 250 kg; nominal 80% and 90%; seed 7; calibration fraction 0.25.

Usage:
  python eval/acme_grouped.py nulls                         # B4, local, acme_weight.py folds
  python eval/acme_grouped.py grouped --root DIR --out DIR  # B2, B3, B4 (Kaggle kernel)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acme_weight as aw  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

BATCHES = ("B2", "B3", "B4")
BANDS_KG = (150.0, 250.0)              # under 150 | 150 to 250 | over 250 (fixed in advance)
BAND_NAMES = ("under_150", "150_to_250", "over_250")
ALPHAS = {"80": 0.20, "90": 0.10}
CAL_FRACTION = 0.25
SEED = 7
SEEDS_REPEAT = (7, 8, 9, 10, 11)
MIN_GROUP = 40
STICKER_TOL = 40                       # max per-channel distance for the tolerant colour match
# Sticker colours in BGR, from the dataset Readme (B2 differs from B3/B4); checked on one mask each.
STICKER_BGR = {"B2": (0, 240, 255), "B3": (255, 117, 0), "B4": (255, 117, 0)}
CATTLE_BGR = {"B2": (0, 255, 0), "B3": (249, 30, 255), "B4": (249, 30, 255)}
SIDE_NAMES = ("wither", "pinbone", "shoulderbone", "front_girth_top", "front_girth_bottom",
              "rear_girth_top", "rear_girth_bottom", "height_top", "height_bottom")
COMMON_NAMES = ("wither", "pinbone", "front_girth_top", "front_girth_bottom", "rear_girth_top", "rear_girth_bottom")
REAR_NAMES = ("top", "bottom", "left", "right")
ALIAS = {"front_top": "front_girth_top", "front_bottom": "front_girth_bottom", "rear_top": "rear_girth_top",
         "rear_bottom": "rear_girth_bottom", "withers": "wither", "pin_bone": "pinbone", "shoulder_bone": "shoulderbone"}
METHODS = ("constant", "sex", "sticker_only", "physics_b1", "physics", "fitted", "fitted_noscale")
PROTOCOLS = {"full": {"vol": ("G", "G", "L"), "cols": ("G", "L", "D", "W", "H")},
             "common": {"vol": ("Df", "Df", "Lw"), "cols": ("Lw", "Df", "Dr")}}

LABEL_NOTE = ("Labels are Acme's tape-formula ('volumetric') weights, truth label L, never a scale. "
              "A model on image girth and length partly re-derives that formula, so within-batch "
              "accuracy flatters the method. Leave-one-batch-out measures transfer across capture "
              "batches against the same kind of label, which is the honest question.")


def log(*a) -> None:
    print(*a, flush=True)


# ---------------------------------------------------------------- names, files, keypoints

def parse_name(fname: str) -> dict | None:
    """<id>_[b4-sub_]<s|r>_<weight>[_<extra>]_<sex>[.jpg][___fuse.png] -> key, view, weight, sex."""
    stem = fname.replace("\\", "/").split("/")[-1].split("___")[0]
    stem = re.sub(r"\.(jpe?g|png)$", "", stem, flags=re.I)
    t = stem.split("_")
    for i, tok in enumerate(t):
        if tok.lower() in ("s", "r") and 1 <= i and i + 2 < len(t):
            try:
                w = float(t[i + 1])
            except ValueError:
                return None
            return {"key": "_".join(t[:i] + t[i + 1:]), "view": tok.lower(), "weight": w,
                    "sex": t[-1].upper(), "stem": stem, "tokens": t, "view_index": i}
    return None


def _norm_kp(name: str) -> str:
    n = re.sub(r"^\d+_", "", name.strip().lower()).replace("-", "_").replace(" ", "_")
    return ALIAS.get(n, n)


def load_keypoints(path: Path, canon: tuple[str, ...]) -> tuple[dict, dict]:
    """Per image: {canonical name: (x, y)} from the last annotation whose named points are all visible.

    Points are matched by NAME. A category with none of the canonical names (B2's numbered
    scheme) is skipped and its images are listed as 'undocumented'.
    """
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    imgs = {i["id"]: i["file_name"].replace("\\", "/").split("/")[-1] for i in c["images"]}
    dims = {imgs[i["id"]]: (i.get("width"), i.get("height")) for i in c["images"]}
    cat_map, notes = {}, []
    for cat in c.get("categories", []):
        kn = [_norm_kp(n) for n in cat.get("keypoints", [])]
        m = {i: n for i, n in enumerate(kn) if n in canon}
        cat_map[cat["id"]] = m or None
        if not m:
            notes.append(f"category '{cat.get('name')}': points {cat.get('keypoints')} carry no canonical name; skipped")
        elif [n for n in kn if n in canon] != [n for n in canon if n in m.values()]:
            notes.append(f"category '{cat.get('name')}': points reordered by name from {kn}")
    out, undocumented, incomplete, multi, seen = {}, set(), 0, 0, set()
    for a in c["annotations"]:
        fn = imgs.get(a["image_id"])
        if fn is None:
            continue
        m = cat_map.get(a.get("category_id"))
        if not m:
            undocumented.add(fn)
            continue
        k = np.array(a.get("keypoints") or [], float).reshape(-1, 3)
        if len(k) < max(m) + 1:
            incomplete += 1
            continue
        if fn in seen:
            multi += 1
        seen.add(fn)
        if (k[:, 2] > 0).all():                   # every point of the annotation visible, as acme_weight.keypoints
            out[fn] = {n: k[i, :2] for i, n in m.items()}
        else:
            incomplete += 1
    info = {"file": Path(path).name, "images": len(imgs), "annotations": len(c["annotations"]),
            "complete": len(out), "incomplete_annotations": incomplete,
            "images_with_multiple_annotations": multi, "undocumented_scheme_images": len(undocumented - set(out)),
            "categories": [{"name": cat.get("name"), "keypoints": cat.get("keypoints")} for cat in c.get("categories", [])],
            "notes": notes}
    return {"kp": out, "dims": dims, "images": set(imgs.values()), "undocumented": undocumented - set(out)}, info


def _major_axis(binary: np.ndarray) -> float | None:
    """Largest component, fitted ellipse, major axis: the logic of acme_weight.sticker_px."""
    st = binary.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(st)
    if n < 2:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cnt, _ = cv2.findContours((lab == big).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(cnt[0]) < 5:
        return None
    (_, _), (a, b), _ = cv2.fitEllipse(cnt[0])
    return float(max(a, b))


def _sticker_job(args: tuple) -> tuple:
    """(stem, exact-colour major axis, tolerant major axis or None, (mask w, h)).

    The tolerant match runs only when the exact colour fails, or when forced."""
    stem, path, bgr, tol, force_tol, img_wh, pts, cattle_bgr = args
    m = cv2.imread(path, cv2.IMREAD_COLOR)
    if m is None:
        return stem, None, None, None, None
    d_ex = _major_axis(np.all(m == np.array(bgr, np.uint8), axis=2))
    d_tol = None
    if force_tol or d_ex is None:
        lo = np.clip(np.array(bgr, int) - tol, 0, 255).astype(np.uint8)
        hi = np.clip(np.array(bgr, int) + tol, 0, 255).astype(np.uint8)
        d_tol = _major_axis(cv2.inRange(m, lo, hi) > 0)
    mask_wh = (int(m.shape[1]), int(m.shape[0]))
    # alignment check: outline keypoints mapped into mask pixels should touch the cattle class
    on_body = None
    f, case = scale_factor(img_wh, mask_wh)
    if f and case in ("same_size", "rescaled", "same_width_height_differs") and pts:
        body = cv2.inRange(m, np.array(cattle_bgr, np.uint8), np.array(cattle_bgr, np.uint8))
        r = max(3, int(0.015 * mask_wh[0]))
        body = cv2.dilate(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        hits = []
        for x, y in pts:
            u, v = int(round(x / f)), int(round(y / f))
            hits.append(0 <= u < mask_wh[0] and 0 <= v < mask_wh[1] and body[v, u] > 0)
        on_body = float(np.mean(hits))
    return stem, d_ex, d_tol, mask_wh, on_body


def scale_factor(img_wh, mask_wh, tol: float = 0.01) -> tuple[float | None, str]:
    """Factor that turns mask pixels into keypoint (image) pixels, and the case that applied.

    Seen in the data: B3 keypoints sit on the 4160x3120 (and other) originals while masks are
    1900x1425; 42 B4 masks are stored portrait (1425x1900) for landscape keypoints; a few B4
    images are listed a few rows shorter than their 1900x1425 mask."""
    if not img_wh or not img_wh[0] or not mask_wh:
        return 1.0, "no_size_info"
    W, H = img_wh
    w, h = mask_wh
    if (W, H) == (w, h):
        return 1.0, "same_size"
    if abs(W / w - H / h) <= tol * W / w:
        return W / w, "rescaled"
    if abs(W / h - H / w) <= tol * W / h:
        return W / h, "rotated" if (W, H) == (h, w) else "rotated_rescaled"
    if W == w:
        return 1.0, "same_width_height_differs"
    return None, "inconsistent"


def colour_census(path: Path, top: int = 6) -> list:
    m = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if m is None:
        return []
    packed = (m[..., 2].astype(np.int64) << 16) | (m[..., 1].astype(np.int64) << 8) | m[..., 0]
    vals, counts = np.unique(packed.ravel(), return_counts=True)
    order = np.argsort(-counts)[:top]
    return [{"rgb": [int(vals[i] >> 16), int((vals[i] >> 8) & 255), int(vals[i] & 255)],
             "frac": round(float(counts[i] / packed.size), 4)} for i in order]


def find_root(base: str = "/kaggle/input", max_depth: int = 6) -> Path:
    """Directory holding Vector/ and Pixel/ with all of B2, B3, B4; never a hardcoded slug."""
    hits, rejects = [], []
    base_p = Path(base)
    for dirpath, dirnames, _ in os.walk(base):
        p = Path(dirpath)
        if len(p.relative_to(base_p).parts) >= max_depth:
            dirnames[:] = []
            continue
        if "Vector" in dirnames and "Pixel" in dirnames:
            ok = all((p / "Vector" / b / "Side").is_dir() and (p / "Vector" / b / "Rear").is_dir()
                     and (p / "Pixel" / b).is_dir() for b in BATCHES)
            (hits if ok else rejects).append(str(p))
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in ("images", "annotations", ".thumbnail")]
    log(f"[root] accepted {len(hits)}: {hits}; rejected {len(rejects)}: {rejects}")
    if not hits:
        raise SystemExit("dataset root with Vector/ and Pixel/ for B2, B3, B4 not found under " + base)
    return Path(hits[0])


# ---------------------------------------------------------------- one batch -> rows + accounting

def _merge(paths: list[Path], canon: tuple[str, ...]):
    kp, dims, images, undocumented, infos, subset = {}, {}, set(), set(), [], {}
    dup = conflict = 0
    for p in paths:
        d, info = load_keypoints(p, canon)
        infos.append(info)
        images |= d["images"]
        undocumented |= d["undocumented"]
        for fn, k in d["kp"].items():
            if fn in kp:
                dup += 1
                common = set(k) & set(kp[fn])
                conflict += int(any(np.abs(kp[fn][n] - k[n]).max() > 2 for n in common))
                continue
            kp[fn], dims[fn], subset[fn] = k, d["dims"].get(fn), p.parent.name
    return {"kp": kp, "dims": dims, "images": images, "undocumented": undocumented - set(kp), "subset": subset,
            "info": {"files": infos, "duplicate_images_across_files": dup, "duplicates_with_different_points": conflict}}


def build_batch(root: Path, batch: str, workers: int = 1, census: int = 3) -> tuple[list[dict], dict]:
    t0 = time.time()
    root = Path(root)
    side_json = sorted((root / "Vector" / batch / "Side").rglob("*.json"))
    rear_json = sorted((root / "Vector" / batch / "Rear").rglob("*.json"))
    mask_files = sorted(p for p in (root / "Pixel" / batch).rglob("*___fuse.png") if ".thumbnail" not in p.parts)
    S, R = _merge(side_json, SIDE_NAMES), _merge(rear_json, REAR_NAMES)
    acc = {"batch": batch, "side_json": [str(p.relative_to(root)) for p in side_json],
           "rear_json": [str(p.relative_to(root)) for p in rear_json],
           "side_keypoints": S["info"], "rear_keypoints": R["info"]}

    masks, unparsed_masks = defaultdict(list), 0
    for p in mask_files:
        pn = parse_name(p.name)
        if pn is None:
            unparsed_masks += 1
        elif pn["view"] == "s":
            masks[pn["stem"]].append(p)
    acc["side_mask_files"] = sum(len(v) for v in masks.values())
    acc["side_mask_stems_with_copies"] = sum(len(v) > 1 for v in masks.values())
    acc["unparsed_mask_names"] = unparsed_masks

    side_listed = [n for n in S["images"] if (parse_name(n) or {}).get("view") == "s"]
    acc["unparsed_side_names"] = sum(parse_name(n) is None for n in S["images"])
    keys_avail = {parse_name(n)["key"] for n in side_listed} | {parse_name(s)["key"] for s in masks}
    side_by_key, rear_by_key, rear_undoc = defaultdict(list), defaultdict(list), set()
    for fn in S["kp"]:
        pn = parse_name(fn)
        if pn and pn["view"] == "s":
            side_by_key[pn["key"]].append(fn)
    for fn in R["kp"]:
        pn = parse_name(fn)
        if pn and pn["view"] == "r":
            rear_by_key[pn["key"]].append(fn)
    for fn in R["undocumented"]:
        pn = parse_name(fn)
        if pn:
            rear_undoc.add(pn["key"])
    side_undoc = {parse_name(fn)["key"] for fn in S["undocumented"] if parse_name(fn)}

    drops = {"full": Counter(), "common": Counter()}
    drop_rows, cand = [], []
    for key in sorted(keys_avail):
        sf = side_by_key.get(key, [])
        if not sf:
            why = "side_keypoints_undocumented_scheme" if key in side_undoc else "side_keypoints_missing_or_incomplete"
            for p in drops:
                drops[p][why] += 1
                drop_rows.append((batch, key, p, why))
            continue
        if len(sf) > 1 or len(rear_by_key.get(key, [])) > 1:
            for p in drops:
                drops[p]["ambiguous_key"] += 1
                drop_rows.append((batch, key, p, "ambiguous_key"))
            continue
        fn = sf[0]
        kd = S["kp"][fn]
        common_ok = all(n in kd for n in COMMON_NAMES)
        full_ok = all(n in kd for n in SIDE_NAMES)
        why_full = None
        if not full_ok:
            why_full = "side_lacks_shoulder_or_height"
        elif key not in rear_by_key:
            why_full = "rear_keypoints_undocumented_scheme" if key in rear_undoc else "rear_keypoints_missing_or_incomplete"
        why_common = None if common_ok else "side_lacks_common_points"
        copies = masks.get(parse_name(fn)["stem"], [])
        if not copies:
            why_full, why_common = why_full or "side_mask_missing", why_common or "side_mask_missing"
        for p, why in (("full", why_full), ("common", why_common)):
            if why:
                drops[p][why] += 1
                drop_rows.append((batch, key, p, why))
        if why_full and why_common:
            continue
        same = [p for p in copies if S["subset"].get(fn) in p.relative_to(root).parts]
        cand.append({"key": key, "fn": fn, "mask": (same or copies)[0],
                     "rear": rear_by_key[key][0] if not why_full else None,
                     "full": why_full is None, "common": why_common is None})

    acc["census"] = [{"mask": c["mask"].name, "top_colours": colour_census(c["mask"])} for c in cand[:census]]
    bgr = STICKER_BGR[batch]
    def run_jobs(force_tol):
        jobs = [(c["fn"], str(c["mask"]), bgr, STICKER_TOL, force_tol, S["dims"].get(c["fn"]),
                 [tuple(map(float, S["kp"][c["fn"]][n])) for n in COMMON_NAMES if n in S["kp"][c["fn"]]],
                 CATTLE_BGR[batch]) for c in cand]
        if workers > 1 and len(jobs) > 50:
            with ProcessPoolExecutor(workers) as ex:
                return list(ex.map(_sticker_job, jobs, chunksize=16))
        return [_sticker_job(j) for j in jobs]

    res = run_jobs(False)
    n_exact = sum(r[1] is not None for r in res)
    rescued = sum(r[1] is None and r[2] is not None for r in res)
    use_tol = len(res) > 0 and n_exact < 0.5 * len(res)
    if use_tol:
        res = run_jobs(True)
    st = {r[0]: r for r in res}
    acc["sticker"] = {"bgr": list(bgr), "masks_read": len(res), "found_exact": n_exact,
                      "exact_failures_a_tolerant_match_would_rescue": rescued,
                      "found_tolerant": sum(r[2] is not None for r in res) if use_tol else None,
                      "tolerance": STICKER_TOL, "match_used": "tolerant" if use_tol else "exact",
                      "rule": "tolerant match only when the exact colour finds the sticker in under half the masks"}

    rows, size_cases = [], Counter()
    for c in cand:
        key, fn = c["key"], c["fn"]
        _, d_ex, d_tol, shape, _ = st[fn]
        d = d_tol if use_tol else d_ex
        flags = [p for p in ("full", "common") if c[p]]
        if not d:
            for p in flags:
                drops[p]["sticker_not_found"] += 1
                drop_rows.append((batch, key, p, "sticker_not_found"))
            continue
        f, case = scale_factor(S["dims"].get(fn), shape)
        size_cases[case] += 1
        if f is None:
            for p in flags:
                drops[p]["mask_keypoint_size_inconsistent"] += 1
                drop_rows.append((batch, key, p, "mask_keypoint_size_inconsistent"))
            continue
        d = d * f                                 # sticker in keypoint (image) pixel units
        k = S["kp"][fn]
        s = 1.0 / d
        pn = parse_name(fn)
        row = {"batch": batch, "key": key, "side_file": fn, "rear_file": c["rear"] or "",
               "group": pn["tokens"][1] if batch == "B4" and pn["view_index"] >= 2 else S["subset"].get(fn, batch),
               "weight": pn["weight"], "sex": pn["sex"], "d_px": float(d)}
        ok = {}
        if c["common"]:
            Lw = np.linalg.norm(k["wither"] - k["pinbone"]) * s
            Df = np.linalg.norm(k["front_girth_top"] - k["front_girth_bottom"]) * s
            Dr = np.linalg.norm(k["rear_girth_top"] - k["rear_girth_bottom"]) * s
            ok["common"] = min(Lw, Df, Dr) > 0
            row.update(Lw=float(Lw), Df=float(Df), Dr=float(Dr))
        if c["full"]:
            kr = R["kp"][c["rear"]]
            L = np.linalg.norm(k["shoulderbone"] - k["pinbone"]) * s      # acme_weight.build: k[2] - k[1]
            D = np.linalg.norm(k["front_girth_top"] - k["front_girth_bottom"]) * s
            H = np.linalg.norm(k["height_top"] - k["height_bottom"]) * s
            rh = np.linalg.norm(kr["top"] - kr["bottom"])
            W = np.linalg.norm(kr["left"] - kr["right"]) / rh * H if rh > 0 else 0.0
            ok["full"] = min(L, D, H, W) > 0
            if ok["full"]:
                row.update(L=float(L), D=float(D), H=float(H), W=float(W),
                           G=float(aw.ellipse_perimeter(D / 2, W / 2)))
        for p in flags:
            if not ok[p]:
                drops[p]["geometry_degenerate"] += 1
                drop_rows.append((batch, key, p, "geometry_degenerate"))
        row["full"] = bool(ok.get("full"))
        row["common"] = bool(ok.get("common"))
        if row["full"] or row["common"]:
            rows.append(row)
    acc["mask_vs_keypoint_image_size"] = dict(size_cases)
    ob = np.array([r[4] for r in res if r[4] is not None], float)
    acc["keypoints_on_cattle_mask"] = {
        "what": "share of the 6 outline keypoints that land on the cattle class (dilated 1.5% of mask width) "
                "after mapping into mask pixels; rotated masks skipped; reported only, never used to drop rows",
        "checked": int(len(ob)), "mean": _r(float(ob.mean())) if len(ob) else None,
        "share_all_6_on_body": _r(float((ob >= 0.999).mean())) if len(ob) else None,
        "share_at_most_3_on_body": _r(float((ob <= 0.5).mean())) if len(ob) else None}
    acc["available_animals"] = len(keys_avail)
    acc["used"] = {p: sum(r[p] for r in rows) for p in ("full", "common")}
    acc["drops"] = {p: dict(v) for p, v in drops.items()}
    acc["drop_rows"] = drop_rows
    acc["seconds"] = round(time.time() - t0, 1)
    for p in ("full", "common"):
        assert acc["available_animals"] == acc["used"][p] + sum(drops[p].values()), (batch, p)
    log(f"[{batch}] available {len(keys_avail)} used {acc['used']} drops {acc['drops']} "
        f"sticker exact {n_exact}/{len(res)} rescuable {rescued} sizes {dict(size_cases)} ({acc['seconds']} s)")
    return rows, acc


# ---------------------------------------------------------------- models

def _vol(rows, proto):
    a, b, c = PROTOCOLS[proto]["vol"]
    return np.log([r[a] * r[b] * r[c] for r in rows])


def _X(method: str, rows: list[dict], proto: str) -> np.ndarray:
    one = np.ones(len(rows))
    male = np.array([r["sex"] == "M" for r in rows], float)
    cols = PROTOCOLS[proto]["cols"]
    if method == "physics":
        return np.c_[one, _vol(rows, proto)]
    if method == "fitted":
        return np.c_[one, np.log([[r[c] for c in cols] for r in rows]), male]
    if method == "fitted_noscale":
        return np.c_[one, np.log([[r[c] * r["d_px"] for c in cols] for r in rows]), male]
    if method == "sticker_only":
        return np.c_[one, np.log([r["d_px"] for r in rows])]
    raise ValueError(method)


def fit_method(method: str, rows: list[dict], proto: str = "full"):
    """Return (predict(rows) -> log kg, params)."""
    w = np.array([r["weight"] for r in rows], float)
    y = np.log(w)
    if method == "constant":
        c = float(np.log(np.median(w)))
        return (lambda R: np.full(len(R), c)), {"median_kg": float(np.median(w))}
    if method == "sex":
        overall = float(np.median(w))
        med = {s: float(np.median([r["weight"] for r in rows if r["sex"] == s])) for s in sorted({r["sex"] for r in rows})}
        return (lambda R: np.log([med.get(r["sex"], overall) for r in R])), {"median_kg_by_sex": med}
    if method == "physics_b1":
        a = float(np.mean(y - _vol(rows, proto)))
        return (lambda R: a + _vol(R, proto)), {"a": round(a, 5)}
    X = _X(method, rows, proto)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return (lambda R: _X(method, R, proto) @ beta), {"beta": [round(float(b), 5) for b in beta]}


def band_of(w) -> np.ndarray:
    w = np.asarray(w, float)
    return np.where(w < BANDS_KG[0], 0, np.where(w <= BANDS_KG[1], 1, 2))


def conformal_q(res: np.ndarray, alpha: float) -> float:
    n = len(res)
    k = math.ceil((n + 1) * (1 - alpha))
    if n == 0 or k > n:
        return math.inf
    return float(np.sort(res)[k - 1])


def run_split(rows: list[dict], tr, te, seed: int, proto: str) -> dict:
    """Fit on 75% of tr, calibrate on the other 25% of tr, predict te. Per-row arrays per method."""
    rng = np.random.default_rng(seed)
    tr, te = np.asarray(tr), np.asarray(te)
    calm = rng.random(len(tr)) < CAL_FRACTION
    proper, cal = tr[~calm], tr[calm]
    sub = lambda idx: [rows[i] for i in idx]  # noqa: E731
    yc = np.log([rows[i]["weight"] for i in cal])
    out = {}
    for m in METHODS:
        pred, params = fit_method(m, sub(proper), proto)
        pc, pt = pred(sub(cal)), pred(sub(te))
        res = np.abs(yc - pc)
        bc, bt = band_of(np.exp(pc)), band_of(np.exp(pt))
        o = {"pred_log": pt, "params": params, "n_proper": int(len(proper)), "n_cal": int(len(cal)),
             "n_cal_by_pred_band": [int((bc == b).sum()) for b in range(3)]}
        for lab, a in ALPHAS.items():
            o["q" + lab] = np.full(len(te), conformal_q(res, a))
            qb = [conformal_q(res[bc == b], a) for b in range(3)]
            o["qm" + lab] = np.array([qb[b] for b in bt], float)
        out[m] = o
    return out


# ---------------------------------------------------------------- metrics

def _r(x, nd=4):
    return None if x is None or not math.isfinite(x) else round(float(x), nd)


def point_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = p - y
    ape = np.abs(err) / y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {"n": int(len(y)), "mape": _r(ape.mean()), "median_ape": _r(np.median(ape)),
            "within_10pct": _r((ape <= 0.10).mean(), 3), "rmse_kg": _r(rmse, 1),
            "nrmse": _r(rmse / y.mean()), "mae_kg": _r(np.abs(err).mean(), 1),
            "bias_kg": _r(err.mean(), 1), "bias_pct": _r((err / y).mean())}


def interval_metrics(y: np.ndarray, p: np.ndarray, q: np.ndarray, pred_band: np.ndarray) -> dict:
    lo, hi = p * np.exp(-q), p * np.exp(q)
    cov = (y >= lo) & (y <= hi)
    width = hi - lo
    fin = np.isfinite(q)
    out = {"coverage": _r(cov.mean(), 3), "unbounded_rows": int((~fin).sum()),
           "median_width_kg": _r(np.median(width), 1) if fin.all() else None,
           "median_half_width_pct": _r((np.exp(np.median(q)) - 1) * 100, 1)}
    tb = band_of(y)
    for tag, bands in (("by_true_band", tb), ("by_pred_band", pred_band)):
        out[tag] = {}
        for b, name in enumerate(BAND_NAMES):
            idx = bands == b
            if idx.sum() == 0:
                out[tag][name] = {"n": 0}
                continue
            out[tag][name] = {"n": int(idx.sum()), "coverage": _r(cov[idx].mean(), 3),
                              "median_width_kg": _r(np.median(width[idx]), 1) if fin[idx].all() else None}
    return out


def weight_range(y: np.ndarray) -> dict:
    return {"min": float(y.min()), "p5": float(np.percentile(y, 5)), "median": float(np.median(y)),
            "p95": float(np.percentile(y, 95)), "max": float(y.max()), "mean": round(float(y.mean()), 1),
            "n_by_band": {n: int((band_of(y) == b).sum()) for b, n in enumerate(BAND_NAMES)}}


def evaluate(y: np.ndarray, per: dict) -> dict:
    """per: method -> concatenated arrays (pred_log, q80, q90, qm80, qm90)."""
    out = {}
    for m, o in per.items():
        p = np.exp(o["pred_log"])
        pb = band_of(p)
        out[m] = {"point": point_metrics(y, p),
                  "conformal": {lab: interval_metrics(y, p, o["q" + lab], pb) for lab in ALPHAS},
                  "mondrian": {lab: interval_metrics(y, p, o["qm" + lab], pb) for lab in ALPHAS}}
    return out


def _concat(parts: list[dict]) -> dict:
    keys = ("pred_log", "q80", "q90", "qm80", "qm90")
    return {m: {k: np.concatenate([p[m][k] for p in parts]) for k in keys} for m in METHODS}


def kfold(rows: list[dict], proto: str, folds: int = 10, seed: int = SEED) -> tuple[np.ndarray, dict]:
    fold = np.random.default_rng(seed).permutation(len(rows)) % folds
    parts, order = [], []
    for f in range(folds):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        parts.append(run_split(rows, tr, te, seed=seed * 100 + f, proto=proto))
        order.append(te)
    idx = np.concatenate(order)
    return np.array([rows[i]["weight"] for i in idx], float), _concat(parts)


def leave_group_out(rows: list[dict], group_key: str, proto: str, seed: int = SEED) -> tuple[dict, dict]:
    groups = sorted({r[group_key] for r in rows})
    per_group, parts, ys = {}, [], []
    for g in groups:
        te = np.array([i for i, r in enumerate(rows) if r[group_key] == g])
        tr = np.array([i for i, r in enumerate(rows) if r[group_key] != g])
        o = run_split(rows, tr, te, seed=seed, proto=proto)
        y = np.array([rows[i]["weight"] for i in te], float)
        per_group[g] = {"n_test": int(len(te)), "n_train": int(len(tr)),
                        "train_groups": sorted({r[group_key] for r in rows} - {g}),
                        "weight_kg": weight_range(y), "metrics": evaluate(y, _concat([o])),
                        "params": {m: o[m]["params"] for m in METHODS},
                        "n_proper": o["fitted"]["n_proper"], "n_cal": o["fitted"]["n_cal"],
                        "n_cal_by_pred_band": {m: o[m]["n_cal_by_pred_band"] for m in ("physics", "fitted")}}
        parts.append(o)
        ys.append(y)
    yall = np.concatenate(ys)
    return per_group, {"weight_kg": weight_range(yall), "metrics": evaluate(yall, _concat(parts))}


def headline(block: dict) -> dict:
    """Compact table: method -> MAPE, nRMSE, MAE, bias, coverage at 80% and 90%."""
    out = {}
    for m in METHODS:
        b = block[m]
        out[m] = {"mape": b["point"]["mape"], "nrmse": b["point"]["nrmse"], "mae_kg": b["point"]["mae_kg"],
                  "bias_pct": b["point"]["bias_pct"], "cov80": b["conformal"]["80"]["coverage"],
                  "cov90": b["conformal"]["90"]["coverage"],
                  "half_width90_pct": b["conformal"]["90"]["median_half_width_pct"],
                  "mondrian_cov90": b["mondrian"]["90"]["coverage"]}
    return out


def batch_diagnostics(rows: list[dict], proto: str) -> dict:
    cols = PROTOCOLS[proto]["cols"]
    w = np.array([r["weight"] for r in rows], float)
    med = lambda k: round(float(np.median([r[k] for r in rows])), 4)  # noqa: E731
    out = {"n": len(rows), "weight_kg": weight_range(w), "sex": dict(Counter(r["sex"] for r in rows)),
           "groups": dict(Counter(r["group"] for r in rows)),
           "median": {k: med(k) for k in ("d_px",) + cols},
           "fit_on_this_batch_alone": {}}
    if proto == "full":
        out["median_ratio"] = {f"{a}/H": round(float(np.median([r[a] / r["H"] for r in rows])), 4) for a in ("L", "D", "W")}
    else:
        out["median_ratio"] = {f"{a}/Lw": round(float(np.median([r[a] / r["Lw"] for r in rows])), 4) for a in ("Df", "Dr")}
    if len(rows) >= MIN_GROUP:
        for m in ("physics_b1", "physics", "fitted"):
            out["fit_on_this_batch_alone"][m] = fit_method(m, rows, proto)[1]
    return out


LOCAL_N = (25, 50, 100)
LOCAL_REPEATS = 20


def local_calibration(rows_all: list[dict], proto: str) -> dict:
    """Extension: a few labelled animals from the NEW batch recalibrate a model trained elsewhere.

    The model is fitted on every animal of the other batches. From the held-out batch, n animals
    are drawn at random (20 repeats). local_cal: the n animals calibrate the interval.
    local_shift_cal: half of them set an intercept shift (median log residual), the other half
    calibrate. Metrics are computed on the rest of the held-out batch."""
    out = {}
    for held in sorted({r["batch"] for r in rows_all}):
        tr = [r for r in rows_all if r["batch"] != held]
        te = [r for r in rows_all if r["batch"] == held]
        y = np.log([r["weight"] for r in te])
        out[held] = {}
        for m in ("physics", "fitted"):
            pred, _ = fit_method(m, tr, proto)
            p = pred(te)
            w = np.exp(y)
            out[held][m] = {"no_local": {"mape": _r(np.mean(np.abs(np.exp(p) - w) / w)), "n_test": len(te)}}
            for n in LOCAL_N:
                if n > len(te) // 2:
                    continue
                agg = defaultdict(list)
                for rep in range(LOCAL_REPEATS):
                    perm = np.random.default_rng(1000 * n + rep).permutation(len(te))
                    loc, ev = perm[:n], perm[n:]
                    h = n // 2
                    shift = float(np.median(y[loc[:h]] - p[loc[:h]]))
                    variants = (("local_cal", np.abs(y[loc] - p[loc]), 0.0),
                                ("local_shift_cal", np.abs(y[loc[h:]] - p[loc[h:]] - shift), shift))
                    yt = w[ev]
                    for var, res, sh in variants:
                        pe = np.exp(p[ev] + sh)
                        agg[var + ".mape"].append(float(np.mean(np.abs(pe - yt) / yt)))
                        for lab, a in ALPHAS.items():
                            q = conformal_q(res, a)
                            agg[f"{var}.cov{lab}"].append(float(np.mean((yt >= pe * np.exp(-q)) & (yt <= pe * np.exp(q)))))
                            agg[f"{var}.half_width{lab}_pct"].append((math.exp(q) - 1) * 100 if math.isfinite(q) else math.nan)
                out[held][m][str(n)] = {k: {"mean": _r(float(np.nanmean(v))), "min": _r(float(np.nanmin(v))),
                                            "max": _r(float(np.nanmax(v)))} for k, v in agg.items()}
    return out


# ---------------------------------------------------------------- grouped run (Kaggle)

def analyse(data: dict[str, list[dict]], proto: str) -> dict:
    t0 = time.time()
    rows_b = {b: [r for r in data[b] if r[proto]] for b in data}
    usable = [b for b in rows_b if len(rows_b[b]) >= MIN_GROUP]
    res = {"batches_used": usable, "n_by_batch": {b: len(v) for b, v in rows_b.items()}}
    if len(usable) < 2:
        res["skipped"] = "fewer than two batches with enough animals"
        return res
    rows_all = [r for b in usable for r in rows_b[b]]
    res["diagnostics"] = {b: batch_diagnostics(rows_b[b], proto) for b in usable}

    lobo, pooled = leave_group_out(rows_all, "batch", proto)
    res["lobo"] = {"per_heldout_batch": lobo, "pooled": pooled}
    rep = defaultdict(lambda: defaultdict(list))
    for s in SEEDS_REPEAT:
        per, pl = leave_group_out(rows_all, "batch", proto, seed=s)
        for g, blk in list(per.items()) + [("pooled", pl)]:
            for m in ("physics_b1", "physics", "fitted"):
                mt = blk["metrics"][m]
                rep[g][m + ".mape"].append(mt["point"]["mape"])
                rep[g][m + ".cov80"].append(mt["conformal"]["80"]["coverage"])
                rep[g][m + ".cov90"].append(mt["conformal"]["90"]["coverage"])
                rep[g][m + ".mondrian_cov90"].append(mt["mondrian"]["90"]["coverage"])
    res["lobo"]["seed_range"] = {g: {k: [min(v), max(v)] for k, v in d.items()} for g, d in rep.items()}
    res["local_calibration"] = local_calibration(rows_all, proto)

    res["random10_within_batch"] = {}
    for b in usable:
        y, per = kfold(rows_b[b], proto)
        res["random10_within_batch"][b] = {"weight_kg": weight_range(y), "metrics": evaluate(y, per)}
    y, per = kfold(rows_all, proto)
    res["random10_pooled"] = {"weight_kg": weight_range(y), "metrics": evaluate(y, per)}

    res["leave_subgroup_out"] = {}
    for b in usable:
        gs = Counter(r["group"] for r in rows_b[b])
        if len(gs) >= 2 and min(gs.values()) >= MIN_GROUP:
            per, pl = leave_group_out(rows_b[b], "group", proto)
            res["leave_subgroup_out"][b] = {"groups": dict(gs), "per_group": per, "pooled": pl}

    gap = {}
    for b in usable:
        gap[b] = {"n": len(rows_b[b]), "random10_within": headline(res["random10_within_batch"][b]["metrics"]),
                  "lobo_heldout": headline(lobo[b]["metrics"])}
    gap["pooled"] = {"n": len(rows_all), "random10_pooled": headline(res["random10_pooled"]["metrics"]),
                     "lobo_pooled": headline(pooled["metrics"]),
                     "lobo_macro_mape": {m: _r(np.mean([lobo[b]["metrics"][m]["point"]["mape"] for b in usable]))
                                         for m in METHODS}}
    for b, v in res["leave_subgroup_out"].items():
        gap[b]["leave_subgroup_out"] = headline(v["pooled"]["metrics"])
    res["headline"] = gap
    res["seconds"] = round(time.time() - t0, 1)
    return res


def run_grouped(root: Path, out_dir: Path, workers: int = 4, keep: dict | None = None) -> int:
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"[env] python {sys.version.split()[0]} cv2 {cv2.__version__} numpy {np.__version__} workers {workers}")
    data, acc = {}, {}
    for b in BATCHES:
        data[b], acc[b] = build_batch(root, b, workers=workers)
    cols = ["batch", "group", "key", "side_file", "rear_file", "weight", "sex", "d_px", "full", "common",
            "Lw", "Df", "Dr", "L", "D", "H", "W", "G"]
    with open(out_dir / "rows.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, restval="")
        wr.writeheader()
        for b in BATCHES:
            wr.writerows(data[b])
    with open(out_dir / "drops.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["batch", "key", "protocol", "reason"])
        for b in BATCHES:
            wr.writerows(acc[b].pop("drop_rows"))
    summary = {"track": "1c Acme grouped weight", "label_note": LABEL_NOTE,
               "evidence_labels": {"input": "R (real photos; keypoints and masks by Acme annotators)",
                                   "truth": "L (tape-formula weight in the file name)"},
               "preregistered": {"bands_kg": "under 150 | 150 to 250 | over 250", "alphas": ALPHAS,
                                 "cal_fraction": CAL_FRACTION, "seed": SEED, "seeds_repeat": SEEDS_REPEAT,
                                 "min_group": MIN_GROUP},
               "methods": METHODS, "protocols": {k: {"cols": v["cols"], "vol": v["vol"]} for k, v in PROTOCOLS.items()},
               "accounting": acc, "results": {}}
    for proto in ("full", "common"):
        summary["results"][proto] = analyse(data, proto)
        log(f"[{proto}] done ({time.time() - t0:.0f} s)")
        log(json.dumps(summary["results"][proto].get("headline"), indent=1))
    summary["seconds"] = round(time.time() - t0, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log(f"[done] {summary['seconds']} s")
    if keep is not None:
        keep["data"], keep["acc"] = data, acc
    return 0


# ---------------------------------------------------------------- B4 nulls (local)

def aw_cv_predictions(rows: list[dict], folds: int = 10, seed: int = 7):
    """acme_weight.cv_eval, step for step (same RNG stream), returning the point predictions."""
    rng = np.random.default_rng(seed)
    n = len(rows)
    y = np.log([r["weight"] for r in rows])
    X_phys = np.c_[np.ones(n), np.log([r["G"] ** 2 * r["L"] for r in rows])]
    X_fit = np.c_[np.ones(n), np.log([[r["G"], r["L"], r["D"], r["W"], r["H"]] for r in rows]),
                  [r["sex"] == "M" for r in rows]]
    fold = rng.permutation(n) % folds
    preds = {}
    for name, X in (("physics", X_phys), ("fitted", X_fit)):
        pred = np.zeros(n)
        for f in range(folds):
            tr, te = fold != f, fold == f
            cal = tr & (rng.random(n) < 0.25)
            beta = np.linalg.lstsq(X[tr & ~cal], y[tr & ~cal], rcond=None)[0]
            pred[te] = X[te] @ beta
        preds[name] = pred
    return fold, preds


def run_nulls(out_dir: Path, workers: int = 4, mine: list[dict] | None = None, acc: dict | None = None) -> int:
    """B4 nulls on the acme_weight.py folds. acme_weight.build reads ROOT/data/acme (on Kaggle the kernel
    links that path to the attached dataset). mine/acc: B4 rows already built by build_batch."""
    out_dir.mkdir(parents=True, exist_ok=True)
    root = ROOT / "data" / "acme"
    t0 = time.time()
    rows_aw = aw.build("B4")                                   # the rows behind the 13.3%
    log(f"[aw] acme_weight.build B4: {len(rows_aw)} rows ({time.time() - t0:.0f} s)")
    if mine is None:
        mine, acc = build_batch(root, "B4", workers=workers)
    acc = {k: v for k, v in (acc or {}).items() if k != "drop_rows"}
    (out_dir / "b4_rows.json").write_text(json.dumps(mine), encoding="utf-8")
    by_aw_key = {aw.animal_of(r["side_file"])[0]: r for r in mine if r["full"]}
    missing = [r["animal"] for r in rows_aw if r["animal"] not in by_aw_key]
    pairs = [(a, by_aw_key[a["animal"]]) for a in rows_aw if a["animal"] in by_aw_key]
    rows = [b for _, b in pairs]
    maxdiff = max(abs(a[k] - b[k]) / abs(a[k]) for a, b in pairs for k in ("L", "D", "H", "W", "G"))
    check = {"acme_weight_rows": len(rows_aw), "acme_grouped_full_rows": sum(r["full"] for r in mine),
             "acme_grouped_common_rows": sum(r["common"] for r in mine), "matched": len(rows),
             "acme_weight_rows_missing_here": len(missing), "max_relative_feature_diff": maxdiff,
             "weights_equal": all(a["weight"] == b["weight"] for a, b in pairs),
             "sex_equal": all(a["sex"] == b["sex"] for a, b in pairs)}
    check["passed"] = not missing and maxdiff <= 1e-9 and check["weights_equal"] and check["sex_equal"]
    log(f"[check] {check}")
    if missing:
        raise SystemExit("acme_grouped loader misses rows that acme_weight.build keeps; folds cannot align")

    ref = aw.cv_eval(rows_aw)                                  # reproduces the published 13.3%
    fold, preds = aw_cv_predictions(rows_aw)
    y = np.array([r["weight"] for r in rows], float)
    out = {"batch": "B4", "animals": len(rows), "weight_kg": weight_range(y),
           "folds": "acme_weight.cv_eval folds: numpy default_rng(7).permutation(n) % 10, rows in acme_weight.build order",
           "loader_check": check, "accounting": acc, "acme_weight_cv_eval": ref,
           "evidence_labels": {"input": "R", "truth": "L"}, "label_note": LABEL_NOTE, "methods": {}}
    for name in ("physics", "fitted"):
        m = point_metrics(y, np.exp(preds[name]))
        assert abs(m["mape"] - ref[name]["mape"]) < 1e-4, (name, m["mape"], ref[name]["mape"])
        out["methods"][name] = {**m, "fit_on": "75% of each training fold (acme_weight.py holds 25% for calibration)",
                                "n_params": 2 if name == "physics" else 7}
    n_params = {"constant": 1, "sex": 2, "sticker_only": 2, "physics_b1": 1, "fitted_noscale": 7,
                "common.physics_b1": 1, "common.physics": 2, "common.fitted": 5}
    for name in n_params:
        proto, meth = ("common", name.split(".")[1]) if name.startswith("common.") else ("full", name)
        p = np.zeros(len(rows))
        for f in range(10):
            tr = [rows[i] for i in np.where(fold != f)[0]]
            te_i = np.where(fold == f)[0]
            pred, _ = fit_method(meth, tr, proto)
            p[te_i] = pred([rows[i] for i in te_i])
        out["methods"][name] = {**point_metrics(y, np.exp(p)), "fit_on": "full training fold", "n_params": n_params[name]}
    # grouped inside B4: leave one sub-batch out (b4-1..b4-4) beside random 10-fold, same conformal pipeline
    per, pooled = leave_group_out(rows, "group", "full")
    yk, perk = kfold(rows, "full")
    ek = evaluate(yk, perk)
    out["b4_leave_one_subbatch_out"] = {
        "per_group": {g: {"n_test": v["n_test"], "weight_kg": v["weight_kg"], "headline": headline(v["metrics"])}
                      for g, v in per.items()},
        "pooled": {"weight_kg": pooled["weight_kg"], "headline": headline(pooled["metrics"]), "metrics": pooled["metrics"]}}
    out["b4_random10_same_pipeline"] = {"headline": headline(ek), "metrics": ek}
    out["subbatch_id_note"] = ("Inside sub-batches b4-1, b4-3 and b4-4 the animal id follows weight order "
                               "(Spearman 0.99 to 1.00); b4-2 is unordered. Ids shared across sub-batches belong to "
                               "different animals: sex agrees at the chance rate (0.76 to 0.79). Ids are never a model input.")
    out["seconds"] = round(time.time() - t0, 1)
    (out_dir / "b4_nulls.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(json.dumps({k: {kk: v[kk] for kk in ("mape", "nrmse", "mae_kg", "bias_pct")} for k, v in out["methods"].items()}, indent=1))
    log("[b4 random10, conformal pipeline]", json.dumps(out["b4_random10_same_pipeline"]["headline"], indent=1))
    log("[b4 leave one sub-batch out]", json.dumps(out["b4_leave_one_subbatch_out"]["pooled"]["headline"], indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["nulls", "grouped"])
    ap.add_argument("--root", default=None, help="dataset root holding Vector/ and Pixel/ (grouped)")
    ap.add_argument("--out", default=str(ROOT / "eval" / "out" / "acme_grouped"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.mode == "nulls":
        return run_nulls(Path(a.out), a.workers)
    root = Path(a.root) if a.root else find_root()
    return run_grouped(root, Path(a.out), a.workers)


if __name__ == "__main__":
    raise SystemExit(main())
