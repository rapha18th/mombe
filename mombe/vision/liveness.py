"""Liveness: is a live animal in front of the phone?

The field threats are a printed muzzle photo or a phone screen showing one, held
up to claim another animal's identity, and an enrolled image file sent again.
Five cues, all OpenCV 5, each a one-sided score where higher means more live:

a. plane vs parallax. A capture is a burst of two or more photos taken from
   slightly different places. ALIKED keypoints are matched across the burst
   (mutual nearest neighbours), then a homography and a fundamental matrix are
   fitted with USAC_MAGSAC. GRIC (Torr) scores both models on the same matches.
   A print or a screen is a plane, so the homography explains it; a live nose
   has depth, so the fundamental matrix wins. Score: GRIC(H) - GRIC(F) per match.
b. too-perfect match. The probe is matched to the enrolled images of its two
   closest animals; the enrolled images that line up best keypoint for
   keypoint go on to a pixel test. ECC (cv2.findTransformECC) refines the
   homography and the enrolled image is warped onto the probe. Two exposures
   of one nose differ in sensor noise, wetness and small motion, so their
   correlation stays below one; a resent file, even cropped, rescaled and
   re-encoded, carries the same pixels. Score: log10(1 - correlation) for the
   closest enrolled image.
c. screen grid energy. A camera sampling a screen's RGB subpixel grid aliases
   it into moire: narrow peaks in the spectrum. Welch-averaged cv2.dft power,
   minus its radial median, gives the peak height. The 8 px JPEG block grid of
   the original file is masked out. Score: minus the peak.
d. specular glint layout. A wet nose throws small mirror glints from bead tops
   all over its surface, bright enough to reach the top of the sensor. Paper
   holds highlights below its white, so a print loses them. Score: how many
   cells of a 4 x 4 grid over the nose hold a glint.
e. tonal range. Ink prints no darker than a few percent reflectance and paper
   white stays far below a wet glint, so a print spans few stops of light. A
   live nose holds deep nostril shadow and bright wet highlights at once.
   Score: stops between the 0.5th and 99.5th luminance percentiles over the
   middle of the frame, in linear light, best frame of the burst.

Fusion. Each cue is compared with the same cue on live bursts of the dev
identities: p = the fraction of live bursts that scored as low or lower (mid
rank for ties). The liveness score is either the product of the cue p values
(Fisher's combination: several weak signs add up) or the smallest p (one clear
sign decides). The rule is picked on dev identities and stored with the
calibration. The capture is live when the score exceeds the threshold.
Calibration and threshold use live dev bursts only; the attacks are synthetic
and only measure error rates (eval/liveness_eval.py).
A burst without enough matches to test depth has p = 0 for parallax: the
officer is asked for two photos taken a hand's width apart.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

CALIBRATION = Path(__file__).with_name("liveness_calibration.json")
CUES = ("parallax", "replay", "moire", "glint", "range")
WORK_SIDE = 640                   # same working size as the identity features

# -- cue a: plane vs parallax ---------------------------------------------------
MIN_MATCHES = 40                  # matches needed before either model is believable
F_THRESH_PX = 1.5                 # MAGSAC thresholds at the working size
H_THRESH_PX = 2.0
SIGMA_FLOOR_PX = 0.5              # keypoint noise floor at the working size

# -- cue b: too-perfect match -------------------------------------------------
REPLAY_ANIMALS = 2                # shortlisted animals checked
REPLAY_IMAGES = 8                 # enrolled images checked per animal, closest embedding first
REPLAY_MIN_INLIERS = 30
NO_TWIN = 9.0                     # twin distance when no enrolled image lines up at all
PIXEL_SIDE = 1024                 # long side for the pixel test
PIXEL_CANDIDATES = 2              # closest keypoint twins that get the pixel test
PIXEL_FLOOR = 1e-6                # 1 - correlation of a byte-identical file

# -- cue c: screen grid energy --------------------------------------------------
MOIRE_SIDE = 1024
MOIRE_TILE = 128
MOIRE_RMIN, MOIRE_RMAX = 6, 60    # annulus in frequency bins of the tile spectrum

# -- cue d: glints ---------------------------------------------------------------
GLINT_V = 245                     # at the top of the sensor
GLINT_TOPHAT = 25                 # brighter than the surrounding beads
GLINT_MAX_FRAC = 0.002            # a glint is small; washout is the quality gate's business
GLINT_GRID = 4

# -- cue e: tonal range ------------------------------------------------------------
RANGE_PCT = (0.5, 99.5)           # percentiles of linear luminance
RANGE_EPS = 1e-4                  # linear-light floor, about 13 stops below white


@dataclass
class LivenessResult:
    live: bool
    score: float                  # fused tail probability (0..1, higher is more live)
    threshold: float
    failed_cue: str | None        # the most suspicious cue when the check fails
    cues: dict = field(default_factory=dict)
    frames: int = 0
    method: str = "plane vs parallax (GRIC), too-perfect match (ECC), DFT moire, glint layout, tonal range"

    def to_dict(self) -> dict:
        return asdict(self)


# -- helpers ----------------------------------------------------------------------
def _work(bgr: np.ndarray, side: int = WORK_SIDE) -> np.ndarray:
    h, w = bgr.shape[:2]
    s = side / max(h, w)
    return cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def match_mnn(da: np.ndarray, db: np.ndarray, ratio: float = 0.9) -> np.ndarray:
    """Mutual nearest neighbours with a ratio test. Returns K x 2 index pairs."""
    if len(da) < 2 or len(db) < 2:
        return np.zeros((0, 2), int)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    ab = bf.knnMatch(da, db, k=2)
    back = {m.queryIdx: m.trainIdx for m in bf.match(db, da)}
    out = [(m[0].queryIdx, m[0].trainIdx) for m in ab
           if len(m) == 2 and m[0].distance < ratio * m[1].distance and back.get(m[0].trainIdx) == m[0].queryIdx]
    return np.array(out, int).reshape(-1, 2)


def _homography_err2(H: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Symmetric transfer error, halved to the geometric scale (2 residual dof)."""
    pa = cv2.perspectiveTransform(a.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
    pb = cv2.perspectiveTransform(b.reshape(-1, 1, 2).astype(np.float64), np.linalg.inv(H)).reshape(-1, 2)
    return 0.25 * (((pa - b) ** 2).sum(1) + ((pb - a) ** 2).sum(1))


def _sampson_err2(F: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """First-order geometric error of the epipolar constraint (1 residual dof)."""
    xa = np.hstack([a, np.ones((len(a), 1))])
    xb = np.hstack([b, np.ones((len(b), 1))])
    Fa = xa @ F.T
    Fb = xb @ F
    num = (xb * Fa).sum(1) ** 2
    den = Fa[:, 0] ** 2 + Fa[:, 1] ** 2 + Fb[:, 0] ** 2 + Fb[:, 1] ** 2
    return num / np.maximum(den, 1e-12)


def gric(err2: np.ndarray, sigma: float, d: int, k: int, r: int = 4) -> float:
    """Torr's Geometric Robust Information Criterion; lower is better."""
    n = len(err2)
    rho = np.minimum(err2 / sigma ** 2, 2.0 * (r - d))
    return float(rho.sum() + math.log(r) * d * n + math.log(r * n) * k)


# -- cue a ---------------------------------------------------------------------------
def plane_parallax(ka: np.ndarray, da: np.ndarray, kb: np.ndarray, db: np.ndarray) -> dict:
    """GRIC(H) - GRIC(F) per match for one frame pair."""
    m = match_mnn(da, db)
    out = {"matches": int(len(m)), "available": False}
    if len(m) < MIN_MATCHES:
        return out
    a, b = ka[m[:, 0]].astype(np.float64), kb[m[:, 1]].astype(np.float64)
    F, fm = cv2.findFundamentalMat(a, b, cv2.USAC_MAGSAC, F_THRESH_PX, 0.999, 10000)
    if F is None or fm is None or F.shape != (3, 3):
        return out
    keep = fm.ravel().astype(bool)
    if keep.sum() < MIN_MATCHES:
        return out
    a, b = a[keep], b[keep]                  # matches consistent with some rigid geometry
    H, _ = cv2.findHomography(a, b, cv2.USAC_MAGSAC, H_THRESH_PX)
    if H is None:
        return out
    eh, ef = _homography_err2(H, a, b), _sampson_err2(F, a, b)
    sigma = max(SIGMA_FLOOR_PX, 1.4826 * float(np.median(np.sqrt(ef))))
    n = len(a)
    gh, gf = gric(eh, sigma, d=2, k=8), gric(ef, sigma, d=3, k=7)
    out.update(available=True, inliers=int(n), sigma_px=round(sigma, 3), gric_h=round(gh, 1), gric_f=round(gf, 1),
               score=round((gh - gf) / n, 4), off_plane=round(float((np.sqrt(eh) > 3 * sigma).mean()), 4),
               median_motion_px=round(float(np.median(np.linalg.norm(b - a, axis=1))), 1))
    return out


# -- cue b ---------------------------------------------------------------------------
def twin_stats(kq: np.ndarray, dq: np.ndarray, kg: np.ndarray, dg: np.ndarray) -> dict | None:
    """How closely a probe lines up with one enrolled image (None: no common geometry)."""
    m = match_mnn(dq, dg)
    if len(m) < REPLAY_MIN_INLIERS:
        return None
    a, b = kq[m[:, 0]].astype(np.float64), kg[m[:, 1]].astype(np.float64)
    H, hm = cv2.findHomography(a, b, cv2.USAC_MAGSAC, 3.0)
    if H is None:
        return None
    inl = hm.ravel().astype(bool)
    if inl.sum() < REPLAY_MIN_INLIERS:
        return None
    err = float(np.median(np.sqrt(_homography_err2(H, a[inl], b[inl]))))
    dd = float(np.median(np.linalg.norm(dq[m[inl, 0]] - dg[m[inl, 1]], axis=1)))
    return {"inliers": int(inl.sum()), "median_err_px": round(err, 4), "median_desc_dist": round(dd, 4),
            "twin_distance": round(math.sqrt(err * dd), 4),
            "inlier_frac": round(float(inl.sum()) / max(1, min(len(kq), len(kg))), 4), "H": H}


def _gray(bgr: np.ndarray, side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    s = min(1.0, side / max(h, w))
    if s < 1.0:
        bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def pixel_residual(probe: np.ndarray, probe_wh: tuple, enrolled: np.ndarray, enrolled_wh: tuple,
                   H: np.ndarray) -> dict:
    """1 - correlation once the enrolled image is warped onto the probe.

    H maps probe working-size keypoints onto enrolled working-size keypoints.
    ECC refines it at PIXEL_SIDE; the correlation is taken over the inner part
    of the overlap. A failed ECC keeps the keypoint homography.
    """
    gp, ge = _gray(probe, PIXEL_SIDE), _gray(enrolled, PIXEL_SIDE)
    sp, se = gp.shape[1] / probe_wh[0], ge.shape[1] / enrolled_wh[0]
    Hp = (np.diag([se, se, 1.0]) @ H @ np.diag([1 / sp, 1 / sp, 1.0])).astype(np.float32)
    refined = True
    try:
        _, Hp = cv2.findTransformECC(gp, ge, Hp, cv2.MOTION_HOMOGRAPHY,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6), None, 3)
    except cv2.error:
        refined = False
    size = (gp.shape[1], gp.shape[0])
    warped = cv2.warpPerspective(ge, Hp, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    valid = cv2.warpPerspective(np.ones_like(ge, np.uint8), Hp, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    valid = cv2.erode(valid, np.ones((9, 9), np.uint8)).astype(bool)
    if valid.sum() < 0.1 * valid.size:
        return {"residual": 1.0, "ecc": refined, "overlap": round(float(valid.mean()), 3)}
    a, b = gp[valid], warped[valid]
    a, b = a - a.mean(), b - b.mean()
    cc = float((a * b).sum() / math.sqrt(max(float((a * a).sum() * (b * b).sum()), 1e-12)))
    return {"residual": max(PIXEL_FLOOR, 1.0 - cc), "ecc": refined, "overlap": round(float(valid.mean()), 3)}


def replay_cue(feats: list, gallery, frames: list | None = None, image_of=None) -> dict:
    """Pixel residual against the enrolled images that line up best with the probe.

    A crop shifts the embedding, so every enrolled image of a shortlisted animal
    is checked keypoint for keypoint, the embedding's favourites first. The
    PIXEL_CANDIDATES closest twins go on to the pixel test; `image_of(animal,
    image_id)` returns an enrolled image (None when it is not on file).
    """
    if gallery is None or not getattr(gallery, "features", None):
        return {"available": False}
    twins = {}
    for qi, q in enumerate(feats):
        for animal, _, idxs in gallery.shortlist(q.embedding, k=REPLAY_ANIMALS):
            rest = [i for i, a in enumerate(gallery.animal_ids) if a == animal and i not in idxs]
            for i in (list(idxs) + rest)[:REPLAY_IMAGES]:
                g = gallery.features[i]
                r = twin_stats(q.kpts, q.desc, g.kpts, g.desc)
                key = gallery.image_ids[i]
                if r and (key not in twins or r["twin_distance"] < twins[key]["twin_distance"]):
                    twins[key] = {**r, "frame": qi, "animal_id": animal, "image_id": key, "_i": i}
    if not twins:
        return {"available": True, "score": 0.0, "twin_distance": NO_TWIN, "residual": 1.0}
    ranked = sorted(twins.values(), key=lambda t: t["twin_distance"])[:PIXEL_CANDIDATES]
    best = None
    for t in ranked:
        img = image_of(t["animal_id"], t["image_id"]) if image_of and frames else None
        if img is None:
            continue
        g = gallery.features[t["_i"]]
        px = pixel_residual(frames[t["frame"]], feats[t["frame"]].size, img, g.size, t["H"])
        if best is None or px["residual"] < best["residual"]:
            best = {**t, **px}
    if best is None:                              # no enrolled pixels on file: keypoints alone cannot tell
        t = ranked[0]
        return {"available": False, "twin_distance": t["twin_distance"], "image_id": t["image_id"],
                "reason": "enrolled image not on file"}
    out = {k: v for k, v in best.items() if k not in ("H", "_i")}
    return {"available": True, "score": round(math.log10(best["residual"]), 4), **out}


# -- cue c ---------------------------------------------------------------------------
_HANN: dict[int, np.ndarray] = {}


def moire_cue(bgr: np.ndarray) -> dict:
    """Narrow spectral peaks: Welch-averaged tile spectra against their radial median."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    s = min(1.0, MOIRE_SIDE / max(h, w))
    if s < 1.0:
        g = cv2.resize(g, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    T = MOIRE_TILE if min(g.shape) >= 2 * MOIRE_TILE else max(32, (min(g.shape) // 2) // 16 * 16)
    win = _HANN.get(T)
    if win is None:
        win = _HANN[T] = cv2.createHanningWindow((T, T), cv2.CV_32F)
    h, w = g.shape
    y0, y1, x0, x1 = int(h * 0.1), int(h * 0.9), int(w * 0.1), int(w * 0.9)
    if y1 - y0 < T or x1 - x0 < T:
        y0, y1, x0, x1 = 0, h, 0, w
    power, n = np.zeros((T, T), np.float32), 0
    for y in range(y0, y1 - T + 1, T // 2):
        for x in range(x0, x1 - T + 1, T // 2):
            tile = g[y:y + T, x:x + T]
            spec = cv2.dft((tile - tile.mean()) * win, flags=cv2.DFT_COMPLEX_OUTPUT)
            power += spec[..., 0] ** 2 + spec[..., 1] ** 2
            n += 1
    if n == 0:
        return {"score": 0.0, "peak": 0.0, "tiles": 0}
    logp = np.log(np.fft.fftshift(power / n) + 1e-6)
    yy, xx = np.mgrid[0:T, 0:T]
    rr = np.hypot(yy - T // 2, xx - T // 2)
    rbin = np.round(rr).astype(int)
    bg = np.zeros_like(logp)
    for r in range(rbin.max() + 1):
        sel = rbin == r
        if sel.any():
            bg[sel] = np.median(logp[sel])
    resid = logp - bg
    ann = (rr >= MOIRE_RMIN * T / MOIRE_TILE) & (rr <= MOIRE_RMAX * T / MOIRE_TILE)
    # JPEG's 8 px block grid: harmonics at k/8 cycles per original pixel, folded to the analysis size
    harm = [(((k / (8 * s)) + 0.5) % 1.0 - 0.5) * T for k in range(8)]
    for fy in harm:
        for fx in harm:
            for sy, sx in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                ann &= np.hypot(yy - T // 2 - sy * fy, xx - T // 2 - sx * fx) > 1.5
    if not ann.any():
        return {"score": 0.0, "peak": 0.0, "tiles": n}
    iy, ix = np.unravel_index(np.argmax(np.where(ann, resid, -np.inf)), resid.shape)
    peak = float(resid[iy, ix])
    return {"score": round(-peak, 4), "peak": round(peak, 4), "tiles": n, "tile": T,
            "peak_cycles_per_px": round(float(rr[iy, ix]) / T, 4)}


# -- cue d ---------------------------------------------------------------------------
def glint_mask(bgr_work: np.ndarray) -> np.ndarray:
    """Small specular glints: at the top of the sensor and brighter than the beads around them."""
    v = bgr_work.max(axis=2)
    th = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    cand = ((v >= GLINT_V) & (th >= GLINT_TOPHAT)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    keep = np.zeros(n, bool)
    area = stats[1:, cv2.CC_STAT_AREA]
    keep[1:] = (area >= 2) & (area <= GLINT_MAX_FRAC * v.size)
    return keep[lab].astype(np.uint8)


def glint_cue(frames: list[np.ndarray]) -> dict:
    """Spread of glints over the nose: occupied cells of a grid, best frame of the burst."""
    cells, counts = [], []
    for f in frames:
        m = glint_mask(_work(f))
        counts.append(int(cv2.connectedComponents(m)[0] - 1))
        h, w = m.shape
        occ = 0
        for i in range(GLINT_GRID):
            for j in range(GLINT_GRID):
                occ += bool(m[i * h // GLINT_GRID:(i + 1) * h // GLINT_GRID, j * w // GLINT_GRID:(j + 1) * w // GLINT_GRID].any())
        cells.append(occ)
    return {"available": bool(frames), "score": float(max(cells, default=0)), "cells": cells, "glints": counts}


# -- cue e ---------------------------------------------------------------------------
def range_cue(frames: list[np.ndarray]) -> dict:
    """Stops of light between the darkest and brightest nose pixels; best frame of the burst."""
    stops = []
    for f in frames:
        g = cv2.cvtColor(_work(f), cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        lin = (g[h // 10:h - h // 10, w // 10:w - w // 10].astype(np.float32) / 255.0) ** 2.2
        lo, hi = np.percentile(lin, RANGE_PCT)
        stops.append(round(math.log2((hi + RANGE_EPS) / (lo + RANGE_EPS)), 3))
    return {"available": bool(frames), "score": max(stops, default=0.0), "stops": stops}


# -- all cues and fusion -------------------------------------------------------------
def cue_values(frames: list[np.ndarray], feats: list, gallery=None, image_of=None) -> dict:
    """Raw cue scores for one capture (frames in burst order, ALIKED features per frame).

    `image_of(animal_id, image_id)` loads an enrolled image for the pixel replay test.
    """
    pairs = [plane_parallax(feats[i].kpts, feats[i].desc, feats[i + 1].kpts, feats[i + 1].desc)
             for i in range(len(feats) - 1)]
    avail = [p for p in pairs if p.get("available")]
    par = max(avail, key=lambda p: p["score"]) if avail else \
        {"available": False, "matches": max((p["matches"] for p in pairs), default=0)}
    moire = min((moire_cue(f) for f in frames), key=lambda m: m["score"])
    return {"frames": len(frames), "parallax": par, "replay": replay_cue(feats, gallery, frames, image_of),
            "moire": moire, "glint": glint_cue(frames), "range": range_cue(frames)}


_CALIB: dict = {}


def load_calibration(path: Path = CALIBRATION) -> dict | None:
    """Read once per process; a new calibration takes effect on restart."""
    if path not in _CALIB and path.exists():
        _CALIB[path] = json.loads(path.read_text())
    return _CALIB.get(path)


def tail_p(value: float, live: list[float]) -> float:
    """Fraction of live dev bursts that scored at or below `value` (mid-rank for ties)."""
    arr = np.asarray(live, float)
    below, tie = float((arr < value).sum()), float((arr == value).sum())
    return (below + 0.5 * tie + 0.5) / (len(arr) + 1)


def cue_p(values: dict, calib: dict) -> dict:
    ps = {}
    for c in calib["cues"]:
        v = values.get(c, {})
        if c == "parallax" and not v.get("available"):
            ps[c] = 0.0                       # no view change, no proof of depth
        elif not v.get("available", True) or "score" not in v:
            ps[c] = 1.0                       # cue silent: no evidence either way
        else:
            ps[c] = tail_p(v["score"], calib["live"][c])
    return ps


def fuse(values: dict, calib: dict) -> tuple[float, str | None, dict]:
    """Fused score (Fisher: product of p; or the minimum p), the most suspicious cue, per-cue p."""
    ps = cue_p(values, calib)
    if not ps:
        return 1.0, None, ps
    worst = min(ps, key=ps.get)
    score = float(np.prod(list(ps.values()))) if calib.get("fusion", "fisher") == "fisher" else ps[worst]
    return score, worst, ps


def assess(frames: list[np.ndarray], feats: list, gallery=None, calib: dict | None = None,
           silent: tuple[str, ...] = (), image_of=None) -> LivenessResult:
    """Liveness of one capture. Cues in `silent` count as no evidence either way:
    a carcass nose dries, so an insurance claim sets silent=("glint", "range")."""
    calib = calib or load_calibration()
    if calib is None:
        raise RuntimeError("liveness calibration missing: run eval/liveness_eval.py")
    vals = cue_values(frames, feats, gallery, image_of)
    for c in silent:
        vals[c] = {**vals.get(c, {}), "available": False, "silenced": True}
    score, worst, ps = fuse(vals, calib)
    live = score > calib["threshold"] and len(frames) >= 2
    cues = {c: {**vals.get(c, {}), "p": round(ps.get(c, 1.0), 4)} for c in CUES}
    return LivenessResult(live=bool(live), score=float(f"{score:.4g}"), threshold=calib["threshold"],
                          failed_cue=None if live else (worst if len(frames) >= 2 else "parallax"),
                          cues=cues, frames=len(frames))
