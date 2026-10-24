"""Mombe's front end on real handheld cattle video: SideViewCows2026 (Zenodo 21605650, CC BY 4.0).

Each barn recording is a handheld 1920x1080 pass along a cow's right flank with
a binary mask on every frame. Per recording this module runs the production
front end, unchanged, and measures it:

  keyframes     the provider's rule: the sharpest frame in each of MAX_FRAMES equal
                slices (mombe.body.provider), checked against the provider itself
  quality gate  mombe.vision.quality.assess on the full keyframe and on the cow's box
  matching      ALIKED + LightGlue + MAGSAC essential matrix (Reconstructor.match) for
                every pair within MATCH_WINDOW; inliers in total and on the cow mask
  registration  Reconstructor.sfm: incremental PnP registration with gated bundle
                adjustment; registered fraction and reprojection error
  stereo pairs  triangulation angle of every registered pair, scale free, to test the
                fixed PAIR_STEP rule in mombe.body.dense against a 5 to 10 degree rule

There is no board in the barn, so Reconstructor.run() cannot give metric scale.
The module calls run()'s steps in run()'s order (board_obs, calibrate, features,
matching, focal_from_pairs, sfm) and stops before scale, ground and to_metric.
The camera never circles the animal, so the loop-closure pairs are switched off
(recon.LOOP = False). Everything reported is scale free except matches per
square decimetre of hide, which uses an approximate pixel scale (see hide_scale).

The data stays on Zenodo: RemoteZip reads barn.zip with HTTP range requests, one
request per run of contiguous members, so a shard downloads only its own cows.
"""

from __future__ import annotations

import io
import json
import math
import struct
import sys
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

RECORD = 21605650
BASE = f"https://zenodo.org/api/records/{RECORD}/files"
BARN_URL = f"{BASE}/barn.zip/content"
MANIFEST_URL = f"{BASE}/manifest.csv/content"
UA = "mombe-sideview-eval/1.0 (research; CC BY 4.0 dataset)"

# ---- pre-registered constants (fixed before any result was seen) -------------------------
# Approximate pixel scale for "matches per square decimetre of hide". The visible right
# flank of the mask spans nose to tail at best, and handheld close views cut it off, so
# the scale comes from the barrel instead: in each mask column the vertical run from back
# line to belly is the body depth. The lower-quartile run over the middle third of the
# mask's columns (leg columns run on to the hoof and sit higher) is taken as the chest
# depth of an adult Holstein Friesian cow, 0.76 m. Published herd means: 76.1 and 77.4 cm
# (Dutch and German lines, Scientific Papers Animal Science 2019 issue 2 art. 2), 69.5 cm
# (a Ukrainian herd, NUBiP). Treat the scale as +-15%, so a density as roughly +-30%.
# Columns that touch the image top or bottom are skipped; a frame with fewer than 20
# usable columns takes the recording's median scale.
BARREL_DEPTH_M = 0.76
# Coat proxy, defined on eroded mask pixels that are not privacy magenta, HSV from OpenCV:
#   dark   V < 70                  (black hide in barn light)
#   light  V > 150 and S < 60      (white hide)
#   red    5 <= H <= 25, S > 90, 40 < V < 220   (Fleckvieh red-brown)
# Class order: red-pied if red >= 0.20; else mostly-black if dark >= 0.60; else
# mostly-white if light >= 0.55; else patched. Texture: mean Sobel magnitude of the grey
# image at WORK_W inside the eroded mask; groups are recording tertiles.
COAT_RULE = {"dark_v": 70, "light_v": 150, "light_s": 60, "red_h": (5, 25), "red_s": 90, "red_v": (40, 220),
             "red_min": 0.20, "dark_min": 0.60, "light_min": 0.55, "erode_px": 15}
MAGENTA_TOL = 60          # L1 distance to (255, 0, 255) in BGR that counts as privacy blanking
ANGLE_BAND = (5.0, 10.0)  # candidate stereo rule under test, degrees
ANGLE_MAX_GAP = 8         # how far ahead (registered keyframes) the angle rule may look
MIN_SHARED = 20           # triangulated points two frames must share for a point-based angle
# Added after shard 0 (v2): Reconstructor.sfm crashed with an IndexError on 2 of 15 recordings.
# The production path is still measured as shipped (crash = 0 registered). The eval adds an
# instrumented _bundle that names the degenerate initial pair, and an eval-only retry that
# drops that pair from the match graph and lets sfm choose again, at most SFM_RETRIES times.
EVAL_VERSION = 2
SFM_RETRIES = 3


# ---- HTTP range access to the Zenodo zip -------------------------------------------------
def http_get(url: str, rng: tuple[int, int] | None = None, tries: int = 8, timeout: float = 120) -> bytes:
    """GET with a byte range, retrying 429 and 5xx with the server's retry-after."""
    delay = 5.0
    for k in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        if rng is not None:
            req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
                if rng is not None and r.status != 206:
                    raise IOError(f"server ignored the range request (HTTP {r.status})")
                if rng is not None and len(data) != rng[1] - rng[0] + 1:
                    raise IOError(f"short read {len(data)} of {rng[1] - rng[0] + 1}")
                return data
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or k == tries - 1:
                raise
            wait = float(e.headers.get("retry-after") or delay)
            print(f"  http {e.code}, retry in {wait:.0f}s", flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, IOError, ConnectionError) as e:
            if k == tries - 1:
                raise
            print(f"  {type(e).__name__}: {e}; retry in {delay:.0f}s", flush=True)
            time.sleep(delay)
        delay = min(delay * 2, 300)
    raise IOError("unreachable")


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file; reads become range requests (zipfile needs few)."""

    def __init__(self, url: str, size: int | None = None):
        self.url, self.pos = url, 0
        if size is None:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                size = int(r.headers["Content-Length"])
        self.size = size
        self.requests = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        data = http_get(self.url, (self.pos, self.pos + n - 1))
        self.requests += 1
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


class RemoteZip:
    """Central directory over HTTP once; members fetched in contiguous spans."""

    def __init__(self, url: str = BARN_URL):
        self.url = url
        self.f = HttpRangeFile(url)
        self.z = zipfile.ZipFile(self.f)
        self.info = {i.filename: i for i in self.z.infolist()}
        offs = sorted(i.header_offset for i in self.info.values())
        cd_start = self.z.start_dir
        self._next = {o: (offs[k + 1] if k + 1 < len(offs) else cd_start) for k, o in enumerate(offs)}
        self.bytes_fetched = 0
        self.span_requests = 0

    def _get_parallel(self, a: int, b: int, chunk: int = 16 << 20, workers: int = 6) -> bytes:
        """One span as parallel range requests: a single Zenodo connection ran at about 0.3 MB/s on Kaggle."""
        from concurrent.futures import ThreadPoolExecutor
        parts = [(x, min(x + chunk, b + 1) - 1) for x in range(a, b + 1, chunk)]
        with ThreadPoolExecutor(min(workers, len(parts))) as ex:
            bufs = list(ex.map(lambda r: http_get(self.url, r), parts))
        self.span_requests += len(parts)
        buf = b"".join(bufs)
        self.bytes_fetched += len(buf)
        return buf

    def fetch(self, names: list[str], gap: int = 4 << 20, max_span: int = 400 << 20) -> dict[str, bytes]:
        infos = sorted((self.info[n] for n in names), key=lambda i: i.header_offset)
        spans, cur = [], []
        for i in infos:
            if cur and (i.header_offset - self._next[cur[-1].header_offset] > gap
                        or self._next[i.header_offset] - cur[0].header_offset > max_span):
                spans.append(cur)
                cur = []
            cur.append(i)
        if cur:
            spans.append(cur)
        out = {}
        for s in spans:
            a, b = s[0].header_offset, self._next[s[-1].header_offset] - 1
            buf = self._get_parallel(a, b)
            for i in s:
                o = i.header_offset - a
                sig, = struct.unpack_from("<I", buf, o)
                if sig != 0x04034B50:
                    raise IOError(f"bad local header for {i.filename}")
                nl, el = struct.unpack_from("<HH", buf, o + 26)
                d = buf[o + 30 + nl + el:o + 30 + nl + el + i.compress_size]
                if i.compress_type == zipfile.ZIP_DEFLATED:
                    d = zlib.decompressobj(-15).decompress(d)
                elif i.compress_type != zipfile.ZIP_STORED:
                    raise IOError(f"unsupported compression {i.compress_type}")
                if zlib.crc32(d) & 0xFFFFFFFF != i.CRC:
                    raise IOError(f"CRC mismatch for {i.filename}")
                out[i.filename] = d
        return out


# ---- manifest ------------------------------------------------------------------------------
def recordings(manifest_csv: str) -> list[dict]:
    """Barn recordings: one per (individual, video start time), frames in order."""
    import csv
    recs: dict[tuple, list] = {}
    for r in csv.DictReader(io.StringIO(manifest_csv)):
        if r["subset"] != "barn":
            continue
        recs.setdefault((r["individual_id"], int(r["time_offset_s"])), []).append(
            (int(r["frame_no"]), r["image_path"], r["mask_path"]))
    out = []
    for (ind, t0), rows in sorted(recs.items(), key=lambda kv: (int(kv[0][0]), kv[0][1])):
        rows.sort()
        out.append({"id": f"{ind}_{t0}", "individual": ind, "t0": t0, "frames": [r[0] for r in rows],
                    "images": [r[1] for r in rows], "masks": [r[2] for r in rows]})
    return out


# ---- keyframes: the provider's rule ----------------------------------------------------------
def select_keyframes(paths: list[str], max_frames: int) -> list[int]:
    """Indices that mombe.body.provider.WalkAroundBody._frames keeps (same code path, indices out)."""
    if len(paths) <= max_frames:
        return list(range(len(paths)))
    picks, sharp = [], []
    for chunk in np.array_split(np.arange(len(paths)), max_frames):
        best, bi = -1.0, chunk[0]
        for i in chunk:
            g = cv2.imread(paths[i], cv2.IMREAD_REDUCED_GRAYSCALE_4)
            v = cv2.Laplacian(g, cv2.CV_32F).var() if g is not None else -1
            if v > best:
                best, bi = v, i
        picks.append(int(bi))
    return picks


def sharpness_all(paths: list[str]) -> np.ndarray:
    out = []
    for p in paths:
        g = cv2.imread(p, cv2.IMREAD_REDUCED_GRAYSCALE_4)
        out.append(float(cv2.Laplacian(g, cv2.CV_32F).var()) if g is not None else -1.0)
    return np.array(out)


# ---- masks, hide scale, coat -----------------------------------------------------------------
def magenta(bgr: np.ndarray) -> np.ndarray:
    d = np.abs(bgr.astype(np.int16) - np.array([255, 0, 255], np.int16)).sum(2)
    return d < MAGENTA_TOL


def hide_scale(mask: np.ndarray) -> dict:
    """Pixels per metre from the barrel depth (BARREL_DEPTH_M); see the constant's note."""
    h, w = mask.shape
    cols = np.where(mask.any(0))[0]
    if len(cols) < 30:
        return {"px_per_m": None, "reason": "mask too small"}
    lo, hi = cols[0] + (cols[-1] - cols[0]) // 3, cols[0] + 2 * (cols[-1] - cols[0]) // 3
    runs = []
    for c in range(lo, hi + 1):
        ys = np.where(mask[:, c])[0]
        if len(ys) == 0 or ys[0] == 0 or ys[-1] == h - 1:
            continue
        # longest vertical run in this column (the barrel; legs below are separate runs)
        br = np.where(np.diff(ys) > 1)[0]
        starts = np.r_[ys[0], ys[br + 1]]
        ends = np.r_[ys[br], ys[-1]]
        runs.append(int((ends - starts).max()) + 1)
    if len(runs) < 20:
        return {"px_per_m": None, "reason": "barrel columns touch the frame edge"}
    # lower quartile: columns that pass through a leg run on to the hoof and sit in the upper quartiles
    depth_px = float(np.quantile(runs, 0.25))
    return {"px_per_m": depth_px / BARREL_DEPTH_M, "barrel_px": round(depth_px, 1), "columns": len(runs)}


def coat_stats(bgr: np.ndarray, mask: np.ndarray, work_w: int) -> dict | None:
    r = COAT_RULE
    k = np.ones((2 * r["erode_px"] + 1,) * 2, np.uint8)
    m = cv2.erode(mask.astype(np.uint8), k) > 0
    m &= ~magenta(bgr)
    if m.sum() < 5000:
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0][m].astype(int), hsv[..., 1][m].astype(int), hsv[..., 2][m].astype(int)
    dark = float((V < r["dark_v"]).mean())
    light = float(((V > r["light_v"]) & (S < r["light_s"])).mean())
    red = float(((H >= r["red_h"][0]) & (H <= r["red_h"][1]) & (S > r["red_s"])
                 & (V > r["red_v"][0]) & (V < r["red_v"][1])).mean())
    s = work_w / bgr.shape[1]
    g = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ms = cv2.resize(m.astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    tex = float(np.sqrt(gx * gx + gy * gy)[ms].mean()) if ms.sum() else float("nan")
    return {"dark": dark, "light": light, "red": red, "texture": tex, "mean_v": float(V.mean())}


def coat_class(dark: float, light: float, red: float) -> str:
    r = COAT_RULE
    if red >= r["red_min"]:
        return "red-pied"
    if dark >= r["dark_min"]:
        return "mostly-black"
    if light >= r["light_min"]:
        return "mostly-white"
    return "patched"


# ---- geometry helpers --------------------------------------------------------------------------
def pair_angles(rec, tri_ids: list[int]) -> dict[tuple[int, int], tuple[float, int]]:
    """Median triangulation angle (degrees) and shared point count for every registered pair."""
    reg = sorted(rec.R)
    C = {f: rec.centre(f) for f in reg}
    pts_of: dict[int, dict[int, np.ndarray]] = {f: {} for f in reg}
    for n, tid in enumerate(tri_ids):
        X = rec.points[n]
        for f, _k in rec.tracks[tid]:
            if f in C:
                pts_of[f][n] = X
    out = {}
    for a in range(len(reg)):
        for b in range(a + 1, min(len(reg), a + 1 + ANGLE_MAX_GAP)):
            fa, fb = reg[a], reg[b]
            shared = set(pts_of[fa]) & set(pts_of[fb])
            if len(shared) < MIN_SHARED:
                continue
            X = rec.points[sorted(shared)]
            ra, rb = X - C[fa], X - C[fb]
            cos = np.sum(ra * rb, 1) / (np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1) + 1e-12)
            out[(fa, fb)] = (float(np.degrees(np.median(np.arccos(np.clip(cos, -1, 1))))), len(shared))
    return out


def reprojection(rec) -> dict:
    """Errors of every observation of every kept 3D point in a registered frame (px)."""
    errs = []
    for n, tid in enumerate(rec.point_track):
        Xh = np.append(rec.points[n], 1)
        for f, k in rec.tracks[tid]:
            if f in rec.R:
                xh = rec.P(f) @ Xh
                if xh[2] > 0:
                    errs.append(float(np.linalg.norm(xh[:2] / xh[2] - rec.feats_kpts[f][k])))
    e = np.array(errs)
    if not len(e):
        return {"obs": 0}
    gate = 4.0   # 2 x REPROJ_PX, the bundle's own keep threshold
    return {"obs": int(len(e)), "median_all_px": round(float(np.median(e)), 3),
            "median_kept_px": round(float(np.median(e[e <= gate])), 3) if (e <= gate).any() else None,
            "frac_within_4px": round(float((e <= gate).mean()), 4)}


def camera_path(rec) -> dict:
    """Scale-free path summary: total heading change and path length over median depth."""
    reg = sorted(rec.R)
    if len(reg) < 2 or not len(rec.points):
        return {}
    C = np.array([rec.centre(f) for f in reg])
    axes = np.array([rec.R[f][2] for f in reg])          # optical axis in world
    step = np.linalg.norm(np.diff(C, axis=0), axis=1)
    depth = float(np.median([np.median((rec.R[f] @ rec.points.T + rec.t[f][:, None])[2]) for f in reg]))
    turn = float(np.degrees(np.arccos(np.clip(axes[0] @ axes[-1], -1, 1))))
    span = float(np.linalg.norm(C[-1] - C[0]))
    return {"heading_change_deg": round(turn, 1), "path_over_depth": round(float(step.sum()) / depth, 3),
            "span_over_depth": round(span / depth, 3), "median_depth_units": depth}


# ---- one recording -----------------------------------------------------------------------------
@dataclass
class Shard:
    """What a kernel shard keeps besides per-recording rows: failure examples for the gallery."""
    gallery: list = field(default_factory=list)


def draw_pair(a: np.ndarray, b: np.ndarray, ma: np.ndarray, mb: np.ndarray, pa: np.ndarray, pb: np.ndarray,
              caption: str, width: int = 1280) -> bytes:
    """Side-by-side pair: mask outline in yellow, verified matches in green; small JPEG."""
    h, w = a.shape[:2]
    s = (width / 2) / w
    def prep(img, m):
        im = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        mm = cv2.resize(m.astype(np.uint8), (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cs, -1, (0, 220, 255), 2)
        return im
    A, B = prep(a, ma), prep(b, mb)
    canvas = np.hstack([A, B])
    for p, q in zip(pa * s, pb * s):
        cv2.line(canvas, (int(p[0]), int(p[1])), (int(q[0] + A.shape[1]), int(q[1])), (60, 200, 60), 1, cv2.LINE_AA)
    bar = np.full((34, canvas.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, caption[:150], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    ok, enc = cv2.imencode(".jpg", np.vstack([canvas, bar]), [cv2.IMWRITE_JPEG_QUALITY, 78])
    return enc.tobytes()


def match_detail(recon, a, b, K) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Reconstructor.match line for line, also returning the raw LightGlue matches and any matcher error."""
    empty = np.zeros((0, 2), int)
    if len(a.kpts) < 16 or len(b.kpts) < 16:
        return empty, empty, None
    try:
        recon.lightglue.setPairInfo(a.kpts_work, b.kpts_work, a.work_size, b.work_size)
        m = recon.lightglue.match(a.desc, b.desc)
    except cv2.error as e:
        return empty, empty, str(e).strip().splitlines()[-1][:200]
    idx = np.array([[x.queryIdx, x.trainIdx] for x in m]).reshape(-1, 2).astype(int)
    if len(m) < 16:
        return idx, empty, None
    E, mask = cv2.findEssentialMat(a.kpts[idx[:, 0]], b.kpts[idx[:, 1]], K, cv2.USAC_MAGSAC, 0.999, 1.0)
    if E is None or mask is None:
        return idx, empty, None
    return idx, idx[mask.ravel().astype(bool)], None


EDGE_DROP_PX = 12   # hide-only variant: keypoints this close to the silhouette sit on the outline, not the hide


def hide_only_features(recon, frames, masks, K, dist):
    """Features restricted to the hide: background filled with the cow's mean colour, outline keypoints dropped."""
    from mombe.body.recon import FrameFeatures
    masked = []
    for f, m in zip(frames, masks):
        g = f.copy()
        if m is not None and m.any():
            g[~m] = np.uint8(f[m].mean(0))
        masked.append(g)
    feats = recon.features(masked, K, dist)
    out = []
    for ft, m in zip(feats, masks):
        if m is None or not len(ft.kpts):
            out.append(ft)
            continue
        dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
        h, w = m.shape
        xy = np.clip(np.round(ft.kpts).astype(int), [0, 0], [w - 1, h - 1])
        keep = dt[xy[:, 1], xy[:, 0]] > EDGE_DROP_PX
        out.append(FrameFeatures(ft.kpts[keep], ft.kpts_work[keep], ft.desc[keep], ft.work_size))
    return out


class DegenerateInit(RuntimeError):
    """sfm's first bundle had nothing to adjust: the chosen initial pair kept no usable 3D point."""

    def __init__(self, msg: str, diag: dict):
        super().__init__(msg)
        self.diag = diag


def init_diag(rec, matches: dict, n_tri: int) -> dict:
    """Why the initial pair failed: its matches, pixel parallax, and the ray angles its pose gives."""
    reg = sorted(rec.R)
    d: dict = {"pair": reg, "points_triangulated": int(n_tri)}
    if len(reg) != 2:
        return d
    i, j = reg
    m = matches.get((i, j))
    if m is None or not len(m):
        return d
    p1 = rec.feats_kpts[i][m[:, 0]].astype(np.float64)
    p2 = rec.feats_kpts[j][m[:, 1]].astype(np.float64)
    X = cv2.triangulatePoints(rec.P(i), rec.P(j), p1.T, p2.T)
    X = (X[:3] / np.where(np.abs(X[3]) < 1e-12, 1e-12, X[3])).T
    zi = (rec.R[i] @ X.T + rec.t[i][:, None])[2]
    zj = (rec.R[j] @ X.T + rec.t[j][:, None])[2]
    ra, rb = X - rec.centre(i), X - rec.centre(j)
    cos = np.sum(ra * rb, 1) / (np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1) + 1e-12)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    front = (zi > 0) & (zj > 0)
    # rotation between the two views versus the pixel motion it explains on its own
    rot = float(np.degrees(np.linalg.norm(cv2.Rodrigues(rec.R[j] @ rec.R[i].T)[0])))
    d.update({"gap": int(j - i), "matches": int(len(m)),
              "px_parallax_median": round(float(np.median(np.linalg.norm(p1 - p2, axis=1))), 1),
              "rotation_deg": round(rot, 2),
              "front_frac": round(float(front.mean()), 4),
              "angle_deg_median_front": round(float(np.median(ang[front])), 3) if front.any() else None,
              "frac_angle_ge_1_5_front": round(float((ang[front] >= 1.5).mean()), 4) if front.any() else None})
    return d


def run_sfm(recon, feats, K, dist, size, matches, seed):
    """recon.sfm with an instrumented _bundle: identical results, a named error when the init is degenerate."""
    orig = recon._bundle

    def bundle(rec, tri, fix, focal=False):
        if "init_pair" not in rec.stats:
            rec.stats["init_pair"] = [int(f) for f in sorted(rec.R)]
        if not tri:
            d = init_diag(rec, matches, 0)
            raise DegenerateInit(f"initial pair {d['pair']} kept 0 points", d)
        try:
            return orig(rec, tri, fix=fix, focal=focal)
        except IndexError as e:      # every observation gated out: the same empty bundle
            d = init_diag(rec, matches, len(tri))
            raise DegenerateInit(f"bundle had no observations ({e}); pair {d['pair']}", d) from e

    recon._bundle = bundle
    try:
        cv2.setRNGSeed(seed)
        return recon.sfm(feats, K, dist, size, matches)
    finally:
        del recon._bundle


def sfm_report(rec, n, feats, masks, on_mask) -> dict:
    reg = sorted(rec.R)
    on3d = []   # where the kept 3D points sit: on the cow or on the barn (first observation's mask)
    for tid in rec.point_track:
        f, k = next(((f, k) for f, k in rec.tracks[tid] if f in rec.R), (None, None))
        if f is not None and masks[f] is not None:
            on3d.append(bool(on_mask(feats[f], np.array([k]), masks[f])[0]))
    return {"ok": True, "registered": len(reg), "registered_frac": round(len(reg) / n, 4),
            "unregistered": [f for f in range(n) if f not in rec.R],
            "pnp_failed": rec.stats.get("unregistered", []),
            "points": int(len(rec.points)), "points_on_cow_frac": round(float(np.mean(on3d)), 4) if on3d else None,
            "tracks": rec.stats.get("tracks"),
            "ba_rms_px": rec.stats.get("ba_rms_px"), "ba_obs": rec.stats.get("ba_obs"),
            "final_focal_px": round(float(rec.K[0, 0]), 1), "init_pair": rec.stats.get("init_pair"),
            "reprojection": reprojection(rec), "path": camera_path(rec)}


def _frontend(variant: str, frames, masks, recon_mod, recon, seed, size, scales, area_px, rec_px_per_m, qf, mag,
              shard, rec_id, threads) -> tuple[dict, dict]:
    """One pass of run()'s steps up to sfm, with every measurement; returns (report, internals)."""
    n = len(frames)
    out: dict = {}
    cv2.setNumThreads(1)
    cv2.setRNGSeed(seed)
    np.random.seed(seed)
    obs = recon.board_obs(frames) if variant == "as_shipped" else {}
    cal = recon.calibrate(obs, size, None)
    K, dist = cal["K"], cal["dist"]
    cv2.setNumThreads(threads)
    t1 = time.time()
    feats = recon.features(frames, K, dist) if variant == "as_shipped" else hide_only_features(recon, frames, masks, K, dist)
    t2 = time.time()
    raw, allm, errs = {}, {}, {}
    for p in recon.pairs(n):
        raw[p], allm[p], e = match_detail(recon, feats[p[0]], feats[p[1]], K)
        if e:
            errs[p] = e
    t3 = time.time()
    # the copy must equal the shipped matcher: check the first pair without a matcher error
    p0 = next((p for p in recon.pairs(n) if p not in errs), None)
    same = None if p0 is None else bool(np.array_equal(recon.match(feats[p0[0]], feats[p0[1]], K), allm[p0]))
    cv2.setNumThreads(1)
    matches = {p: m for p, m in allm.items() if len(m) >= recon_mod.MIN_PAIR_INLIERS}
    foc = recon.focal_from_pairs(feats, matches, size)
    board_ok = cal["method"] == "ChArUco" and cal["rms"] is not None and cal["rms"] < 1.0 and cal["frames"] >= 5
    if foc["f"] and not board_ok:
        K = K.copy()
        K[0, 0] = K[1, 1] = foc["f"]
    if variant == "as_shipped":
        out["board_frames"] = len(obs)
    out["keyframes"] = n
    out["match_copy_equal"] = same
    out["matcher_errors"] = {"pairs": len(errs), "messages": sorted(set(errs.values()))[:3],
                             "where": [f"{i}-{j}" for i, j in sorted(errs)][:20]}
    out["focal"] = {"prior_px": round(float(cal["K"][0, 0]), 1), "pairs_px": foc["f"] and round(foc["f"], 1),
                    "pairs": foc["pairs"], "source": "pairs" if foc["f"] and not board_ok else "prior"}
    out["timing_s"] = {"features": round(t2 - t1, 1), "matching": round(t3 - t2, 1), "pairs": len(allm)}

    def on_mask(ft, idx, m):
        xy = np.clip(np.round(ft.kpts[idx]).astype(int), [0, 0], [size[0] - 1, size[1] - 1])
        return m[xy[:, 1], xy[:, 0]]

    kp_on, dens_kp = [], []
    for fi, (ft, m) in enumerate(zip(feats, masks)):
        if m is None or not len(ft.kpts):
            kp_on.append(0); dens_kp.append(None)
            continue
        on = int(on_mask(ft, np.arange(len(ft.kpts)), m).sum())
        kp_on.append(on)
        s = scales[fi] or rec_px_per_m
        dens_kp.append(on / (area_px[fi] / (s / 10) ** 2) if s and area_px[fi] else None)
    pairs_rows = []
    for (i, j), mm in sorted(allm.items()):
        rw = raw[(i, j)]
        row = {"i": i, "j": j, "gap": j - i, "raw": int(len(rw)), "inliers": int(len(mm)), "on_cow": 0, "raw_on_cow": 0,
               "lg_error": (i, j) in errs}
        if masks[i] is not None and masks[j] is not None:
            if len(rw):
                row["raw_on_cow"] = int((on_mask(feats[i], rw[:, 0], masks[i]) & on_mask(feats[j], rw[:, 1], masks[j])).sum())
            if len(mm):
                on = on_mask(feats[i], mm[:, 0], masks[i]) & on_mask(feats[j], mm[:, 1], masks[j])
                row["on_cow"] = int(on.sum())
            s = scales[i] or rec_px_per_m
            if s and area_px[i]:
                row["on_cow_per_dm2"] = round(row["on_cow"] / (area_px[i] / (s / 10) ** 2), 3)
        pairs_rows.append(row)
    out["pairs"] = pairs_rows
    nb = [r for r in pairs_rows if r["gap"] == 1]
    med = lambda v: float(np.median(v)) if len(v) else None
    out["neighbour"] = {
        "pairs": len(nb), "inliers_median": med([r["inliers"] for r in nb]), "raw_median": med([r["raw"] for r in nb]),
        "on_cow_median": med([r["on_cow"] for r in nb]), "raw_on_cow_median": med([r["raw_on_cow"] for r in nb]),
        "on_cow_frac_median": med([r["on_cow"] / r["inliers"] for r in nb if r["inliers"]]),
        "raw_on_cow_kept_median": med([r["on_cow"] / r["raw_on_cow"] for r in nb if r["raw_on_cow"]]),
        "per_dm2_median": med([r["on_cow_per_dm2"] for r in nb if "on_cow_per_dm2" in r]),
        "frac_ge_min": round(sum(r["inliers"] >= recon_mod.MIN_PAIR_INLIERS for r in nb) / len(nb), 4) if nb else None}
    out["keypoints"] = {"median_total": med([len(f.kpts) for f in feats]), "median_on_cow": med(kp_on),
                        "on_cow_per_dm2_median": med([d for d in dens_kp if d is not None])}

    rec = None
    degen = None
    try:
        rec = run_sfm(recon, feats, K, dist, size, matches, seed)
    except DegenerateInit as e:
        degen = e
        out["sfm"] = {"ok": False, "error": f"DegenerateInit: {e}", "registered": 0, "registered_frac": 0.0,
                      "init_diag": e.diag}
    except RuntimeError as e:
        out["sfm"] = {"ok": False, "error": str(e), "registered": 0, "registered_frac": 0.0}
    except Exception as e:  # report, keep the shard alive
        out["sfm"] = {"ok": False, "error": f"{type(e).__name__}: {e}", "registered": 0, "registered_frac": 0.0}
    if degen is not None:
        # eval-only: drop the degenerate initial pair from the match graph and let sfm choose again
        tries, m2, e2 = [], dict(matches), degen
        for _ in range(SFM_RETRIES):
            m2.pop(tuple(e2.diag["pair"]), None)
            try:
                r2 = run_sfm(recon, feats, K, dist, size, m2, seed)
                out["sfm_retry"] = {**sfm_report(r2, n, feats, masks, on_mask), "dropped_pairs": tries + [e2.diag["pair"]]}
                break
            except DegenerateInit as e3:
                tries.append(e2.diag["pair"])
                e2 = e3
            except Exception as e3:
                out["sfm_retry"] = {"ok": False, "error": f"{type(e3).__name__}: {e3}",
                                    "dropped_pairs": tries + [e2.diag["pair"]], "registered_frac": 0.0}
                break
        else:
            out["sfm_retry"] = {"ok": False, "error": f"DegenerateInit after {SFM_RETRIES} drops",
                                "dropped_pairs": tries + [e2.diag["pair"]], "registered_frac": 0.0,
                                "last_init_diag": e2.diag}
    if rec is not None:
        out["sfm"] = sfm_report(rec, n, feats, masks, on_mask)
        reg = sorted(rec.R)
        ang = pair_angles(rec, rec.point_track)
        inl = {(r["i"], r["j"]): r["inliers"] for r in pairs_rows}
        out["angles"] = [{"i": a, "j": b, "deg": round(d, 3), "shared": s, "inliers": inl.get((a, b))}
                         for (a, b), (d, s) in sorted(ang.items())]
        dense_mod = sys.modules.get("mombe.body.dense")
        step = int(getattr(dense_mod, "PAIR_STEP", 2) or 2)
        rows = []
        for k, f1 in enumerate(reg):
            r = {"f": f1}
            if k + step < len(reg):
                f2 = reg[k + step]
                r["step_partner"] = f2
                r["step_deg"] = ang.get((f1, f2), (None, 0))[0]
            cands = [(reg[k + g], ang[(f1, reg[k + g])][0]) for g in range(1, ANGLE_MAX_GAP)
                     if k + g < len(reg) and (f1, reg[k + g]) in ang]
            inband = [(f2, d) for f2, d in cands if ANGLE_BAND[0] <= d <= ANGLE_BAND[1]]
            if inband:
                f2, d = min(inband, key=lambda t: abs(t[1] - sum(ANGLE_BAND) / 2))
                r["band_partner"], r["band_deg"] = f2, round(d, 3)
            r["max_deg_in_reach"] = round(max((d for _, d in cands), default=0.0), 3)
            rows.append(r)
        out["stereo"] = rows

    if shard is not None:
        size_px = size[0] * size[1]
        blank = lambda f: np.zeros(f.shape[:2], bool)
        clean = [r for r in nb if not r["lg_error"]]
        weakest = min(clean, key=lambda r: r["inliers"]) if clean else None
        if weakest is not None and weakest["inliers"] < recon_mod.MIN_PAIR_INLIERS:
            i, j = weakest["i"], weakest["j"]
            mm = allm[(i, j)]
            cap = (f"[{variant}] {rec_id} kf{i}-kf{j}: {weakest['raw']} raw, {weakest['inliers']} verified "
                   f"(<{recon_mod.MIN_PAIR_INLIERS}); sharp {qf[i].sharpness:.0f}/{qf[j].sharpness:.0f}; "
                   f"kp on cow {kp_on[i]}/{kp_on[j]}; mask {area_px[i] / size_px:.2f}/{area_px[j] / size_px:.2f}")
            shard.gallery.append({"rec": rec_id, "variant": variant, "kind": "weak_pair", "i": i, "j": j,
                                  "raw": weakest["raw"], "inliers": weakest["inliers"],
                                  "sharp": [qf[i].sharpness, qf[j].sharpness], "kp_on": [kp_on[i], kp_on[j]],
                                  "mask_frac": [area_px[i] / size_px, area_px[j] / size_px], "magenta": [mag[i], mag[j]],
                                  "caption": cap,
                                  "jpg": draw_pair(frames[i], frames[j],
                                                   masks[i] if masks[i] is not None else blank(frames[i]),
                                                   masks[j] if masks[j] is not None else blank(frames[j]),
                                                   feats[i].kpts[mm[:, 0]] if len(mm) else np.zeros((0, 2)),
                                                   feats[j].kpts[mm[:, 1]] if len(mm) else np.zeros((0, 2)), cap)})
        if degen is not None and len(degen.diag.get("pair", [])) == 2:
            i, j = degen.diag["pair"]
            dg = degen.diag
            mm = allm.get((i, j), np.zeros((0, 2), int))
            rt = out.get("sfm_retry", {})
            cap = (f"[{variant}] {rec_id} init pair kf{i}-kf{j}: {dg.get('matches')} verified, parallax "
                   f"{dg.get('px_parallax_median')} px, rotation {dg.get('rotation_deg')} deg, median ray angle "
                   f"{dg.get('angle_deg_median_front')} deg, {dg.get('points_triangulated')} pts kept -> sfm crash; "
                   f"retry reg {rt.get('registered_frac')}")
            shard.gallery.append({"rec": rec_id, "variant": variant, "kind": "degenerate_init", "i": i, "j": j,
                                  "inliers": int(len(mm)), "diag": dg, "caption": cap,
                                  "jpg": draw_pair(frames[i], frames[j],
                                                   masks[i] if masks[i] is not None else blank(frames[i]),
                                                   masks[j] if masks[j] is not None else blank(frames[j]),
                                                   feats[i].kpts[mm[:, 0]] if len(mm) else np.zeros((0, 2)),
                                                   feats[j].kpts[mm[:, 1]] if len(mm) else np.zeros((0, 2)), cap)})
        unreg = out["sfm"].get("unregistered") if out["sfm"]["ok"] else []
        if unreg and rec is not None:
            f = unreg[0]
            g = min(rec.R, key=lambda x: abs(x - f))
            a, b = min(f, g), max(f, g)
            mm = allm.get((a, b), np.zeros((0, 2), int))
            cap = (f"[{variant}] {rec_id} kf{f} unregistered (nearest registered kf{g}); pair {a}-{b} "
                   f"{len(raw.get((a, b), []))} raw, {len(mm)} verified; sharp {qf[f].sharpness:.0f}; "
                   f"kp on cow {kp_on[f]}; reg {len(rec.R)}/{n}")
            shard.gallery.append({"rec": rec_id, "variant": variant, "kind": "unregistered", "i": a, "j": b, "frame": f,
                                  "inliers": int(len(mm)), "sharp": [qf[f].sharpness], "kp_on": [kp_on[f]],
                                  "mask_frac": [area_px[f] / size_px], "magenta": [mag[f]], "caption": cap,
                                  "jpg": draw_pair(frames[a], frames[b],
                                                   masks[a] if masks[a] is not None else blank(frames[a]),
                                                   masks[b] if masks[b] is not None else blank(frames[b]),
                                                   feats[a].kpts[mm[:, 0]] if len(mm) else np.zeros((0, 2)),
                                                   feats[b].kpts[mm[:, 1]] if len(mm) else np.zeros((0, 2)), cap)})
        if out["sfm"]["ok"] and not unreg and variant == "as_shipped" and not any(g.get("kind") == "healthy" for g in shard.gallery):
            i, j = nb[len(nb) // 2]["i"], nb[len(nb) // 2]["j"]
            mm = allm[(i, j)]
            cap = (f"[{variant}] {rec_id} kf{i}-kf{j}: healthy reference, {len(mm)} verified, "
                   f"{nb[len(nb) // 2]['on_cow']} on the cow; reg {len(rec.R)}/{n}")
            shard.gallery.append({"rec": rec_id, "variant": variant, "kind": "healthy", "i": i, "j": j, "caption": cap,
                                  "jpg": draw_pair(frames[i], frames[j], masks[i] if masks[i] is not None else blank(frames[i]),
                                                   masks[j] if masks[j] is not None else blank(frames[j]),
                                                   feats[i].kpts[mm[:, 0]], feats[j].kpts[mm[:, 1]], cap)})
    return out, {"rec": rec}


def evaluate_recording(paths: list[str], masks: list[np.ndarray | None], kidx: list[int], recon_mod, recon,
                       quality_mod, seed: int = 7, shard: Shard | None = None, rec_id: str = "",
                       variants: tuple[str, ...] = ("as_shipped", "hide_only")) -> dict:
    """Run the front end on one recording's keyframes; `masks` aligns with `kidx`.

    as_shipped  the production path: every ALIKED keypoint in the frame, barn included
    hide_only   the same code fed frames whose background is filled flat, keypoints near
                the outline dropped: what the front end does once a cow mask exists (track 2a)
    """
    t0 = time.time()
    frames = [cv2.imread(paths[i]) for i in kidx]
    ok = [f is not None for f in frames]
    frames = [f for f in frames if f is not None]
    masks = [m for m, o in zip(masks, ok) if o]
    kidx = [i for i, o in zip(kidx, ok) if o]
    n = len(frames)
    size = (frames[0].shape[1], frames[0].shape[0])
    out: dict = {"id": rec_id, "eval_version": EVAL_VERSION, "keyframes": n, "keyframe_idx": kidx}

    # quality gate on body keyframes (the gate was calibrated on muzzle ROIs)
    qf, qr = [], []
    for f, m in zip(frames, masks):
        qf.append(quality_mod.assess(f))
        if m is not None and m.any():
            ys, xs = np.where(m)
            qr.append(quality_mod.assess(f, (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
                                             int(ys.max() - ys.min() + 1))))

    def qsum(qs):
        if not qs:
            return {}
        iss: dict[str, int] = {}
        for q in qs:
            for c in q.issues:
                iss[c] = iss.get(c, 0) + 1
        return {"n": len(qs), "pass_rate": round(sum(q.ok for q in qs) / len(qs), 4), "issues": iss,
                "sharpness_median": round(float(np.median([q.sharpness for q in qs])), 1),
                "mean_median": round(float(np.median([q.mean for q in qs])), 1)}
    out["quality_full"] = qsum(qf)
    out["quality_cow_roi"] = qsum(qr)

    # masks: hide area, scale, coat
    area_px, scales, coat, mag = [], [], [], []
    for f, m in zip(frames, masks):
        if m is None:
            area_px.append(0); scales.append(None); mag.append(0.0)
            continue
        area_px.append(int(m.sum()))
        scales.append(hide_scale(m).get("px_per_m"))
        mag.append(float(magenta(f).mean()))
        cs = coat_stats(f, m, recon_mod.WORK_W)
        if cs:
            coat.append(cs)
    good_scales = [s for s in scales if s]
    rec_px_per_m = float(np.median(good_scales)) if good_scales else None
    out["hide"] = {"mask_frac_median": round(float(np.median(area_px)) / (size[0] * size[1]), 4),
                   "frames_with_scale": len(good_scales), "px_per_m_median": rec_px_per_m and round(rec_px_per_m, 1),
                   "magenta_frac_max": round(max(mag), 4)}
    if coat:
        d, l, r = (float(np.median([c[k] for c in coat])) for k in ("dark", "light", "red"))
        out["coat"] = {"dark": round(d, 3), "light": round(l, 3), "red": round(r, 3),
                       "texture": round(float(np.median([c["texture"] for c in coat])), 2),
                       "mean_v": round(float(np.median([c["mean_v"] for c in coat])), 1),
                       "class": coat_class(d, l, r)}

    threads = cv2.getNumThreads()
    try:
        for v in variants:
            res, _ = _frontend(v, frames, masks, recon_mod, recon, seed, size, scales, area_px, rec_px_per_m, qf, mag,
                               shard, rec_id, threads)
            if v == "as_shipped":
                out.update(res)          # the production numbers sit at the top level
            else:
                out[v] = res
    finally:
        cv2.setNumThreads(threads)
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ---- aggregation (runs locally on the pulled shard files) ---------------------------------------
def _q(v, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(v):
        return None
    return {"n": int(len(v)), "mean": round(float(v.mean()), 4),
            **{f"p{int(q * 100)}": round(float(np.quantile(v, q)), 4) for q in qs}}


def _frac(v):
    v = list(v)
    return round(float(np.mean(v)), 4) if v else None


FLOORS = [0.5, 0.54, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]


def _agg_variant(rs: list[dict]) -> dict:
    """Matching, registration and stereo-angle summary for one variant's per-recording dicts."""
    s: dict = {}
    allp = [p for r in rs for p in r.get("pairs", [])]
    s["match_copy_equal"] = f"{sum(r.get('match_copy_equal') is True for r in rs)}/{sum(r.get('match_copy_equal') is not None for r in rs)}"
    s["matcher_errors"] = {"pairs_with_error": int(sum(p.get("lg_error", False) for p in allp)), "pairs": len(allp),
                           "recordings_with_error": int(sum(r.get("matcher_errors", {}).get("pairs", 0) > 0 for r in rs)),
                           "messages": sorted({m for r in rs for m in r.get("matcher_errors", {}).get("messages", [])})[:5]}
    s["pair_inliers_by_gap"] = {str(g): _q([p["inliers"] for p in allp if p["gap"] == g]) for g in (1, 2, 3, 4)}
    s["pair_frac_ge_30_by_gap"] = {str(g): _frac(p["inliers"] >= 30 for p in allp if p["gap"] == g) for g in (1, 2, 3, 4)}
    nb = [r["neighbour"] for r in rs if r.get("neighbour")]
    s["neighbour_raw_median"] = _q([x["raw_median"] for x in nb])
    s["neighbour_inliers_median"] = _q([x["inliers_median"] for x in nb])
    s["neighbour_frac_ge_30"] = _q([x["frac_ge_min"] for x in nb])
    s["neighbour_on_cow_median"] = _q([x["on_cow_median"] for x in nb])
    s["neighbour_raw_on_cow_median"] = _q([x["raw_on_cow_median"] for x in nb])
    s["neighbour_on_cow_frac"] = _q([x["on_cow_frac_median"] for x in nb])
    s["neighbour_on_cow_survival"] = _q([x["raw_on_cow_kept_median"] for x in nb])
    s["neighbour_on_cow_per_dm2"] = _q([x["per_dm2_median"] for x in nb])
    s["keypoints_total"] = _q([r["keypoints"]["median_total"] for r in rs if r.get("keypoints")])
    s["keypoints_on_cow"] = _q([r["keypoints"]["median_on_cow"] for r in rs if r.get("keypoints")])
    s["keypoints_on_cow_per_dm2"] = _q([r["keypoints"]["on_cow_per_dm2_median"] for r in rs if r.get("keypoints")])
    sf = [r["sfm"] for r in rs if "sfm" in r]
    ok = [x for x in sf if x["ok"]]
    s["sfm_ok"] = f"{len(ok)}/{len(sf)}"
    s["sfm_errors"] = sorted({x["error"] for x in sf if not x["ok"]})
    s["registered_frac"] = _q([x["registered_frac"] for x in sf])
    s["registered_all"] = f"{sum(x['registered_frac'] == 1.0 for x in sf)}/{len(sf)}"
    s["registration_floor_curve"] = {f"{f:.2f}": _frac(x["registered_frac"] >= f for x in sf) for f in FLOORS}
    s["sfm_degenerate_init"] = {
        "recordings": int(sum(1 for x in sf if not x["ok"] and x.get("error", "").startswith(("DegenerateInit", "IndexError")))),
        "diags": [x["init_diag"] for x in sf if x.get("init_diag")]}
    rt = [r for r in rs if "sfm_retry" in r]
    with_retry = [r["sfm_retry"]["registered_frac"] if "sfm_retry" in r else r["sfm"]["registered_frac"]
                  for r in rs if "sfm" in r]
    s["eval_only_retry"] = {
        "note": "eval-only: degenerate initial pair dropped from the match graph, sfm rerun; production crashes",
        "recordings": len(rt), "ok": int(sum(r["sfm_retry"].get("ok", False) for r in rt)),
        "registered_frac": [r["sfm_retry"]["registered_frac"] for r in rt],
        "dropped_pairs": [r["sfm_retry"].get("dropped_pairs") for r in rt],
        "reproj_median_all_px": [r["sfm_retry"].get("reprojection", {}).get("median_all_px") for r in rt],
        "registered_frac_with_retry": _q(with_retry),
        "registration_floor_curve_with_retry": {f"{f:.2f}": _frac(v >= f for v in with_retry) for f in FLOORS}}
    # why keyframes stay unregistered: PnP ran and was rejected, or registration stalled (no keyframe
    # left with 12 correspondences to triangulated points), or sfm never started (crash / no init pair)
    cause = {"recordings_partial": 0, "frames_pnp_rejected": 0, "frames_stalled": 0, "recordings_stall_only": 0,
             "recordings_not_started": int(sum(1 for x in sf if not x["ok"])),
             "frames_not_started": int(sum(r["keyframes"] for r in rs if "sfm" in r and not r["sfm"]["ok"]))}
    for x in ok:
        un, pf = set(x.get("unregistered", [])), set(x.get("pnp_failed", []))
        if un:
            cause["recordings_partial"] += 1
            cause["frames_pnp_rejected"] += len(un & pf)
            cause["frames_stalled"] += len(un - pf)
            cause["recordings_stall_only"] += int(not (un & pf))
    s["unregistered_cause"] = cause
    s["init_pair_gap"] = _q([x["init_pair"][1] - x["init_pair"][0] for x in ok if x.get("init_pair")])
    s["ba_rms_px"] = _q([x.get("ba_rms_px") for x in ok])
    s["reproj_median_all_px"] = _q([x["reprojection"].get("median_all_px") for x in ok])
    s["reproj_median_kept_px"] = _q([x["reprojection"].get("median_kept_px") for x in ok])
    s["reproj_frac_within_4px"] = _q([x["reprojection"].get("frac_within_4px") for x in ok])
    s["points"] = _q([x.get("points") for x in ok])
    s["points_on_cow_frac"] = _q([x.get("points_on_cow_frac") for x in ok])
    s["focal_pairs_px"] = _q([r["focal"]["pairs_px"] for r in rs if r.get("focal")])
    s["focal_source_counts"] = {k: sum(1 for r in rs if r.get("focal", {}).get("source") == k) for k in ("pairs", "prior")}
    s["heading_change_deg"] = _q([x["path"].get("heading_change_deg") for x in ok if x.get("path")])
    s["span_over_depth"] = _q([x["path"].get("span_over_depth") for x in ok if x.get("path")])

    gaps: dict[int, list] = {}
    for r in rs:
        if not r.get("sfm", {}).get("ok"):
            continue
        un = set(r["sfm"]["unregistered"])
        pos = {f: k for k, f in enumerate(f for f in range(r["keyframes"]) if f not in un)}
        for a in r.get("angles", []):
            if a["i"] in pos and a["j"] in pos:
                gaps.setdefault(pos[a["j"]] - pos[a["i"]], []).append(a["deg"])
    s["angle_by_registered_gap"] = {str(g): _q(v) for g, v in sorted(gaps.items())}
    st = [x for r in rs for x in r.get("stereo", [])]
    step = [x["step_deg"] for x in st if x.get("step_deg") is not None]
    s["step2_rule"] = {"pairs": len(step), "deg": _q(step), "frac_below_2": _frac(d < 2 for d in step),
                       "frac_below_5": _frac(d < 5 for d in step),
                       "frac_5_to_10": _frac(5 <= d <= 10 for d in step), "frac_above_10": _frac(d > 10 for d in step)}
    s["band_rule"] = {"frames": len(st), "frac_with_partner": _frac("band_partner" in x for x in st),
                      "partner_gap_keyframes": _q([x["band_partner"] - x["f"] for x in st if "band_partner" in x]),
                      "frac_max_reach_below_5": _frac(x["max_deg_in_reach"] < 5 for x in st)}
    s["per_recording_frac_in_band"] = {
        "band_rule": _q([float(np.mean(["band_partner" in x for x in r["stereo"]])) for r in rs if r.get("stereo")]),
        "step2_rule": _q([float(np.mean([5 <= x["step_deg"] <= 10 for x in r["stereo"] if x.get("step_deg") is not None]))
                          for r in rs if any(x.get("step_deg") is not None for x in r.get("stereo", []))])}
    # added at aggregation (v3, after the shards ran): the widest angle any registered pair reaches over the
    # whole recording, to tell "the band rule looks too near" apart from "the take never has the baseline"
    reach = [max(a["deg"] for a in r["angles"]) for r in rs if r.get("angles")]
    s["max_angle_any_pair_deg"] = _q(reach)
    s["recordings_reaching_band"] = {f"ge_{t}": f"{sum(x >= t for x in reach)}/{len(reach)}" for t in (2, 5, 10)}
    s["path_over_depth"] = _q([x["path"].get("path_over_depth") for x in ok if x.get("path")])
    ang = [a for r in rs for a in r.get("angles", [])]
    bins = [(0, 2), (2, 5), (5, 10), (10, 15), (15, 90)]
    s["inliers_by_angle_bin"] = {f"{a}-{b}": _q([x["inliers"] for x in ang if x["inliers"] is not None and a <= x["deg"] < b])
                                 for a, b in bins}
    s["shared_points_by_angle_bin"] = {f"{a}-{b}": _q([x["shared"] for x in ang if a <= x["deg"] < b]) for a, b in bins}
    s["seconds_matching"] = _q([r.get("timing_s", {}).get("matching") for r in rs])
    return s


def aggregate(rows: list[dict], attempted: int, synthetic_floor_ref: float = 0.54) -> dict:
    """Summary over recordings; every number keeps its N."""
    done = [r for r in rows if "sfm" in r]
    s: dict = {"recordings_attempted": attempted, "recordings_measured": len(done),
               "recordings_missing": attempted - len(done),
               "keyframes": _q([r["keyframes"] for r in done]),
               "keyframes_total": int(sum(r["keyframes"] for r in done)),
               "board_frames_total": int(sum(r.get("board_frames", 0) for r in done))}
    s["quality_full_pass_rate"] = _q([r["quality_full"].get("pass_rate") for r in done])
    s["quality_cow_roi_pass_rate"] = _q([r["quality_cow_roi"].get("pass_rate") for r in done])
    iss_f: dict = {}
    iss_r: dict = {}
    for r in done:
        for k, v in r["quality_full"].get("issues", {}).items():
            iss_f[k] = iss_f.get(k, 0) + v
        for k, v in r["quality_cow_roi"].get("issues", {}).items():
            iss_r[k] = iss_r.get(k, 0) + v
    nf = int(sum(r["quality_full"].get("n", 0) for r in done))
    nr = int(sum(r["quality_cow_roi"].get("n", 0) for r in done))
    s["quality_keyframes"] = {
        "full_frame": {"n": nf, "pass": int(sum(round(r["quality_full"].get("pass_rate", 0) * r["quality_full"].get("n", 0)) for r in done)),
                       "issues": iss_f},
        "cow_roi": {"n": nr, "pass": int(sum(round(r["quality_cow_roi"].get("pass_rate", 0) * r["quality_cow_roi"].get("n", 0)) for r in done)),
                    "issues": iss_r},
        "sharpness_full_median": _q([r["quality_full"].get("sharpness_median") for r in done]),
        "sharpness_roi_median": _q([r["quality_cow_roi"].get("sharpness_median") for r in done])}
    s["px_per_m"] = _q([r["hide"]["px_per_m_median"] for r in done])
    s["mask_frac"] = _q([r["hide"]["mask_frac_median"] for r in done])
    s["recordings_with_magenta"] = sum(r["hide"]["magenta_frac_max"] > 0.001 for r in done)
    s["as_shipped"] = _agg_variant(done)
    ho = [r["hide_only"] for r in done if "hide_only" in r]
    if ho:
        s["hide_only"] = _agg_variant(ho)
    # The merged gate (mombe.agent.policy.REGISTERED_FRAC_FLOOR, 0.80) is the floor that matters.
    # COOL sweep_21's 0.54 is girth-loop coverage (README, synthetic walk-around rows); it registered 48 of 48 keyframes.
    floor = 0.80
    s["production_registration_floor"] = {
        "floor": floor, "source": "mombe/agent/policy.py REGISTERED_FRAC_FLOOR",
        **{v: {"pass": int(sum((x["sfm"]["registered_frac"] if v == "as_shipped" else x.get(v, {}).get("sfm", {}).get("registered_frac", 0)) >= floor
                               for x in done)), "n": len(done)} for v in ("as_shipped", "hide_only")},
        "note": "single right-flank pass, not an orbit; the gate's orbit coverage and loop checks do not apply here. "
                "COOL sweep_21's 0.54 is girth-loop coverage (README, synthetic walk-around rows), a different quantity from the registered fraction."}

    # coat groups (proxy fixed in COAT_RULE before results)
    cls: dict[str, list] = {}
    for r in done:
        if r.get("coat"):
            cls.setdefault(r["coat"]["class"], []).append(r)
    tex = [r["coat"]["texture"] for r in done if r.get("coat")]
    t1, t2 = (np.quantile(tex, [1 / 3, 2 / 3]) if len(tex) >= 3 else (np.nan, np.nan))

    def grp(g):
        d = {"recordings": len(g), "quality_roi_pass": _q([r["quality_cow_roi"].get("pass_rate") for r in g])}
        for v in ("as_shipped", "hide_only"):
            vs = [r if v == "as_shipped" else r.get(v) for r in g]
            vs = [x for x in vs if x and "sfm" in x]
            d[v] = {"registered_frac": _q([x["sfm"]["registered_frac"] for x in vs]),
                    "neighbour_inliers": _q([x["neighbour"]["inliers_median"] for x in vs if x.get("neighbour")]),
                    "neighbour_on_cow": _q([x["neighbour"]["on_cow_median"] for x in vs if x.get("neighbour")]),
                    "on_cow_per_dm2": _q([x["neighbour"]["per_dm2_median"] for x in vs if x.get("neighbour")]),
                    "kp_on_cow_per_dm2": _q([x["keypoints"]["on_cow_per_dm2_median"] for x in vs if x.get("keypoints")])}
        return d
    s["by_coat_class"] = {k: grp(v) for k, v in sorted(cls.items())}
    tg: dict[str, list] = {"low": [], "mid": [], "high": []}
    for r in done:
        if r.get("coat"):
            t = r["coat"]["texture"]
            tg["low" if t <= t1 else "mid" if t <= t2 else "high"].append(r)
    s["by_texture_tertile"] = {"cuts": [round(float(t1), 2), round(float(t2), 2)], **{k: grp(v) for k, v in tg.items()}}
    s["seconds_per_recording"] = _q([r.get("seconds") for r in done])
    return s


def recording_line(r: dict) -> dict:
    """One compact row per recording for summary.json: the headline numbers of both variants."""
    def v(x: dict) -> dict:
        sf, nb = x.get("sfm", {}), x.get("neighbour", {})
        return {"registered_frac": sf.get("registered_frac"), "sfm_ok": sf.get("ok"),
                "sfm_error": None if sf.get("ok") else sf.get("error"),
                "reproj_median_px": sf.get("reprojection", {}).get("median_all_px") if sf.get("ok") else None,
                "nb_inliers_median": nb.get("inliers_median"), "nb_on_cow_median": nb.get("on_cow_median"),
                "nb_on_cow_per_dm2": nb.get("per_dm2_median"),
                "kp_on_cow_per_dm2": x.get("keypoints", {}).get("on_cow_per_dm2_median")}
    out = {"id": r["id"], "individual": r.get("individual"), "frames": r.get("frames"),
           "keyframes": r.get("keyframes"), "eval_version": r.get("eval_version", 1),
           "coat": r.get("coat", {}).get("class"), "texture": r.get("coat", {}).get("texture"),
           "quality_full_pass": r.get("quality_full", {}).get("pass_rate"),
           "quality_roi_pass": r.get("quality_cow_roi", {}).get("pass_rate"),
           "px_per_m": r.get("hide", {}).get("px_per_m_median"),
           "mask_frac": r.get("hide", {}).get("mask_frac_median"),
           "as_shipped": v(r)}
    if "hide_only" in r:
        out["hide_only"] = v(r["hide_only"])
    return out


def to_json(obj) -> str:
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (bytes, bytearray)):
            return None
        raise TypeError(type(o))
    return json.dumps(obj, default=conv, indent=1)


def main(argv: list[str] | None = None) -> int:
    """Merge pulled shard files into summary.json (+ per-recording rows).

    python -m eval.sideview_frontend --raw eval/out/sideview/raw --out eval/out/sideview
    """
    import argparse
    import glob
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="eval/out/sideview/raw")
    ap.add_argument("--out", default="eval/out/sideview")
    ap.add_argument("--attempted", type=int, default=71)
    ap.add_argument("--exclude", nargs="*", default=["probe"], help="raw subfolders to leave out")
    a = ap.parse_args(argv)
    rows: dict[str, dict] = {}
    shards = []
    for f in sorted(glob.glob(f"{a.raw}/*/rows_s*.jsonl")):
        if Path(f).parent.name in a.exclude:
            continue
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                old = rows.get(r["id"])
                # a successful retry replaces a failure; a newer eval version replaces an older row
                if (old is None or ("error" in old and "error" not in r)
                        or r.get("eval_version", 1) > old.get("eval_version", 1)):
                    rows[r["id"]] = r
    for f in sorted(glob.glob(f"{a.raw}/*/summary_s*.json")):
        if Path(f).parent.name not in a.exclude:
            shards.append(json.load(open(f, encoding="utf-8")))
    rs = sorted(rows.values(), key=lambda r: r["id"])
    s = aggregate(rs, a.attempted)
    s["errors"] = [{"id": r["id"], "error": r["error"]} for r in rs if "error" in r]
    s["eval_versions"] = {str(k): sum(r.get("eval_version", 1) == k for r in rs) for k in (1, 2)}
    s["per_recording"] = [recording_line(r) for r in rs if "error" not in r]
    s["shards"] = [{k: sh.get(k) for k in ("shard", "nshards", "attempted", "measured", "failed", "skipped_budget",
                                         "elapsed_h", "sysinfo", "micro_timing", "provider_keyframe_check",
                                         "bytes_fetched")} for sh in shards]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(to_json(s), encoding="utf-8")
    slim = [{k: v for k, v in r.items() if k not in ("pairs", "angles", "stereo", "traceback")} for r in rs]
    for r in slim:
        if "hide_only" in r:
            r["hide_only"] = {k: v for k, v in r["hide_only"].items() if k not in ("pairs", "angles", "stereo")}
    (out / "recordings.json").write_text(to_json(slim), encoding="utf-8")
    gal = []
    for f in sorted(glob.glob(f"{a.raw}/*/gallery_s*.jsonl")):
        if Path(f).parent.name in a.exclude:
            continue
        for line in open(f, encoding="utf-8"):
            if line.strip():
                g = json.loads(line)
                g["dir"] = str(Path(f).parent / "gallery")
                gal.append(g)
    (out / "gallery_index.json").write_text(to_json([{k: v for k, v in g.items() if k != "dir"} for g in gal]),
                                            encoding="utf-8")
    print(f"{len(rs)} recordings, {len(s['errors'])} errors, {len(gal)} gallery candidates -> {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
