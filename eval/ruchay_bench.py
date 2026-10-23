"""Ruchay real-cattle benchmark for the body measurement back end (track 1a).

Runs mombe.body.measure.measure() unchanged on the public multi-Kinect clouds of
103 Hereford cattle (github.com/ruchaya/CowDatabase) and scores it against the
manual tape measurements and live weights in github.com/ruchaya/CowDB.

The data carry no licence file. Use them for research only. This module holds
code; clouds and per-animal rows stay out of the repository.

Per animal:
  1. read the aligned cloud (left, top and right Kinect clouds in the top
     camera's frame, stored as three blocks of 512 x 424 points), drop the
     invalid pixels, tag each point with its camera;
  2. voxel downsample;
  3. RANSAC the floor plane, level to z-up, floor at z = 0, metres;
  4. crop to the animal: the largest footprint of body-height points in a
     top-down grid, expanded by a margin, then the largest 3D cluster;
  5. find the front: the end whose body extends furthest past its legs
     (the head and neck), on the variant's own points;
  6. call measure(points, [{"centre": front}]): one pseudo-board without a
     "normal", so drop_boards removes nothing and only the axis sign is set.

Variants: "fused" keeps all three cameras; "no_top" drops the overhead camera
to mimic a phone walk-around that sees only the flanks.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if (_ROOT / "mombe" / "body" / "measure.py").exists() and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:  # in the repo
    from mombe.body.measure import measure, schaeffer_kg, LEG_BAND
except ImportError:  # in the Kaggle kernel the exact source sits next to this file
    from measure import measure, schaeffer_kg, LEG_BAND  # type: ignore

VIEW_PIXELS = 512 * 424
VIEWS = ("left", "top", "right")      # block order inside <i>_AutoAligned.ply, verified on the data

# Every preprocessing parameter. Chosen on CowDatabase animals 1-10 only, then frozen.
PARAMS = {
    "voxel_m": 0.01,               # voxel downsample edge
    "max_cam_range_m": 4.5,        # drop points further than this from the top camera (background, walls)
    "ransac_iters": 400,
    "ransac_thresh_m": 0.02,       # floor inlier distance
    "ransac_max_tilt_deg": 25.0,   # floor normal must lie within this of the top camera's optical axis
    "ransac_seed": 0,
    "ransac_sample": 60000,        # points sampled for RANSAC scoring
    "floor_refine_m": 0.015,       # least-squares refit on inliers within this distance
    "z_max_m": 2.2,                # nothing on a cow stands higher
    "body_band_m": (0.6, 1.8),     # heights of the back and upper flank used to find the footprint
    "grid_m": 0.05,                # top-down grid for the footprint
    "grid_min_pts": 3,             # points for a grid cell to count as occupied
    "crop_margin_m": 0.35,         # footprint bounding box expanded by this (legs, head)
    "per_view_floor": True,        # re-seat each camera on the common floor (rigid, about its own centre)
    "view_floor_band_m": 0.06,     # points this close to the common floor feed each camera's floor fit
    "trunk_zmin_frac": 0.5,        # front call: body width measured on points above this fraction of top
    "trunk_step_m": 0.05,          # front call: station spacing along the axis
    "trunk_frac": 0.6,             # front call: trunk = stations at least this fraction of the widest
    "front_min_margin_m": 0.10,    # front call is flagged "low confidence" below this overhang difference
    # Which end is the head. "auto": the overhang rule above (run v1). "+x" / "-x": a recording-setup
    # label, set by main(--front). Run v2 uses it: the v1 visual audit of all 103 top-down renders
    # showed every CowDatabase animal facing +x of the levelled top-camera frame, and the auto rule
    # flipped 3 of them (35, 37, 95). No tape value enters this label.
    "front_source": "auto",
    # Diagnostic only (outside measure()): station this far behind the trunk front.
    # Fitted to tape on animals 1-10 (scan of 0.05-0.20 m; 0.10 m gave the lowest dev MAPE).
    "diag_station_offset_m": 0.10,
}


# ---------------------------------------------------------------- reading
def read_ply(path) -> tuple[np.ndarray, list[str]]:
    """Vertex xyz of a PCL PLY (ascii or binary little endian), float64, in file order."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header")
            line = line.decode("ascii", "replace").strip()
            header.append(line)
            if line == "end_header":
                break
        fmt = next(h for h in header if h.startswith("format")).split()[1]
        elems, cur = [], None
        for h in header:
            if h.startswith("element"):
                _, name, n = h.split()
                cur = {"name": name, "n": int(n), "props": []}
                elems.append(cur)
            elif h.startswith("property") and cur is not None:
                parts = h.split()
                cur["props"].append((parts[1], parts[2]))
        v = elems[0]
        assert v["name"] == "vertex"
        if fmt == "ascii":
            ncol = len(v["props"])
            raw = f.read()
            # the vertex block is the first n lines; parse fast with fromstring
            lines = raw.split(b"\n", v["n"])
            block = b"\n".join(lines[: v["n"]])
            arr = np.array(block.split(), dtype=np.float64)
            arr = arr.reshape(v["n"], ncol)
            return arr[:, :3].copy(), header
        tmap = {"float": "<f4", "uchar": "u1", "double": "<f8", "int": "<i4", "uint": "<u4"}
        dt = np.dtype([(n, tmap[t]) for t, n in v["props"]])
        arr = np.frombuffer(f.read(v["n"] * dt.itemsize), dtype=dt, count=v["n"])
        return np.c_[arr["x"], arr["y"], arr["z"]].astype(np.float64), header


def parse_matrices(text: str) -> dict:
    """The CowDatabase .matrices file: 'Left -> Top' and 'Right -> Top' 4x4 transforms."""
    out, key, rows = {}, None, []
    for line in text.splitlines():
        line = line.strip()
        if "->" in line:
            if key and rows:
                out[key] = np.array(rows, float)
            key, rows = line.split("->")[0].strip().lower(), []
        elif line:
            rows.append([float(x) for x in line.split()])
    if key and rows:
        out[key] = np.array(rows, float)
    return out


def split_aligned(xyz: np.ndarray, matrices: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Drop invalid pixels and tag each point with its camera (0 left, 1 top, 2 right).

    Invalid Kinect pixels are stored as (0, 0, 0) in each camera frame, so after the
    transform they collapse onto that camera's translation. The block whose sentinel
    is the origin is the top camera (identity transform). When the matrices are
    given, the other two sentinels must equal the left and right translations.
    """
    assert len(xyz) == 3 * VIEW_PIXELS, f"expected 3 x {VIEW_PIXELS} points, got {len(xyz)}"
    blocks, sents = [], []
    for k in range(3):
        blk = xyz[k * VIEW_PIXELS:(k + 1) * VIEW_PIXELS]
        vals, counts = np.unique(np.round(blk, 6), axis=0, return_counts=True)
        blocks.append(blk)
        sents.append(vals[np.argmax(counts)])
    # The top camera is the block whose sentinel is the origin (identity transform).
    # CowDatabase stores (left, top, right); CowDatabase2 stores the top block first.
    origin = [k for k in range(3) if np.abs(sents[k]).max() < 1e-4]
    top_k = origin[0] if len(origin) == 1 else 1
    order = [k for k in range(3) if k != top_k]
    order = [order[0], top_k, order[1]]        # view ids: 0 side A, 1 top, 2 side B
    keep, view, info = [], [], {"sentinels": [], "block_order": order}
    for vid, k in enumerate(order):
        blk, sent = blocks[k], sents[k]
        info["sentinels"].append([round(float(x), 4) for x in sent])
        ok = np.isfinite(blk).all(1) & (np.abs(blk - sent).sum(1) > 1e-5)
        keep.append(blk[ok])
        view.append(np.full(int(ok.sum()), vid, np.int8))
    # verify the block order
    s = np.array(info["sentinels"])
    info["top_block_is_origin"] = bool(np.abs(s[1]).max() < 1e-4)
    if matrices is not None and "left" in matrices and "right" in matrices:
        info["left_block_matches_matrix"] = bool(np.abs(s[0] - matrices["left"][:3, 3]).max() < 0.006)
        info["right_block_matches_matrix"] = bool(np.abs(s[2] - matrices["right"][:3, 3]).max() < 0.006)
    return np.vstack(keep), np.concatenate(view), info


def fuse_views(left: np.ndarray, right: np.ndarray, top: np.ndarray, matrices: dict) -> np.ndarray:
    """Rebuild the aligned cloud from the per-camera clouds and the published transforms."""
    def tf(X, M):
        return X @ M[:3, :3].T + M[:3, 3]
    return np.vstack([tf(left, matrices["left"]), top, tf(right, matrices["right"])])


# ---------------------------------------------------------------- tape table
def read_xlsx_rows(path) -> list[list]:
    """First worksheet of an .xlsx as rows of python values (stdlib only)."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", ns):
                shared.append("".join(t.text or "" for t in si.iter("{%s}t" % ns["m"])))
        sheets = sorted(n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n))
        root = ET.fromstring(z.read(sheets[0]))
    rows = []
    for r in root.iter("{%s}row" % ns["m"]):
        vals = {}
        for c in r.findall("m:c", ns):
            ref = c.get("r")
            col = 0
            for ch in re.match(r"[A-Z]+", ref).group(0):
                col = col * 26 + ord(ch) - 64
            v = c.find("m:v", ns)
            t = c.get("t")
            if t == "inlineStr":
                is_ = c.find("m:is", ns)
                val = "".join(x.text or "" for x in is_.iter("{%s}t" % ns["m"])) if is_ is not None else None
            elif v is None:
                val = None
            elif t == "s":
                val = shared[int(v.text)]
            elif t == "str":
                val = v.text
            else:
                try:
                    f = float(v.text)
                    val = int(f) if f.is_integer() else f
                except ValueError:
                    val = v.text
            vals[col] = val
        if vals:
            rows.append([vals.get(i) for i in range(1, max(vals) + 1)])
    return rows


TAPE_COLS = {"live weithg": "weight_kg", "withers height": "withers_cm", "hip height": "hip_height_cm",
             "chest depth": "chest_depth_cm", "chest width": "chest_width_cm", "ilium width": "ilium_width_cm",
             "hip joint width": "hjw_cm", "oblique body length": "obl_cm", "hip length": "hip_length_cm",
             "heart girth": "girth_cm"}


def read_tape(path) -> dict[int, dict]:
    rows = read_xlsx_rows(path)
    hdr = [str(h).strip().lower() if h is not None else "" for h in rows[0]]
    out = {}
    for r in rows[1:]:
        if not r or r[0] is None:
            continue
        rec = {}
        for i, h in enumerate(hdr):
            if h in TAPE_COLS and i < len(r) and r[i] is not None:
                rec[TAPE_COLS[h]] = float(r[i])
            elif h.endswith("timestamp") and i < len(r):
                rec[h.replace(" ", "_")] = r[i]
        out[int(r[0])] = rec
    return out


def premap(cdb_id: int) -> int | None:
    """Pre-registered row map: CowDatabase 1-45 = CowDB 1-45, 46-103 = CowDB 97-154."""
    if 1 <= cdb_id <= 45:
        return cdb_id
    if 46 <= cdb_id <= 103:
        return cdb_id + 51
    return None


PREREG_DROP_CDB = {43, 44, 45}     # pairing mismatch


def tape_ok(rec: dict) -> tuple[bool, str]:
    """Pre-registered plausibility rules on the tape row."""
    if rec.get("obl_cm", 0) > 250:
        return False, f"OBL {rec['obl_cm']:.0f} cm > 250"
    if rec.get("hjw_cm", 99) < 20:
        return False, f"HJW {rec['hjw_cm']:.0f} cm < 20"
    return True, ""


# ---------------------------------------------------------------- geometry
def voxel_down(P: np.ndarray, view: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """Centroid per occupied voxel; the voxel takes the camera with most points in it."""
    q = np.floor(P / voxel).astype(np.int64)
    q -= q.min(0)
    dims = q.max(0) + 1
    key = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    uniq, inv = np.unique(key, return_inverse=True)
    n = len(uniq)
    cnt = np.bincount(inv, minlength=n).astype(float)
    out = np.c_[np.bincount(inv, P[:, 0], n), np.bincount(inv, P[:, 1], n), np.bincount(inv, P[:, 2], n)] / cnt[:, None]
    vc = np.stack([np.bincount(inv, (view == k).astype(float), n) for k in range(3)], 1)
    return out, np.argmax(vc, 1).astype(np.int8)


def fit_floor(P: np.ndarray, up_hint: np.ndarray, p: dict) -> tuple[np.ndarray, float, dict]:
    """RANSAC plane n.x + d = 0 with n near up_hint; returns unit normal pointing up_hint-wards."""
    rng = np.random.default_rng(p["ransac_seed"])
    S = P[rng.choice(len(P), min(len(P), p["ransac_sample"]), replace=False)]
    cos_tilt = math.cos(math.radians(p["ransac_max_tilt_deg"]))
    best, best_n, best_d = -1, None, None
    for _ in range(p["ransac_iters"]):
        a, b, c = S[rng.choice(len(S), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if n @ up_hint < 0:
            n = -n
        if n @ up_hint < cos_tilt:
            continue
        d = -n @ a
        score = int((np.abs(S @ n + d) < p["ransac_thresh_m"]).sum())
        if score > best:
            best, best_n, best_d = score, n, d
    if best_n is None:
        raise RuntimeError("floor RANSAC found no plane within the tilt limit")
    # least-squares refit on the inliers of the full cloud
    inl = np.abs(P @ best_n + best_d) < p["floor_refine_m"]
    Q = P[inl]
    cq = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - cq, full_matrices=False)
    n = vt[-1]
    if n @ up_hint < 0:
        n = -n
    d = -n @ cq
    resid = np.abs((Q - cq) @ n)
    return n, d, {"inliers": int(inl.sum()), "inlier_frac": round(float(inl.mean()), 3),
                  "rms_mm": round(float(np.sqrt(np.mean(resid ** 2)) * 1000), 2),
                  "tilt_deg": round(float(np.degrees(np.arccos(np.clip(n @ up_hint, -1, 1)))), 2)}


def level(P: np.ndarray, n: np.ndarray, d: float, ref_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate so n -> +z and the floor sits at z = 0. x follows ref_axis projected on the floor."""
    z = n
    x = ref_axis - (ref_axis @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])            # rows: new axes in old coordinates
    Q = P @ R.T
    Q[:, 2] += d                       # n.p + d is the signed height above the floor
    return Q, R


def _rot_to_z(n: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector n onto +z."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z)
    s = np.linalg.norm(v)
    c = float(n @ z)
    if s < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def refine_views(Q: np.ndarray, view: np.ndarray, cams: np.ndarray, p: dict) -> tuple[np.ndarray, dict]:
    """Seat each camera on the common floor.

    The published transforms carry two decimals, so a camera's floor can sit a
    few centimetres off and a degree or two tilted against the others. Each
    camera's floor points are fitted with a plane and that camera's points are
    rotated about the camera centre until its floor is level, then shifted to z = 0.
    """
    Q = Q.copy()
    info = {}
    for k in range(3):
        m = view == k
        F = Q[m & (np.abs(Q[:, 2]) < p["view_floor_band_m"])]
        if len(F) < 500:
            info[VIEWS[k]] = {"skipped": f"{len(F)} floor points"}
            continue
        A = np.c_[F[:, 0], F[:, 1], np.ones(len(F))]
        sel = np.ones(len(F), bool)
        for _ in range(4):
            coef, *_ = np.linalg.lstsq(A[sel], F[sel, 2], rcond=None)
            sel = np.abs(F[:, 2] - A @ coef) < p["floor_refine_m"]
        n = np.array([-coef[0], -coef[1], 1.0])
        n /= np.linalg.norm(n)
        Rk = _rot_to_z(n)
        C = cams[k]
        Qk = (Q[m] - C) @ Rk.T + C
        h = float(np.median(((F[sel] - C) @ Rk.T + C)[:, 2]))
        Qk[:, 2] -= h
        Q[m] = Qk
        info[VIEWS[k]] = {"tilt_deg": round(float(np.degrees(np.arccos(np.clip(n[2], -1, 1)))), 3),
                          "offset_mm": round(h * 1000, 1), "floor_pts": int(sel.sum())}
    return Q, info


def footprint_crop(Q: np.ndarray, p: dict, centre_xy: np.ndarray) -> tuple[np.ndarray, dict]:
    """Mask of points inside the animal's footprint box.

    Body-height points are binned on a top-down grid; the connected component
    with the most points (ties broken towards the top camera) is the animal.
    """
    from scipy import ndimage
    lo, hi = p["body_band_m"]
    B = Q[(Q[:, 2] > lo) & (Q[:, 2] < hi)]
    if len(B) < 200:
        raise RuntimeError("no body-height points")
    g = p["grid_m"]
    mn = B[:, :2].min(0)
    ij = np.floor((B[:, :2] - mn) / g).astype(int)
    shape = ij.max(0) + 1
    cnt = np.zeros(shape, int)
    np.add.at(cnt, (ij[:, 0], ij[:, 1]), 1)
    occ = cnt >= p["grid_min_pts"]
    lab, nlab = ndimage.label(occ, structure=np.ones((3, 3)))
    if nlab == 0:
        raise RuntimeError("empty footprint grid")
    mass = ndimage.sum(cnt, lab, range(1, nlab + 1))
    k = int(np.argmax(mass)) + 1
    cells = np.argwhere(lab == k)
    xy_lo = mn + cells.min(0) * g - p["crop_margin_m"]
    xy_hi = mn + (cells.max(0) + 1) * g + p["crop_margin_m"]
    m = (Q[:, 0] >= xy_lo[0]) & (Q[:, 0] <= xy_hi[0]) & (Q[:, 1] >= xy_lo[1]) & (Q[:, 1] <= xy_hi[1])
    m &= (Q[:, 2] < p["z_max_m"])
    second = float(np.sort(mass)[-2] / mass.max()) if nlab > 1 else 0.0
    return m, {"footprint_cells": int(len(cells)), "components": int(nlab), "second_mass_ratio": round(second, 3),
               "box_m": [round(float(v), 3) for v in (*xy_lo, *xy_hi)]}


def leg_runs(P: np.ndarray, ax: np.ndarray, c: np.ndarray, top: float):
    """The two biggest runs of leg points along the body axis (same rule as measure())."""
    legs = P[(P[:, 2] > LEG_BAND[0] * top) & (P[:, 2] < LEG_BAND[1] * top)]
    if len(legs) < 20:
        return []
    ls = (legs[:, :2] - c) @ ax
    hist, edges = np.histogram(ls, bins=np.arange(ls.min() - 0.02, ls.max() + 0.04, 0.02))
    occupied = hist > max(3, 0.05 * hist.max())
    runs, start = [], None
    for i, o in enumerate(occupied):
        if o and start is None:
            start = i
        if (not o or i == len(occupied) - 1) and start is not None:
            end = i if o else i - 1
            runs.append((float(edges[start]), float(edges[end + 1]), int(hist[start:end + 1].sum())))
            start = None
    return sorted(runs, key=lambda r: -r[2])[:2]


def body_frame(P: np.ndarray):
    """measure()'s own body axis: principal direction of the points above 60% of the top height."""
    top = P[:, 2].max()
    upper = P[P[:, 2] > 0.6 * top]
    c = upper[:, :2].mean(0)
    w, v = np.linalg.eigh(np.cov((upper[:, :2] - c).T))
    return c, v[:, -1], top


def trunk_profile(P: np.ndarray, c: np.ndarray, ax: np.ndarray, top: float, p: dict):
    """Width of the body (points above trunk_zmin_frac of the top) at stations along the axis."""
    lat_dir = np.array([-ax[1], ax[0]])
    B = P[P[:, 2] > p["trunk_zmin_frac"] * top]
    s = (B[:, :2] - c) @ ax
    lat = (B[:, :2] - c) @ lat_dir
    st = p["trunk_step_m"]
    stations = np.arange(np.percentile(s, 0.5), np.percentile(s, 99.5) + st, st)
    widths = np.zeros(len(stations))
    for i, x in enumerate(stations):
        m = np.abs(s - x) < st / 2
        if m.sum() >= 10:
            widths[i] = np.percentile(lat[m], 98) - np.percentile(lat[m], 2)
    return stations, widths, s


def find_front(P: np.ndarray, p: dict) -> dict:
    """Which end of the body axis is the head.

    The trunk is the run of stations at least trunk_frac as wide as the widest
    station, grown from the widest one. The neck and head reach far past the
    trunk at the front; the rump ends close behind it. The end with the longer
    overhang beyond the trunk is the front.
    """
    c, ax, top = body_frame(P)
    stations, widths, s = trunk_profile(P, c, ax, top, p)
    if (widths > 0).sum() < 5:
        return {"ok": False, "reason": "too few stations for the front call"}
    wmax = np.percentile(widths[widths > 0], 95)
    wide = widths >= p["trunk_frac"] * wmax
    i0 = int(np.argmax(np.where(wide, widths, -1)))
    lo = hi = i0
    while lo > 0 and wide[lo - 1]:
        lo -= 1
    while hi < len(stations) - 1 and wide[hi + 1]:
        hi += 1
    s_lo, s_hi = float(stations[0]), float(stations[-1])
    over_hi = s_hi - float(stations[hi])
    over_lo = float(stations[lo]) - s_lo
    auto_sign = 1.0 if over_hi >= over_lo else -1.0
    src = p.get("front_source", "auto")
    if src in ("+x", "-x"):
        # recording-setup label: every animal in the database walks the chute facing this way
        want = 1.0 if src == "+x" else -1.0
        sign = 1.0 if ax[0] * want >= 0 else -1.0
    else:
        sign = auto_sign
    s_front = s_hi if sign > 0 else s_lo
    front_xy = c + ax * s_front
    margin = abs(over_hi - over_lo)
    t_lo, t_hi = float(stations[lo]) - p["trunk_step_m"] / 2, float(stations[hi]) + p["trunk_step_m"] / 2
    trunk_s = [t_lo, t_hi] if sign > 0 else [-t_hi, -t_lo]      # front-positive, measure()'s frame
    return {"ok": True, "centre": [float(front_xy[0]), float(front_xy[1]), float(0.9 * top)],
            "axis": [float(ax[0] * sign), float(ax[1] * sign)],
            "trunk_s_m": [round(trunk_s[0], 3), round(trunk_s[1], 3)],
            "trunk_m": round(t_hi - t_lo, 3),
            "overhang_front_m": round(float(over_hi if sign > 0 else over_lo), 3),
            "overhang_rear_m": round(float(over_lo if sign > 0 else over_hi), 3), "margin_m": round(float(margin), 3),
            "confident": bool(margin >= p["front_min_margin_m"]),
            "front_source": src, "auto_agrees": bool(sign == auto_sign)}


# ---------------------------------------------------------------- one animal
def prepare(xyz: np.ndarray, view: np.ndarray, cams: np.ndarray, p: dict = PARAMS) -> dict:
    """Level and crop the fused cloud (all cameras); the frame is shared by both variants.

    cams: (3, 3) camera centres in the top camera's frame (left, top, right).
    """
    # top camera sits at the origin looking along +z; the floor is far along +z
    rng_ok = np.linalg.norm(xyz, axis=1) < p["max_cam_range_m"]
    P0, v0 = xyz[rng_ok], view[rng_ok]
    P, v = voxel_down(P0, v0, p["voxel_m"])
    up_hint = np.array([0.0, 0.0, -1.0])        # the top camera looks down: up is -z in its frame
    n, d, finfo = fit_floor(P, up_hint, p)
    Q, R = level(P, n, d, np.array([1.0, 0.0, 0.0]))
    C = np.asarray(cams, float) @ R.T
    C[:, 2] += d
    vinfo = {}
    if p["per_view_floor"]:
        Q, vinfo = refine_views(Q, v, C, p)
    m, cinfo = footprint_crop(Q, p, C[1, :2])
    return {"Q": Q[m], "view": v[m], "R": R.tolist(), "floor": finfo, "view_floor": vinfo, "crop": cinfo,
            "cams_levelled": np.round(C, 3).tolist(),
            "n_raw": int(len(xyz)), "n_range": int(len(P0)), "n_voxel": int(len(P)), "n_crop": int(m.sum())}


def run_variant(Q: np.ndarray, view: np.ndarray, variant: str, p: dict = PARAMS) -> dict:
    keep = np.ones(len(Q), bool) if variant == "fused" else (view != 1)
    P = Q[keep]
    out = {"variant": variant, "n_points": int(len(P))}
    if len(P) < 500:
        return {**out, "ok": False, "stage": "crop", "reason": "too few points"}
    # find the front on the same cluster measure() will keep
    from scipy import ndimage  # noqa: F401  (measure imports it lazily)
    try:
        from mombe.body.measure import _largest_cluster
    except ImportError:
        from measure import _largest_cluster  # type: ignore
    Pc = _largest_cluster(P[P[:, 2] > 0.05])
    fr = find_front(Pc, p)
    out["front"] = fr
    if not fr["ok"]:
        return {**out, "ok": False, "stage": "front", "reason": fr["reason"]}
    res = measure(P, [{"centre": fr["centre"]}])
    out["measure"] = res
    out["diag"] = diag_girth(P, fr["centre"], fr["trunk_s_m"][1], p)
    if not res.get("ok"):
        return {**out, "ok": False, "stage": "measure", "reason": res.get("reason", "?")}
    out["ok"] = True
    # diagnostic only: does measure()'s girth station sit inside the trunk found by the front call?
    t0, t1 = fr["trunk_s_m"]
    out["station_in_trunk"] = bool(t0 <= res["girth_station_m"] <= t1)
    out["station_from_trunk_front_m"] = round(t1 - res["girth_station_m"], 3)
    return out


def diag_girth(P: np.ndarray, front_centre, trunk_front_s: float, p: dict = PARAMS) -> dict:
    """DIAGNOSTIC, outside measure(): the slice girth at an independently placed station.

    Same cluster, frame, slice width, slice height cut and girth_of_slice() as
    measure(), so only the station differs. The station sits
    diag_station_offset_m behind the front of the trunk found by the front call
    (where the body narrows into the neck). The offset was fitted to tape on
    CowDatabase animals 1-10, so only animals 11+ test it.
    """
    try:
        from mombe.body.measure import girth_of_slice, _largest_cluster, SLICE_HALF_M
    except ImportError:
        from measure import girth_of_slice, _largest_cluster, SLICE_HALF_M  # type: ignore
    Pc = _largest_cluster(P[P[:, 2] > 0.05])
    c, ax, top = body_frame(Pc)
    if (np.asarray(front_centre)[:2] - c) @ ax < 0:
        ax = -ax
    lat_dir = np.array([-ax[1], ax[0]])
    s = (Pc[:, :2] - c) @ ax
    lat = (Pc[:, :2] - c) @ lat_dir
    s_girth = float(trunk_front_s) - p["diag_station_offset_m"]
    sl = (np.abs(s - s_girth) < SLICE_HALF_M) & (Pc[:, 2] > LEG_BAND[1] * top)
    if sl.sum() < 12:
        return {"ok": False, "reason": "diagnostic slice empty"}
    g = girth_of_slice(lat[sl], Pc[sl, 2])
    return {"ok": bool(np.isfinite(g.get("girth_m", np.nan))), "girth": g, "station_m": round(s_girth, 3)}


def run_animal(xyz: np.ndarray, view: np.ndarray, cams: np.ndarray, p: dict = PARAMS,
               variants=("fused", "no_top")) -> tuple[dict, dict]:
    """Both variants. "no_top" removes the overhead camera before anything else, so its floor,
    camera seating, crop, front call and measurement all come from the two flank cameras."""
    keys = ("floor", "view_floor", "crop", "n_raw", "n_range", "n_voxel", "n_crop")
    prep = prepare(xyz, view, cams, p)
    rec = {k: prep[k] for k in keys}
    rec["variants"] = {}
    if "fused" in variants:
        rec["variants"]["fused"] = run_variant(prep["Q"], prep["view"], "fused", p)
    if "no_top" in variants:
        keep = view != 1
        try:
            prep2 = prepare(xyz[keep], view[keep], cams, p)
            rec["no_top_prep"] = {k: prep2[k] for k in keys}
            rec["variants"]["no_top"] = run_variant(prep2["Q"], prep2["view"], "no_top", p)
        except Exception as e:
            rec["variants"]["no_top"] = {"variant": "no_top", "ok": False, "stage": "prepare",
                                         "reason": f"{type(e).__name__}: {e}"}
    return rec, prep


# ---------------------------------------------------------------- scoring
def _t975(df: int) -> float:
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except Exception:
        return 1.96


def agreement(meas: np.ndarray, ref: np.ndarray) -> dict:
    """Measured vs reference (same units): MAE, MAPE, bias with 95% CI, Bland-Altman, Pearson r."""
    meas, ref = np.asarray(meas, float), np.asarray(ref, float)
    ok = np.isfinite(meas) & np.isfinite(ref)
    meas, ref = meas[ok], ref[ok]
    n = len(meas)
    if n < 3:
        return {"n": n}
    e = meas - ref
    sd = float(np.std(e, ddof=1))
    bias = float(np.mean(e))
    half = _t975(n - 1) * sd / math.sqrt(n)
    r = float(np.corrcoef(meas, ref)[0, 1])
    return {"n": n, "mae": round(float(np.mean(np.abs(e))), 3), "mape_pct": round(float(np.mean(np.abs(e) / ref) * 100), 3),
            "bias": round(bias, 3), "bias_ci95": [round(bias - half, 3), round(bias + half, 3)],
            "bias_pct": round(float(np.mean(e / ref) * 100), 3), "sd_diff": round(sd, 3),
            "loa95": [round(bias - 1.96 * sd, 3), round(bias + 1.96 * sd, 3)], "pearson_r": round(r, 4),
            "rmse": round(float(np.sqrt(np.mean(e ** 2))), 3)}


def weight_scores(pred: np.ndarray, w: np.ndarray) -> dict:
    pred, w = np.asarray(pred, float), np.asarray(w, float)
    ok = np.isfinite(pred) & np.isfinite(w)
    pred, w = pred[ok], w[ok]
    if len(w) < 3:
        return {"n": int(len(w))}
    e = (pred - w) / w * 100
    return {"n": int(len(w)), "mape_pct": round(float(np.mean(np.abs(e))), 3), "bias_pct": round(float(np.mean(e)), 3),
            "mae_kg": round(float(np.mean(np.abs(pred - w))), 2),
            "rmse_kg": round(float(np.sqrt(np.mean((pred - w) ** 2))), 2),
            "pearson_r": round(float(np.corrcoef(pred, w)[0, 1]), 4)}


def kfold_ids(n: int, k: int = 10, seed: int = 0) -> np.ndarray:
    """Fold index per row: a seeded permutation dealt round-robin into k folds."""
    perm = np.random.default_rng(seed).permutation(n)
    fold = np.empty(n, int)
    fold[perm] = np.arange(n) % k
    return fold


def cv_loglog(G_cm: np.ndarray, L_cm, W: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Out-of-fold predictions of log W = a + b log G + c log L (ordinary least squares).
    L_cm=None fits girth alone: log W = a + b log G."""
    X = np.c_[np.ones(len(W)), np.log(G_cm)] if L_cm is None else np.c_[np.ones(len(W)), np.log(G_cm), np.log(L_cm)]
    y = np.log(W)
    pred = np.full(len(W), np.nan)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        pred[te] = np.exp(X[te] @ beta)
    return pred


def bland_altman_arrays(meas, ref) -> dict:
    meas, ref = np.asarray(meas, float), np.asarray(ref, float)
    ok = np.isfinite(meas) & np.isfinite(ref)
    mean = (meas[ok] + ref[ok]) / 2
    diff = meas[ok] - ref[ok]
    return {"mean": [round(float(x), 2) for x in mean], "diff": [round(float(x), 2) for x in diff]}


# ---------------------------------------------------------------- the full run
# Broad physical envelope for adult cattle, written before the full run. Outputs outside it
# are counted; every headline statistic is reported on all outputs, inside or outside.
ENVELOPE = {"girth_cm": (100.0, 300.0), "length_cm": (80.0, 250.0), "withers_cm": (80.0, 180.0)}


def verify_rowmap(cdb_tape: dict, cowdb_tape: dict, cowdb_names: list[str] | None) -> dict:
    """Check the pre-registered map by tape values and, given the CowDB file list, by capture timestamps."""
    by_left_ts, ts_of = {}, {}
    for nm in cowdb_names or []:
        m = re.match(r"^(\d+)/(left|right|top)-([\d.]+)\.ply$", nm)
        if m:
            ts_of.setdefault(int(m.group(1)), {})[m.group(2)] = m.group(3)
            if m.group(2) == "left":
                by_left_ts[m.group(3)] = int(m.group(1))
    fields = list(TAPE_COLS.values())
    rows, n_val, n_ts = [], 0, 0
    for i in sorted(cdb_tape):
        j = premap(i)
        a, b = cdb_tape[i], cowdb_tape.get(j, {})
        val_eq = all(a.get(f) == b.get(f) for f in fields)
        n_val += val_eq
        rec = {"cdb": i, "cowdb": j, "values_equal": val_eq}
        if cowdb_names:
            ts = (a.get("left_timestamp"), a.get("top_timestamp"), a.get("right_timestamp"))
            tj = ts_of.get(j, {})
            ts_eq = ts == (tj.get("left"), tj.get("top"), tj.get("right"))
            n_ts += ts_eq
            rec["timestamps_equal"] = ts_eq
            rec["cowdb_folder_by_timestamp"] = by_left_ts.get(ts[0])
        rows.append(rec)
    out = {"n_cdb": len(cdb_tape), "n_values_equal": n_val,
           "mismatches": [r for r in rows if not r["values_equal"] or r.get("timestamps_equal") is False]}
    if cowdb_names:
        out["n_timestamps_equal"] = n_ts
    return out


def _f(x, nd=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return ""
    return "" if not np.isfinite(x) else round(x, nd)


# Post-hoc diagnostic, outside measure() and outside every headline number. Set after run v1:
# among measured animals the frontmost leg run had a median length of 1.24 m where the girth
# station fell far behind the trunk front, and 0.26-0.56 m elsewhere. Two front legs in stride
# span well under 0.6 m along the body.
DIAG_PARAMS = {"merged_run_m": 0.6}


def station_class(vr: dict) -> str:
    """Why measure()'s heart-girth station sits where it does, from its own leg runs.

    legs_not_separated: measure() found fewer than two leg runs and stopped.
    head_in_leg_band:   the frontmost run starts ahead of the trunk front, so it is the
                        lowered head or neck, and the station lands on the neck.
    legs_merged:        the frontmost run is longer than merged_run_m: front legs, hind legs
                        in stride and belly or sheath points fused into one run, so the
                        station lands near the hind legs.
    legs_separated:     the rule worked as designed.
    """
    m = vr.get("measure") or {}
    if not vr.get("ok"):
        return "legs_not_separated" if "legs" in str(vr.get("reason", "")) else f"failed_{vr.get('stage', '?')}"
    t1 = (vr.get("front") or {}).get("trunk_s_m", [np.nan, np.nan])[1]
    lo, hi = m["legs"]["front"]
    if lo > t1:
        return "head_in_leg_band"
    if hi - lo > DIAG_PARAMS["merged_run_m"]:
        return "legs_merged"
    return "legs_separated"


def flatten(cdb_id: int, cowdb_id, tape: dict, eligible: bool, excl: str, rec: dict | None, split: str | None = None) -> dict:
    tg, tl = tape.get("girth_cm", np.nan), tape.get("obl_cm", np.nan)
    tape = {"girth_cm": tg, "obl_cm": tl, "withers_cm": tape.get("withers_cm", np.nan),
            "weight_kg": tape.get("weight_kg", np.nan)}
    row = {"cdb_id": cdb_id, "cowdb_id": cowdb_id, "split": split or ("dev" if cdb_id <= 10 else "test"),
           "eligible": int(eligible), "exclusion": excl,
           "tape_girth_cm": _f(tape["girth_cm"], 1), "tape_obl_cm": _f(tape["obl_cm"], 1),
           "tape_withers_cm": _f(tape["withers_cm"], 1), "tape_weight_kg": _f(tape["weight_kg"], 1),
           "tape_schaeffer_kg": _f(tg ** 2 * tl / 10838.0, 1)}
    if not rec:
        return row
    row.update(floor_tilt_deg=rec.get("floor", {}).get("tilt_deg"), floor_rms_mm=rec.get("floor", {}).get("rms_mm"),
               n_crop=rec.get("n_crop"))
    for k, vf in rec.get("view_floor", {}).items():
        row[f"viewfloor_{k}_tilt_deg"] = vf.get("tilt_deg")
        row[f"viewfloor_{k}_offset_mm"] = vf.get("offset_mm")
    for vn, vr in rec.get("variants", {}).items():
        m = vr.get("measure", {}) or {}
        g = m.get("girth", {}) or {}
        fr = vr.get("front", {}) or {}
        dg = vr.get("diag", {}) or {}
        dgg = dg.get("girth", {}) or {}
        okv = bool(vr.get("ok"))
        G = m.get("heart_girth_m", np.nan) * 100 if okv else np.nan
        L = m.get("body_length_m", np.nan) * 100 if okv else np.nan
        WH = m.get("withers_height_m", np.nan) * 100 if okv else np.nan
        GH = g.get("hull_m", np.nan) * 100 if okv else np.nan
        GD = dgg.get("girth_m", np.nan) * 100 if dg.get("ok") else np.nan
        GDH = dgg.get("hull_m", np.nan) * 100 if dg.get("ok") else np.nan
        sk = G ** 2 * L / 10838.0 if np.isfinite(G) and np.isfinite(L) else np.nan
        row.update({
            f"{vn}_ok": int(okv), f"{vn}_stage": vr.get("stage", ""), f"{vn}_reason": vr.get("reason", ""),
            f"{vn}_n_points": vr.get("n_points"),
            f"{vn}_front_margin_m": fr.get("margin_m"), f"{vn}_front_confident": int(bool(fr.get("confident"))),
            f"{vn}_front_axis_x": _f(fr.get("axis", [np.nan])[0], 3),
            f"{vn}_front_auto_agrees": int(bool(fr.get("auto_agrees", True))) if fr.get("ok") else "",
            f"{vn}_trunk_front_m": _f(fr.get("trunk_s_m", [np.nan, np.nan])[1], 3),
            f"{vn}_front_run_lo_m": _f((m.get("legs") or {}).get("front", [np.nan, np.nan])[0], 3),
            f"{vn}_front_run_len_m": _f(np.subtract(*(m.get("legs") or {}).get("front", [np.nan, np.nan])[::-1]), 3),
            f"{vn}_station_class": station_class(vr) if fr.get("ok") else f"failed_{vr.get('stage', '?')}",
            f"{vn}_girth_cm": _f(G), f"{vn}_girth_hull_cm": _f(GH), f"{vn}_length_cm": _f(L), f"{vn}_withers_cm": _f(WH),
            f"{vn}_length_direct_cm": _f(m.get("length_direct_m", np.nan) * 100) if okv else "",
            f"{vn}_coverage": g.get("coverage", ""), f"{vn}_filled_bins": g.get("filled_bins", ""),
            f"{vn}_station_m": m.get("girth_station_m", ""),
            f"{vn}_station_in_trunk": int(bool(vr.get("station_in_trunk"))),
            f"{vn}_station_behind_trunk_front_m": vr.get("station_from_trunk_front_m", ""),
            f"{vn}_diag_girth_cm": _f(GD), f"{vn}_diag_girth_hull_cm": _f(GDH),
            f"{vn}_diag_coverage": dgg.get("coverage", ""),
            f"{vn}_schaeffer_kg": _f(sk, 1),
            f"{vn}_girth_err_cm": _f(G - tape["girth_cm"]), f"{vn}_length_err_cm": _f(L - tape["obl_cm"]),
            f"{vn}_withers_err_cm": _f(WH - tape["withers_cm"]), f"{vn}_diag_girth_err_cm": _f(GD - tape["girth_cm"]),
            f"{vn}_schaeffer_err_pct": _f((sk - tape["weight_kg"]) / tape["weight_kg"] * 100),
        })
    return row


def _col(rows, key):
    out = []
    for r in rows:
        v = r.get(key, "")
        out.append(float(v) if v not in ("", None) else np.nan)
    return np.array(out, float)


def score_set(rows: list[dict], variants, folds_seed: int = 0) -> dict:
    """All statistics for one set of eligible animals."""
    out = {"n_eligible": len(rows)}
    tw = _col(rows, "tape_weight_kg")
    tg, tl, twh = _col(rows, "tape_girth_cm"), _col(rows, "tape_obl_cm"), _col(rows, "tape_withers_cm")
    out["tape_schaeffer_all_eligible"] = weight_scores(tg ** 2 * tl / 10838.0, tw)
    te = np.isfinite(tg) & np.isfinite(tw)
    if te.sum() >= 20:
        folds = kfold_ids(int(te.sum()), 10, folds_seed)
        out["tape_cv10_loglog_girth_only_all_eligible"] = weight_scores(cv_loglog(tg[te], None, tw[te], folds), tw[te])
        if np.isfinite(tl[te]).all():
            out["tape_cv10_loglog_all_eligible"] = weight_scores(cv_loglog(tg[te], tl[te], tw[te], folds), tw[te])
    for vn in variants:
        ok = _col(rows, f"{vn}_ok") == 1
        G, GH, L, WH = (_col(rows, f"{vn}_{k}") for k in ("girth_cm", "girth_hull_cm", "length_cm", "withers_cm"))
        GD, GDH = _col(rows, f"{vn}_diag_girth_cm"), _col(rows, f"{vn}_diag_girth_hull_cm")
        stages = {}
        for r in rows:
            if r.get(f"{vn}_ok") != 1:
                key = f"{r.get(f'{vn}_stage') or 'load'}: {r.get(f'{vn}_reason') or r.get('load_error', '?')}"
                stages[key] = stages.get(key, 0) + 1
        inside = ok.copy()
        for arr, (lo, hi) in ((G, ENVELOPE["girth_cm"]), (L, ENVELOPE["length_cm"]), (WH, ENVELOPE["withers_cm"])):
            with np.errstate(invalid="ignore"):
                inside &= (arr >= lo) & (arr <= hi)
        cls = [r.get(f"{vn}_station_class", "") for r in rows]
        tax = {}
        for c in sorted(set(cls)):
            mk = np.array([x == c for x in cls]) & ok
            tax[c] = {"n": int(sum(x == c for x in cls)),
                      "girth_hull_vs_tape": agreement(GH[mk], tg[mk]) if mk.sum() >= 3 else {"n": int(mk.sum())},
                      "girth_ellipse_vs_tape": agreement(G[mk], tg[mk]) if mk.sum() >= 3 else {"n": int(mk.sum())}}
        v = {"n_attempted": len(rows), "n_measured": int(ok.sum()), "failures": stages,
             "n_inside_envelope": int(inside.sum()),
             "n_station_in_trunk": int((_col(rows, f"{vn}_station_in_trunk") == 1)[ok].sum()),
             "n_front_confident": int((_col(rows, f"{vn}_front_confident") == 1).sum()),
             "front_auto_disagrees_ids": [r["cdb_id"] for r in rows if r.get(f"{vn}_front_auto_agrees") == 0],
             "station_taxonomy": {"note": "DIAGNOSTIC: why measure()'s girth station sits where it does; see station_class()",
                                  "merged_run_m": DIAG_PARAMS["merged_run_m"], "classes": tax},
             "girth_ellipse_vs_tape": agreement(G[ok], tg[ok]),
             "girth_hull_vs_tape": agreement(GH[ok], tg[ok]),
             "length_vs_tape_obl": agreement(L[ok], tl[ok]),
             "withers_vs_tape": agreement(WH[ok], twh[ok]),
             "inside_envelope": {"girth_ellipse_vs_tape": agreement(G[inside], tg[inside]),
                                 "girth_hull_vs_tape": agreement(GH[inside], tg[inside]),
                                 "length_vs_tape_obl": agreement(L[inside], tl[inside]),
                                 "withers_vs_tape": agreement(WH[inside], twh[inside])}}
        dok = np.isfinite(GD)
        v["diag_girth"] = {"note": "DIAGNOSTIC outside measure(): girth_of_slice at a station "
                                   f"{PARAMS['diag_station_offset_m']} m behind the trunk front; offset fitted on animals 1-10",
                           "n": int(dok.sum()), "ellipse_vs_tape": agreement(GD[dok], tg[dok]),
                           "hull_vs_tape": agreement(GDH[dok], tg[dok])}
        # weight: Schaeffer on measured dims, beside Schaeffer on tape dims for the same animals
        wm = ok & np.isfinite(G) & np.isfinite(L) & (G > 0) & (L > 0) & np.isfinite(tw)
        has_tl = bool(np.isfinite(tl[wm]).all()) and wm.sum() > 0
        v["weight"] = {"schaeffer_measured": weight_scores(G[wm] ** 2 * L[wm] / 10838.0, tw[wm])}
        if has_tl:
            v["weight"]["schaeffer_tape_same_animals"] = weight_scores(tg[wm] ** 2 * tl[wm] / 10838.0, tw[wm])
        if wm.sum() >= 20:
            folds = kfold_ids(int(wm.sum()), 10, folds_seed)
            v["weight"]["cv10_loglog_measured"] = weight_scores(cv_loglog(G[wm], L[wm], tw[wm], folds), tw[wm])
            v["weight"]["cv10_loglog_measured_girth_only"] = weight_scores(cv_loglog(G[wm], None, tw[wm], folds), tw[wm])
            if has_tl:
                v["weight"]["cv10_loglog_tape_same_animals"] = weight_scores(cv_loglog(tg[wm], tl[wm], tw[wm], folds), tw[wm])
            if np.isfinite(tg[wm]).all():
                v["weight"]["cv10_loglog_tape_girth_only_same_animals"] = weight_scores(cv_loglog(tg[wm], None, tw[wm], folds), tw[wm])
            v["weight"]["cv_folds"] = (f"10 folds: numpy default_rng({folds_seed}).permutation over the {int(wm.sum())} "
                                       "animals in id order, dealt round-robin; same folds for measured and tape")
        wd = dok & wm
        v["weight_diag"] = {"note": "DIAGNOSTIC: diagnostic girth with measure() length",
                            "schaeffer": weight_scores(GD[wd] ** 2 * L[wd] / 10838.0, tw[wd]),
                            "schaeffer_tape_same_animals": weight_scores(tg[wd] ** 2 * tl[wd] / 10838.0, tw[wd])}
        if wd.sum() >= 20:
            folds = kfold_ids(int(wd.sum()), 10, folds_seed)
            v["weight_diag"]["cv10_loglog"] = weight_scores(cv_loglog(GD[wd], L[wd], tw[wd], folds), tw[wd])
            v["weight_diag"]["cv10_loglog_girth_only"] = weight_scores(cv_loglog(GD[wd], None, tw[wd], folds), tw[wd])
            if np.isfinite(tl[wd]).all():
                v["weight_diag"]["cv10_loglog_tape_same_animals"] = weight_scores(cv_loglog(tg[wd], tl[wd], tw[wd], folds), tw[wd])
        out[vn] = v
    return out


def front_audit_tile(Q: np.ndarray, rec: dict, label: str, size=(240, 120)) -> np.ndarray:
    """Small top-down render (colour = height) with the front call circled, for a visual audit."""
    import cv2
    W_, H_ = size
    img = np.full((H_, W_, 3), 255, np.uint8)
    B = Q[Q[:, 2] > 0.3]
    if len(B) == 0:
        return img
    c = B[:, :2].mean(0)
    sc = W_ / 3.2
    x = ((B[:, 0] - c[0]) * sc + W_ / 2).astype(int)
    y = ((B[:, 1] - c[1]) * sc + H_ / 2).astype(int)
    ok = (x >= 0) & (x < W_) & (y >= 0) & (y < H_)
    z = np.clip(B[ok, 2] / 1.4, 0, 1)
    img[y[ok], x[ok]] = np.stack([255 * (1 - z), 120 * np.ones_like(z), 255 * z], 1).astype(np.uint8)
    fr = rec.get("variants", {}).get("fused", {}).get("front", {})
    if fr.get("ok"):
        fx = int((fr["centre"][0] - c[0]) * sc + W_ / 2)
        fy = int((fr["centre"][1] - c[1]) * sc + H_ / 2)
        cv2.circle(img, (fx, fy), 6, (0, 0, 0), 2)
    cv2.putText(img, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return img


def main(argv=None) -> int:
    import argparse
    import time
    ap = argparse.ArgumentParser(description="Ruchay real-cattle benchmark for measure()")
    ap.add_argument("--cdb", required=True, help="CowDatabase checkout (AlignedClouds/, TopClouds/, Matrices/, Measurements.xlsx)")
    ap.add_argument("--cowdb-xlsx", required=True, help="CowDB Manual_measurements.xlsx")
    ap.add_argument("--cowdb-names", default=None, help="text file listing CowDB paths, for the timestamp check")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default="1-103")
    ap.add_argument("--source", choices=["aligned", "views"], default="aligned")
    ap.add_argument("--front", choices=["auto", "+x", "-x"], default="auto",
                    help="auto: overhang rule; +x/-x: recording-setup label for the head direction")
    a = ap.parse_args(argv)
    PARAMS["front_source"] = a.front
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = (int(x) for x in a.ids.split("-"))
    ids = list(range(lo, hi + 1))
    cdb = Path(a.cdb)
    cdb_tape = read_tape(cdb / "Measurements.xlsx")
    cowdb_tape = read_tape(a.cowdb_xlsx)
    names = Path(a.cowdb_names).read_text().split() if a.cowdb_names else None
    rowmap = verify_rowmap(cdb_tape, cowdb_tape, names)
    (out / "rowmap_check.json").write_text(json.dumps(rowmap, indent=1))
    print("row map:", {k: v for k, v in rowmap.items() if k != "mismatches"}, "mismatches:", rowmap["mismatches"], flush=True)

    rows, failures, tiles = [], [], []
    jl = open(out / "animals.jsonl", "w")
    variants = ("fused", "no_top")
    for i in ids:
        t0 = time.time()
        j = premap(i)
        tape = cowdb_tape.get(j, {})
        excl = ""
        if i in PREREG_DROP_CDB:
            folder = next((r.get("cowdb_folder_by_timestamp") for r in rowmap["mismatches"] if r["cdb"] == i), None)
            excl = "pre-registered drop: pairing mismatch" + (f" (cloud timestamp matches CowDB folder {folder})" if folder else "")
        else:
            ok_t, why = tape_ok(tape)
            if not ok_t:
                excl = "pre-registered tape filter: " + why
        eligible = not excl
        rec = None
        try:
            mpath = cdb / "Matrices" / f"{i}.matrices"
            mats = parse_matrices(mpath.read_text()) if mpath.exists() else None
            if a.source == "aligned":
                xyz, _ = read_ply(cdb / "AlignedClouds" / f"{i}_AutoAligned.ply")
            else:
                xyz = fuse_views(read_ply(cdb / "LeftClouds" / f"{i}.ply")[0], read_ply(cdb / "RightClouds" / f"{i}.ply")[0],
                                 read_ply(cdb / "TopClouds" / f"{i}.ply")[0], mats)
            P, v, info = split_aligned(xyz, mats)
            top_file = cdb / "TopClouds" / f"{i}.ply"
            if top_file.exists():
                T, _ = read_ply(top_file)
                Tv = T[np.abs(T).sum(1) > 0]
                Pt = P[v == 1]
                info["top_block_equals_TopClouds"] = bool(len(Tv) == len(Pt) and np.allclose(Tv, Pt, atol=1e-4))
            rec, prep = run_animal(P, v, np.array(info["sentinels"]))
            rec["split_info"] = info
            tiles.append(front_audit_tile(prep["Q"], rec, f"{i}"))
        except Exception as e:  # keep going; the failure is recorded
            import traceback
            failures.append({"cdb_id": i, "variant": "all", "stage": "load/prepare", "reason": f"{type(e).__name__}: {e}",
                             "eligible": eligible})
            traceback.print_exc()
        row = flatten(i, j, tape, eligible, excl, rec)
        if rec is None:
            row["load_error"] = failures[-1]["reason"]
        else:
            for vn in variants:
                vr = rec["variants"][vn]
                if not vr.get("ok"):
                    failures.append({"cdb_id": i, "variant": vn, "stage": vr.get("stage"), "reason": vr.get("reason"),
                                     "eligible": eligible})
        if excl:
            failures.append({"cdb_id": i, "variant": "all", "stage": "exclusion", "reason": excl, "eligible": False})
        rows.append(row)
        jl.write(json.dumps({"cdb_id": i, "cowdb_id": j, "eligible": eligible, "exclusion": excl, "rec": rec},
                            default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        jl.flush()
        print(f"[{i}] {time.time() - t0:.1f}s eligible={eligible} " + " ".join(
            f"{vn}: ok={row.get(f'{vn}_ok')} G={row.get(f'{vn}_girth_cm')} L={row.get(f'{vn}_length_cm')} "
            f"WH={row.get(f'{vn}_withers_cm')} Gd={row.get(f'{vn}_diag_girth_cm')}" for vn in variants)
              + f" | tape G={tape.get('girth_cm')} L={tape.get('obl_cm')} WH={tape.get('withers_cm')}", flush=True)
    jl.close()

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out / "per_animal.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (out / "failures.json").write_text(json.dumps(failures, indent=1))

    elig = [r for r in rows if r["eligible"] == 1]
    sets = {"all": elig, "test_11plus": [r for r in elig if r["cdb_id"] >= 11],
            "dev_1to10": [r for r in elig if r["cdb_id"] <= 10]}
    ks = sorted(cowdb_tape)
    allw = np.array([cowdb_tape[k]["weight_kg"] for k in ks])
    allg = np.array([cowdb_tape[k]["girth_cm"] for k in ks])
    alll = np.array([cowdb_tape[k]["obl_cm"] for k in ks])
    clean = np.array([tape_ok(cowdb_tape[k])[0] for k in ks])
    summary = {
        "what": "mombe.body.measure.measure() unchanged on Ruchay CowDatabase multi-Kinect clouds vs CowDB tape and live weight",
        "labels": {"measurements_vs_tape": "R/P (real capture, physical tape or stick truth)",
                   "weight": "R/L in the ledger sense: real capture; live weight with weighing method unstated in the source"},
        "params": {k: list(v) if isinstance(v, tuple) else v for k, v in PARAMS.items()},
        "envelope": {k: list(v) for k, v in ENVELOPE.items()},
        "source": a.source,
        "row_map": {k: v for k, v in rowmap.items() if k != "mismatches"},
        "row_map_mismatches": rowmap["mismatches"],
        "n": {"cdb_animals": len(ids), "excluded": sum(1 for r in rows if not r["eligible"]),
              "eligible": len(elig), "load_failures": sum(1 for r in rows if "load_error" in r),
              "eligible_test_11plus": len(sets["test_11plus"]), "eligible_dev_1to10": len(sets["dev_1to10"])},
        "tape_only_schaeffer_cowdb": {"n_rows": len(ks), "all": weight_scores(allg ** 2 * alll / 10838.0, allw),
                                      "prereg_clean": weight_scores(allg[clean] ** 2 * alll[clean] / 10838.0, allw[clean])},
        "published_baseline": {"bao2023_heart_girth_mape_pct": {"pose_normalised": 5.21, "raw": 5.59},
                               "source": "Bao et al. 2023, ISPRS Annals X-1-W1-2023, same 103 animals"},
        "sets": {name: score_set(rs, variants) for name, rs in sets.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    write_ba(out, elig, variants)
    write_audit(out, tiles)
    print(json.dumps({k: summary[k] for k in ("n", "row_map")}, indent=1))
    return 0


def write_ba(out: Path, elig: list[dict], variants) -> None:
    """Bland-Altman arrays (mean, difference, animal id) per variant and measurement, for a figure."""
    ba = {}
    for vn in variants:
        okr = [r for r in elig if r.get(f"{vn}_ok") == 1]
        ba[vn] = {}
        for name, mk, tk in (("girth_ellipse", f"{vn}_girth_cm", "tape_girth_cm"),
                             ("girth_hull", f"{vn}_girth_hull_cm", "tape_girth_cm"),
                             ("length", f"{vn}_length_cm", "tape_obl_cm"), ("withers", f"{vn}_withers_cm", "tape_withers_cm")):
            mv, tv = _col(okr, mk), _col(okr, tk)
            d = bland_altman_arrays(mv, tv)
            d["cdb_id"] = [r["cdb_id"] for r, x, y in zip(okr, mv, tv) if np.isfinite(x) and np.isfinite(y)]
            d.update({k: v for k, v in agreement(mv, tv).items() if k in ("n", "bias", "loa95")})
            ba[vn][name] = d
        dr = [r for r in elig if r.get(f"{vn}_diag_girth_cm") not in ("", None)]
        mv, tv = _col(dr, f"{vn}_diag_girth_cm"), _col(dr, "tape_girth_cm")
        d = bland_altman_arrays(mv, tv)
        d["cdb_id"] = [r["cdb_id"] for r in dr]
        d.update({k: v for k, v in agreement(mv, tv).items() if k in ("n", "bias", "loa95")})
        ba[vn]["diag_girth_ellipse"] = d
    (out / "bland_altman.json").write_text(json.dumps(ba))


def write_audit(out: Path, tiles: list) -> None:
    """All top-down audit tiles, ten per row, with each front call circled."""
    if not tiles:
        return
    import cv2
    per = 10
    blank = np.full_like(tiles[0], 255)
    rows_img = [np.hstack(tiles[k:k + per] + [blank] * (per - len(tiles[k:k + per]))) for k in range(0, len(tiles), per)]
    cv2.imwrite(str(out / "front_audit.png"), np.vstack(rows_img))


# ---------------------------------------------------------------- CowDatabase2 (Black Angus), zero re-tuning
CDB2_COLS = {"frame number": "frame", "live weigth": "weight_kg", "withers height": "withers_cm",
             "hip height": "hip_height_cm", "chest width": "chest_width_cm", "chest heart": "girth_cm"}


def read_tape_cdb2(path) -> dict[int, dict]:
    """CowDatabase2 Database2022v2.xlsx, keyed by 'Frame number' (the cloud folder's name).

    'Chest heart' is read as heart girth: its values (2xx cm for 550-750 kg Angus) fit no
    other girth or depth. The sheet does not state how the animals were measured.
    """
    rows = read_xlsx_rows(path)
    hdr = [str(h).strip().lower() if h is not None else "" for h in rows[0]]
    idx = {}
    for i, h in enumerate(hdr):
        if h in CDB2_COLS and CDB2_COLS[h] not in idx:
            idx[CDB2_COLS[h]] = i
    out = {}
    fi = idx["frame"]
    for r in rows[1:]:
        if not r or not isinstance(r[0], (int, float)) or fi >= len(r) or not isinstance(r[fi], (int, float)):
            continue
        rec = {k: float(r[i]) for k, i in idx.items() if k != "frame" and i < len(r) and isinstance(r[i], (int, float))}
        rec["row_no"] = int(r[0])
        out[int(r[fi])] = rec
    return out


def main_cdb2(argv=None) -> int:
    """measure() on CowDatabase2 with every parameter frozen from CowDatabase animals 1-10."""
    import argparse
    import time
    ap = argparse.ArgumentParser(description="CowDatabase2 replication for measure()")
    ap.add_argument("--cdb2", required=True, help="directory holding <frame>/aligned.ply")
    ap.add_argument("--xlsx", required=True, help="CowDatabase2 Database2022v2.xlsx")
    ap.add_argument("--out", required=True)
    ap.add_argument("--front", choices=["auto", "+x", "-x"], default="auto")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    PARAMS["front_source"] = a.front
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tape = read_tape_cdb2(a.xlsx)
    root = Path(a.cdb2)
    folders = sorted(int(q.name) for q in root.iterdir() if q.name.isdigit() and (q / "aligned.ply").exists())
    frames = sorted(tape)[: a.limit] if a.limit else sorted(tape)
    rows, failures, tiles = [], [], []
    jl = open(out / "animals.jsonl", "w")
    variants = ("fused", "no_top")
    for i in frames:
        t0 = time.time()
        tp = tape[i]
        excl = "" if ("weight_kg" in tp and "girth_cm" in tp) else "sheet row lacks live weight or girth"
        if not excl and i not in folders:
            excl = "no cloud folder for this frame number"
        eligible = not excl
        rec = None
        if i in folders:
            try:
                xyz, _ = read_ply(root / str(i) / "aligned.ply")
                P, v, info = split_aligned(xyz, None)
                rec, prep = run_animal(P, v, np.array(info["sentinels"]))
                rec["split_info"] = info
                tiles.append(front_audit_tile(prep["Q"], rec, f"{i}"))
            except Exception as e:
                import traceback
                failures.append({"cdb_id": i, "variant": "all", "stage": "load/prepare", "reason": f"{type(e).__name__}: {e}",
                                 "eligible": eligible})
                traceback.print_exc()
        row = flatten(i, tp.get("row_no"), tp, eligible, excl, rec, split="untouched")
        if rec is None and i in folders:
            row["load_error"] = failures[-1]["reason"]
        elif rec is not None:
            for vn in variants:
                vr = rec["variants"][vn]
                if not vr.get("ok"):
                    failures.append({"cdb_id": i, "variant": vn, "stage": vr.get("stage"), "reason": vr.get("reason"),
                                     "eligible": eligible})
        if excl:
            failures.append({"cdb_id": i, "variant": "all", "stage": "exclusion", "reason": excl, "eligible": False})
        rows.append(row)
        jl.write(json.dumps({"frame": i, "eligible": eligible, "exclusion": excl, "rec": rec},
                            default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        jl.flush()
        print(f"[cdb2 {i}] {time.time() - t0:.1f}s eligible={eligible} " + " ".join(
            f"{vn}: ok={row.get(f'{vn}_ok')} G={row.get(f'{vn}_girth_cm')} L={row.get(f'{vn}_length_cm')} "
            f"WH={row.get(f'{vn}_withers_cm')}" for vn in variants)
              + f" | sheet G={tp.get('girth_cm')} WH={tp.get('withers_cm')} W={tp.get('weight_kg')}", flush=True)
    jl.close()
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out / "per_animal.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (out / "failures.json").write_text(json.dumps(failures, indent=1))
    elig = [r for r in rows if r["eligible"] == 1]
    summary = {
        "what": "mombe.body.measure.measure() unchanged on CowDatabase2 (Black Angus, Voronezh) aligned multi-Kinect "
                "clouds vs the database sheet; every preprocessing parameter frozen from CowDatabase animals 1-10",
        "labels": {"measurements_vs_sheet": "R/P assumed: real capture; the sheet does not state the measuring method",
                   "weight": "R/P assumed: real capture; live weight, scale not described in the source"},
        "params": {k: list(v) if isinstance(v, tuple) else v for k, v in PARAMS.items()},
        "mapping": {"rule": "cloud folder name = sheet 'Frame number'",
                    "check": "every sheet frame names an existing folder; folders 10 and 11 are absent from the "
                             "repository and the sheet skips both; folder 1 is a calibration capture of a person",
                    "folders_present": len(folders), "sheet_rows": len(tape),
                    "sheet_frames_without_folder": [i for i in sorted(tape) if i not in folders],
                    "folders_without_sheet_row": [i for i in folders if i not in tape]},
        "n": {"sheet_rows": len(tape), "attempted": len(frames), "eligible": len(elig),
              "load_failures": sum(1 for r in rows if "load_error" in r)},
        "sets": {"all_untouched": score_set(elig, variants)},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    write_ba(out, elig, variants)
    write_audit(out, tiles)
    print(json.dumps({k: summary[k] for k in ("n", "mapping")}, indent=1))
    return 0


def null_baselines(out_dir) -> dict:
    """What a constant guess achieves on the same animals, from per_animal.csv alone.

    Girth and withers: the mean tape value of the other animals (leave one out).
    Weight: the training-fold mean weight, with the same 10 folds as the log-log fits.
    A method that cannot beat these carries no information about the individual animal.
    """
    out_dir = Path(out_dir)
    rows = list(csv.DictReader(open(out_dir / "per_animal.csv")))
    for r in rows:
        for k in list(r):
            if k.endswith("_ok") or k in ("eligible", "cdb_id"):
                r[k] = int(r[k]) if r[k] not in ("", None) else ""
    elig = [r for r in rows if r["eligible"] == 1]
    sets = {"all": elig}
    if any(r["split"] == "test" for r in elig):
        sets["test_11plus"] = [r for r in elig if r["cdb_id"] >= 11]
    res = {"note": "constant-guess references on the same animals as each variant's measured set"}
    for name, rs in sets.items():
        res[name] = {}
        for vn in ("fused", "no_top"):
            ok = _col(rs, f"{vn}_ok") == 1
            d = {}
            for key, tk in (("girth", "tape_girth_cm"), ("withers", "tape_withers_cm")):
                t = _col(rs, tk)[ok]
                t = t[np.isfinite(t)]
                if len(t) >= 3:
                    loo = (t.sum() - t) / (len(t) - 1)
                    d[f"{key}_loo_mean_mape_pct"] = round(float(np.mean(np.abs(loo - t) / t) * 100), 3)
                    d[f"{key}_loo_mean_mae"] = round(float(np.mean(np.abs(loo - t))), 3)
            G, L, tw = _col(rs, f"{vn}_girth_cm"), _col(rs, f"{vn}_length_cm"), _col(rs, "tape_weight_kg")
            wm = ok & np.isfinite(G) & np.isfinite(L) & (G > 0) & (L > 0) & np.isfinite(tw)
            if wm.sum() >= 20:
                w = tw[wm]
                folds = kfold_ids(int(wm.sum()), 10, 0)
                pred = np.array([w[folds != f].mean() for f in folds])
                d["weight_cv10_mean"] = weight_scores(pred, w)
            d["n_measured"] = int(ok.sum())
            res[name][vn] = d
    return res


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "nulls":
        s = json.loads((Path(sys.argv[2]) / "summary.json").read_text())
        s["null_baselines"] = null_baselines(sys.argv[2])
        (Path(sys.argv[2]) / "summary.json").write_text(json.dumps(s, indent=1))
        print(json.dumps(s["null_baselines"], indent=1))
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "cdb2":
        sys.exit(main_cdb2(sys.argv[2:]))
    sys.exit(main())
