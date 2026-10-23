"""Body measurements, version 2: robust to chute clutter, merged legs and lowered heads.

Version 1 (measure.py) took every height as a fraction of the cloud's highest
point and put the tape 6 cm behind the frontmost run of "leg" points. On real
chute captures (Ruchay CowDatabase, 103 Hereford) the highest point is a chute
bar, so the leg band reached into the trunk, the legs merged into one run and
the station drifted to the hind legs or onto a lowered neck.

Version 2:

1. clean: keep the largest solid blob in the top-down and the side
   silhouettes after a morphological opening. Thin bars and seams vanish;
   legs, trunk, neck and head stay.
2. back height zb: the median top of the body over the middle half of its
   length. Every height below is a fraction of zb.
3. body axis: principal direction of the top-down footprint cells of the body
   (each cell counts once, so dense flanks do not pull it), refined on the trunk.
4. trunk: the longest run of stations whose vertical extent above the leg zone
   is at least TRUNK_FRAC of the largest. A lowered head and the neck fall out.
5. station: the mean position of forearm-height points in the front half of the
   trunk, plus GIRTH_FROM_FOREARM_M. On walking chute cattle the forelegs swing
   fore and aft, and the tape line sits over the top of the forward foreleg.
   Without forearm points the station sits GIRTH_FROM_TRUNK_FRONT_M from the
   trunk front.
6. girth: girth_of_slice() from version 1 on three cleaned slices around the
   station (detached blobs and points off a fitted ellipse dropped), median of
   each statistic. The ellipse is the girth.
6b. barrel: the largest hull girth over the middle 60% of the trunk, and
   BARREL_TO_HEART times it as a second, herd-calibrated heart-girth estimate.
7. length: point of shoulder (SHOULDER_FROM_GIRTH_M ahead of the station) to
   the rear surface at pin height, oblique over the height difference.
8. withers: the highest station top over the front half of the trunk.

Every fixed offset and ratio was chosen on Ruchay CowDatabase animals 1-30
only (eval/out/ruchay/v2_dev_log.md). They fit walking chute Herefords seen by
three Kinect v2 cameras; the synthetic sweep animals do not share that anatomy.

The return schema matches measure().
"""

from __future__ import annotations

import math

import numpy as np

from mombe.body.measure import _largest_cluster, drop_boards, girth_of_slice

CELL_M = 0.02            # projection grid
OPEN_CELLS = 3           # opening radius in cells: structures thinner than ~2 x 6 cm vanish
STEP_M = 0.02            # station spacing along the body
LEG_CUT = 0.40           # below this fraction of zb a point may be leg
TRUNK_FRAC = 0.80        # trunk stations: vertical extent at least this fraction of the largest
LAT_SPLIT_M = 0.02       # forearm points closer than this to the midline are dropped (dewlap, sheath)
FOREARM_BAND = (0.32, 0.42)     # forearm heights, fraction of zb
GIRTH_FROM_FOREARM_M = 0.12     # station = forearm centroid + this (chosen on Ruchay dev 1-30)
GIRTH_FROM_TRUNK_FRONT_M = -0.14  # fallback station relative to the trunk front (chosen on dev)
BARREL_TO_HEART = 0.826         # tape heart girth / barrel hull, Ruchay dev 1-30 mean (herd-specific)
SLICE_HALF_M = 0.015
SLICE_SPREAD_M = 0.02    # three slices at -spread, 0, +spread
SHOULDER_FRAC = 0.62     # point of shoulder height, fraction of zb
SHOULDER_FROM_GIRTH_M = 0.05  # point of shoulder ahead of the girth station (chosen on dev for OBL)
PIN_FRAC = 0.85          # pin bone height, fraction of zb


def _disk(r: int) -> np.ndarray:
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r + 0.5


def _solid_mask(u: np.ndarray, v: np.ndarray, cell: float = CELL_M, open_r: int = OPEN_CELLS,
                min_pts: int = 2) -> np.ndarray:
    """Per point: does it fall in the largest solid blob of the (u, v) silhouette after opening?"""
    from scipy import ndimage
    pad = open_r + 3
    iu = np.floor((u - u.min()) / cell).astype(int) + pad
    iv = np.floor((v - v.min()) / cell).astype(int) + pad
    H = np.zeros((iu.max() + pad + 1, iv.max() + pad + 1), np.int32)
    np.add.at(H, (iu, iv), 1)
    occ = ndimage.binary_closing(H >= min_pts, structure=_disk(1))
    op = ndimage.binary_opening(occ, structure=_disk(open_r))
    lab, n = ndimage.label(op)
    if n == 0:
        return np.ones(len(u), bool)
    sizes = ndimage.sum(op, lab, range(1, n + 1))
    keep = ndimage.binary_dilation(lab == 1 + int(np.argmax(sizes)), structure=_disk(1))
    return keep[iu, iv]


def _axis_from_cells(xy: np.ndarray, cell: float = CELL_M) -> tuple[np.ndarray, np.ndarray]:
    q = np.unique(np.floor(xy / cell).astype(np.int64), axis=0)
    C = (q + 0.5) * cell
    c = C.mean(0)
    w, v = np.linalg.eigh(np.cov((C - c).T))
    return c, v[:, -1]


def _runs(mask: np.ndarray, bridge: int = 1) -> list[tuple[int, int]]:
    """Index runs of True, bridging gaps of up to `bridge` False entries."""
    m = mask.copy()
    if bridge > 0:
        idx = np.flatnonzero(m)
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= bridge + 1:
                m[a:b] = True
    out, start = [], None
    for i, x in enumerate(m):
        if x and start is None:
            start = i
        if (not x or i == len(m) - 1) and start is not None:
            out.append((start, i if x else i - 1))
            start = None
    return out


def _station_stats(s: np.ndarray, z: np.ndarray, stations: np.ndarray, fn, min_pts: int = 10) -> np.ndarray:
    order = np.argsort(s)
    ss, zz = s[order], z[order]
    lo = np.searchsorted(ss, stations - STEP_M / 2)
    hi = np.searchsorted(ss, stations + STEP_M / 2)
    out = np.full(len(stations), np.nan)
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b - a >= min_pts:
            out[i] = fn(zz[a:b])
    return out


def _clean_slice(lat: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Keep the cross-section's main loop: drop small detached blobs (bars, stray points)."""
    from scipy import ndimage
    cell = 0.02
    pad = 3
    iu = np.floor((lat - lat.min()) / cell).astype(int) + pad
    iv = np.floor((z - z.min()) / cell).astype(int) + pad
    occ = np.zeros((iu.max() + pad + 1, iv.max() + pad + 1), bool)
    occ[iu, iv] = True
    grown = ndimage.binary_dilation(occ, structure=_disk(2))
    lab, n = ndimage.label(grown)
    if n <= 1:
        return np.ones(len(lat), bool)
    sizes = np.bincount(lab[iu, iv], minlength=n + 1)
    sizes[0] = 0
    return lab[iu, iv] == int(np.argmax(sizes))


def _slice_keep(lat: np.ndarray, z: np.ndarray, tol: float = 0.06, iters: int = 3) -> np.ndarray:
    """Points of the girth loop: detached blobs dropped, then points far from a fitted ellipse dropped.

    A chute bar touching the hide or the top of a leg leaves a branch off the loop;
    the ellipse through the per-angle median profile ignores it, and the branch's
    points sit well off that ellipse.
    """
    import cv2
    keep = _clean_slice(lat, z)
    pts = np.c_[lat, z].astype(np.float64)
    for _ in range(iters):
        q = pts[keep]
        if len(q) < 12:
            break
        c = np.median(q, axis=0)
        rel = q - c
        ang = np.arctan2(rel[:, 1], rel[:, 0])
        rad = np.hypot(rel[:, 0], rel[:, 1])
        nb = 72
        idx = np.floor((ang + np.pi) / (2 * np.pi) * nb).astype(int) % nb
        prof = np.array([np.median(rad[idx == b]) if (idx == b).sum() >= 3 else np.nan for b in range(nb)])
        ok = np.isfinite(prof)
        if ok.sum() < 8:
            break
        cen = (np.arange(nb) + 0.5) / nb * 2 * np.pi - np.pi
        loop = np.c_[c[0] + prof[ok] * np.cos(cen[ok]), c[1] + prof[ok] * np.sin(cen[ok])].astype(np.float32)
        (ex, ez), (w, h), rot = cv2.fitEllipse(loop)
        th = math.radians(rot)
        d = pts - np.array([ex, ez])
        u = d[:, 0] * math.cos(th) + d[:, 1] * math.sin(th)
        v = -d[:, 0] * math.sin(th) + d[:, 1] * math.cos(th)
        r = np.hypot(u, v)
        phi = np.arctan2(v, u)
        a2, b2 = w / 2, h / 2
        r_e = a2 * b2 / np.sqrt((b2 * np.cos(phi)) ** 2 + (a2 * np.sin(phi)) ** 2)
        new = keep & (np.abs(r - r_e) < tol)
        if new.sum() == keep.sum():
            break
        keep = new
    return keep


def measure_v2(points: np.ndarray, boards: list[dict], debug: bool = False) -> dict:
    P = drop_boards(points[points[:, 2] > 0.05], boards)
    P = _largest_cluster(P)
    if len(P) < 500:
        return {"ok": False, "reason": "too little surface", "points": len(P)}

    # 1. clean in the top-down silhouette (levelled frame, no axis needed yet)
    P = P[_solid_mask(P[:, 0], P[:, 1])]
    if len(P) < 500:
        return {"ok": False, "reason": "too little surface after cleaning", "points": len(P)}
    z99 = float(np.percentile(P[:, 2], 99.5))

    # 3. axis: footprint cells of the upper body, then refined on the trunk below
    c, ax = _axis_from_cells(P[P[:, 2] > 0.5 * z99, :2])
    if boards:
        front = np.mean([np.asarray(b["centre"])[:2] for b in boards], axis=0)
        if (front - c) @ ax < 0:
            ax = -ax
    lat_dir = np.array([-ax[1], ax[0]])
    s = (P[:, :2] - c) @ ax

    # 1b. clean in the side silhouette
    keep = _solid_mask(s, P[:, 2])
    P, s = P[keep], s[keep]
    z = P[:, 2]

    # 2. back height
    stations = np.arange(np.percentile(s, 0.5), np.percentile(s, 99.5) + STEP_M, STEP_M)
    ztop = _station_stats(s, z, stations, lambda a: np.percentile(a, 98))
    mid = (stations > np.percentile(s, 25)) & (stations < np.percentile(s, 75)) & np.isfinite(ztop)
    zb = float(np.median(ztop[mid])) if mid.sum() >= 3 else z99

    # 4. trunk: vertical extent above the leg zone
    upper = z > LEG_CUT * zb
    zlo = _station_stats(s[upper], z[upper], stations, lambda a: np.percentile(a, 2))
    ext = ztop - zlo
    ext_ok = np.isfinite(ext)
    if ext_ok.sum() < 10:
        return {"ok": False, "reason": "too few stations", "points": len(P)}
    emax = float(np.percentile(ext[ext_ok], 95))
    wide = ext_ok & (ext >= TRUNK_FRAC * emax)
    runs = _runs(wide, bridge=2)
    if not runs:
        return {"ok": False, "reason": "no trunk", "points": len(P)}
    r0, r1 = max(runs, key=lambda r: r[1] - r[0])
    t_rear, t_front = float(stations[r0] - STEP_M / 2), float(stations[r1] + STEP_M / 2)

    # refine the axis on the trunk footprint and recompute the frame
    tm = (s >= t_rear) & (s <= t_front) & (z > 0.5 * zb)
    if tm.sum() > 200:
        c2, ax2 = _axis_from_cells(P[tm, :2])
        if ax2 @ ax < 0:
            ax2 = -ax2
        if abs(float(ax2 @ ax)) > math.cos(math.radians(30)):
            c, ax = c2, ax2
            lat_dir = np.array([-ax[1], ax[0]])
            s = (P[:, :2] - c) @ ax
            # carry the trunk ends into the new frame through the trunk's own points
            t_rear = float(np.percentile(s[tm], 0.5))
            t_front = float(np.percentile(s[tm], 99.5))
    lat = (P[:, :2] - c) @ lat_dir

    # 5. station: the front legs' forearms, then a fixed step forward to where the tape sits
    mid_s = 0.5 * (t_rear + t_front)
    fa = ((z > FOREARM_BAND[0] * zb) & (z < FOREARM_BAND[1] * zb) & (np.abs(lat) > LAT_SPLIT_M)
          & (s > mid_s + 0.1) & (s < t_front + 0.05))
    legs = {"n_forearm": int(fa.sum())}
    if fa.sum() >= 20:
        s_fa = float(np.mean(s[fa]))
        s_girth = s_fa + GIRTH_FROM_FOREARM_M
        method = "forearm_centroid"
        legs["front"] = [round(float(np.percentile(s[fa], 5)), 3), round(float(np.percentile(s[fa], 95)), 3)]
        legs["forearm_centroid_m"] = round(s_fa, 3)
    else:
        s_fa = None
        s_girth = t_front + GIRTH_FROM_TRUNK_FRONT_M
        method = "trunk_front"
    if not (mid_s < s_girth < t_front):
        s_girth = t_front + GIRTH_FROM_TRUNK_FRONT_M
        method = "trunk_front"
    rear_legs = (z > FOREARM_BAND[0] * zb) & (z < FOREARM_BAND[1] * zb) & (s < mid_s - 0.1) & (s > t_rear - 0.05)
    if rear_legs.sum() >= 20:
        legs["rear"] = [round(float(np.percentile(s[rear_legs], 5)), 3), round(float(np.percentile(s[rear_legs], 95)), 3)]

    # 6. girth: three cleaned slices around the station, median of each statistic
    zcut = LEG_CUT * zb

    def slice_girth(x: float) -> dict | None:
        sl = (np.abs(s - x) < SLICE_HALF_M) & (z > zcut)
        if sl.sum() < 30:
            return None
        k = _slice_keep(lat[sl], z[sl])
        if k.sum() < 30:
            return None
        g_ = girth_of_slice(lat[sl][k], z[sl][k])
        return g_ if np.isfinite(g_.get("girth_m", np.nan)) else None

    gs = [g_ for g_ in (slice_girth(s_girth + d) for d in (-SLICE_SPREAD_M, 0.0, SLICE_SPREAD_M)) if g_]
    if not gs:
        return {"ok": False, "reason": "girth slice empty", "points": int(len(P))}
    g = sorted(gs, key=lambda g_: g_["ellipse_m"])[len(gs) // 2]
    g = dict(g, slices=len(gs), hull_m=float(np.median([x["hull_m"] for x in gs])),
             ellipse_m=float(np.median([x["ellipse_m"] for x in gs])),
             hull_slices_m=[x["hull_m"] for x in gs])
    g["girth_m"] = g["ellipse_m"]

    # 6b. barrel: the largest trunk circumference over the middle of the trunk (hull, 3-station median)
    L_t = t_front - t_rear
    bst = np.arange(t_rear + 0.2 * L_t, t_front - 0.2 * L_t, STEP_M)
    bh = np.array([(lambda r: r["hull_m"] if r else np.nan)(slice_girth(x)) for x in bst])
    barrel = float("nan")
    if np.isfinite(bh).sum() >= 5:
        okb = np.isfinite(bh)
        bh_s = np.interp(bst, bst[okb], bh[okb])
        bh_m = np.array([np.median(bh_s[max(0, i - 1):i + 2]) for i in range(len(bh_s))])
        barrel = float(bh_m.max())

    # 7. length: point of shoulder (a fixed step ahead of the girth station) to the rear at pin height, oblique
    zs, zp = SHOULDER_FRAC * zb, PIN_FRAC * zb
    s_sh = s_girth + SHOULDER_FROM_GIRTH_M
    rb_ = (np.abs(z - zp) < 0.05 * zb) & (s < s_girth) & (np.abs(lat) < 0.2)
    s_pin = float(np.percentile(s[rb_], 1)) if rb_.sum() >= 10 else t_rear
    horiz = s_sh - s_pin
    length = math.hypot(horiz, zp - zs)

    # 8. withers: the highest station top over the front half of the trunk
    # the station tops above were taken in the first frame; recompute them in the refined one
    st2 = np.arange(np.percentile(s, 0.5), np.percentile(s, 99.5) + STEP_M, STEP_M)
    zt2 = _station_stats(s, z, st2, lambda a_: np.percentile(a_, 98))
    wf = np.isfinite(zt2) & (st2 > mid_s) & (st2 < t_front - 0.05)
    withers = float(np.max(zt2[wf])) if wf.sum() >= 3 else float("nan")

    out = {"ok": True, "heart_girth_m": g["girth_m"], "girth": g, "body_length_m": round(length, 4),
           "withers_height_m": round(withers, 4), "length_direct_m": round(horiz, 4),
           "legs": {"front": legs.get("front"), "rear": legs.get("rear")},
           "girth_station_m": round(float(s_girth), 3), "points": int(len(P)),
           "axis": [round(float(ax[0]), 4), round(float(ax[1]), 4)], "centre_m": [round(float(c[0]), 3), round(float(c[1]), 3)],
           "version": 2, "station_method": method, "back_height_m": round(zb, 4),
           "trunk_m": [round(t_rear, 3), round(t_front, 3)], "shoulder_s_m": round(s_sh, 3), "pin_s_m": round(s_pin, 3),
           "barrel_girth_m": round(barrel, 4), "heart_girth_from_barrel_m": round(BARREL_TO_HEART * barrel, 4)}
    if debug:
        sl = (np.abs(s - s_girth) < SLICE_HALF_M) & (z > zcut)
        out["debug"] = {"legs": legs, "z99": z99, "stations": stations, "ztop": ztop, "zlo": zlo, "emax": emax,
                        "s_elbow": s_fa, "s": s, "z": z, "lat": lat, "slice": sl, "zcut": zcut}
    return out
