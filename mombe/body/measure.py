"""Body measurements from the fused metric surface.

The tape measure is the reference every livestock officer trusts, so the
measurements follow it:

* heart girth: the loop a tape makes just behind the front legs. A tape
  bridges hollows, so the slice's convex hull is the physical analogue; where
  the cameras never saw the underside, an ellipse fitted to the seen arc
  closes the loop instead.
* body length: point of shoulder to pin bone, read along the body axis.
* height at the withers.

The animal's front is where the board was held: the protocol puts it on the
shoulders.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

LEG_BAND = (0.12, 0.45)     # fraction of withers height that is leg, not body
SLICE_HALF_M = 0.015
TAPE_OFFSET_M = 0.06        # tape sits this far behind the front legs


def _largest_cluster(P: np.ndarray, voxel: float = 0.04) -> np.ndarray:
    """Keep the largest 26-connected blob of occupied voxels: the animal, not clutter."""
    from scipy import ndimage
    if len(P) == 0:
        return P
    q = np.floor((P - P.min(0)) / voxel).astype(int)
    grid = np.zeros(q.max(0) + 1, bool)
    grid[tuple(q.T)] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    if n == 0:
        return P
    sizes = ndimage.sum(grid, lab, range(1, n + 1))
    keep = lab[tuple(q.T)] == (1 + int(np.argmax(sizes)))
    return P[keep]


def ellipse_perimeter(a: float, b: float) -> float:
    h = ((a - b) / (a + b)) ** 2
    return math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))


def girth_of_slice(lat: np.ndarray, z: np.ndarray, bin_deg: float = 5.0) -> dict:
    """Tape length around one slice.

    Points are binned by angle around the slice centre and each bin keeps its
    median radius, so surface noise cancels. Bins the cameras never saw (the
    belly, mostly) take the radius of an ellipse fitted to the seen profile.
    The convex hull of that loop is what a tape lying on the hide measures.
    """
    pts = np.c_[lat, z].astype(np.float64)
    c = np.median(pts, axis=0)
    rel = pts - c
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    rad = np.hypot(rel[:, 0], rel[:, 1])
    nb = int(round(360 / bin_deg))
    idx = np.floor((ang + np.pi) / (2 * np.pi) * nb).astype(int) % nb
    prof = np.full(nb, np.nan)
    for b in range(nb):
        r = rad[idx == b]
        if len(r) >= 3:
            prof[b] = np.median(r)
    seen = np.isfinite(prof)
    coverage = float(seen.mean())
    centres = (np.arange(nb) + 0.5) / nb * 2 * np.pi - np.pi
    out = {"coverage": round(coverage, 3), "points": int(len(pts))}
    if seen.sum() < 8:
        return {**out, "girth_m": float("nan"), "method": "too little of the loop seen"}
    loop = np.c_[c[0] + prof[seen] * np.cos(centres[seen]), c[1] + prof[seen] * np.sin(centres[seen])]
    filled = 0
    if (~seen).any():
        (ex, ez), (w, h), rot = cv2.fitEllipse(loop.astype(np.float32))
        th = np.radians(rot)
        a2, b2 = w / 2, h / 2
        for ang_b in centres[~seen]:
            # ray from the slice centre to the ellipse, in the ellipse frame
            dx, dz = math.cos(ang_b), math.sin(ang_b)
            ox, oz = c[0] - ex, c[1] - ez
            rx, rz = dx * math.cos(th) + dz * math.sin(th), -dx * math.sin(th) + dz * math.cos(th)
            px, pz = ox * math.cos(th) + oz * math.sin(th), -ox * math.sin(th) + oz * math.cos(th)
            A = (rx / a2) ** 2 + (rz / b2) ** 2
            B = 2 * (px * rx / a2 ** 2 + pz * rz / b2 ** 2)
            C = (px / a2) ** 2 + (pz / b2) ** 2 - 1
            t = (-B + math.sqrt(max(B * B - 4 * A * C, 0))) / (2 * A)
            loop = np.vstack([loop, [c[0] + t * dx, c[1] + t * dz]])
            filled += 1
        out["ellipse_axes_m"] = [round(w, 3), round(h, 3)]
    hull = cv2.convexHull(loop.astype(np.float32))
    out.update(girth_m=round(float(cv2.arcLength(hull, True)), 4), filled_bins=filled,
               method="median polar profile, convex hull" + (", ellipse fills unseen arc" if filled else ""))
    return out


def drop_boards(P: np.ndarray, boards: list[dict], plane_m: float = 0.035, radius_m: float = 0.27) -> np.ndarray:
    """Remove the printed boards: points near each board's plane within its card."""
    keep = np.ones(len(P), bool)
    for b in boards:
        if "normal" not in b:
            continue
        d = P - np.asarray(b["centre"])
        off = d @ np.asarray(b["normal"])
        inplane = np.linalg.norm(d - off[:, None] * np.asarray(b["normal"]), axis=1)
        keep &= ~((np.abs(off) < plane_m) & (inplane < radius_m))
    return P[keep]


def measure(points: np.ndarray, boards: list[dict]) -> dict:
    P = drop_boards(points[points[:, 2] > 0.05], boards)
    P = _largest_cluster(P)
    if len(P) < 500:
        return {"ok": False, "reason": "too little surface", "points": len(P)}
    top = P[:, 2].max()
    upper = P[P[:, 2] > 0.6 * top]
    # body axis: principal direction of the upper body on the ground plane
    c = upper[:, :2].mean(0)
    w, v = np.linalg.eigh(np.cov((upper[:, :2] - c).T))
    ax = v[:, -1]
    if boards:
        front = np.mean([np.asarray(b["centre"])[:2] for b in boards], axis=0)
        if (front - c) @ ax < 0:
            ax = -ax
    lat_dir = np.array([-ax[1], ax[0]])
    s = (P[:, :2] - c) @ ax                     # along the body, front positive
    lat = (P[:, :2] - c) @ lat_dir

    # withers height: highest body point over the front half, short of the neck
    legs = P[(P[:, 2] > LEG_BAND[0] * top) & (P[:, 2] < LEG_BAND[1] * top)]
    ls = (legs[:, :2] - c) @ ax
    hist, edges = np.histogram(ls, bins=np.arange(ls.min() - 0.02, ls.max() + 0.04, 0.02))
    occupied = hist > max(3, 0.05 * hist.max())
    runs, start = [], None
    for i, o in enumerate(occupied):
        if o and start is None:
            start = i
        if (not o or i == len(occupied) - 1) and start is not None:
            end = i if o else i - 1
            runs.append((edges[start], edges[end + 1], hist[start:end + 1].sum()))
            start = None
    runs = sorted(runs, key=lambda r: -r[2])[:2]
    if len(runs) < 2:
        return {"ok": False, "reason": "front and rear legs not separated", "points": len(P)}
    front_run = max(runs, key=lambda r: r[0])
    rear_run = min(runs, key=lambda r: r[0])
    s_girth = front_run[0] - TAPE_OFFSET_M
    sl = (np.abs(s - s_girth) < SLICE_HALF_M) & (P[:, 2] > LEG_BAND[1] * top)
    if sl.sum() < 12:
        return {"ok": False, "reason": "girth slice empty", "points": len(P)}
    g = girth_of_slice(lat[sl], P[sl, 2])

    # length: barrel band between the slice's bottom and middle, so the neck stays out
    zb, zt = P[sl, 2].min(), P[sl, 2].max()
    band = (P[:, 2] > zb + 0.2 * (zt - zb)) & (P[:, 2] < zb + 0.55 * (zt - zb))
    sb = s[band]
    length = float(np.percentile(sb, 99.5) - np.percentile(sb, 0.5)) if band.sum() > 50 else float("nan")
    near_front = (s > front_run[0] - 0.15) & (s < front_run[1] + 0.05)
    withers = float(np.percentile(P[near_front, 2], 99)) if near_front.sum() > 20 else float("nan")
    return {"ok": True, "heart_girth_m": g["girth_m"], "girth": g, "body_length_m": round(length, 4),
            "withers_height_m": round(withers, 4), "legs": {"front": [round(front_run[0], 3), round(front_run[1], 3)],
                                                              "rear": [round(rear_run[0], 3), round(rear_run[1], 3)]},
            "girth_station_m": round(float(s_girth), 3), "points": int(len(P))}


def schaeffer_kg(girth_m: float, length_m: float) -> float:
    """Schaeffer's formula: kg = girth(cm)^2 x length(cm) / 10838."""
    return (girth_m * 100) ** 2 * (length_m * 100) / 10838.0
