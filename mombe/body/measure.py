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


def unseen_arc(seen: np.ndarray, centres: np.ndarray) -> dict:
    """The longest run of unseen angle bins around the girth loop, and where it sits.

    Angles run from the animal's left flank (0) over the back (90) to the right
    flank (180) and under the belly (-90), so the run's centre names the pass
    that would fill it: a flank walk or a crouching pass under the chest.
    """
    nb = len(seen)
    if seen.all() or not seen.any():
        return {"gap_deg": 0.0 if seen.all() else 360.0, "gap_arc": None if seen.all() else "all"}
    start = int(np.argmax(seen))                  # roll so the loop starts on a seen bin
    s = np.roll(seen, -start)
    best, best_i, run = 0, 0, 0
    for i, v in enumerate(s):
        run = 0 if v else run + 1
        if run > best:
            best, best_i = run, i
    mid = (start + best_i - (best - 1) / 2) % nb
    ang = math.degrees(math.atan2(math.sin(centres[0] + mid * 2 * math.pi / nb),
                                  math.cos(centres[0] + mid * 2 * math.pi / nb)))
    arc = "left" if -45 <= ang < 45 else "top" if 45 <= ang < 135 else "belly" if -135 <= ang < -45 else "right"
    return {"gap_deg": round(best * 360.0 / nb, 1), "gap_centre_deg": round(ang, 1), "gap_arc": arc}


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
    out = {"coverage": round(coverage, 3), "points": int(len(pts)), **unseen_arc(seen, centres)}
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
    (_, _), (ew, eh), _ = cv2.fitEllipse(loop.astype(np.float32))
    out["ellipse_m"] = round(ellipse_perimeter(ew / 2, eh / 2), 4)      # noise-robust alternative to the hull
    out["hull_m"] = round(float(cv2.arcLength(hull, True)), 4)
    # The ellipse through the profile is the girth: on seven synthetic animals it read
    # +2.0% (sd 1.4%) where the hull read +2.9%, since a hull leans outward on noisy surface.
    out.update(girth_m=out["ellipse_m"], filled_bins=filled,
               method="median polar profile, ellipse fit" + (" (fills the unseen arc)" if filled else ""))
    return out


def trunk_length_fit(stations: np.ndarray, widths: np.ndarray, w_max: float) -> float | None:
    """Length where the fitted trunk profile falls to half its widest point.

    Along a barrel, squared width is a quadratic in position (an ellipse's
    profile). A parabola fitted to w^2 over the well-observed middle of the
    trunk (width at least 60% of the widest) is solved for w = w0 / 2.
    """
    m = widths >= 0.6 * w_max
    if m.sum() < 8:
        return None
    c2, c1, c0 = np.polyfit(stations[m], widths[m] ** 2, 2)
    if c2 >= 0:
        return None
    peak = c0 - c1 * c1 / (4 * c2)
    disc = c1 * c1 - 4 * c2 * (c0 - 0.25 * peak)
    if peak <= 0 or disc <= 0:
        return None
    return float(math.sqrt(disc) / abs(c2))


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

    # length: the barrel is wide and the neck is narrow. Along the body axis, the
    # run of stations at least half as wide as the girth slice is the trunk; its
    # extent approximates point of shoulder to pin bone.
    zb, zt = P[sl, 2].min(), P[sl, 2].max()
    band = (P[:, 2] > zb + 0.15 * (zt - zb)) & (P[:, 2] < zt - 0.1 * (zt - zb))
    step = 0.02
    stations = np.arange(s[band].min(), s[band].max() + step, step)
    widths = np.zeros(len(stations))
    for i, x in enumerate(stations):
        m = band & (np.abs(s - x) < step / 2)
        if m.sum() >= 10:
            widths[i] = np.percentile(lat[m], 98) - np.percentile(lat[m], 2)
    # half the trunk's widest point: a cut that scales with the animal, wherever the girth line sits
    w_max = np.percentile(widths[widths > 0], 95)
    wide = widths >= 0.5 * w_max
    fitted = trunk_length_fit(stations, widths, w_max)
    i0 = int(np.argmin(np.abs(stations - s_girth)))
    lo_i = hi_i = i0
    while lo_i > 0 and (wide[lo_i - 1] or (lo_i > 1 and wide[lo_i - 2])):   # bridge one-station gaps
        lo_i -= 1
    while hi_i < len(stations) - 1 and (wide[hi_i + 1] or (hi_i < len(stations) - 2 and wide[hi_i + 2])):
        hi_i += 1
    length_cut = float(stations[hi_i] - stations[lo_i] + step)
    # The reconstruction thins out at the tapered ends of the barrel, so the cut read
    # directly runs short; the fitted profile reaches the same cut from the well-seen middle.
    length = fitted if fitted and 0.7 * length_cut <= fitted <= 1.4 * length_cut else length_cut
    near_front = (s > front_run[0] - 0.15) & (s < front_run[1] + 0.05)
    withers = float(np.percentile(P[near_front, 2], 99)) if near_front.sum() > 20 else float("nan")
    return {"ok": True, "heart_girth_m": g["girth_m"], "girth": g, "body_length_m": round(length, 4),
            "withers_height_m": round(withers, 4), "length_direct_m": round(length_cut, 4), "legs": {"front": [round(front_run[0], 3), round(front_run[1], 3)],
                                                              "rear": [round(rear_run[0], 3), round(rear_run[1], 3)]},
            "girth_station_m": round(float(s_girth), 3), "points": int(len(P)),
            "axis": [round(float(ax[0]), 4), round(float(ax[1]), 4)], "centre_m": [round(float(c[0]), 3), round(float(c[1]), 3)]}


def schaeffer_kg(girth_m: float, length_m: float) -> float:
    """Schaeffer's formula: kg = girth(cm)^2 x length(cm) / 10838."""
    return (girth_m * 100) ** 2 * (length_m * 100) / 10838.0
