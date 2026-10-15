"""Chain of custody: one unbroken take from the nose to the body.

A swapped animal shows up as a cut: between two consecutive frames the tracked
points stop agreeing on a single smooth motion. Lucas-Kanade tracks corners
frame to frame; MAGSAC fits the dominant motion; a frame pair whose surviving
inlier fraction collapses, or whose appearance jumps, is a break.
"""

from __future__ import annotations

import cv2
import numpy as np

MIN_INLIER_FRAC = 0.35
MAX_HIST_DIST = 0.45


def _gray(img: np.ndarray, side: int = 480) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    s = side / max(g.shape)
    return cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def check(frames: list[np.ndarray]) -> dict:
    breaks, pairs = [], []
    prev = _gray(frames[0])
    for i in range(1, len(frames)):
        cur = _gray(frames[i])
        if cur.shape != prev.shape:                      # a still and a video frame can differ in size
            cur = cv2.resize(cur, (prev.shape[1], prev.shape[0]), interpolation=cv2.INTER_AREA)
        p0 = cv2.goodFeaturesToTrack(prev, 400, 0.01, 7)
        frac, hist = 0.0, 1.0
        if p0 is not None and len(p0) >= 8:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(21, 21), maxLevel=3)
            ok = st.ravel() == 1
            if ok.sum() >= 8:
                H, mask = cv2.findHomography(p0[ok], p1[ok], cv2.USAC_MAGSAC, 3.0)
                frac = float(mask.sum()) / len(p0) if mask is not None else 0.0
        h0 = cv2.calcHist([prev], [0], None, [32], [0, 256])
        h1 = cv2.calcHist([cur], [0], None, [32], [0, 256])
        hist = float(cv2.compareHist(cv2.normalize(h0, h0), cv2.normalize(h1, h1), cv2.HISTCMP_BHATTACHARYYA))
        broken = frac < MIN_INLIER_FRAC or hist > MAX_HIST_DIST
        pairs.append({"pair": [i - 1, i], "inlier_frac": round(frac, 3), "hist_dist": round(hist, 3), "break": broken})
        if broken:
            breaks.append(i)
        prev = cur
    return {"continuous": not breaks, "breaks": len(breaks), "break_at": breaks, "pairs": len(pairs),
            "min_inlier_frac": round(min((p["inlier_frac"] for p in pairs), default=1.0), 3),
            "method": "LK optical flow + MAGSAC motion consistency"}
