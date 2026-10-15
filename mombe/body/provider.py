"""The body provider the hub's tools call: real frames in, measurements with honest errors out.

Same four calls as the simulated provider, so the evidence policy reads real
numbers through an unchanged interface:

  continuity  optical-flow chain from the nose close-up into the walk-around
  scale       metric reconstruction; board frames and the scale error
  measure     dense surface, heart girth, body length, which sides were seen
  weight      girth-and-length formula with a 90% interval built from the scale
              error, the measurement error and the formula's own spread

The reconstruction runs once per capture and is cached for the later calls.
"""

from __future__ import annotations

import math
import os

import cv2
import numpy as np

from mombe.body.dense import DenseFusion
from mombe.body.measure import measure as tape_measure, schaeffer_kg
from mombe.body.recon import Reconstructor

# Girth measurement error on synthetic walk-arounds (stereo + TSDF + slice), fraction.
GIRTH_SIGMA = float(os.environ.get("MOMBE_GIRTH_SIGMA", 0.04))
# Spread of the girth-and-length formula around true weight, fraction (log scale).
FORMULA_SIGMA = float(os.environ.get("MOMBE_FORMULA_SIGMA", 0.08))
Z90 = 1.645


class WalkAroundBody:
    def __init__(self):
        self._recon = None
        self._dense = None
        self._cache: dict[tuple, dict] = {}

    def _frames(self, c) -> list[np.ndarray]:
        frames = [cv2.imread(p) for p in c.body]
        return [f for f in frames if f is not None]

    def _run(self, c) -> dict:
        key = tuple(c.body)
        if key in self._cache:
            return self._cache[key]
        frames = self._frames(c)
        out: dict = {"frames": len(frames)}
        if len(frames) < 12:
            out["error"] = "walk-around too short: fewer than 12 frames"
            self._cache[key] = out
            return out
        self._recon = self._recon or Reconstructor()
        try:
            rec = self._recon.run(frames)
        except RuntimeError as e:
            out["error"] = f"reconstruction failed: {e}"
            self._cache[key] = out
            return out
        out["rec"] = rec
        self._cache[key] = out
        return out

    def _measured(self, c) -> dict:
        r = self._run(c)
        if "rec" not in r or "measure" in r:
            return r
        self._dense = self._dense or DenseFusion()
        dense = self._dense.run_stereo(r["rec"], self._frames(c))
        r["measure"] = tape_measure(dense.points, r["rec"].boards)
        r["dense_s"] = dense.seconds
        return r

    # -- the four calls ---------------------------------------------------------
    def continuity(self, c) -> dict:
        """One animal, one take: the nose close-up flows into the first body frame, and
        every body frame registers into the same reconstruction. A spliced take of
        another animal fails one or the other."""
        from mombe.vision.continuity import check
        link = None
        if c.muzzle and c.body:
            pair = [cv2.imread(c.muzzle[-1]), cv2.imread(c.body[0])]
            if all(f is not None for f in pair):
                link = check(pair)
        r = self._run(c)
        if "rec" not in r:
            return {"continuous": False, "breaks": 1, "method": "reconstruction registration", "reason": r.get("error")}
        rec = r["rec"]
        unregistered = sorted(set(range(r["frames"])) - set(rec.R))
        breaks = len(unregistered) + (0 if link is None or link["continuous"] else 1)
        return {"continuous": breaks == 0, "breaks": breaks, "unregistered_frames": unregistered,
                "nose_to_body": None if link is None else link["continuous"],
                "method": "every frame registers into one reconstruction; LK flow from the nose to the body"}

    def scale(self, c) -> dict:
        r = self._run(c)
        if "rec" not in r:
            return {"board_frames": 0, "scale_rel_err": 1.0, "method": "ChArUco walk-around", "reason": r.get("error")}
        rec = r["rec"]
        d = rec.scale_detail
        return {"board_frames": d.get("board_frames", 0), "placements": d.get("placements", 0),
                "scale_rel_err": round(float(rec.scale_rel_err), 5), "registered_frames": rec.stats.get("registered"),
                "frames": r["frames"], "reprojection_px": rec.stats.get("ba_rms_px"),
                "focal": rec.stats.get("focal"), "method": "board corners as tracks; jackknife within, spread between placements"}

    def measure(self, c) -> dict:
        r = self._measured(c)
        m = r.get("measure")
        if not m or not m.get("ok"):
            return {"coverage": 0.0, "missing_side": None, "reason": (m or {}).get("reason") or r.get("error")}
        rec = r["rec"]
        # which sides of the animal the camera saw: camera bearings around the body
        cams = np.array([rec.centre(f) for f in rec.R])[:, :2]
        bearing = np.degrees(np.arctan2(cams[:, 1], cams[:, 0]))
        seen = np.unique(np.floor((bearing + 180) / 30).astype(int))
        coverage = len(seen) / 12
        left = np.any((bearing > 30) & (bearing < 150))
        right = np.any((bearing < -30) & (bearing > -150))
        missing = None if (left and right) else ("left" if not left else "right")
        return {"coverage": round(float(coverage), 3), "missing_side": missing,
                "girth_cm": round(m["heart_girth_m"] * 100, 1), "length_cm": round(m["body_length_m"] * 100, 1),
                "withers_cm": round(m["withers_height_m"] * 100, 1), "girth_loop_coverage": m["girth"]["coverage"],
                "girth_method": m["girth"]["method"], "dense_s": r.get("dense_s")}

    def weight(self, c) -> dict:
        r = self._measured(c)
        m = r.get("measure")
        if not m or not m.get("ok"):
            return {"kg": None, "reason": (m or {}).get("reason") or r.get("error")}
        kg = schaeffer_kg(m["heart_girth_m"], m["body_length_m"])
        s_err = float(r["rec"].scale_rel_err)
        # weight goes with girth^2 x length: scale error enters three times, girth error twice
        sigma = math.sqrt((3 * s_err) ** 2 + (2 * GIRTH_SIGMA) ** 2 + FORMULA_SIGMA ** 2)
        lo, hi = kg * math.exp(-Z90 * sigma), kg * math.exp(Z90 * sigma)
        return {"kg": round(kg, 1), "lo": round(lo, 1), "hi": round(hi, 1), "sigma_rel": round(sigma, 4),
                "interval": "90%: scale, girth measurement and formula spread",
                "formula": "Schaeffer: girth^2 x length / 10838 (cm, kg)"}
