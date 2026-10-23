"""The body provider the hub's tools call: real frames in, measurements with honest errors out.

Same four calls as the simulated provider, so the evidence policy reads real
numbers through an unchanged interface:

  continuity  optical-flow chain from the nose close-up into the walk-around
  scale       metric reconstruction; registered keyframes, board frames and the
              scale error, raw and calibrated (mombe/body/calibration.py)
  measure     dense surface, heart girth, body length, withers height, which
              sides of the animal and how much of the girth loop were seen
  weight      girth-and-length formula with a 90% interval built from the
              calibrated scale error, the measurement error and the formula's spread

The reconstruction runs once per capture and is cached for the later calls.
"""

from __future__ import annotations

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

import cv2
import numpy as np

from mombe.body.calibration import Z90, scale_claim
from mombe.body.dense import DenseFusion
from mombe.body.measure import measure as tape_measure, schaeffer_kg
from mombe.body.recon import Reconstructor

# Weight error from measurement alone (stereo, TSDF, girth and length), fraction: seven synthetic
# walk-arounds read -1.4% mean, sd 3.6% against the formula on true dimensions; both enter.
MEASURE_SIGMA = float(os.environ.get("MOMBE_MEASURE_SIGMA", 0.04))
# Spread of the girth-and-length formula around scale weight, fraction (log scale). An assumption
# until scale weights arrive: studies report high correlation with low absolute agreement
# (e.g. Gyr x Holstein crossbreds, Rev. Mex. Agroecosistemas); no clean beef-cattle figure.
FORMULA_SIGMA = float(os.environ.get("MOMBE_FORMULA_SIGMA", 0.08))
MAX_FRAMES = int(os.environ.get("MOMBE_MAX_FRAMES", 48))   # keyframes kept from the phone's 2 fps walk
WAIT_S = float(os.environ.get("MOMBE_BODY_WAIT_S", 40))    # a tool call waits this long, then answers pending
# Stability probe: the reconstruction runs a second time with the calibrated focal nudged by this
# fraction. COOL and the stock wheel differ by about 1e-10 in calibrateCamera (track 1d), and on
# sweep_21 that difference alone moved the true scale error from -5.9% to -8.9%; stage swaps between
# the builds spread it from -1.4% to -41%. On either build a nudge this size moves the orbit of the six
# other synthetic takes by 1.2% or less in 35 of 36 probes (3.4% once, COOL sweep_22) and the orbit of
# sweep_21 by up to 14.7%, so a fragile walk-around shows itself on any single build. Costs one more
# reconstruction. 0 switches the probe off.
STABILITY_EPS = float(os.environ.get("MOMBE_STABILITY_EPS", 1e-9))


class WalkAroundBody:
    def __init__(self):
        self._recon = None
        self._dense = None
        self._cache: dict[tuple, dict] = {}
        self._jobs: dict[tuple, object] = {}
        self._pool = ThreadPoolExecutor(1)       # one walk at a time: each uses every core
        self._lock = threading.Lock()

    # -- background processing -------------------------------------------------
    def start(self, c, on_done=None) -> None:
        """Begin the full pipeline as soon as the frames land; tool calls then read the cache."""
        key = tuple(c.body)
        with self._lock:
            if not key or key in self._jobs:
                return
            fut = self._pool.submit(self._measured, c)
            self._jobs[key] = fut
        if on_done:
            fut.add_done_callback(lambda _f: on_done())

    def _ready(self, c) -> dict | None:
        """Wait briefly for this capture's job; None means still processing."""
        key = tuple(c.body)
        if key not in self._jobs:
            self.start(c)
        fut = self._jobs.get(key)
        if fut is None:
            return {"frames": 0, "error": "no walk-around frames in this capture"}
        try:
            fut.result(timeout=WAIT_S)
        except FutureTimeout:
            return None
        except Exception as e:  # a failed job answers with its error
            self._cache.setdefault(key, {"frames": len(c.body), "error": f"{type(e).__name__}: {e}"})
        return self._cache.get(key)

    @staticmethod
    def _pending(c) -> dict:
        return {"pending": True, "status": "processing the walk-around", "capture": getattr(c, "index", None),
                "note": "the session parks; the hub wakes it when the measurements are ready"}

    def _frames(self, c) -> list[np.ndarray]:
        """Keyframes: the sharpest frame in each of MAX_FRAMES equal slices of the walk."""
        paths = list(c.body)
        if len(paths) > MAX_FRAMES:
            picks = []
            for chunk in np.array_split(np.arange(len(paths)), MAX_FRAMES):
                best, bi = -1.0, chunk[0]
                for i in chunk:
                    g = cv2.imread(paths[i], cv2.IMREAD_REDUCED_GRAYSCALE_4)
                    v = cv2.Laplacian(g, cv2.CV_32F).var() if g is not None else -1
                    if v > best:
                        best, bi = v, i
                picks.append(paths[bi])
            paths = picks
        frames = [cv2.imread(p) for p in paths]
        return [f for f in frames if f is not None]

    def _run(self, c) -> dict:
        key = tuple(c.body)
        if key in self._cache:
            return self._cache[key]
        frames = self._frames(c)
        out: dict = {"frames": len(frames), "images": frames}
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
        out["stability"] = self._stability(frames, rec)
        self._cache[key] = out
        return out

    @staticmethod
    def _orbit_radius_m(rec) -> float:
        c = np.array([rec.centre(f) for f in sorted(rec.R)])
        return float(np.median(np.linalg.norm(c - c.mean(0), axis=1)))

    def _stability(self, frames, rec) -> dict:
        """Rerun the reconstruction with the focal nudged by STABILITY_EPS; compare the metric orbits.

        Both runs end in metres (to_metric), so the median camera distance from the orbit's centre
        is comparable without any truth. Its relative shift is the probe's reading.
        """
        if not STABILITY_EPS:
            return {"ok": None, "reason": "probe off"}
        probe = Reconstructor()
        probe.seed = self._recon.seed
        orig = probe.calibrate

        def calibrate(obs, size, K0=None):
            cal = orig(obs, size, K0)
            K = cal["K"].copy()
            K[0, 0] *= 1 + STABILITY_EPS
            K[1, 1] *= 1 + STABILITY_EPS
            return {**cal, "K": K}
        probe.calibrate = calibrate
        try:
            r2 = probe.run(frames)
        except RuntimeError as e:
            return {"ok": False, "shift_rel": None, "reason": f"the nudged reconstruction failed: {e}"}
        a, b = self._orbit_radius_m(rec), self._orbit_radius_m(r2)
        return {"ok": True, "shift_rel": round(abs(b / a - 1), 5), "orbit_radius_m": [round(a, 4), round(b, 4)],
                "registered": [len(rec.R), len(r2.R)], "eps": STABILITY_EPS}

    def _measured(self, c) -> dict:
        r = self._run(c)
        if "rec" not in r or "measure" in r:
            return r
        self._dense = self._dense or DenseFusion()
        dense = self._dense.run_stereo(r["rec"], r["images"])
        r["measure"] = tape_measure(dense.points, r["rec"].boards)
        r["dense_s"] = dense.seconds
        return r

    # -- the four calls ---------------------------------------------------------
    def continuity(self, c) -> dict:
        """One animal, one take.

        The video opens on the nose and pulls back, so its first frames belong to
        the head, not the body model: optical flow must carry the nose photo
        through them to the first frame that registers. From there on, frames
        must register into the one reconstruction; a spliced take of another
        animal falls out of it. A few blurred frames are tolerated.
        """
        from mombe.vision.continuity import check
        r = self._ready(c)
        if r is None:
            return self._pending(c)
        if "rec" not in r:
            return {"continuous": False, "breaks": 1, "method": "reconstruction registration", "reason": r.get("error")}
        rec, images = r["rec"], r["images"]
        reg = sorted(rec.R)
        first = reg[0]
        chain = None
        nose = cv2.imread(c.muzzle[-1]) if c.muzzle else None
        if nose is not None:
            chain = check([nose] + images[:first + 1])
        dropped = [i for i in range(first, len(images)) if i not in rec.R]
        allowed = max(1, int(0.1 * (len(images) - first)))
        breaks = (0 if chain is None or chain["continuous"] else chain["breaks"]) + max(0, len(dropped) - allowed)
        return {"continuous": breaks == 0, "breaks": breaks, "opening_frames": first,
                "nose_to_body": None if chain is None else chain["continuous"],
                "unregistered_after_opening": dropped, "tolerated": allowed,
                "method": "LK flow from the nose to the first registered frame; registration after it"}

    def scale(self, c) -> dict:
        r = self._ready(c)
        if r is None:
            return self._pending(c)
        if "rec" not in r:
            return {"board_frames": 0, "scale_rel_err": 1.0, "registered_frames": 0, "frames": r.get("frames", 0),
                    "method": "ChArUco walk-around", "reason": r.get("error")}
        rec = r["rec"]
        d = rec.scale_detail
        raw = float(rec.scale_rel_err)
        return {"board_frames": d.get("board_frames", 0), "placements": d.get("placements", 0),
                "scale_rel_err": round(scale_claim(raw), 5), "scale_rel_err_raw": round(raw, 5),
                "registered_frames": rec.stats.get("registered"),
                "frames": r["frames"], "reprojection_px": rec.stats.get("ba_rms_px"),
                "focal": rec.stats.get("focal"), "stability": r.get("stability"),
                "method": "board corners as tracks; jackknife within, spread between placements; calibrated claim"}

    def measure(self, c) -> dict:
        r = self._ready(c)
        if r is None:
            return self._pending(c)
        m = r.get("measure")
        if not m or not m.get("ok"):
            return {"coverage": 0.0, "missing_side": None, "girth_loop_coverage": 0.0,
                    "reason": (m or {}).get("reason") or r.get("error")}
        rec = r["rec"]
        # which sides of the animal the camera saw: camera bearings around the body, from its own axis
        # (0 ahead of the animal, +90 its left flank, -90 its right flank)
        ax, ctr = np.asarray(m["axis"]), np.asarray(m["centre_m"])
        rel = np.array([rec.centre(f) for f in rec.R])[:, :2] - ctr
        bearing = np.degrees(np.arctan2(rel @ np.array([-ax[1], ax[0]]), rel @ ax))
        seen = np.unique(np.floor((bearing + 180) / 30).astype(int))
        coverage = len(seen) / 12
        left = np.any((bearing > 30) & (bearing < 150))
        right = np.any((bearing < -30) & (bearing > -150))
        missing = None if (left and right) else ("left" if not left else "right")
        g = m["girth"]
        return {"coverage": round(float(coverage), 3), "missing_side": missing,
                "girth_cm": round(m["heart_girth_m"] * 100, 1), "length_cm": round(m["body_length_m"] * 100, 1),
                "length_cut_cm": round(m["length_direct_m"] * 100, 1),
                "withers_cm": round(m["withers_height_m"] * 100, 1), "girth_loop_coverage": g["coverage"],
                "girth_gap_deg": g.get("gap_deg"), "girth_gap_arc": g.get("gap_arc"),
                "girth_method": g["method"], "dense_s": r.get("dense_s")}

    def weight(self, c) -> dict:
        r = self._ready(c)
        if r is None:
            return self._pending(c)
        m = r.get("measure")
        if not m or not m.get("ok"):
            return {"kg": None, "reason": (m or {}).get("reason") or r.get("error")}
        kg = schaeffer_kg(m["heart_girth_m"], m["body_length_m"])
        s_err = scale_claim(float(r["rec"].scale_rel_err))
        # weight goes with girth^2 x length: the scale error enters three times
        sigma = math.sqrt((3 * s_err) ** 2 + MEASURE_SIGMA ** 2 + FORMULA_SIGMA ** 2)
        lo, hi = kg * math.exp(-Z90 * sigma), kg * math.exp(Z90 * sigma)
        return {"kg": round(kg, 1), "lo": round(lo, 1), "hi": round(hi, 1), "sigma_rel": round(sigma, 4),
                "scale_rel_err": round(s_err, 5),
                "interval": "90%: calibrated scale, measurement and formula spread",
                "formula": "Schaeffer: girth^2 x length / 10838 (cm, kg)"}
