"""Per-stage dump of one walk-around reconstruction, for bisecting two OpenCV builds.

Runs the real Reconstructor.run on a take. Hooks on the instance (and thin
recording wrappers around a few cv2 functions) record every stage's output
without changing any arithmetic. With --inject, chosen stages return another
build's recorded output instead, so a stage can be swapped between builds.

Stages recorded, in pipeline order:
  frames    decoded pixels (sha256 per frame)
  gray      cvtColor BGR2GRAY used by board detection
  board     ChArUco corners and ids per frame
  calib     K, distortion, rms
  work      INTER_AREA work image fed to ALIKED
  aliked    keypoints (work pixels), responses, descriptors
  kpts      undistorted full-resolution keypoints
  lg        LightGlue raw matches per pair
  ess       MAGSAC essential inliers per pair (findEssentialMat in match)
  matches   pair graph after the 30-inlier floor
  focal     Mendonca-Cipolla focal from the pairs
  sfm       initial pair, registration order, PnP inliers, bundle steps
  rec       final poses, points, focal, stats
  scale     per placement scale, jackknife, final scale and claim
  score     true scale error against the synthetic truth
  dense     (optional) stereo + TSDF + girth

Usage:
  python bisect_take.py --take DIR --out PREFIX [--seed 7] [--dense]
                        [--inject board,calib,feats,lg,matches --src OTHER.pkl] [--plain]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = Path(os.environ.get("MOMBE_CODE", HERE.parent))     # dir holding the mombe package and tools/
sys.path.insert(0, str(CODE))

import cv2  # noqa: E402

from mombe.body.recon import Reconstructor  # noqa: E402


def h(a) -> str | None:
    if a is None:
        return None
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _threads():
    try:
        return cv2.getNumThreads()
    except cv2.error:
        return None


def env_info() -> dict:
    bi = cv2.getBuildInformation()
    keep = [l.strip() for l in bi.splitlines() if any(k in l for k in ("Baseline", "Custom HAL", "Lapack", "C++ flags (Release)", "JPEG:"))]
    import scipy
    return {"opencv": cv2.__version__, "cv2_file": cv2.__file__, "numpy": np.__version__, "scipy": scipy.__version__,
            "python": platform.python_version(), "machine": platform.machine(), "node": platform.node(),
            "cpu_features": cv2.getCPUFeaturesLine(), "threads": _threads(), "build": keep}


class Recorder:
    """Instance hooks on one Reconstructor."""

    def __init__(self, R: Reconstructor, inject: set[str], src: dict | None):
        self.R, self.inject, self.src = R, inject, src or {}
        self.d: dict = {"board": {}, "gray": {}, "work": [], "aliked": [], "lg": {}, "ess": {}, "pnp": [], "sfm_steps": [],
                        "fundamental": []}
        self.fid: dict[int, int] = {}
        self.cur_pair = None
        self._ali_i = 0
        self._wrap()

    def _wrap(self):
        R, d = self.R, self.d
        rec_self = self

        # board: per-frame detection, recorded with the gray image hash
        orig_detect = R.board.detect
        frame_i = {"i": 0}

        def detect(bgr):
            i = frame_i["i"]
            frame_i["i"] += 1
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
            d["gray"][i] = h(gray)
            c, ids = orig_detect(bgr)
            d["board"][i] = (None if c is None else np.array(c, np.float32), None if ids is None else np.array(ids))
            return c, ids
        R.board.detect = detect

        orig_board_obs = R.board_obs

        def board_obs(frames):
            obs = orig_board_obs(frames)
            if "board" in self.inject:
                obs = {int(i): (np.asarray(c, np.float32), np.asarray(ids)) for i, (c, ids) in self.src["board_obs"].items()}
            d["board_obs"] = {i: (c.copy(), ids.copy()) for i, (c, ids) in obs.items()}
            return obs
        R.board_obs = board_obs

        orig_cal = R.calibrate

        def calibrate(obs, size, K0=None):
            cal = orig_cal(obs, size, K0)
            if "calib" in self.inject:
                s = self.src["calib"]
                cal = {**s, "K": np.array(s["K"]), "dist": np.array(s["dist"])}
            d["calib"] = {k: (np.array(v) if isinstance(v, np.ndarray) else v) for k, v in cal.items()}
            return cal
        R.calibrate = calibrate

        # ALIKED: input image and raw outputs
        real_aliked = R.aliked

        class AlikedProxy:
            def detectAndCompute(self_, img, mask):
                i = rec_self._ali_i
                rec_self._ali_i += 1
                d["work"].append({"hash": h(img), "shape": img.shape})
                if "feats" in rec_self.inject:
                    a = rec_self.src["aliked"][i]
                    kps = [cv2.KeyPoint(float(x), float(y), float(s), float(an), float(r))
                           for (x, y), s, an, r in zip(a["pts"], a["size"], a["angle"], a["resp"])]
                    desc = a["desc"].copy() if a["desc"] is not None else None
                else:
                    kps, desc = real_aliked.detectAndCompute(img, mask)
                d["aliked"].append({"pts": np.float32([k.pt for k in kps]).reshape(-1, 2),
                                    "size": np.float32([k.size for k in kps]), "angle": np.float32([k.angle for k in kps]),
                                    "resp": np.float32([k.response for k in kps]),
                                    "desc": None if desc is None else np.array(desc)})
                return kps, desc
        R.aliked = AlikedProxy()

        orig_features = R.features

        def features(frames, K, dist):
            feats = orig_features(frames, K, dist)
            self.fid = {id(f): n for n, f in enumerate(feats)}
            d["kpts"] = [f.kpts.copy() for f in feats]
            d["work_size"] = [f.work_size for f in feats]
            return feats
        R.features = features

        # LightGlue raw matches per pair
        real_lg = R.lightglue

        class LGProxy:
            def setPairInfo(self_, *a):
                return real_lg.setPairInfo(*a)

            def match(self_, d1, d2):
                p = rec_self.cur_pair
                if "lg" in rec_self.inject and p in rec_self.src["lg"]:
                    s = rec_self.src["lg"][p]
                    if isinstance(s, str):
                        d["lg"][p] = s
                        raise cv2.error(s)
                    m = [cv2.DMatch(int(q), int(t), float(dd)) for (q, t), dd in zip(s["idx"], s["dist"])]
                else:
                    try:
                        m = real_lg.match(d1, d2)
                    except cv2.error as e:
                        d["lg"][p] = f"cv2.error: {str(e)[:80]}"
                        raise
                d["lg"][p] = {"idx": np.array([[x.queryIdx, x.trainIdx] for x in m], int).reshape(-1, 2),
                              "dist": np.float32([x.distance for x in m])}
                return m
        R.lightglue = LGProxy()

        orig_match = R.match

        def match(a, b, K):
            self.cur_pair = (self.fid.get(id(a)), self.fid.get(id(b)))
            out = orig_match(a, b, K)
            d["ess"][self.cur_pair] = out.copy()
            return out
        R.match = match

        orig_match_all = R.match_all

        def match_all(feats, K):
            m = orig_match_all(feats, K)
            if "matches" in self.inject:
                m = {tuple(p): np.asarray(v) for p, v in self.src["matches"].items()}
            d["matches"] = {p: v.copy() for p, v in m.items()}
            return m
        R.match_all = match_all

        orig_focal = R.focal_from_pairs

        def focal_from_pairs(feats, matches, size):
            out = orig_focal(feats, matches, size)
            d["focal"] = dict(out)
            return out
        R.focal_from_pairs = focal_from_pairs

        orig_tri = R._triangulate_new
        seen = {"reg": set()}

        def _triangulate_new(rec, tri, registered):
            orig_tri(rec, tri, registered)
            now = set(rec.R)
            new = sorted(now - seen["reg"])
            seen["reg"] = now
            d["sfm_steps"].append({"op": "tri", "new": new, "registered": len(now), "points": len(tri),
                                   "unregistered": list(rec.stats.get("unregistered", []))})
        R._triangulate_new = _triangulate_new

        orig_bundle = R._bundle

        def _bundle(rec, tri, fix, focal=False):
            orig_bundle(rec, tri, fix, focal)
            d["sfm_steps"].append({"op": "ba", "focal": bool(focal), "registered": len(rec.R), "points": len(tri),
                                   "rms_px": rec.stats.get("ba_rms_px"), "obs": rec.stats.get("ba_obs"),
                                   "rejected": rec.stats.get("ba_rejected"), "f": float(rec.K[0, 0])})
        R._bundle = _bundle

        orig_scale = R.scale

        def scale(rec, obs):
            orig_scale(rec, obs)
            d["scale"] = {"metres_per_unit": rec.metres_per_unit, "scale_rel_err": rec.scale_rel_err,
                          "detail": json.loads(json.dumps(rec.scale_detail, default=float)),
                          "boards": [{"centre": np.array(b["centre"]), "normal": np.array(b["normal"]), "frames": list(b["frames"])}
                                     for b in rec.boards]}
        R.scale = scale


class CvSpy:
    """Thin recording wrappers around cv2 functions the geometry calls. Arithmetic unchanged."""

    def __init__(self, d: dict):
        self.d, self.saved = d, {}

    def __enter__(self):
        d = self.d
        real_pnp, real_ess, real_rp = cv2.solvePnPRansac, cv2.findEssentialMat, cv2.recoverPose

        def pnp(obj, img, *a, **k):
            out = real_pnp(obj, img, *a, **k)
            ok, rv, tv, inl = out
            d["pnp"].append({"n": len(obj), "ok": bool(ok), "inliers": 0 if inl is None else int(len(inl)),
                             "inl_hash": h(inl), "rvec": None if rv is None else np.array(rv).ravel(),
                             "tvec": None if tv is None else np.array(tv).ravel(), "obj_hash": h(obj), "img_hash": h(img)})
            return out

        def ess(p1, p2, *a, **k):
            out = real_ess(p1, p2, *a, **k)
            E, mask = out
            d.setdefault("ess_calls", []).append({"n": len(p1), "in_hash": h(np.hstack([p1, p2])),
                                                  "inliers": None if mask is None else int(mask.sum()),
                                                  "mask_hash": h(mask), "E": None if E is None else np.array(E)})
            return out

        def rp(*a, **k):
            out = real_rp(*a, **k)
            d.setdefault("recover_pose", []).append({"inliers": int(out[0]), "R": np.array(out[1]), "t": np.array(out[2]).ravel()})
            return out
        self.saved = {"solvePnPRansac": real_pnp, "findEssentialMat": real_ess, "recoverPose": real_rp}
        cv2.solvePnPRansac, cv2.findEssentialMat, cv2.recoverPose = pnp, ess, rp
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(cv2, k, v)


def summary_of(rec, sc: dict | None) -> dict:
    return {"registered": len(rec.R), "frames": rec.stats.get("frames"), "points": int(len(rec.points)),
            "tracks": rec.stats.get("tracks"), "unregistered": rec.stats.get("unregistered", []),
            "ba_rms_px": rec.stats.get("ba_rms_px"), "final_focal_px": float(rec.K[0, 0]),
            "metres_per_unit_scale_claim": rec.scale_rel_err, "scale_detail": rec.scale_detail,
            "ground": {k: v for k, v in rec.ground.items() if k in ("found", "inliers", "camera_height_m")},
            "focal": rec.stats.get("focal"), "calibration": rec.stats.get("calibration"), "score": sc}


def load_take(take: Path):
    truth = json.loads((take / "truth.json").read_text())
    frames = [cv2.imread(str(p)) for p in sorted(take.glob("frame_*.jpg"))]
    return truth, frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--inject", default="")
    ap.add_argument("--src")
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--plain", action="store_true", help="also run Reconstructor.run without hooks and compare")
    a = ap.parse_args()
    from tools.eval_recon import score

    take = Path(a.take)
    truth, frames = load_take(take)
    inject = {s for s in a.inject.split(",") if s}
    src = pickle.loads(Path(a.src).read_bytes()) if a.src else None
    out: dict = {"env": env_info(), "take": take.name, "seed": a.seed, "inject": sorted(inject),
                 "src": a.src and Path(a.src).name, "frames": [h(f) for f in frames]}

    if a.plain:
        t0 = time.time()
        R0 = Reconstructor()
        R0.seed = a.seed
        rec0 = R0.run(frames)
        rec0.feats_kpts = rec0.feats_kpts or []
        out["plain"] = summary_of(rec0, score(rec0, truth))
        out["plain"]["seconds"] = round(time.time() - t0, 1)
        out["plain"]["R_hash"] = h(np.array([rec0.R[f] for f in sorted(rec0.R)]))

    t0 = time.time()
    R = Reconstructor()
    R.seed = a.seed
    rc = Recorder(R, inject, src)
    with CvSpy(rc.d):
        rec = R.run(frames)
    rec.feats_kpts = rec.feats_kpts or []
    sc = score(rec, truth)
    d = rc.d
    d["rec"] = {"R": {f: rec.R[f].copy() for f in rec.R}, "t": {f: rec.t[f].copy() for f in rec.t}, "K": rec.K.copy(),
                "points": rec.points.copy(), "point_track": list(rec.point_track), "n_tracks": len(rec.tracks),
                "tracks_hash": h(np.array([x for t in rec.tracks for o in t for x in o])),
                "stats": json.loads(json.dumps({k: v for k, v in rec.stats.items() if k != "pairs"}, default=float)),
                "pairs": dict(rec.stats.get("pairs", {}))}
    d["score"] = sc
    out["hooked"] = summary_of(rec, sc)
    out["hooked"]["seconds"] = round(time.time() - t0, 1)
    out["hooked"]["R_hash"] = h(np.array([rec.R[f] for f in sorted(rec.R)]))
    if a.plain:
        out["hooks_transparent"] = (out["plain"]["R_hash"] == out["hooked"]["R_hash"]
                                    and out["plain"]["score"] == out["hooked"]["score"])

    if a.dense:
        from mombe.body.dense import DenseFusion
        from mombe.body.measure import measure
        t1 = time.time()
        dn = DenseFusion().run_stereo(rec, frames)
        m = measure(dn.points, rec.boards)
        dd = {"seconds": round(time.time() - t1, 1), "points": int(len(dn.points)), "points_hash": h(dn.points)}
        if m.get("ok"):
            dd.update(girth_m=m["heart_girth_m"], length_m=m["body_length_m"], loop_coverage=m["girth"]["coverage"],
                      girth_err=round(m["heart_girth_m"] / truth["heart_girth_m"] - 1, 4),
                      length_err=round(m["body_length_m"] / truth["body_length_m"] - 1, 4))
        else:
            dd["measure_failed"] = m.get("reason")
        out["dense"] = dd
        d["dense"] = dd

    # compact per-stage fingerprints for a quick look without the pickle
    out["fp"] = {
        "gray": hashlib.sha256("".join(v for _, v in sorted(d["gray"].items())).encode()).hexdigest()[:16],
        "board_frames": sum(1 for c, _ in d["board"].values() if c is not None),
        "board_hash": h(np.concatenate([c.ravel() for c, _ in d["board"].values() if c is not None] or [np.zeros(1)])),
        "calib_K": np.asarray(d["calib"]["K"]).round(6).tolist(), "calib_dist": np.asarray(d["calib"]["dist"]).round(8).tolist(),
        "calib_rms": d["calib"].get("rms"),
        "work_hash": h(np.array([w["hash"] for w in d["work"]])),
        "aliked_counts": [len(x["pts"]) for x in d["aliked"]],
        "aliked_hash": h(np.concatenate([x["pts"].ravel() for x in d["aliked"]])),
        "desc_hash": h(np.concatenate([x["desc"].ravel() for x in d["aliked"] if x["desc"] is not None])),
        "lg_total": int(sum(len(v["idx"]) for v in d["lg"].values() if isinstance(v, dict))),
        "lg_hash": h(np.concatenate([v["idx"].ravel() for k, v in sorted(d["lg"].items()) if isinstance(v, dict)])),
        "ess_total": int(sum(len(v) for v in d["ess"].values())),
        "pairs_kept": len(d["matches"]), "focal": d.get("focal"),
        "registration_order": [s["new"] for s in d["sfm_steps"] if s["op"] == "tri" and s["new"]],
        "pnp_inliers": [p["inliers"] for p in d["pnp"]],
    }
    Path(a.out + ".json").write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    d["env"] = out["env"]
    d["frames_hash"] = out["frames"]
    Path(a.out + ".pkl").write_bytes(pickle.dumps(d))
    print(json.dumps({"take": take.name, "inject": sorted(inject), "registered": out["hooked"]["registered"],
                      "claim": round(rec.scale_rel_err, 5), "true": sc["animal_scale_err"],
                      "transparent": out.get("hooks_transparent"), "dense": out.get("dense")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
