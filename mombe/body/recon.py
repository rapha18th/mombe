"""Metric multi-view reconstruction of a walk-around.

1. Intrinsics: the ChArUco board self-calibrates the phone (focal, k1, k2).
2. Structure from motion: ALIKED keypoints, LightGlue matches between nearby
   frames and across the loop closure, USAC_MAGSAC essential matrices, then
   incremental PnP registration and bundle adjustment.
3. Scale: every board corner is a track with a known position on the print.
   Triangulated in the reconstruction and fitted to the board's geometry, the
   corners fix metres per unit. Leaving one board frame out at a time gives a
   jackknife error on that scale, the number the weight gate reads.
4. Ground: a MAGSAC-style plane fit on the scaled sparse points sets "up" and
   puts the origin on the ground under the walk-around centre.

All geometry runs in full-resolution undistorted pixels with the calibrated K.
"""

from __future__ import annotations

import math
from collections import Counter as collections_counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from mombe.vision.board import ScaleEstimator

MODELS = Path(__file__).resolve().parents[2] / "models"
WORK_W = 960            # feature extraction width
MATCH_WINDOW = 4        # each frame matches its next few neighbours
LOOP = True             # walk-arounds close: the last frames match the first
REPROJ_PX = 2.0
SINGLE_BOARD_FLOOR = 0.007   # scale error seen on synthetic walk-arounds with one placement
MIN_PAIR_INLIERS = 30   # weaker pairs carry more wrong matches than information


# -- small geometry helpers ---------------------------------------------------
def rodrigues(rv: np.ndarray) -> np.ndarray:
    """Batched axis-angle to rotation matrices, N x 3 -> N x 3 x 3."""
    th = np.linalg.norm(rv, axis=1, keepdims=True)
    k = rv / np.maximum(th, 1e-12)
    K = np.zeros((len(rv), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = k[:, 2], -k[:, 1], k[:, 0]
    s, c = np.sin(th)[:, :, None], np.cos(th)[:, :, None]
    return np.eye(3)[None] + s * K + (1 - c) * K @ K


def triangulate_multi(Ps: list[np.ndarray], xs: list[np.ndarray]) -> np.ndarray:
    """Linear multi-view triangulation of one point (DLT)."""
    A = []
    for P, x in zip(Ps, xs):
        A.append(x[0] * P[2] - P[0])
        A.append(x[1] * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.asarray(A))
    X = vt[-1]
    return X[:3] / X[3]


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity dst ~ s R src + t (least squares)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / a.var(0).sum())
    return s, R, mu_d - s * R @ mu_s


# -- data --------------------------------------------------------------------
@dataclass
class FrameFeatures:
    kpts: np.ndarray        # N x 2 full-res undistorted pixels
    kpts_work: np.ndarray   # N x 2 work-size pixels (LightGlue wants these)
    desc: np.ndarray
    work_size: tuple


@dataclass
class Reconstruction:
    K: np.ndarray
    dist: np.ndarray
    size: tuple
    R: dict = field(default_factory=dict)        # frame -> world->cam rotation
    t: dict = field(default_factory=dict)        # frame -> world->cam translation
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    point_track: list = field(default_factory=list)   # point -> track id
    tracks: list = field(default_factory=list)        # track -> list of (frame, kp)
    feats: list = field(default_factory=list)
    metres_per_unit: float = float("nan")
    scale_rel_err: float = float("nan")
    scale_detail: dict = field(default_factory=dict)
    ground: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    def P(self, i: int) -> np.ndarray:
        return self.K @ np.hstack([self.R[i], self.t[i][:, None]])

    def centre(self, i: int) -> np.ndarray:
        return -self.R[i].T @ self.t[i]

    def to_metric(self) -> "Reconstruction":
        """Apply the scale and the ground frame: metres, z up, origin on the ground."""
        s = self.metres_per_unit
        G = self.ground.get("R_world", np.eye(3))
        o = self.ground.get("origin", np.zeros(3))
        self.points = (G @ (self.points * s - o).T).T
        for i in list(self.R):
            c = G @ (self.centre(i) * s - o)
            R = self.R[i] @ G.T
            self.R[i], self.t[i] = R, -R @ c
        self.metres_per_unit = 1.0
        return self


class Reconstructor:
    def __init__(self, models: Path = MODELS):
        ap = cv2.ALIKED_Params()
        ap.inputSize = (640, 640)
        self.aliked = cv2.ALIKED_create(str(models / "aliked-n16rot-top1k-640.onnx"), ap)
        self.lightglue = cv2.LightGlueMatcher_create(str(models / "aliked_lightglue.onnx"), 0.0,
                                                     cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
        self.board = ScaleEstimator()
        self.refine_focal = True

    # 1. intrinsics -----------------------------------------------------------
    def board_obs(self, frames: list[np.ndarray]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        obs = {}
        for i, f in enumerate(frames):
            c, ids = self.board.detect(f)
            if c is not None:
                obs[i] = (c.astype(np.float32), ids)
        return obs

    def calibrate(self, obs: dict, size: tuple, K0: np.ndarray | None = None) -> dict:
        w, h = size
        K = K0.copy() if K0 is not None else np.array([[0.85 * w, 0, w / 2], [0, 0.85 * w, h / 2], [0, 0, 1]], float)
        if len(obs) < 3:
            return {"K": K, "dist": np.zeros(5), "rms": None, "frames": len(obs), "method": "prior"}
        board = self.board.board
        objp = [board.getChessboardCorners()[ids].astype(np.float32) for c, ids in obs.values()]
        imgp = [c for c, _ in obs.values()]
        # Few, similar board views constrain the focal well and the principal point poorly:
        # fix the centre and the higher distortion terms.
        # Phone video arrives rectified by the phone; a board seen from one shoulder
        # cannot separate lens distortion from pose, so distortion is estimated
        # only when the board was seen in many frames.
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
                 | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
        if len(obs) < 20:
            flags |= cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2
        rms, K, dist, _, _ = cv2.calibrateCamera(objp, imgp, size, K, np.zeros(5), flags=flags)
        return {"K": K, "dist": dist.ravel(), "rms": float(rms), "frames": len(obs), "method": "ChArUco"}

    @staticmethod
    def focal_from_pairs(feats: list[FrameFeatures], matches: dict, size: tuple) -> dict:
        """Focal length that makes every pair's essential matrix valid (Mendonca-Cipolla).

        With the principal point at the centre, K^T F K has two equal singular
        values only at the true focal. Across a walk-around the pairs see the
        scene from every side, so the median over pairs pins the focal far
        better than a board seen from one shoulder.
        """
        w, h = size
        Fs = []
        for (i, j), m in matches.items():
            F, mask = cv2.findFundamentalMat(feats[i].kpts[m[:, 0]], feats[j].kpts[m[:, 1]], cv2.USAC_MAGSAC, 1.0, 0.999)
            if F is not None and F.shape == (3, 3) and mask is not None and mask.sum() > 40:
                Fs.append(F)
        if len(Fs) < 10:
            return {"f": None, "pairs": len(Fs)}

        def cost(f):
            K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
            c = [np.linalg.svd(K.T @ F @ K, compute_uv=False) for F in Fs]
            return float(np.median([(s[0] - s[1]) / (s[0] + s[1]) for s in c]))
        grid = np.linspace(0.5 * w, 1.5 * w, 201)
        f0 = grid[int(np.argmin([cost(f) for f in grid]))]
        fine = np.linspace(f0 - 0.01 * w, f0 + 0.01 * w, 81)
        cs = [cost(f) for f in fine]
        return {"f": float(fine[int(np.argmin(cs))]), "pairs": len(Fs), "cost": round(min(cs), 5)}

    # 2. features and matches ------------------------------------------------
    def features(self, frames: list[np.ndarray], K: np.ndarray, dist: np.ndarray) -> list[FrameFeatures]:
        out = []
        for f in frames:
            s = WORK_W / f.shape[1]
            work = cv2.resize(f, (WORK_W, round(f.shape[0] * s)), interpolation=cv2.INTER_AREA)
            kps, desc = self.aliked.detectAndCompute(work, None)
            kw = np.float32([k.pt for k in kps]).reshape(-1, 2)
            full = cv2.undistortPoints((kw / s).reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2) if len(kw) else kw
            out.append(FrameFeatures(full, kw, desc if desc is not None else np.zeros((0, 128), np.float32),
                                     (work.shape[1], work.shape[0])))
        return out

    def match(self, a: FrameFeatures, b: FrameFeatures, K: np.ndarray) -> np.ndarray:
        """Geometrically verified matches as M x 2 index pairs."""
        if len(a.kpts) < 16 or len(b.kpts) < 16:
            return np.zeros((0, 2), int)
        try:
            self.lightglue.setPairInfo(a.kpts_work, b.kpts_work, a.work_size, b.work_size)
            m = self.lightglue.match(a.desc, b.desc)
        except cv2.error:
            return np.zeros((0, 2), int)
        if len(m) < 16:
            return np.zeros((0, 2), int)
        idx = np.array([[x.queryIdx, x.trainIdx] for x in m])
        E, mask = cv2.findEssentialMat(a.kpts[idx[:, 0]], b.kpts[idx[:, 1]], K, cv2.USAC_MAGSAC, 0.999, 1.0)
        if E is None or mask is None:
            return np.zeros((0, 2), int)
        return idx[mask.ravel().astype(bool)]

    def pairs(self, n: int) -> list[tuple[int, int]]:
        ps = {(i, j) for i in range(n) for j in range(i + 1, min(n, i + 1 + MATCH_WINDOW))}
        if LOOP and n > 2 * MATCH_WINDOW:
            ps |= {(i, j) for i in range(MATCH_WINDOW) for j in range(n - MATCH_WINDOW + i, n)}
        return sorted(ps)

    @staticmethod
    def tracks(n_kp: list[int], matches: dict) -> list[list[tuple[int, int]]]:
        off = np.cumsum([0] + n_kp)
        parent = np.arange(off[-1])

        def find(x):
            r = x
            while parent[r] != r:
                r = parent[r]
            while parent[x] != r:
                parent[x], x = r, parent[x]
            return r
        for (i, j), m in matches.items():
            for a, b in m:
                ra, rb = find(off[i] + a), find(off[j] + b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
        groups: dict[int, list] = {}
        for f in range(len(n_kp)):
            for k in range(n_kp[f]):
                groups.setdefault(find(off[f] + k), []).append((f, k))
        out = []
        for obs in groups.values():
            frames = [f for f, _ in obs]
            if len(obs) >= 2 and len(set(frames)) == len(frames):      # one keypoint per frame
                out.append(obs)
        return out

    # 3. incremental SfM -------------------------------------------------------
    def match_all(self, feats: list[FrameFeatures], K: np.ndarray) -> dict:
        out = {p: self.match(feats[p[0]], feats[p[1]], K) for p in self.pairs(len(feats))}
        return {p: m for p, m in out.items() if len(m) >= MIN_PAIR_INLIERS}

    def sfm(self, feats: list[FrameFeatures], K: np.ndarray, dist: np.ndarray, size: tuple,
            matches: dict | None = None) -> Reconstruction:
        n = len(feats)
        matches = self.match_all(feats, K) if matches is None else matches
        tracks = self.tracks([len(f.kpts) for f in feats], matches)
        rec = Reconstruction(K, dist, size, tracks=tracks, feats=feats)
        rec.stats["pairs"] = {f"{i}-{j}": len(m) for (i, j), m in matches.items()}
        track_of = [dict() for _ in range(n)]          # frame -> kp -> track
        for ti, obs in enumerate(tracks):
            for f, k in obs:
                track_of[f][k] = ti

        # initial pair: strong matches with real parallax
        best, init = -1, None
        for (i, j), m in matches.items():
            if j - i < 2 or len(m) < 60:
                continue
            p1, p2 = feats[i].kpts[m[:, 0]], feats[j].kpts[m[:, 1]]
            E, mask = cv2.findEssentialMat(p1, p2, K, cv2.USAC_MAGSAC, 0.999, 1.0)
            if E is None:
                continue
            _, R, t, mask2 = cv2.recoverPose(E[:3], p1, p2, K, mask=mask)
            par = np.median(np.linalg.norm(p1 - p2, axis=1))
            score = int((mask2 > 0).sum()) * min(par, 80)
            if score > best:
                best, init = score, (i, j, R, t.ravel())
        if init is None:
            raise RuntimeError("no initial pair with enough matches and parallax")
        i, j, R, t = init
        rec.R[i], rec.t[i] = np.eye(3), np.zeros(3)
        rec.R[j], rec.t[j] = R, t
        tri: dict[int, np.ndarray] = {}
        self._triangulate_new(rec, tri, {i, j})
        self._bundle(rec, tri, fix={i})

        while len(rec.R) < n:
            cand = []
            for f in range(n):
                if f in rec.R:
                    continue
                corr = [(k, ti) for k, ti in track_of[f].items() if ti in tri]
                cand.append((len(corr), f, corr))
            cand.sort(reverse=True)
            if not cand or cand[0][0] < 12:
                break
            _, f, corr = cand[0]
            obj = np.array([tri[ti] for _, ti in corr], np.float64)
            img = feats[f].kpts[[k for k, _ in corr]].astype(np.float64)
            ok, rv, tv, inl = cv2.solvePnPRansac(obj, img, K, None, reprojectionError=REPROJ_PX * 2,
                                                 iterationsCount=2000, flags=cv2.SOLVEPNP_SQPNP)
            if not ok or inl is None or len(inl) < 10:
                rec.stats.setdefault("unregistered", []).append(f)
                track_of[f] = {}
                continue
            rv, tv = cv2.solvePnPRefineLM(obj[inl.ravel()], img[inl.ravel()], K, None, rv, tv)
            rec.R[f], rec.t[f] = cv2.Rodrigues(rv)[0], tv.ravel()
            self._triangulate_new(rec, tri, set(rec.R))
            if len(rec.R) % 4 == 0:
                # The board sees a narrow range of views, so its focal is a prior; once the
                # orbit spans enough frames, every bundle refines it.
                self._bundle(rec, tri, fix={i}, focal=self.refine_focal and len(rec.R) >= 8)
        self._bundle(rec, tri, fix={i})
        self._triangulate_new(rec, tri, set(rec.R))
        self._bundle(rec, tri, fix={i}, focal=self.refine_focal)
        self._bundle(rec, tri, fix={i}, focal=self.refine_focal)
        rec.stats.pop("_rejected", None)
        ids = sorted(tri)
        rec.points = np.array([tri[t] for t in ids]) if ids else np.zeros((0, 3))
        rec.point_track = ids
        rec.stats.update(registered=len(rec.R), frames=n, points=len(ids), tracks=len(tracks))
        return rec

    def _triangulate_new(self, rec: Reconstruction, tri: dict, registered: set) -> None:
        for ti, obs in enumerate(rec.tracks):
            if ti in tri:
                continue
            seen = [(f, k) for f, k in obs if f in rec.R]
            if len(seen) < 2:
                continue
            cs = [rec.centre(f) for f, _ in seen]
            # widest baseline pair decides whether the ray angle is usable
            Ps = [rec.P(f) for f, _ in seen]
            xs = [rec.feats[f].kpts[k] for f, k in seen]
            X = triangulate_multi(Ps, xs)
            rays = [X - c for c in cs]
            cosang = min(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                         for a in rays for b in rays)
            if cosang > math.cos(math.radians(1.5)):
                continue
            good = True
            for (f, k), P in zip(seen, Ps):
                xh = P @ np.append(X, 1)
                if xh[2] <= 0 or np.linalg.norm(xh[:2] / xh[2] - rec.feats[f].kpts[k]) > REPROJ_PX * 2:
                    good = False
                    break
            if good:
                tri[ti] = X

    def _bundle(self, rec: Reconstruction, tri: dict, fix: set, focal: bool = False) -> None:
        """Bundle adjustment over the observations that agree with the current model.

        An observation joins only if it reprojects within GATE_PX now; after the
        solve, any that ends worse than 2 x REPROJ_PX is rejected for good, and
        points left with fewer than two observations are dropped. With `focal`,
        one shared focal length is refined too (the board calibrates it from a
        narrow range of views; the whole orbit pins it far better).
        """
        from scipy.optimize import least_squares
        from scipy.sparse import lil_matrix

        GATE_PX = 12.0
        bad = rec.stats.setdefault("_rejected", set())
        cams = sorted(rec.R)
        ci = {f: n for n, f in enumerate(cams)}
        obs_c, obs_t, obs_x = [], [], []
        for t in sorted(tri):
            for f, k in rec.tracks[t]:
                if f not in ci or (t, f) in bad:
                    continue
                x = rec.feats[f].kpts[k]
                xh = rec.P(f) @ np.append(tri[t], 1)
                if xh[2] <= 0 or np.linalg.norm(xh[:2] / xh[2] - x) > GATE_PX:
                    bad.add((t, f))
                    continue
                obs_c.append(ci[f]); obs_t.append(t); obs_x.append(x)
        cnt = collections_counter(obs_t)
        keep = [n for n, t in enumerate(obs_t) if cnt[t] >= 2]
        obs_c = np.array([obs_c[n] for n in keep]); obs_x = np.array([obs_x[n] for n in keep])
        pts = sorted({obs_t[n] for n in keep})
        pi = {t: n for n, t in enumerate(pts)}
        obs_p = np.array([pi[obs_t[n]] for n in keep])
        for t in list(tri):
            if t not in pi:
                tri.pop(t)
        K = rec.K
        nc = len(cams)
        x0 = np.concatenate([np.concatenate([cv2.Rodrigues(rec.R[f])[0].ravel(), rec.t[f]]) for f in cams]
                            + [np.array([tri[t] for t in pts]).ravel(), [K[0, 0]] if focal else []])
        free = np.array([f not in fix for f in cams])
        npt = 6 * nc + 3 * len(pts)

        def residual(x):
            cam = x[:6 * nc].reshape(nc, 6)
            X = x[6 * nc:npt].reshape(-1, 3)
            f = x[npt] if focal else K[0, 0]
            Rm = rodrigues(cam[:, :3])
            Xc = np.einsum("nij,nj->ni", Rm[obs_c], X[obs_p]) + cam[obs_c, 3:]
            proj = Xc[:, :2] / Xc[:, 2:3] * f + [K[0, 2], K[1, 2]]
            return (proj - obs_x).ravel()

        m = len(obs_c) * 2
        A = lil_matrix((m, len(x0)), dtype=int)
        r = np.arange(len(obs_c))
        fr = free[obs_c]
        for s in range(6):
            A[2 * r[fr], 6 * obs_c[fr] + s] = 1
            A[2 * r[fr] + 1, 6 * obs_c[fr] + s] = 1
        for s in range(3):
            A[2 * r, 6 * nc + 3 * obs_p + s] = 1
            A[2 * r + 1, 6 * nc + 3 * obs_p + s] = 1
        if focal:
            A[:, npt] = 1
        res = least_squares(residual, x0, jac_sparsity=A, loss="huber", f_scale=1.0, x_scale="jac",
                            method="trf", max_nfev=80 if focal else 40, verbose=0)
        cam = res.x[:6 * nc].reshape(nc, 6)
        X = res.x[6 * nc:npt].reshape(-1, 3)
        for n, f in enumerate(cams):
            rec.R[f], rec.t[f] = cv2.Rodrigues(cam[n, :3])[0], cam[n, 3:].copy()
        if focal:
            rec.K = K.copy()
            rec.K[0, 0] = rec.K[1, 1] = float(res.x[npt])
        for n, t in enumerate(pts):
            tri[t] = X[n]
        err = np.linalg.norm(residual(res.x).reshape(-1, 2), axis=1)
        for n in np.where(err > 2 * REPROJ_PX)[0]:
            bad.add((pts[obs_p[n]], cams[obs_c[n]]))
        good = err <= 2 * REPROJ_PX
        rec.stats["ba_rms_px"] = round(float(np.sqrt(np.mean(err[good] ** 2))), 3)
        rec.stats["ba_obs"] = int(good.sum())
        rec.stats["ba_rejected"] = len(bad)

    # 4. metric scale from the board ------------------------------------------
    def scale(self, rec: Reconstruction, obs: dict) -> None:
        board = self.board.board.getChessboardCorners()
        frames = [f for f in obs if f in rec.R]
        if len(frames) < 3:
            rec.scale_detail = {"board_frames": len(frames), "reason": "board registered in fewer than 3 frames"}
            rec.scale_rel_err = 1.0
            return
        und = {f: cv2.undistortPoints(obs[f][0].reshape(-1, 1, 2), rec.K, rec.dist, P=rec.K).reshape(-1, 2)
               for f in frames}

        def fit(use: list[int]) -> tuple[float, float]:
            by_id: dict[int, list] = {}
            for f in use:
                for x, cid in zip(und[f], obs[f][1]):
                    by_id.setdefault(int(cid), []).append((rec.P(f), x))
            ids = [c for c, v in by_id.items() if len(v) >= 2]
            if len(ids) < 6:
                return float("nan"), float("nan")
            X = np.array([triangulate_multi([p for p, _ in by_id[c]], [x for _, x in by_id[c]]) for c in ids])
            s, R, t = umeyama(board[ids].astype(float), X)        # units per metre
            resid = np.linalg.norm((s * (R @ board[ids].T)).T + t - X, axis=1) / s
            return 1.0 / s, float(np.sqrt(np.mean(resid ** 2)))

        # One ruler per board placement. The protocol moves the board from the
        # left shoulder to the right, so frames group by the board's normal in
        # the reconstruction. Each placement gives its own scale; jackknife
        # within a placement sees corner noise, the gap between placements sees
        # how the reconstruction bends between the two sides of the animal.
        normals = {}
        for f in frames:
            ok, rv, _ = cv2.solvePnP(board[obs[f][1]].astype(np.float64), und[f].astype(np.float64), rec.K, None,
                                     flags=cv2.SOLVEPNP_IPPE)
            if ok:
                normals[f] = rec.R[f].T @ cv2.Rodrigues(rv)[0][:, 2]
        groups: list[list[int]] = []
        for f in sorted(normals):
            for g in groups:
                if float(normals[f] @ normals[g[0]]) > 0.5:
                    g.append(f)
                    break
            else:
                groups.append([f])
        groups = [g for g in groups if len(g) >= 3]
        if not groups:
            groups = [frames]
        per = []
        for g in groups:
            full, fit_rms = fit(g)
            jk = np.array([v for v in (fit([x for x in g if x != f])[0] for f in g) if np.isfinite(v)])
            n = len(jk)
            se = float(np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2))) if n >= 2 else float("nan")
            if np.isfinite(full):
                per.append({"frames": len(g), "metres_per_unit": full, "jackknife_rel_err": se / full if np.isfinite(se) else 1.0,
                            "fit_rms_mm": round(fit_rms * 1000, 2)})
        if not per:
            rec.scale_rel_err = 1.0
            rec.scale_detail = {"board_frames": len(frames), "reason": "board corners did not triangulate"}
            return
        w = np.array([1 / max(p["jackknife_rel_err"], 1e-4) ** 2 for p in per])
        m = np.array([p["metres_per_unit"] for p in per])
        rec.metres_per_unit = float((w * m).sum() / w.sum())
        jk_rel = float(1 / np.sqrt(w.sum()))
        if len(per) >= 2:
            between = float(np.std(m, ddof=1) / rec.metres_per_unit)
            rel = math.sqrt(jk_rel ** 2 + between ** 2)
        else:
            # One placement cannot see bending: assume the floor measured on synthetic walk-arounds.
            between = None
            rel = math.sqrt(jk_rel ** 2 + SINGLE_BOARD_FLOOR ** 2)
        rec.scale_rel_err = rel
        rec.scale_detail = {"board_frames": len(frames), "placements": len(per),
                            "per_placement": [{**p, "metres_per_unit": round(p["metres_per_unit"], 6),
                                               "jackknife_rel_err": round(p["jackknife_rel_err"], 5)} for p in per],
                            "between_placements_rel": between and round(between, 5), "scale_rel_err": round(rel, 5),
                            "method": "board corners as tracks; jackknife within, spread between placements"}

    # 5. ground plane ------------------------------------------------------------
    def ground(self, rec: Reconstruction, thresh_m: float = 0.03, iters: int = 3000, seed: int = 0) -> None:
        P = rec.points * rec.metres_per_unit
        cams = np.array([rec.centre(f) for f in rec.R]) * rec.metres_per_unit
        rng = np.random.default_rng(seed)
        best, bn, bd = 0, None, 0.0
        for _ in range(iters):
            s = P[rng.choice(len(P), 3, replace=False)]
            nrm = np.cross(s[1] - s[0], s[2] - s[0])
            if np.linalg.norm(nrm) < 1e-9:
                continue
            nrm /= np.linalg.norm(nrm)
            d = -nrm @ s[0]
            side = cams @ nrm + d
            if not (np.all(side > 0.3) or np.all(side < -0.3)):      # every camera on one side, held above it
                continue
            cnt = int((np.abs(P @ nrm + d) < thresh_m).sum())
            if cnt > best:
                best, bn, bd = cnt, nrm, d
        if bn is None:
            rec.ground = {"found": False}
            return
        inl = np.abs(P @ bn + bd) < thresh_m
        c = P[inl].mean(0)
        _, _, vt = np.linalg.svd(P[inl] - c)
        nrm = vt[2]
        if np.mean(cams @ nrm - c @ nrm) < 0:
            nrm = -nrm
        orbit = cams.mean(0)
        origin = orbit - ((orbit - c) @ nrm) * nrm
        x = np.cross([0, 1, 0], nrm) if abs(nrm[1]) < 0.9 else np.cross([1, 0, 0], nrm)
        x /= np.linalg.norm(x)
        y = np.cross(nrm, x)
        rec.ground = {"found": True, "inliers": int(inl.sum()), "R_world": np.stack([x, y, nrm]), "origin": origin,
                      "camera_height_m": round(float(np.median(cams @ nrm - c @ nrm)), 3)}

    # -- all of it -------------------------------------------------------------------
    def run(self, frames: list[np.ndarray], K0: np.ndarray | None = None, cache: Path | None = None) -> Reconstruction:
        import pickle
        size = (frames[0].shape[1], frames[0].shape[0])
        obs = self.board_obs(frames)
        cal = self.calibrate(obs, size, K0)
        K, dist = cal["K"], cal["dist"]
        cached = cache and (cache / "sfm_cache.pkl").exists()
        if cached:
            feats, matches = pickle.loads((cache / "sfm_cache.pkl").read_bytes())
        else:
            feats = self.features(frames, K, dist)
            matches = self.match_all(feats, K)
            if cache:
                (cache / "sfm_cache.pkl").write_bytes(pickle.dumps((feats, matches)))
        foc = self.focal_from_pairs(feats, matches, size)
        board_ok = cal["method"] == "ChArUco" and cal["rms"] is not None and cal["rms"] < 1.0 and cal["frames"] >= 5
        if foc["f"] and not board_ok:
            # No trustworthy board calibration: the pairs set the focal.
            K = K.copy()
            K[0, 0] = K[1, 1] = foc["f"]
        focal_source = "board" if board_ok or not foc["f"] else "pairs"
        rec = self.sfm(feats, K, dist, size, matches)
        rec.stats["calibration"] = {k: v for k, v in cal.items() if k not in ("K", "dist")}
        rec.stats["focal"] = {"board_px": round(float(cal["K"][0, 0]), 1), "pairs_px": foc["f"] and round(foc["f"], 1),
                              "pairs": foc["pairs"], "final_px": round(float(rec.K[0, 0]), 1), "source": focal_source,
                              "board_vs_pairs": foc["f"] and round(float(cal["K"][0, 0]) / foc["f"] - 1, 4)}
        self.scale(rec, obs)
        self.ground(rec)
        rec.to_metric()
        return rec
