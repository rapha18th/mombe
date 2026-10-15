"""Dense metric surface: calibrated stereo between neighbouring frames, fused by TSDF.

The reconstruction already knows every camera in metres, so each pair of
neighbouring frames is a calibrated stereo rig with a baseline of about a
third of a metre. Rectified with stereoRectify and matched with StereoSGBM, a
quarter-pixel disparity error is a few millimetres of depth on the animal.
Depth-Anything-V2 (OpenCV DNN) runs as the second opinion: aligned per frame to
the stereo depth, it fills the pixels stereo leaves empty. OpenCV 5's TSDF
volume fuses everything in the ground frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from mombe.body.recon import Reconstruction

MODELS = Path(__file__).resolve().parents[2] / "models"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
DA_H = 518                 # network height; width keeps the aspect, a multiple of 14
FUSE_W = int(__import__("os").environ.get("MOMBE_FUSE_W", 1920))               # fusion resolution
VOXEL_M = 0.01
VOLUME_M = (3.6, 3.6, 2.4)  # x, y, z extent centred on the walk-around, ground upwards
GROUND_CLEAR_M = 0.03       # the volume starts just above the ground plane
MAX_DEPTH_M = 5.0
MIN_DEPTH_M = 1.2
import os as _os
PAIR_STEP = int(_os.environ.get("MOMBE_PAIR_STEP", 2))   # stereo partner: frames ahead; wider baseline, finer depth
EDGE_PX = int(_os.environ.get("MOMBE_EDGE_PX", 9))   # stereo fattens silhouettes by half a block; trim past it


@dataclass
class DenseResult:
    points: np.ndarray                       # N x 3 metric, ground frame
    normals: np.ndarray
    frames: list = field(default_factory=list)   # per-frame alignment report
    seconds: float = 0.0


class DenseFusion:
    def __init__(self, models: Path = MODELS):
        cv2.ocl.setUseOpenCL(False)              # the hub has no OpenCL; keep every machine on one path
        self.net = cv2.dnn.readNetFromONNX(str(models / "depth_anything_v2_small.onnx"), cv2.dnn.ENGINE_NEW)

    def disparity(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        dw = round(DA_H * w / h / 14) * 14
        blob = cv2.dnn.blobFromImage(bgr, 1 / 255.0, (dw, DA_H), swapRB=True)
        self.net.setInput(((blob - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32))
        return self.net.forward().reshape(DA_H, dw)

    @staticmethod
    def align(disp: np.ndarray, uv: np.ndarray, z: np.ndarray, full_size: tuple) -> tuple[float, float, dict]:
        """Robust fit of 1/z = a d + b at the sparse points; returns a, b and the fit report."""
        sx, sy = disp.shape[1] / full_size[0], disp.shape[0] / full_size[1]
        d = cv2.remap(disp, (uv[:, 0] * sx).astype(np.float32).reshape(-1, 1),
                      (uv[:, 1] * sy).astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR).ravel()
        inv = 1.0 / z
        keep = (z > 0.3) & (z < MAX_DEPTH_M)
        a = b = float("nan")
        for _ in range(4):
            if keep.sum() < 20:
                return float("nan"), float("nan"), {"points": int(keep.sum()), "ok": False}
            A = np.c_[d[keep], np.ones(keep.sum())]
            a, b = np.linalg.lstsq(A, inv[keep], rcond=None)[0]
            r = np.abs(a * d + b - inv) / inv
            mad = np.median(r[keep]) + 1e-6
            keep = keep & (r < 3 * mad + 0.02)
        zfit = 1.0 / np.maximum(a * d[keep] + b, 1e-6)
        rel = np.abs(zfit - z[keep]) / z[keep]
        return float(a), float(b), {"points": int(keep.sum()), "median_rel_err": round(float(np.median(rel)), 4),
                                     "ok": bool(a > 0 and np.median(rel) < 0.05)}

    @staticmethod
    def edge_mask(depth: np.ndarray, rel: float = 0.04) -> np.ndarray:
        """Pixels where depth jumps: the network blends animal into background there."""
        gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3) / 8
        gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3) / 8
        jump = np.sqrt(gx * gx + gy * gy) / np.maximum(depth, 1e-3)
        return cv2.dilate((jump > rel).astype(np.uint8), np.ones((EDGE_PX, EDGE_PX), np.uint8)) > 0

    @staticmethod
    def stereo_depth(rec: Reconstruction, f1: int, f2: int, img1: np.ndarray, img2: np.ndarray,
                     Kf: np.ndarray, size: tuple) -> np.ndarray:
        """Metric depth in frame f1's own camera (fusion resolution) from the rectified pair (f1, f2)."""
        R1, t1, R2, t2 = rec.R[f1], rec.t[f1], rec.R[f2], rec.t[f2]
        Rr = R2 @ R1.T
        tr = t2 - Rr @ t1
        w, h = size
        z5 = np.zeros(5)
        Rl, Rrr, Pl, Pr, Q, _, _ = cv2.stereoRectify(Kf, z5, Kf, z5, (w, h), Rr, tr.reshape(3, 1), alpha=1,
                                                     flags=cv2.CALIB_ZERO_DISPARITY)
        m1 = cv2.initUndistortRectifyMap(Kf, z5, Rl, Pl, (w, h), cv2.CV_32FC1)
        m2 = cv2.initUndistortRectifyMap(Kf, z5, Rrr, Pr, (w, h), cv2.CV_32FC1)
        g1 = cv2.remap(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), *m1, cv2.INTER_LINEAR)
        g2 = cv2.remap(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), *m2, cv2.INTER_LINEAR)
        if abs(Pr[1, 3]) > abs(Pr[0, 3]):
            return np.zeros((h, w), np.float32)              # a walk-around moves sideways; skip vertical pairs
        fB = abs(Pr[0, 3])                                   # focal x baseline, rectified
        dmin = max(0, int(fB / MAX_DEPTH_M) // 16 * 16)
        ndisp = int(np.clip(np.ceil((fB / MIN_DEPTH_M - dmin) / 16) * 16, 16, 256))
        if dmin + ndisp > w // 2:
            return np.zeros((h, w), np.float32)
        bs = 5
        sgbm = cv2.StereoSGBM_create(minDisparity=dmin, numDisparities=ndisp, blockSize=bs, P1=8 * bs * bs,
                                     P2=32 * bs * bs, disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100,
                                     speckleRange=2, mode=cv2.StereoSGBM_MODE_SGBM_3WAY)
        if Pr[0, 3] < 0:                                     # frame 1 is the left camera
            disp = sgbm.compute(g1, g2).astype(np.float32) / 16.0
        else:                                                # frame 1 is the right camera: mirror both
            disp = sgbm.compute(g1[:, ::-1].copy(), g2[:, ::-1].copy()).astype(np.float32)[:, ::-1] / 16.0
        # Dense depth in frame 1's own pixels: each pixel's ray, rotated into the
        # rectified view, picks up its disparity there (nearest, so edges stay sharp).
        v, u = np.mgrid[0:h, 0:w].astype(np.float64)
        rays = np.linalg.inv(Kf) @ np.vstack([u.ravel(), v.ravel(), np.ones(u.size)])
        rr = Rl @ rays
        pr = Pl[:, :3] @ rr
        mx = (pr[0] / pr[2]).reshape(h, w).astype(np.float32)
        my = (pr[1] / pr[2]).reshape(h, w).astype(np.float32)
        ds = cv2.remap(disp, mx, my, cv2.INTER_NEAREST, borderValue=0)
        valid = ds > max(dmin, 1)
        # depth along the rectified axis, then along frame 1's optical axis
        zr = np.where(valid, fB / np.maximum(ds, 1e-3), 0)
        depth = (zr * (rays[2].reshape(h, w) / rr[2].reshape(h, w))).astype(np.float32)
        depth[~valid | (depth > MAX_DEPTH_M) | (depth < MIN_DEPTH_M * 0.8)] = 0
        return depth

    @staticmethod
    def clean_edges(depth: np.ndarray) -> np.ndarray:
        """Drop flying pixels at depth jumps; holes are filled only to test for jumps."""
        filled = depth.copy()
        for _ in range(3):
            grown = cv2.dilate(filled, np.ones((3, 3), np.uint8))
            filled = np.where(filled > 0, filled, grown)
        out = depth.copy()
        out[DenseFusion.edge_mask(np.where(filled > 0, filled, MAX_DEPTH_M))] = 0
        return out

    def run_stereo(self, rec: Reconstruction, frames: list[np.ndarray], fill: bool = False) -> DenseResult:
        """Fuse stereo depth from every pair of neighbouring registered frames."""
        import time
        t0 = time.time()
        W, H = rec.size
        s = FUSE_W / W
        fw, fh = FUSE_W, round(H * s)
        Kf = rec.K.copy() * s
        Kf[2, 2] = 1
        vol = self._volume(Kf, fw, fh)
        small = {f: cv2.resize(frames[f], (fw, fh), interpolation=cv2.INTER_AREA) for f in rec.R}
        reg = sorted(rec.R)
        depths: dict[int, np.ndarray] = {}
        for n, f1 in enumerate(reg):
            f2 = reg[(n + PAIR_STEP) % len(reg)]
            if np.linalg.norm(rec.centre(f1) - rec.centre(f2)) < 0.1:
                continue
            depths[f1] = self.clean_edges(self.stereo_depth(rec, f1, f2, small[f1], small[f2], Kf, (fw, fh)))
        report = []
        for n, f1 in enumerate(reg):
            if f1 not in depths:
                continue
            others = [reg[(n + k) % len(reg)] for k in (-3, -2, 2, 3)]
            depth = self.agree(rec, Kf, f1, depths[f1], {g: depths[g] for g in others if g in depths})
            report.append({"frame": f1, "valid_raw": round(float((depths[f1] > 0).mean()), 3),
                           "valid": round(float((depth > 0).mean()), 3)})
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = rec.R[f1].T
            c2w[:3, 3] = -rec.R[f1].T @ rec.t[f1]
            vol.integrate(depth.astype(np.float32), c2w)
        pts, nrm = vol.fetchPointsNormals()
        pts = pts.reshape(-1, pts.shape[-1])[:, :3] if pts is not None and len(pts) else np.zeros((0, 3))
        nrm = nrm.reshape(-1, nrm.shape[-1])[:, :3] if nrm is not None and len(nrm) else np.zeros((0, 3))
        return DenseResult(pts.astype(np.float64), nrm.astype(np.float64), report, round(time.time() - t0, 1))

    @staticmethod
    def agree(rec: Reconstruction, Kf: np.ndarray, f: int, depth: np.ndarray, others: dict,
              rel: float = 0.02, need: int = 2) -> np.ndarray:
        """Keep a depth pixel only where at least `need` other views see the same surface.

        Each pixel is lifted to the world, projected into a neighbouring frame and
        compared with that frame's own stereo depth there. Mismatches against a
        far, weakly textured background agree with nothing and drop out.
        """
        h, w = depth.shape
        v, u = np.nonzero(depth > 0)
        if len(u) == 0 or not others:
            return depth
        z = depth[v, u]
        Xc = np.linalg.inv(Kf) @ np.vstack([u, v, np.ones(len(u))]) * z
        Xw = rec.R[f].T @ (Xc - rec.t[f][:, None])
        votes = np.zeros(len(u), int)
        for g, dg in others.items():
            Xg = rec.R[g] @ Xw + rec.t[g][:, None]
            pg = Kf @ Xg
            ug = np.round(pg[0] / pg[2]).astype(int)
            vg = np.round(pg[1] / pg[2]).astype(int)
            inside = (Xg[2] > 0) & (ug >= 0) & (ug < w) & (vg >= 0) & (vg < h)
            seen = np.zeros(len(u))
            seen[inside] = dg[vg[inside], ug[inside]]
            votes += (seen > 0) & (np.abs(seen - Xg[2]) < rel * Xg[2])
        out = np.zeros_like(depth)
        keep = votes >= need
        out[v[keep], u[keep]] = z[keep]
        return out

    @staticmethod
    def _volume(Kf: np.ndarray, fw: int, fh: int):
        vs = cv2.VolumeSettings(cv2.VolumeType_TSDF)
        vs.setVoxelSize(VOXEL_M)
        vs.setVolumeResolution(np.array([round(v / VOXEL_M) for v in VOLUME_M], np.int32))
        vs.setTsdfTruncateDistance(4 * VOXEL_M)
        vs.setCameraIntegrateIntrinsics(Kf.astype(np.float32))
        vs.setIntegrateWidth(fw)
        vs.setIntegrateHeight(fh)
        vs.setDepthFactor(1.0)
        vs.setMaxDepth(MAX_DEPTH_M)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = [-VOLUME_M[0] / 2, -VOLUME_M[1] / 2, GROUND_CLEAR_M]
        vs.setVolumePose(pose)
        return cv2.Volume(cv2.VolumeType_TSDF, vs)

    def run(self, rec: Reconstruction, frames: list[np.ndarray], step: int = 2) -> DenseResult:
        import time
        t0 = time.time()
        W, H = rec.size
        s = FUSE_W / W
        fw, fh = FUSE_W, round(H * s)
        Kf = rec.K.copy() * s
        Kf[2, 2] = 1
        vs = cv2.VolumeSettings(cv2.VolumeType_TSDF)
        res = np.array([round(v / VOXEL_M) for v in VOLUME_M], np.int32)
        vs.setVoxelSize(VOXEL_M)
        vs.setVolumeResolution(res)
        vs.setTsdfTruncateDistance(4 * VOXEL_M)
        vs.setCameraIntegrateIntrinsics(Kf.astype(np.float32))
        vs.setIntegrateWidth(fw)
        vs.setIntegrateHeight(fh)
        vs.setDepthFactor(1.0)
        vs.setMaxDepth(MAX_DEPTH_M)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = [-VOLUME_M[0] / 2, -VOLUME_M[1] / 2, GROUND_CLEAR_M]
        vs.setVolumePose(pose)
        vol = cv2.Volume(cv2.VolumeType_TSDF, vs)

        # sparse observations per frame
        per_frame: dict[int, list] = {}
        for n, tid in enumerate(rec.point_track):
            for f, k in rec.tracks[tid]:
                if f in rec.R:
                    per_frame.setdefault(f, []).append((n, k))
        report = []
        for f in sorted(rec.R)[::step]:
            obs = per_frame.get(f, [])
            if len(obs) < 20:
                continue
            X = rec.points[[n for n, _ in obs]]
            z = (X @ rec.R[f].T + rec.t[f])[:, 2]
            uv = np.array([rec.feats_kpts[f][k] for _, k in obs])
            # Undistorted keypoints vs distorted image: fine at the distortion this pipeline accepts.
            disp = self.disparity(frames[f])
            a, b, rep = self.align(disp, uv, z, (W, H))
            rep["frame"] = f
            report.append(rep)
            if not rep["ok"]:
                continue
            dfull = cv2.resize(disp, (fw, fh), interpolation=cv2.INTER_LINEAR)
            depth = 1.0 / np.maximum(a * dfull + b, 1e-6)
            depth[(a * dfull + b) <= 1.0 / MAX_DEPTH_M] = 0
            depth[self.edge_mask(depth)] = 0
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = rec.R[f].T
            c2w[:3, 3] = -rec.R[f].T @ rec.t[f]
            vol.integrate(depth.astype(np.float32), c2w)
        pts, nrm = vol.fetchPointsNormals()
        pts = pts.reshape(-1, pts.shape[-1])[:, :3] if pts is not None and len(pts) else np.zeros((0, 3))
        nrm = nrm.reshape(-1, nrm.shape[-1])[:, :3] if nrm is not None and len(nrm) else np.zeros((0, 3))
        return DenseResult(pts.astype(np.float64), nrm.astype(np.float64), report, round(time.time() - t0, 1))
