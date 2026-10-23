"""OpenCV worker for op-level diffs: one process per build, driven over stdin/stdout.

Protocol: 8-byte little-endian length, then a pickle. Request (cmd, args...);
reply is a pickle of the result or {"error": text}. Nothing else is written to
stdout. Started by ops_diff.py through run_env.sh, so each worker loads its own
cv2 build.
"""

from __future__ import annotations

import os
import pickle
import struct
import sys
import traceback
from pathlib import Path

import numpy as np

CODE = Path(os.environ.get("MOMBE_CODE", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(CODE))
OUT = sys.stdout.buffer
sys.stdout = sys.stderr          # stray prints must never corrupt the pipe

import cv2  # noqa: E402

from mombe.vision import quality  # noqa: E402
from mombe.vision.board import ScaleEstimator, make_board  # noqa: E402
from mombe.vision.muzzle import IMAGENET_MEAN, IMAGENET_STD, preprocess  # noqa: E402

MODELS = CODE / "models"
THREADS = cv2.getNumThreads()


class W:
    def __init__(self):
        ap = cv2.ALIKED_Params()
        ap.inputSize = (640, 640)
        self.aliked = cv2.ALIKED_create(str(MODELS / "aliked-n16rot-top1k-640.onnx"), ap)
        self.lg = cv2.LightGlueMatcher_create(str(MODELS / "aliked_lightglue.onnx"), 0.0,
                                              cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
        self.dino_net = cv2.dnn.readNetFromONNX(str(MODELS / "dinov2_small.onnx"), cv2.dnn.ENGINE_NEW)
        self.alinet = cv2.dnn.readNetFromONNX(str(MODELS / "aliked-n16rot-top1k-640.onnx"), cv2.dnn.ENGINE_NEW)
        self.se = ScaleEstimator()
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.board_img = make_board().generateImage((700, 1000), marginSize=0, borderBits=1)

    def info(self):
        return {"opencv": cv2.__version__, "file": cv2.__file__, "numpy": np.__version__, "threads": THREADS,
                "useOptimized": cv2.useOptimized()}

    def set_optimized(self, flag):
        cv2.setUseOptimized(bool(flag))
        return cv2.useOptimized()

    def chain(self, path):
        o = {}
        img = cv2.imread(path)
        o["imread"] = img
        h, w = img.shape[:2]
        r960 = cv2.resize(img, (960, round(h * 960 / w)), interpolation=cv2.INTER_AREA)
        o["resize_area_960"] = r960
        s = 640 / max(h, w)
        r640 = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        o["resize_area_640"] = r640
        lab = cv2.cvtColor(r640, cv2.COLOR_BGR2LAB)
        o["cvt_bgr2lab"] = lab
        o["clahe_L"] = self.clahe.apply(lab[:, :, 0])
        o["preprocess"] = preprocess(img)
        r512 = quality._normalise(img)
        o["resize_area_512"] = r512
        g512 = cv2.cvtColor(r512, cv2.COLOR_BGR2GRAY)
        o["laplacian_var"] = float(cv2.Laplacian(g512, cv2.CV_64F, ksize=3).var())
        q = quality.assess(img)
        o["quality"] = {"sharpness": q.sharpness, "glare": q.glare, "mean": q.mean, "clipped": q.clipped, "issues": list(q.issues)}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        o["cvt_bgr2gray"] = gray
        o["gaussian_7x7"] = cv2.GaussianBlur(gray, (7, 7), 0)
        g640 = cv2.cvtColor(r640, cv2.COLOR_BGR2GRAY)
        o["gray640"] = g640
        o["resize_linear_up2_u8"] = cv2.resize(g640, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        o["resize_linear_f32"] = cv2.resize(g640.astype(np.float32) / 255.0, (1920, round(1920 * g640.shape[0] / g640.shape[1])),
                                            interpolation=cv2.INTER_LINEAR)
        o["blob_dinov2"] = cv2.dnn.blobFromImage(o["preprocess"], 1 / 255.0, (224, 224), swapRB=True, crop=False)
        dw = round(518 * r960.shape[1] / r960.shape[0] / 14) * 14
        o["blob_depthanything"] = cv2.dnn.blobFromImage(r960, 1 / 255.0, (dw, 518), swapRB=True)
        hh, ww = r960.shape[:2]
        K = np.array([[0.9 * ww, 0, ww / 2], [0, 0.9 * ww, hh / 2], [0, 0, 1]])
        Rr = cv2.Rodrigues(np.array([0.01, -0.02, 0.005]))[0]
        mx, my = cv2.initUndistortRectifyMap(K, np.array([-0.08, 0.02, 0, 0, 0]), Rr, K, (ww, hh), cv2.CV_32FC1)
        o["undistort_map"] = np.stack([mx, my])
        o["remap_linear"] = cv2.remap(r960, mx, my, cv2.INTER_LINEAR)
        H = np.array([[0.98, 0.03, 12], [-0.02, 0.99, 8], [1e-5, 2e-5, 1]])
        o["warp_perspective"] = cv2.warpPerspective(r960, H, (ww, hh))
        o["gftt"] = cv2.goodFeaturesToTrack(g640, 400, 0.01, 7)
        g960 = cv2.cvtColor(r960, cv2.COLOR_BGR2GRAY)
        sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=5, P1=200, P2=800, disp12MaxDiff=1,
                                     uniquenessRatio=10, speckleWindowSize=100, speckleRange=2, mode=cv2.StereoSGBM_MODE_SGBM_3WAY)
        o["stereo_sgbm"] = sgbm.compute(g960, np.ascontiguousarray(np.roll(g960, -9, axis=1)))
        o["sobel_f32"] = cv2.Sobel(g960.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        return o

    def composite(self, img, i):
        """The board warped onto the photo at a pose fixed by the image index."""
        rng = np.random.default_rng(1000 + i)
        h, w = img.shape[:2]
        bw = rng.uniform(0.18, 0.32) * w
        cx, cy = rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h
        src = np.float32([[0, 0], [700, 0], [700, 1000], [0, 1000]])
        jit = rng.uniform(-0.12, 0.12, (4, 2)) * bw
        dst = np.float32([[cx - bw / 2, cy - bw * 0.7], [cx + bw / 2, cy - bw * 0.7], [cx + bw / 2, cy + bw * 0.7],
                          [cx - bw / 2, cy + bw * 0.7]]) + jit.astype(np.float32)
        Hm = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(self.board_img, Hm, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        mask = cv2.warpPerspective(np.full_like(self.board_img, 255), Hm, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out = np.where(mask > 0, warped, g)
        return out

    def detect(self, gray):
        cv2.setNumThreads(1)                 # as in Reconstructor.run
        try:
            c, ids = self.se.detect(gray)
        finally:
            cv2.setNumThreads(THREADS)
        return {"corners": c, "ids": ids}

    def subpix(self, gray, pts):
        p = pts.astype(np.float32).copy()
        return cv2.cornerSubPix(gray, p, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def lk(self, g0, g1, pts):
        nxt, st, err = cv2.calcOpticalFlowPyrLK(g0, g1, pts.astype(np.float32), None, winSize=(21, 21), maxLevel=3)
        return {"next": nxt, "status": st, "err": err}

    def homog(self, p1, p2):
        H, mask = cv2.findHomography(p1, p2, cv2.USAC_MAGSAC, 4.0)
        return {"H": H, "mask": mask}

    def ess(self, p1, p2, K):
        cv2.setNumThreads(1)
        try:
            E, mask = cv2.findEssentialMat(p1, p2, K, cv2.USAC_MAGSAC, 0.999, 1.0)
        finally:
            cv2.setNumThreads(THREADS)
        return {"E": E, "mask": mask}

    def fundamental(self, p1, p2):
        cv2.setNumThreads(1)
        try:
            F, mask = cv2.findFundamentalMat(p1, p2, cv2.USAC_MAGSAC, 1.0, 0.999)
        finally:
            cv2.setNumThreads(THREADS)
        return {"F": F, "mask": mask}

    def aliked_kp(self, img):
        kps, desc = self.aliked.detectAndCompute(img, None)
        return {"pts": np.float32([k.pt for k in kps]).reshape(-1, 2), "resp": np.float32([k.response for k in kps]),
                "desc": desc}

    def aliked_net(self, blob):
        self.alinet.setInput(blob)
        outs = self.alinet.forward(list(self.alinet.getUnconnectedOutLayersNames()))
        return {n: np.array(o) for n, o in zip(self.alinet.getUnconnectedOutLayersNames(), outs)}

    def dino(self, blob):
        b = ((blob - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)
        self.dino_net.setInput(b)
        tok = self.dino_net.forward()[0]
        v = np.concatenate([tok[0], tok[1:].mean(0)]).astype(np.float32)
        return {"tokens": tok, "emb": v / (np.linalg.norm(v) + 1e-8)}

    def lightglue(self, ka, kb, da, db, sa, sb):
        try:
            self.lg.setPairInfo(ka, kb, sa, sb)
            m = self.lg.match(da, db)
        except cv2.error as e:
            return {"error": str(e)[:80]}
        return {"idx": np.array([[x.queryIdx, x.trainIdx] for x in m], int).reshape(-1, 2),
                "dist": np.float32([x.distance for x in m])}


def read():
    hdr = sys.stdin.buffer.read(8)
    if len(hdr) < 8:
        return None
    n = struct.unpack("<Q", hdr)[0]
    return pickle.loads(sys.stdin.buffer.read(n))


def write(obj):
    b = pickle.dumps(obj, protocol=4)
    OUT.write(struct.pack("<Q", len(b)))
    OUT.write(b)
    OUT.flush()


def main():
    w = W()
    while True:
        req = read()
        if req is None or req[0] == "quit":
            break
        try:
            write(getattr(w, req[0])(*req[1:]))
        except Exception as e:  # report and keep serving
            write({"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-600:]})


if __name__ == "__main__":
    main()
