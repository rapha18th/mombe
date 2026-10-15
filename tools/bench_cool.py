"""COOL versus stock OpenCV 5 on the same Graviton4 instance type.

Runs on a hub instance against its own models and gallery images:

  kernels   core image operations (resize, colour, blur, threshold, contours,
            CLAHE, warp, Laplacian, optical flow) at 1 thread and at all cores
  pipeline  Mombe's stages: quality gate, preprocess, ALIKED, DINOv2, LightGlue
            + MAGSAC verify, end-to-end identify, continuity, ChArUco calibration
  registry  feature extraction throughput (enrolments per second)

Each measurement: warm-up, then N timed repeats; median and p90 in ms.

Usage: python tools/bench_cool.py --out results.json [--repeats 30]
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def timeit(fn, repeats: int, warmup: int = 3) -> dict:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return {"median_ms": round(statistics.median(ts), 3), "p90_ms": round(ts[int(0.9 * (len(ts) - 1))], 3), "n": repeats}


def images(n: int) -> list[np.ndarray]:
    meta = json.loads((ROOT / "var" / "gallery.json").read_text())
    out = []
    for im in meta["image_ids"]:
        p = ROOT / "data" / "muzzle" / "_".join(im.split("_")[:2]) / f"{im}.jpg"
        img = cv2.imread(str(p))
        if img is not None and min(img.shape[:2]) >= 400:
            out.append(img)
        if len(out) >= n:
            break
    return out


def kernels(img: np.ndarray, repeats: int) -> dict:
    big = cv2.resize(img, (1920, 1440))
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (640, 480))
    nxt = np.roll(small, 3, axis=1)
    pts = cv2.goodFeaturesToTrack(small, 400, 0.01, 7)
    H = np.array([[0.98, 0.03, 12], [-0.02, 0.99, 8], [1e-5, 2e-5, 1]])
    clahe = cv2.createCLAHE(2.5, (8, 8))
    ops = {
        "resize_area_1920_to_640": lambda: cv2.resize(big, (640, 480), interpolation=cv2.INTER_AREA),
        "resize_linear_up_2x": lambda: cv2.resize(small, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR),
        "cvtColor_bgr2gray": lambda: cv2.cvtColor(big, cv2.COLOR_BGR2GRAY),
        "cvtColor_bgr2lab": lambda: cv2.cvtColor(big, cv2.COLOR_BGR2LAB),
        "gaussian_blur_7x7": lambda: cv2.GaussianBlur(gray, (7, 7), 0),
        "adaptive_threshold_gaussian": lambda: cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5),
        "find_contours": lambda: cv2.findContours(cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE),
        "clahe": lambda: clahe.apply(gray),
        "warp_perspective": lambda: cv2.warpPerspective(big, H, (1920, 1440)),
        "laplacian_var": lambda: cv2.Laplacian(gray, cv2.CV_64F).var(),
        "lk_optical_flow": lambda: cv2.calcOpticalFlowPyrLK(small, nxt, pts, None, winSize=(21, 21), maxLevel=3),
    }
    res = {}
    for threads in (1, cv2.getNumberOfCPUs()):
        cv2.setNumThreads(threads)
        res[f"threads_{threads}"] = {k: timeit(f, repeats) for k, f in ops.items()}
    cv2.setNumThreads(cv2.getNumberOfCPUs())
    return res


def pipeline(imgs: list[np.ndarray], repeats: int) -> dict:
    from mombe.vision import continuity, quality
    from mombe.vision.board import ScaleEstimator, make_board, SQUARE_MM
    from mombe.vision.muzzle import Gallery, MuzzleEngine, identify, preprocess
    eng = MuzzleEngine()
    a, b = imgs[0], imgs[1]
    pa = preprocess(a)
    fa, fb = eng.extract(a), eng.extract(b)
    g = Gallery.load(ROOT / "var" / "gallery.npz")
    q = eng.extract(imgs[2])

    # synthetic board views and a pan sequence, identical on both hosts
    board = make_board().generateImage((700, 1000), marginSize=0, borderBits=1)
    W, Hh, f = 1280, 960, 1100.0
    K = np.array([[f, 0, W / 2], [0, f, Hh / 2], [0, 0, 1]])
    bw, bh = 7 * SQUARE_MM / 1000, 10 * SQUARE_MM / 1000
    src = np.float32([[0, 0], [700, 0], [700, 1000], [0, 1000]])
    obj = np.float32([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]])
    rng = np.random.default_rng(1)
    views = []
    for _ in range(10):
        r = rng.uniform([-.5, -.6, -.3], [.5, .6, .3]); t = np.array([-bw / 2, -bh / 2, rng.uniform(.9, 1.6)])
        dst, _ = cv2.projectPoints(obj, r, t, K, None)
        Hm = cv2.getPerspectiveTransform(src, dst.reshape(4, 2).astype(np.float32))
        views.append(cv2.warpPerspective(cv2.cvtColor(board, cv2.COLOR_GRAY2BGR), Hm, (W, Hh), borderValue=(90, 110, 120)))
    scene = cv2.resize(a, (2000, 1500))
    seq = [scene[100:820, 60 + i * 25:1020 + i * 25].copy() for i in range(14)]
    se = ScaleEstimator()

    r = max(5, repeats // 3)
    return {
        "quality_assess": timeit(lambda: quality.assess(a), repeats),
        "preprocess_clahe_resize": timeit(lambda: preprocess(a), repeats),
        "aliked_extract": timeit(lambda: eng.aliked.detectAndCompute(pa, None), r),
        "dinov2_embed": timeit(lambda: eng.embed(pa), r),
        "lightglue_magsac_verify": timeit(lambda: eng.verify(fa, fb), r),
        "identify_end_to_end": timeit(lambda: identify(eng, g, [q]), max(3, r // 2)),
        "continuity_14_frames": timeit(lambda: continuity.check(seq), r),
        "charuco_calibrate_10_views": timeit(lambda: se.calibrate(views), max(3, r // 2)),
    }


def registry(imgs: list[np.ndarray]) -> dict:
    from mombe.vision.muzzle import MuzzleEngine
    eng = MuzzleEngine()
    eng.extract(imgs[0])
    t = time.perf_counter()
    for im in imgs:
        eng.extract(im)
    dt = time.perf_counter() - t
    return {"images": len(imgs), "seconds": round(dt, 2), "enrolments_per_s": round(len(imgs) / dt, 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--registry-images", type=int, default=60)
    a = ap.parse_args()
    imgs = images(max(a.registry_images, 3))
    out = {
        "host": {"machine": platform.machine(), "opencv": cv2.__version__, "cpus": cv2.getNumberOfCPUs(),
                 "cpu_features": cv2.getCPUFeaturesLine(), "file": cv2.__file__},
        "build_simd": [l.strip() for l in cv2.getBuildInformation().splitlines() if "Baseline" in l or "Dispatched" in l or "KleidiCV" in l or "HAL" in l],
        "kernels": kernels(imgs[0], a.repeats),
        "pipeline": pipeline(imgs, a.repeats),
        "registry": registry(imgs[: a.registry_images]),
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"opencv": out["host"]["opencv"], "registry": out["registry"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
