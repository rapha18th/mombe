"""Does cv2.setUseOptimized(False) switch KleidiCV off? Outputs and timings say.

Runs ops that KleidiCV and carotene accelerate on a fixed image with
useOptimized True and False, and reports per-op median time and whether the
output changed. An op whose output and time do not move is not gated by the
switch. Also lists any KleidiCV or HAL strings inside the loaded libraries.

Usage: python kleidi_probe.py --image PATH --out OUT.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import statistics
import time
from pathlib import Path

import cv2
import numpy as np


def h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def bench(fn, n=25):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t) * 1000)
    return out, statistics.median(ts)


def lib_strings() -> dict:
    d = os.path.dirname(cv2.__file__)
    libs = glob.glob(os.path.join(d, "*.so")) + glob.glob(os.path.join(d, "**", "*.so*"), recursive=True)
    libs += glob.glob(os.path.join(os.path.dirname(d), "opencv*libs", "*.so*"))
    libs += glob.glob("/opt/cool/cpp_sdk/lib/libopencv_*.so.5.1.0")
    pat = re.compile(rb"[\x20-\x7e]{6,}")
    found = {}
    for p in sorted(set(libs)):
        try:
            data = Path(p).read_bytes()
        except OSError:
            continue
        hits = sorted({m.group().decode() for m in pat.finditer(data)
                       if re.search(rb"(?i)kleidi|OPENCV_[A-Z_]*(HAL|OPTIM|DISABLE|SIMD|DISPATCH)", m.group())})
        if hits:
            found[os.path.basename(p)] = hits[:40]
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    img = cv2.imread(a.image)
    big = cv2.resize(img, (1920, 1440), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    f32 = gray.astype(np.float32) / 255
    H = np.array([[0.98, 0.03, 12], [-0.02, 0.99, 8], [1e-5, 2e-5, 1]])
    ops = {
        "cvtColor_bgr2gray": lambda: cv2.cvtColor(big, cv2.COLOR_BGR2GRAY),
        "cvtColor_bgr2lab": lambda: cv2.cvtColor(big, cv2.COLOR_BGR2LAB),
        "resize_area_to_960": lambda: cv2.resize(big, (960, 720), interpolation=cv2.INTER_AREA),
        "resize_linear_2x_u8": lambda: cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR),
        "resize_linear_f32": lambda: cv2.resize(f32, (1000, 750), interpolation=cv2.INTER_LINEAR),
        "gaussian_7x7_u8": lambda: cv2.GaussianBlur(gray, (7, 7), 0),
        "gaussian_5x5_f32": lambda: cv2.GaussianBlur(f32, (5, 5), 1.2),
        "blur_9x9_f32": lambda: cv2.blur(f32, (9, 9)),
        "sobel_3x3_f32": lambda: cv2.Sobel(f32, cv2.CV_32F, 1, 0, ksize=3),
        "threshold_u8": lambda: cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1],
        "adaptive_threshold": lambda: cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 7),
        "warp_perspective": lambda: cv2.warpPerspective(big, H, (1920, 1440)),
        "dilate_3x3": lambda: cv2.dilate(gray, np.ones((3, 3), np.uint8)),
        "laplacian_64f": lambda: cv2.Laplacian(gray, cv2.CV_64F, ksize=3),
        "minMaxLoc": lambda: np.array(cv2.minMaxLoc(gray)[:2]),
        "add_weighted": lambda: cv2.addWeighted(gray, 0.5, gray, 0.5, 0),
    }
    cv2.setNumThreads(1)
    res = {}
    for name, fn in ops.items():
        cv2.setUseOptimized(True)
        o1, t1 = bench(fn)
        cv2.setUseOptimized(False)
        o2, t2 = bench(fn)
        cv2.setUseOptimized(True)
        res[name] = {"ms_optimized": round(t1, 3), "ms_unoptimized": round(t2, 3), "slowdown": round(t2 / t1, 2),
                     "output_changed": h(o1) != h(o2), "hash": h(o1)}
    out = {"opencv": cv2.__version__, "file": cv2.__file__, "ops": res, "strings": lib_strings(),
           "custom_hal": [l.strip() for l in cv2.getBuildInformation().splitlines() if "HAL" in l]}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for k, v in res.items():
        print(f"{k:24s} {v['ms_optimized']:8.3f} {v['ms_unoptimized']:8.3f}  x{v['slowdown']}  changed={v['output_changed']}")
    print(json.dumps(out["strings"])[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
