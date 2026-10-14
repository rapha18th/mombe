"""Metric scale from a printed ChArUco board.

The board is the ruler in every walk-around: printed on A3 at exact size and
held flat against the animal's shoulder. Per frame, OpenCV 5's CharucoDetector
finds its corners and solvePnP gives the board pose. Across frames, the board's
known square size fixes millimetres per unit of the reconstruction.

Volume goes with scale cubed, so a 1% scale error is a 3% weight error. The
tool reports the relative scale error it achieved, and the policy gate holds
weight back until it is at most 1%.
"""

from __future__ import annotations

import cv2
import numpy as np

# A3 portrait (297 x 420 mm) leaves margins around a 7 x 10 board of 38 mm squares.
SQUARES_X, SQUARES_Y = 7, 10
SQUARE_MM, MARKER_MM = 38.0, 28.0
DICTIONARY = cv2.aruco.DICT_5X5_100


def make_board() -> cv2.aruco.CharucoBoard:
    d = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    return cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_MM / 1000.0, MARKER_MM / 1000.0, d)


def render_print(dpi: int = 300, page_mm=(297, 420)) -> np.ndarray:
    """A3 page at the given dpi with the board at exact physical size and a check ruler."""
    board = make_board()
    px_per_mm = dpi / 25.4
    bw, bh = round(SQUARES_X * SQUARE_MM * px_per_mm), round(SQUARES_Y * SQUARE_MM * px_per_mm)
    img = board.generateImage((bw, bh), marginSize=0, borderBits=1)
    page = np.full((round(page_mm[1] * px_per_mm), round(page_mm[0] * px_per_mm)), 255, np.uint8)
    y0, x0 = (page.shape[0] - bh) // 2, (page.shape[1] - bw) // 2
    page[y0:y0 + bh, x0:x0 + bw] = img
    # 100 mm check bar: measure it with a ruler after printing; it must read 100 mm.
    y = y0 + bh + round(8 * px_per_mm)
    cv2.rectangle(page, (x0, y), (x0 + round(100 * px_per_mm), y + round(3 * px_per_mm)), 0, -1)
    cv2.putText(page, f"MOMBE scale board  {SQUARES_X}x{SQUARES_Y}  square {SQUARE_MM:g} mm  bar = 100 mm  print at 100%",
                (x0, y + round(12 * px_per_mm)), cv2.FONT_HERSHEY_SIMPLEX, 1.6 * dpi / 300, 0, 3, cv2.LINE_AA)
    return page


class ScaleEstimator:
    def __init__(self):
        self.board = make_board()
        self.detector = cv2.aruco.CharucoDetector(self.board)

    def detect(self, bgr: np.ndarray):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        corners, ids, _, _ = self.detector.detectBoard(gray)
        if ids is None or len(ids) < 6:
            return None, None
        return corners.reshape(-1, 2), ids.ravel()

    def calibrate(self, frames: list[np.ndarray]) -> dict:
        """Self-calibrate the phone from the board frames, then pose each frame."""
        obj_all, img_all, used = [], [], []
        size = frames[0].shape[1], frames[0].shape[0]
        for i, f in enumerate(frames):
            c, ids = self.detect(f)
            if c is None:
                continue
            obj = self.board.getChessboardCorners()[ids]
            obj_all.append(obj.astype(np.float32))
            img_all.append(c.astype(np.float32))
            used.append(i)
        out = {"board_frames": len(used), "frames": len(frames), "method": "ChArUco self-calibration + solvePnP"}
        if len(used) < 4:
            out.update(scale_rel_err=1.0, reason="board seen in fewer than 4 frames")
            return out
        flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_all, img_all, size, None, None, flags=flags)
        dists = [float(np.linalg.norm(t)) for t in tvecs]
        # Scale error from reprojection noise: a corner error of rms pixels at
        # distance Z and focal f is rms * Z / f metres on a board of known size;
        # averaged over the corners seen in every frame.
        f = float(K[0, 0])
        n_corners = sum(len(c) for c in img_all)
        board_span = SQUARE_MM / 1000.0 * (min(SQUARES_X, SQUARES_Y) - 1)
        per_corner = rms * np.median(dists) / f
        rel = float(per_corner / board_span / np.sqrt(max(n_corners, 1)) * np.sqrt(len(used)))
        out.update(scale_rel_err=round(rel, 5), reprojection_rms_px=round(float(rms), 3), focal_px=round(f, 1),
                   board_distance_m={"median": round(float(np.median(dists)), 3), "min": round(min(dists), 3)},
                   K=np.round(K, 2).tolist(), dist=np.round(dist.ravel(), 5).tolist())
        return out
