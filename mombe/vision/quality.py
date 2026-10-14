"""Capture quality gate.

Every measurement downstream inherits the quality of the frame it came from, so
the gate runs first and its findings become concrete instructions for the
officer holding the phone. Each issue carries an instruction code the agent can
pass straight to request_capture.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

# Thresholds are per-ROI on a 512 px long side, calibrated on 300 curated images
# of the public muzzle dataset against synthetic blur and sun washout:
#   sharpness: clean p1 = 134, motion-blurred p99 = 164
#   washout:   clean p99 = 0.18 (pink noses and flash specks stay legal)
SHARPNESS_MIN = 150.0         # variance of Laplacian
GLARE_MAX = 0.20              # fraction of the nose washed out: bright and textureless
DARK_MEAN_MIN = 45.0
BRIGHT_MEAN_MAX = 215.0
CLIP_MAX = 0.08               # fraction of pixels clipped at either end
MIN_ROI_PX = 60               # shortest ROI side; curated crops run p1 62 px, median 214 px, and still verify

INSTRUCTIONS = {
    "BLUR": "Hold the phone still and tap the nose to focus.",
    "GLARE": "Shade the nose or step to the side so the light falls across it.",
    "DARK": "Turn the animal toward the light or switch on the torch.",
    "BRIGHT": "Move out of direct sun; the nose is washed out.",
    "TOO_FAR": "Move closer until the nose fills the frame.",
}


@dataclass
class Quality:
    sharpness: float
    glare: float
    mean: float
    clipped: float
    roi_min_side: int
    issues: list[str] = field(default_factory=list)
    score: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.issues

    def instructions(self) -> list[dict]:
        return [{"code": c, "message": INSTRUCTIONS[c]} for c in self.issues]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ok"] = self.ok
        d["instructions"] = self.instructions()
        return d


def _normalise(img: np.ndarray, long_side: int = 512) -> np.ndarray:
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if abs(s - 1.0) > 1e-3:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return img


def sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var())


def glare_fraction(bgr: np.ndarray) -> float:
    """Washout: bright pixels whose 9 x 9 neighbourhood has lost its texture.

    Pink skin and small flash specks keep their bead texture and pass; sun
    glare that erases the beads does not.
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mu = cv2.blur(g, (9, 9))
    sd = np.sqrt(np.maximum(cv2.blur(g * g, (9, 9)) - mu * mu, 0))
    m = ((g >= 230) & (sd < 5)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return float(m.mean())


def assess(bgr: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> Quality:
    """Score one frame (or one ROI of it). roi is x, y, w, h in pixels."""
    if roi is not None:
        x, y, w, h = roi
        bgr = bgr[y:y + h, x:x + w]
    roi_min = int(min(bgr.shape[:2]))
    small = _normalise(bgr)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    q = Quality(
        sharpness=sharpness(gray),
        glare=glare_fraction(small),
        mean=float(gray.mean()),
        clipped=float(((gray <= 3) | (gray >= 252)).mean()),
        roi_min_side=roi_min,
    )
    if q.sharpness < SHARPNESS_MIN:
        q.issues.append("BLUR")
    if q.glare > GLARE_MAX:
        q.issues.append("GLARE")
    if q.mean < DARK_MEAN_MIN:
        q.issues.append("DARK")
    elif q.mean > BRIGHT_MEAN_MAX or q.clipped > CLIP_MAX:
        q.issues.append("BRIGHT")
    if roi_min < MIN_ROI_PX:
        q.issues.append("TOO_FAR")

    # A single comparable number for ranking frames: sharpness in log space,
    # penalised by glare and clipping.
    q.score = float(np.log1p(q.sharpness) * (1 - min(1.0, q.glare * 5)) * (1 - min(1.0, q.clipped * 4)))
    return q


def best_frames(frames: list[np.ndarray], k: int = 3) -> list[tuple[int, Quality]]:
    scored = [(i, assess(f)) for i, f in enumerate(frames)]
    scored.sort(key=lambda t: (t[1].ok, t[1].score), reverse=True)
    return scored[:k]
