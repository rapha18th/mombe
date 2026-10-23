"""Synthetic presentation attacks on muzzle identity.

Every image this module makes is SYNTHETIC. It renders the three field threats
from real bona fide muzzle photos, so liveness can be measured before anyone
films a real attack:

* print: the muzzle photo printed on paper and held up as a card. The card is a
  plane (with an optional slight curl) that moves in 3D between the frames of a
  burst. Paper limits the dynamic range, adds fibre texture and ink dither, and
  glossy paper throws a broad sheen.
* screen: the photo shown on a phone or laptop screen. The camera samples the
  RGB subpixel grid, which aliases into moire. Screens shift gamma and colour,
  lift the black level and reflect a light source off the glass.
* replay: an enrolled image file sent again, exact or lightly cropped and
  re-encoded.

Print and screen frames pass through one camera model: auto exposure on the
mean, lens blur, sensor noise and JPEG. All randomness flows from one seed.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

MAX_SIDE = 1024           # long side of a rendered attack frame
CARD_WIDTH_CM = 12.0      # printed or displayed nose, close to life size

KINDS = ("print", "screen", "replay")


# -- colour and camera ------------------------------------------------------
def _linear(bgr: np.ndarray) -> np.ndarray:
    return (bgr.astype(np.float32) / 255.0) ** 2.2


def _encode(lin: np.ndarray) -> np.ndarray:
    return np.clip(np.clip(lin, 0, 1) ** (1 / 2.2) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _luma(lin: np.ndarray) -> np.ndarray:
    return lin[..., 0] * 0.114 + lin[..., 1] * 0.587 + lin[..., 2] * 0.299


def _camera(lin: np.ndarray, target_mean: float, rng: np.random.Generator) -> np.ndarray:
    """Phone ISP black point, auto exposure on the mean, lens blur, sensor noise, JPEG."""
    floor = float(np.percentile(_luma(lin), 1)) * rng.uniform(0.0, 0.8)
    lin = np.clip((lin - floor) / max(1.0 - floor, 1e-3), 0, None)
    gain = target_mean * rng.uniform(0.85, 1.15) / max(float(_luma(lin).mean()), 1e-4)
    lin = lin * gain
    lin = cv2.GaussianBlur(lin, (0, 0), rng.uniform(0.4, 1.1))
    shot = rng.uniform(0.002, 0.008)
    lin = lin + rng.normal(0, 1, lin.shape).astype(np.float32) * np.sqrt(np.clip(lin, 0, 1) * shot ** 2 + 1e-6)
    img = _encode(lin)
    img = np.clip(img.astype(np.float32) + rng.normal(0, rng.uniform(0.8, 2.5), img.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(85, 96))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _out_size(src: np.ndarray) -> tuple[int, int]:
    h, w = src.shape[:2]
    s = min(1.0, MAX_SIDE / max(h, w))
    return round(w * s), round(h * s)


# -- geometry ---------------------------------------------------------------
def _rot(yaw: float, pitch: float, roll: float) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.array([pitch, yaw, 0.0]))
    Rz, _ = cv2.Rodrigues(np.array([0.0, 0.0, roll]))
    return Rz @ R


def _look_at(center: np.ndarray, eye: np.ndarray, roll: float) -> np.ndarray:
    """Rotation of a camera at `eye` aimed at `center` (reference camera at the origin looks down +z)."""
    z = center - eye
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])                    # world -> camera rows
    c, s = math.cos(roll), math.sin(roll)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R


def burst_poses(n: int, rng: np.random.Generator, depth_cm: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Handheld burst: the phone drifts a few centimetres and re-aims at the card."""
    center = np.array([0.0, 0.0, depth_cm])
    base = rng.normal(0, 1, 3) * np.array([1.0, 1.0, 0.5])
    direction = rng.normal(0, 1, 3) * np.array([1.0, 0.5, 0.25])
    direction /= np.linalg.norm(direction)
    step = rng.uniform(0.6, 3.0)                  # cm between frames
    out = []
    for k in range(n):
        eye = base + direction * step * (k - (n - 1) / 2)
        R = _look_at(center, eye, rng.normal(0, math.radians(2.0)))
        out.append((R, -R @ eye))                 # X_cam = R (X_world - eye)
    return out


def _intrinsics(w: int, h: int, fov_deg: float) -> np.ndarray:
    f = 0.5 * w / math.tan(math.radians(fov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)


def _surface_map(src_wh: tuple[int, int], out_wh: tuple[int, int], R: np.ndarray, t: np.ndarray,
                 K: np.ndarray, depth_cm: float, curl_cm: float, zoom: float,
                 pixels: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Inverse map (output pixel -> source pixel) for a card at `depth_cm`, curled by `curl_cm` sagitta.

    The reference camera sees the card frontally and the card fills its view.
    The card plane is z = depth; curl bends it into a shallow parabola along x.
    Solved by fixed-point iteration from the flat-card homography. `pixels`
    limits the map to given output coordinates (default: the whole frame).
    """
    sw, sh = src_wh
    ow, oh = out_wh
    cm_per_px = CARD_WIDTH_CM / sw

    def forward(u, v, curl):
        X = (u - sw / 2) * cm_per_px
        Y = (v - sh / 2) * cm_per_px
        xn = X / (CARD_WIDTH_CM / 2)
        Z = depth_cm - curl * (1 - xn ** 2)       # edges stay put, the middle bows toward the camera
        P = np.stack([X, Y, Z], -1) @ R.T + t
        x = K[0, 0] * P[..., 0] / P[..., 2] + K[0, 2]
        y = K[1, 1] * P[..., 1] / P[..., 2] + K[1, 2]
        return (x - ow / 2) * zoom + ow / 2, (y - oh / 2) * zoom + oh / 2

    # flat-card homography as the starting point
    corners = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], np.float64)
    fx, fy = forward(corners[:, 0], corners[:, 1], 0.0)
    H = cv2.getPerspectiveTransform(np.float32(corners), np.float32(np.stack([fx, fy], 1)))
    Hi = np.linalg.inv(H)
    if pixels is None:
        xs, ys = np.meshgrid(np.arange(ow, dtype=np.float64), np.arange(oh, dtype=np.float64))
    else:
        xs, ys = pixels
    den =Hi[2, 0] * xs + Hi[2, 1] * ys + Hi[2, 2]
    u = (Hi[0, 0] * xs + Hi[0, 1] * ys + Hi[0, 2]) / den
    v = (Hi[1, 0] * xs + Hi[1, 1] * ys + Hi[1, 2]) / den
    if curl_cm:
        for _ in range(4):
            px, py = forward(u, v, curl_cm)
            d = Hi[2, 0] * px + Hi[2, 1] * py + Hi[2, 2]
            u = u + (Hi[0, 0] * xs + Hi[0, 1] * ys + Hi[0, 2]) / den - (Hi[0, 0] * px + Hi[0, 1] * py + Hi[0, 2]) / d
            v = v + (Hi[1, 0] * xs + Hi[1, 1] * ys + Hi[1, 2]) / den - (Hi[1, 0] * px + Hi[1, 1] * py + Hi[1, 2]) / d
    return u.astype(np.float32), v.astype(np.float32)


def _fit_zoom(src_wh, out_wh, R, t, K, depth_cm, curl_cm) -> float:
    """Smallest zoom at which the card covers the whole frame (the attacker fills the view)."""
    sw, sh = src_wh
    ow, oh = out_wh
    xr, yr = np.linspace(0, ow - 1, 64), np.linspace(0, oh - 1, 64)
    border = (np.concatenate([xr, xr, np.zeros(64), np.full(64, ow - 1.0)]),
              np.concatenate([np.zeros(64), np.full(64, oh - 1.0), yr, yr]))
    zoom = 1.05
    for _ in range(60):
        u, v = _surface_map(src_wh, out_wh, R, t, K, depth_cm, curl_cm, zoom, pixels=border)
        if u.min() >= 1 and u.max() <= sw - 2 and v.min() >= 1 and v.max() <= sh - 2:
            return zoom
        zoom *= 1.03
    return zoom


def _sheen(shape: tuple[int, int], rng: np.random.Generator, amp: float, center: tuple[float, float]) -> np.ndarray:
    """A broad soft reflection of a window or lamp."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, math.pi)
    sx, sy = rng.uniform(0.12, 0.35) * w, rng.uniform(0.05, 0.18) * h
    dx, dy = xx - center[0], yy - center[1]
    a = (dx * math.cos(ang) + dy * math.sin(ang)) / sx
    b = (-dx * math.sin(ang) + dy * math.cos(ang)) / sy
    return (amp * np.exp(-0.5 * (a * a + b * b))).astype(np.float32)


# -- print ------------------------------------------------------------------
def render_print(src: np.ndarray, n: int, rng: np.random.Generator) -> tuple[list[np.ndarray], dict]:
    ow, oh = _out_size(src)
    sw, sh = ow * 2, oh * 2                       # print raster at twice the frame size
    lin = _linear(cv2.resize(src, (sw, sh), interpolation=cv2.INTER_CUBIC))
    glossy = bool(rng.random() < 0.5)
    black = rng.uniform(0.03, 0.06) if glossy else rng.uniform(0.07, 0.12)
    white = rng.uniform(0.80, 0.90) if glossy else rng.uniform(0.70, 0.82)
    gamma = rng.uniform(0.9, 1.15)
    sat = rng.uniform(0.75, 0.95)
    grey = _luma(lin)[..., None]
    lin = grey + (lin - grey) * sat
    refl = black + (white - black) * np.clip(lin, 0, 1) ** gamma
    # ink dither and paper fibre
    dither = rng.normal(0, rng.uniform(0.008, 0.02), refl.shape).astype(np.float32)
    fibre = cv2.GaussianBlur(rng.normal(0, 1, (sh, sw)).astype(np.float32), (0, 0), rng.uniform(1.0, 2.5))
    fibre = 1 + fibre / (fibre.std() + 1e-6) * rng.uniform(0.01, 0.035)
    refl = np.clip(refl * fibre[..., None] + dither, 0, 1)
    curl = float(rng.uniform(0.0, 0.6)) if rng.random() < 0.7 else 0.0
    fov = rng.uniform(45, 70)
    K = _intrinsics(ow, oh, fov)
    depth = CARD_WIDTH_CM * K[0, 0] / ow
    poses = burst_poses(n, rng, depth)
    light = rng.uniform(-0.12, 0.12, 2)
    yy, xx = np.mgrid[0:oh, 0:ow].astype(np.float32)
    sheen_amp = rng.uniform(0.0, 0.6) if glossy else 0.0
    sheen_c = np.array([rng.uniform(0.2, 0.8) * ow, rng.uniform(0.2, 0.8) * oh])
    target = float(_luma(_linear(src)).mean())
    frames = []
    for k, (R, t) in enumerate(poses):
        zoom = _fit_zoom((sw, sh), (ow, oh), R, t, K, depth, curl)
        mx, my = _surface_map((sw, sh), (ow, oh), R, t, K, depth, curl, zoom)
        img = cv2.remap(refl, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        shade = 1 + light[0] * (xx / ow - 0.5) + light[1] * (yy / oh - 0.5)
        img = img * shade[..., None]
        if sheen_amp:
            c = sheen_c + (R[:2, 2] * ow * 3.0)    # the sheen slides as the view turns
            img = img + _sheen((oh, ow), rng, sheen_amp, (c[0], c[1]))[..., None]
        frames.append(_camera(img, target, rng))
    meta = {"kind": "print", "paper": "glossy" if glossy else "matte", "curl_cm": round(curl, 3),
            "black": round(black, 3), "white": round(white, 3), "fov_deg": round(fov, 1),
            "sheen": round(sheen_amp, 3), "synthetic": True}
    return frames, meta


# -- screen -----------------------------------------------------------------
def render_screen(src: np.ndarray, n: int, rng: np.random.Generator, supersample: int = 3) -> tuple[list[np.ndarray], dict]:
    ow, oh = _out_size(src)
    # Display pixels across the nose: phone screens near 400 ppi, laptops near 150.
    disp_w = int(rng.uniform(380, 1100))
    disp_h = max(8, round(disp_w * oh / ow))
    shown = _linear(cv2.resize(src, (disp_w, disp_h), interpolation=cv2.INTER_AREA))
    gamma = rng.uniform(0.8, 1.25)
    tint = np.array([rng.uniform(1.0, 1.2), rng.uniform(0.95, 1.05), rng.uniform(0.82, 0.98)], np.float32)  # B, G, R
    black = rng.uniform(0.01, 0.05)
    shown = black + (1 - black) * np.clip(shown, 0, 1) ** gamma * tint
    fill_x, fill_y = rng.uniform(0.55, 0.85), rng.uniform(0.65, 0.92)
    fov = rng.uniform(45, 70)
    K = _intrinsics(ow, oh, fov)
    depth = CARD_WIDTH_CM * K[0, 0] / ow
    poses = burst_poses(n, rng, depth)
    S = supersample
    glare_amp = rng.uniform(0.1, 0.7) if rng.random() < 0.6 else 0.0
    glare_c = np.array([rng.uniform(0.15, 0.85) * ow, rng.uniform(0.15, 0.85) * oh])
    target = float(_luma(_linear(src)).mean())
    frames = []
    for R, t in poses:
        zoom = _fit_zoom((disp_w, disp_h), (ow, oh), R, t, K, depth, 0.0)
        # map supersampled sensor positions to screen pixel coordinates
        u, v = _surface_map((disp_w, disp_h), (ow * S, oh * S), R, t, _scale_K(K, S), depth, 0.0, zoom)
        iu, iv = np.floor(u), np.floor(v)
        fu, fv = u - iu, v - iv
        pix = cv2.remap(shown, iu.astype(np.float32), iv.astype(np.float32), cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_REPLICATE)
        on_y = (fv < fill_y).astype(np.float32)
        sub = np.empty_like(pix)
        for ch, slot in ((2, 0), (1, 1), (0, 2)):  # R, G, B stripes left to right
            lo = slot / 3.0
            sub[..., ch] = ((fu >= lo) & (fu < lo + fill_x / 3.0)).astype(np.float32) * on_y * 3.0 / (fill_x * fill_y)
        sensor = cv2.resize(pix * sub, (ow, oh), interpolation=cv2.INTER_AREA)  # pixel area integration
        if glare_amp:
            c = glare_c + R[:2, 2] * ow * 3.0
            sensor = sensor + _sheen((oh, ow), rng, glare_amp, (c[0], c[1]))[..., None]
        frames.append(_camera(sensor, target, rng))
    meta = {"kind": "screen", "display_px": disp_w, "camera_px_per_screen_px": round(ow / disp_w, 3),
            "gamma": round(gamma, 3), "glare": round(glare_amp, 3), "fov_deg": round(fov, 1), "synthetic": True}
    return frames, meta


def _scale_K(K: np.ndarray, s: int) -> np.ndarray:
    Ks = K.copy()
    Ks[:2] *= s
    return Ks


# -- replay -----------------------------------------------------------------
def render_replay(enrolled: list[np.ndarray], rng: np.random.Generator) -> tuple[list[np.ndarray | None], dict]:
    """Enrolled images sent again. None means 'send the original file unchanged'."""
    mode = "exact" if rng.random() < 0.5 else "crop"
    if mode == "exact":
        return [None] * len(enrolled), {"kind": "replay", "mode": "exact", "synthetic": True}
    out = []
    for img in enrolled:
        h, w = img.shape[:2]
        l, r, t, b = (rng.uniform(0.01, 0.06, 4) * np.array([w, w, h, h])).astype(int)
        crop = img[t:h - b, l:w - r]
        s = rng.uniform(0.8, 1.0)
        crop = cv2.resize(crop, (max(8, round(crop.shape[1] * s)), max(8, round(crop.shape[0] * s))),
                          interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(80, 96))])
        out.append(cv2.imdecode(buf, cv2.IMREAD_COLOR))
    return out, {"kind": "replay", "mode": "crop", "synthetic": True}


def render(kind: str, src: np.ndarray, n: int, rng: np.random.Generator) -> tuple[list[np.ndarray], dict]:
    if kind == "print":
        return render_print(src, n, rng)
    if kind == "screen":
        return render_screen(src, n, rng)
    raise ValueError(kind)
