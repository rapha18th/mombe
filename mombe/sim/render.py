"""Ray-cast walk-around of a synthetic animal with known geometry.

The body pipeline needs ground truth that farm day cannot give: the exact
camera poses, depth, and the heart girth a perfect tape would read. This
renders a phone walk-around of an analytic animal (ellipsoid barrel, neck and
head, cylinder legs) with a hide texture, the A3 ChArUco board held flat
against the left shoulder, textured ground and a distant wall.

World frame: metres, z up, the animal faces +x and stands at the origin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from mombe.vision.board import MARKER_MM, SQUARE_MM, SQUARES_X, SQUARES_Y, make_board

CARD_MM = (297.0, 420.0)       # A3 card the board is printed on, portrait


def _noise_texture(size: int, seed: int, scales=(4, 12, 40), weights=(0.5, 0.3, 0.2)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    acc = np.zeros((size, size), np.float32)
    for s, w in zip(scales, weights):
        small = rng.random((s, s)).astype(np.float32)
        acc += w * cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
    acc += 0.15 * rng.random((size, size)).astype(np.float32)       # fine grain for keypoints
    acc -= acc.min()
    return acc / acc.max()


def _hide_texture(seed: int, size: int = 1024) -> np.ndarray:
    """Coat with patches (Nguni-like) over hair-scale noise, BGR uint8."""
    rng = np.random.default_rng(seed)
    base = _noise_texture(size, seed)
    patches = np.zeros((size, size), np.float32)
    for _ in range(26):
        c = rng.integers(0, size, 2)
        ax = rng.integers(size // 30, size // 9, 2)
        cv2.ellipse(patches, (int(c[0]), int(c[1])), (int(ax[0]), int(ax[1])), float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
    patches = cv2.GaussianBlur(patches, (0, 0), 3)
    dark = np.array([40, 50, 70], np.float32)
    light = np.array([150, 175, 205], np.float32)
    t = np.clip(0.75 * patches + 0.35 * base, 0, 1)[..., None]
    tex = dark * t + light * (1 - t)
    tex *= (0.8 + 0.4 * _noise_texture(size, seed + 1, scales=(64, 160), weights=(0.6, 0.4)))[..., None]
    return np.clip(tex, 0, 255).astype(np.uint8)


def _ground_texture(seed: int, size: int = 1024) -> np.ndarray:
    n = _noise_texture(size, seed, scales=(6, 30, 120), weights=(0.3, 0.4, 0.3))
    soil = np.array([60, 95, 125], np.float32) * (0.6 + 0.8 * n[..., None])
    return np.clip(soil, 0, 255).astype(np.uint8)


def _wall_texture(seed: int, size: int = 1024) -> np.ndarray:
    n = _noise_texture(size, seed, scales=(8, 24, 90), weights=(0.4, 0.4, 0.2))
    bush = np.array([50, 110, 70], np.float32) * (0.5 + n[..., None])
    return np.clip(bush, 0, 255).astype(np.uint8)


def _board_card(px_per_mm: float = 4.0) -> np.ndarray:
    board = make_board()
    bw, bh = round(SQUARES_X * SQUARE_MM * px_per_mm), round(SQUARES_Y * SQUARE_MM * px_per_mm)
    img = board.generateImage((bw, bh), marginSize=0, borderBits=1)
    card = np.full((round(CARD_MM[1] * px_per_mm), round(CARD_MM[0] * px_per_mm)), 255, np.uint8)
    y0, x0 = (card.shape[0] - bh) // 2, (card.shape[1] - bw) // 2
    card[y0:y0 + bh, x0:x0 + bw] = img
    return cv2.cvtColor(card, cv2.COLOR_GRAY2BGR)


@dataclass
class Animal:
    """Analytic body. Defaults are a medium Mashona-sized animal."""
    barrel: tuple = (0.80, 0.27, 0.31)       # semi-axes x, y, z
    barrel_z: float = 1.02                   # barrel centre height
    neck: tuple = (0.30, 0.13, 0.16)
    neck_c: tuple = (0.92, 0.0, 1.18)
    head: tuple = (0.22, 0.10, 0.11)
    head_c: tuple = (1.20, 0.0, 1.05)
    leg_r: float = 0.065
    legs_x: tuple = (0.50, -0.55)            # front, rear
    legs_y: float = 0.15
    seed: int = 3

    def girth_x(self) -> float:
        """Where a tape goes: just behind the front legs."""
        return self.legs_x[0] - self.leg_r - 0.06

    def heart_girth_m(self) -> float:
        a, b, c = self.barrel
        k = math.sqrt(max(0.0, 1 - (self.girth_x() / a) ** 2))
        p, q = b * k, c * k
        h = ((p - q) / (p + q)) ** 2
        return math.pi * (p + q) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))       # Ramanujan

    def body_length_m(self) -> float:
        """Point of shoulder to pin bone, approximated on the barrel at 85% of its length."""
        return 2 * self.barrel[0] * 0.85

    def board_pose(self, side: str = "left") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Card centre and in-plane axes (column direction, row direction) on one shoulder."""
        a, b, c = self.barrel
        x = self.legs_x[0] - 0.05
        y = b * math.sqrt(1 - (x / a) ** 2) * (1 if side == "left" else -1)
        n = np.array([x / a ** 2, y / b ** 2, 0.0])
        n /= np.linalg.norm(n)                               # held flat: tangent to the shoulder
        centre = np.array([x, y, self.barrel_z - 0.02]) + 0.012 * n
        # Seen from outside, image right is up x n and image down is -z.
        up = np.array([0.0, 0.0, 1.0])
        return centre, np.cross(up, n), -up


@dataclass
class Camera:
    K: np.ndarray
    R: np.ndarray          # world -> camera rotation
    t: np.ndarray          # world -> camera translation
    size: tuple = (1920, 1080)

    @property
    def centre(self) -> np.ndarray:
        return -self.R.T @ self.t


def look_at(eye, target, up=(0, 0, 1)) -> tuple[np.ndarray, np.ndarray]:
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ eye


def default_K(size=(960, 540), hfov_deg: float = 65.0) -> np.ndarray:
    w, h = size
    f = w / 2 / math.tan(math.radians(hfov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], float)


def walk_around(n: int = 72, radius: float = 2.8, height: float = 1.4, seed: int = 0,
                size=(1920, 1080), start_deg: float = 90.0, sweep_deg: float = 360.0) -> list[Camera]:
    """A handheld circle: start at the left flank (board side), small shake in position and aim."""
    rng = np.random.default_rng(seed)
    K = default_K(size)
    cams = []
    for i in range(n):
        th = math.radians(start_deg + sweep_deg * i / n)
        r = radius * (1 + 0.04 * rng.standard_normal())
        eye = (0.1 + r * math.cos(th), r * math.sin(th), height + 0.05 * rng.standard_normal())
        target = (0.1 + 0.08 * rng.standard_normal(), 0.08 * rng.standard_normal(), 0.85 + 0.05 * rng.standard_normal())
        R, t = look_at(eye, target)
        cams.append(Camera(K, R, t, size))
    return cams


@dataclass
class Scene:
    animal: Animal = field(default_factory=Animal)
    wall_r: float = 9.0
    sun: tuple = (0.4, 0.5, 0.75)

    def __post_init__(self):
        s = self.animal.seed
        self.tex = {"hide": _hide_texture(s), "ground": _ground_texture(s + 10, 2048), "wall": _wall_texture(s + 20),
                    "board": _board_card()}
        L = np.asarray(self.sun, float)
        self.L = L / np.linalg.norm(L)

    def render(self, cam: Camera, board: str | None = "left") -> dict:
        """Image (BGR), depth along the optical axis (m), and object ids per pixel."""
        w, h = cam.size
        u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        rays_c = np.stack([(u - cam.K[0, 2]) / cam.K[0, 0], (v - cam.K[1, 2]) / cam.K[1, 1], np.ones_like(u)], -1)
        d = rays_c.reshape(-1, 3) @ cam.R                       # camera -> world directions
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        o = cam.centre
        N = d.shape[0]
        best_t = np.full(N, np.inf)
        obj = np.full(N, -1, np.int8)
        a = self.animal

        def take(t, k):
            m = (t > 1e-4) & (t < best_t)
            best_t[m] = t[m]
            obj[m] = k

        # 0 ground
        with np.errstate(divide="ignore", invalid="ignore"):
            take(np.where(d[:, 2] < -1e-6, -o[2] / d[:, 2], np.inf), 0)
        # 1 wall (inside of a vertical cylinder)
        A = d[:, 0] ** 2 + d[:, 1] ** 2
        B = 2 * (o[0] * d[:, 0] + o[1] * d[:, 1])
        C = o[0] ** 2 + o[1] ** 2 - self.wall_r ** 2
        take((-B + np.sqrt(np.maximum(B * B - 4 * A * C, 0))) / (2 * A + 1e-12), 1)
        # 2 barrel, 3 neck, 4 head
        for k, (c, s) in ((2, ((0, 0, a.barrel_z), a.barrel)), (3, (a.neck_c, a.neck)), (4, (a.head_c, a.head))):
            take(_ray_ellipsoid(o, d, np.asarray(c, float), np.asarray(s, float)), k)
        # 5 legs
        for lx in a.legs_x:
            for ly in (a.legs_y, -a.legs_y):
                take(_ray_cylinder(o, d, lx, ly, a.leg_r, 0.0, a.barrel_z), 5)
        # 6 board card
        bc, bu, bv = a.board_pose(board or "left")
        n = np.cross(bv, bu)                                     # faces +y
        with np.errstate(divide="ignore", invalid="ignore"):
            tb = ((bc - o) @ n) / (d @ n)
        p = o + tb[:, None] * d
        cu = (p - bc) @ bu / (CARD_MM[0] / 1000) + 0.5
        cv_ = (p - bc) @ bv / (CARD_MM[1] / 1000) + 0.5
        inside = (cu >= 0) & (cu <= 1) & (cv_ >= 0) & (cv_ <= 1) & np.isfinite(tb) & (board is not None)
        take(np.where(inside, tb, np.inf), 6)

        P = o + best_t[:, None] * d
        img = np.zeros((N, 3), np.float32)
        normal = np.zeros((N, 3))
        for k in range(7):
            m = obj == k
            if not m.any():
                continue
            q = P[m]
            if k == 0:
                col = _sample(self.tex["ground"], q[:, 0] / 8.0 + 0.5, q[:, 1] / 8.0 + 0.5)
                normal[m] = (0, 0, 1)
            elif k == 1:
                col = _sample(self.tex["wall"], np.arctan2(q[:, 1], q[:, 0]) / (2 * np.pi), q[:, 2] / 4)
                normal[m] = -np.c_[q[:, :2], np.zeros(len(q))] / self.wall_r
            elif k in (2, 3, 4, 5):
                ang = np.arctan2(q[:, 2] - a.barrel_z, q[:, 1]) / (2 * np.pi)
                col = _sample(self.tex["hide"], q[:, 0] / 1.8 + 0.31 * k, ang * 1.6 + (q[:, 2] / 3 if k == 5 else 0))
                if k == 5:
                    lx = np.where(q[:, 0] > 0, a.legs_x[0], a.legs_x[1])
                    ly = np.where(q[:, 1] > 0, a.legs_y, -a.legs_y)
                    normal[m] = np.c_[q[:, 0] - lx, q[:, 1] - ly, np.zeros(len(q))]
                else:
                    c, s = {2: ((0, 0, a.barrel_z), a.barrel), 3: (a.neck_c, a.neck), 4: (a.head_c, a.head)}[k]
                    normal[m] = (q - np.asarray(c)) / np.asarray(s) ** 2
            else:
                col = _sample(self.tex["board"], cu[m], cv_[m], wrap=False)
                normal[m] = n
            img[m] = col
        normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
        shade = 0.38 + 0.62 * np.clip(normal @ self.L, 0, 1)
        shade[obj == 6] = 0.55 + 0.45 * shade[obj == 6]                 # paper reflects more
        img = np.clip(img * shade[:, None], 0, 255)
        depth = ((P - o) @ cam.R[2]).reshape(h, w)
        depth[~np.isfinite(best_t.reshape(h, w))] = 0
        return {"image": img.reshape(h, w, 3).astype(np.uint8), "depth": depth.astype(np.float32),
                "obj": obj.reshape(h, w)}


def _sample(tex: np.ndarray, u: np.ndarray, v: np.ndarray, wrap: bool = True) -> np.ndarray:
    th, tw = tex.shape[:2]
    if wrap:
        u, v = u % 1.0, v % 1.0
    n = len(u)
    cols = 1024                                   # remap maps must stay under SHRT_MAX on each side
    pad = (-n) % cols
    mx = np.pad((u * (tw - 1)).astype(np.float32), (0, pad)).reshape(-1, cols)
    my = np.pad((v * (th - 1)).astype(np.float32), (0, pad)).reshape(-1, cols)
    border = cv2.BORDER_WRAP if wrap else cv2.BORDER_REPLICATE
    return cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=border).reshape(-1, 3)[:n].astype(np.float32)


def _ray_ellipsoid(o, d, c, s):
    oo = (o - c) / s
    dd = d / s
    A = (dd * dd).sum(1)
    B = 2 * dd @ oo
    C = oo @ oo - 1
    disc = B * B - 4 * A * C
    t = (-B - np.sqrt(np.maximum(disc, 0))) / (2 * A)
    return np.where(disc > 0, t, np.inf)


def _ray_cylinder(o, d, cx, cy, r, z0, z1):
    ox, oy = o[0] - cx, o[1] - cy
    A = d[:, 0] ** 2 + d[:, 1] ** 2
    B = 2 * (ox * d[:, 0] + oy * d[:, 1])
    C = ox * ox + oy * oy - r * r
    disc = B * B - 4 * A * C
    t = (-B - np.sqrt(np.maximum(disc, 0))) / (2 * A + 1e-12)
    z = o[2] + t * d[:, 2]
    return np.where((disc > 0) & (z >= z0) & (z <= z1), t, np.inf)
