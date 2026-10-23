"""Closed-loop field simulator.

Plays three roles so the agent's loop can be evaluated end to end:

* the officer and the animal: serves muzzle frames from held-out images of a
  real animal, with scripted defects (glare, blur, a partial nose, a wrong
  animal) that a re-capture fixes;
* the fraudster: SYNTHETIC presentation attacks (a printed photo, a screen, a
  replayed enrolled file) rendered by mombe.sim.attacks on every capture, since
  a re-capture cannot conjure the live animal;
* the body pipeline: until farm data lands, measurement outputs come from an
  error model whose noise follows the capture conditions (scale error,
  coverage, continuity). Every simulated body result says so;
* the approving human: decides approvals from the evidence summary.

Muzzle identity is real OpenCV 5 work on real images in every scenario.
"""

from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from mombe.server.store import Capture, LocalStore
from mombe.sim import attacks

ROOT = Path(__file__).resolve().parents[2]
DATA_MUZZLE = ROOT / "data" / "muzzle"


# -- image defects ---------------------------------------------------------
def add_glare(img: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    # Sun washout across a quarter to a half of the nose.
    for _ in range(rng.randint(2, 3)):
        c = (rng.randint(w // 3, 2 * w // 3), rng.randint(h // 3, 2 * h // 3))
        ax = (rng.randint(w // 5, w // 3), rng.randint(h // 7, h // 4))
        mask = np.zeros((h, w), np.uint8)
        cv2.ellipse(mask, c, ax, rng.uniform(0, 180), 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), w / 120)
        a = (mask.astype(np.float32) / 255.0)[..., None]
        out = (out * (1 - a) + 255 * a).astype(np.uint8)
    return out


def add_blur(img: np.ndarray, rng: random.Random) -> np.ndarray:
    k = rng.randint(31, 51)
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2, k / 2), rng.uniform(0, 180), 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    return cv2.filter2D(img, -1, kernel / kernel.sum())


def partial_nose(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """An edge-on, heavily cropped nose: real texture, too little of it."""
    h, w = img.shape[:2]
    x0 = rng.choice([0, int(w * 0.62)])
    crop = img[int(h * 0.3):int(h * 0.75), x0:x0 + int(w * 0.38)]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)


DEFECTS = {"glare": add_glare, "blur": add_blur, "partial": partial_nose}


# -- body error model ------------------------------------------------------
@dataclass
class BodyCondition:
    true_kg: float
    continuous: bool = True
    board_frames: int = 14
    scale_rel_err: float = 0.004
    coverage: float = 0.85
    missing_side: str = "left"
    weight_bias: float = 0.0


class SimBodyProvider:
    """Measurement outputs from a capture-conditioned error model."""

    BASE_SIGMA = 0.030   # residual model error after calibration, fraction of weight

    def _cond(self, c: Capture) -> BodyCondition:
        return BodyCondition(**c.meta["body_sim"])

    def continuity(self, c: Capture) -> dict:
        b = self._cond(c)
        return {"continuous": b.continuous, "breaks": 0 if b.continuous else 1,
                "method": "optical-flow chain", "simulated": True}

    def scale(self, c: Capture) -> dict:
        b = self._cond(c)
        return {"board_frames": b.board_frames, "scale_rel_err": b.scale_rel_err,
                "method": "ChArUco pose per frame", "simulated": True}

    def measure(self, c: Capture) -> dict:
        b = self._cond(c)
        rng = np.random.default_rng(c.index * 7919 + int(b.true_kg))
        # Allometry consistent with girth^2 x length: girth from weight.
        girth = 0.0113 * b.true_kg ** 0.5 * 100 + 95
        length = girth * 0.86
        noise = 1 + rng.normal(0, 3 * b.scale_rel_err)
        return {"coverage": b.coverage, "missing_side": b.missing_side,
                "girth_cm": round(girth * noise, 1), "length_cm": round(length * noise, 1),
                "simulated": True}

    def weight(self, c: Capture) -> dict:
        b = self._cond(c)
        rng = np.random.default_rng(c.index * 104729 + int(b.true_kg))
        cov_pen = max(0.0, 0.8 - b.coverage) * 0.25
        sigma = math.sqrt(self.BASE_SIGMA ** 2 + (3 * b.scale_rel_err) ** 2 + cov_pen ** 2)
        kg = b.true_kg * (1 + b.weight_bias + rng.normal(0, sigma))
        half = 1.645 * sigma * kg
        return {"kg": round(kg, 1), "lo": round(kg - half, 1), "hi": round(kg + half, 1),
                "sigma_rel": round(sigma, 4), "interval": "90% (simulated error model)", "simulated": True}


# -- scenarios ------------------------------------------------------------
@dataclass
class Scenario:
    id: str
    task: str
    animal: str                      # whose nose the phone sees
    expected: str                    # expected terminal outcome
    defects: list[str] = field(default_factory=list)   # applied to the first capture only
    claimed_owner: str | None = None
    policy_animal: str | None = None
    loan_threshold_usd: float | None = None
    body: dict | None = None         # first-capture BodyCondition overrides
    body_fix: dict | None = None     # what a re-capture fixes
    approve: bool = True             # what the human approver decides
    note: str = ""
    attack: str | None = None        # print | screen | replay: SYNTHETIC, on every capture
    prior_owner: str | None = None   # enrol_double: this animal was enrolled to prior_owner first


class FieldSim:
    def __init__(self, store: LocalStore, query_images: dict[str, list[str]], seed: int = 0,
                 enrolled_images: dict[str, list[str]] | None = None):
        self.store = store
        self.query_images = query_images
        self.enrolled_images = enrolled_images or {}   # registry photos a replay attacker resends
        self.rng = random.Random(seed)
        self.seed = seed
        self.scenarios: dict[str, Scenario] = {}
        self.cursor: dict[str, int] = {}

    def _frames(self, animal: str, n: int = 2, pool: dict[str, list[str]] | None = None) -> list[str]:
        imgs = (pool or self.query_images)[animal]
        key = animal if pool is None else f"enrolled:{animal}"
        i = self.cursor.get(key, 0)
        self.cursor[key] = i + n
        return [imgs[(i + k) % len(imgs)] for k in range(n)]

    def _attack(self, sid: str, sc: Scenario, index: int) -> tuple[list, dict]:
        rng = np.random.default_rng([self.seed, zlib.crc32(sid.encode()), index])
        if sc.attack == "replay":
            paths = self._frames(sc.animal, pool=self.enrolled_images)
            frames, meta = attacks.render_replay([cv2.imread(p) for p in paths], rng)
            return [p if f is None else f for f, p in zip(frames, paths)], meta
        src = cv2.imread(self._frames(sc.animal)[0])
        return attacks.render(sc.attack, src, 2, rng)

    def start(self, sc: Scenario) -> str:
        s = self.store.create(sc.task, id=sc.id, claimed_owner=sc.claimed_owner,
                              policy_animal=sc.policy_animal, loan_threshold_usd=sc.loan_threshold_usd)
        self.scenarios[s.id] = sc
        self._capture(s.id, first=True)
        return s.id

    def _capture(self, sid: str, first: bool, instruction: dict | None = None) -> Capture:
        sc = self.scenarios[sid]
        meta = {"answers": instruction, "synthetic_defects": sc.defects if first else []}
        if sc.attack:
            index = len(self.store.load(sid).captures)
            muzzle, meta["synthetic_attack"] = self._attack(sid, sc, index)
            return self.store.add_capture(sid, muzzle=muzzle, meta=meta)
        paths = self._frames(sc.animal)
        muzzle: list = []
        for p in paths:
            if first and sc.defects:
                img = cv2.imread(p)
                for d in sc.defects:
                    if d in DEFECTS:
                        img = DEFECTS[d](img, self.rng)
                muzzle.append(img)
            else:
                muzzle.append(p)
        if sc.task == "value":
            base = {"true_kg": (sc.body or {}).get("true_kg", 420.0)}
            cond = {**base, **(sc.body or {})} if first else {**base, **(sc.body or {}), **(sc.body_fix or {})}
            if not first and instruction and instruction.get("code") == "SECOND_PASS":
                cond["scale_rel_err"] = min(cond.get("scale_rel_err", 0.004), 0.003)
                cond["board_frames"] = max(cond.get("board_frames", 14), 20)
            meta["body_sim"] = cond
        return self.store.add_capture(sid, muzzle=muzzle, meta=meta)

    def on_request_capture(self, sid: str, instruction: dict) -> Capture:
        return self._capture(sid, first=False, instruction=instruction)

    def approver(self, sid: str, approval: dict) -> str:
        # A reviewer accepts the best available capture when a quality gate ran
        # out of re-captures; fraud, ownership and credit follow the scenario.
        if approval.get("kind") == "adjudicate" and (approval.get("summary") or {}).get("gate"):
            return "approved"
        return "approved" if self.scenarios[sid].approve else "declined"
