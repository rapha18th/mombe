"""Muzzle identity: shortlist with DINOv2, decide with geometry.

Two stages, both inside OpenCV 5:

1. DINOv2-small (new DNN engine) embeds each muzzle; cosine similarity against
   the registry gallery returns a shortlist.
2. ALIKED keypoints, LightGlue matches and a MAGSAC++ homography verify each
   shortlisted animal. The verified inlier count is the identity score: a
   number a vet can see drawn on the two images.

The score lands in one of three bands, calibrated on held-out animals:
match, new animal, or ambiguous. Ambiguous is the band the agent acts on.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

MODELS = Path(__file__).resolve().parents[2] / "models"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)

# Defaults until calibrate.py writes measured thresholds.
T_MATCH = 60      # verified inliers at or above: same animal
T_NEW = 18        # best verified inliers at or below: animal absent from registry
MAGSAC_PX = 4.0   # reprojection threshold at the 640 px working size
WORK_SIDE = 640


@dataclass
class PipelineConfig:
    """Everything the tuner may change. Defaults reproduce the calibrated pipeline."""
    work_side: int = 640             # preprocessing long side
    aliked_input: int = 640          # ALIKED network input (square)
    aliked_engine: str = "new"       # new | classic
    aliked_fp16: bool = False        # FP16 target (classic engine only)
    dino_engine: str = "new"
    dino_fp16: bool = False
    lightglue_fp16: bool = False
    threads: int = 0                 # 0 = all cores
    verify_top: int = 3

    @classmethod
    def from_json(cls, path) -> "PipelineConfig":
        import json as _j
        from pathlib import Path as _P
        p = _P(path)
        return cls(**_j.loads(p.read_text())) if p.exists() else cls()


_ENGINE = {"new": cv2.dnn.ENGINE_NEW, "classic": cv2.dnn.ENGINE_CLASSIC, "auto": cv2.dnn.ENGINE_AUTO}


@dataclass
class Features:
    kpts: np.ndarray          # N x 2 float32, working-size pixels
    desc: np.ndarray          # N x 128 float32
    size: tuple[int, int]     # working (w, h)
    embedding: np.ndarray     # 384 float32, L2-normalised


@dataclass
class Verification:
    animal_id: str
    image_id: str
    matches: int
    inliers: int
    inlier_ratio: float
    homography: list | None = None


@dataclass
class IdentityResult:
    decision: str                                  # match | new | ambiguous
    animal_id: str | None
    score: int                                     # best verified inliers
    margin: int                                    # best minus runner-up animal
    shortlist: list[dict] = field(default_factory=list)
    verifications: list[dict] = field(default_factory=list)
    thresholds: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def preprocess(bgr: np.ndarray, side: int = WORK_SIDE) -> np.ndarray:
    """Working-size image with local contrast lifted so beads and ridges read."""
    h, w = bgr.shape[:2]
    s = side / max(h, w)
    img = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class MuzzleEngine:
    def __init__(self, models: Path = MODELS, t_match: int = T_MATCH, t_new: int = T_NEW,
                 config: PipelineConfig | None = None):
        c = self.config = config or PipelineConfig()
        if c.threads:
            cv2.setNumThreads(c.threads)
        cpu, fp16 = cv2.dnn.DNN_TARGET_CPU, cv2.dnn.DNN_TARGET_CPU_FP16
        ap = cv2.ALIKED_Params()
        ap.inputSize = (c.aliked_input, c.aliked_input)
        ap.engine = _ENGINE[c.aliked_engine]
        ap.target = fp16 if c.aliked_fp16 else cpu
        self.aliked = cv2.ALIKED_create(str(models / "aliked-n16rot-top1k-640.onnx"), ap)
        self.lightglue = cv2.LightGlueMatcher_create(str(models / "aliked_lightglue.onnx"), 0.0,
                                                     cv2.dnn.DNN_BACKEND_OPENCV, fp16 if c.lightglue_fp16 else cpu)
        self.dino = cv2.dnn.readNetFromONNX(str(models / "dinov2_small.onnx"), _ENGINE[c.dino_engine])
        if c.dino_fp16:
            self.dino.setPreferableTarget(fp16)
        self.t_match, self.t_new = t_match, t_new

    # -- features -----------------------------------------------------------
    def embed(self, bgr: np.ndarray) -> np.ndarray:
        blob = cv2.dnn.blobFromImage(bgr, 1 / 255.0, (224, 224), swapRB=True, crop=False)
        blob = ((blob - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)
        self.dino.setInput(blob)
        tokens = self.dino.forward()[0]                      # 257 x 384
        # CLS plus mean patch token: CLS carries the global shape, the patch
        # mean carries texture, and muzzles differ mostly in texture.
        v = np.concatenate([tokens[0], tokens[1:].mean(0)]).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8)

    def extract(self, bgr: np.ndarray) -> Features:
        img = preprocess(bgr, self.config.work_side)
        kps, desc = self.aliked.detectAndCompute(img, None)
        pts = np.float32([k.pt for k in kps]).reshape(-1, 2)
        if desc is None:
            desc = np.zeros((0, 128), np.float32)
        return Features(pts, desc, (img.shape[1], img.shape[0]), self.embed(img))

    # -- verification -------------------------------------------------------
    def verify(self, q: Features, g: Features) -> tuple[int, int, np.ndarray | None, np.ndarray | None]:
        """Returns (matches, inliers, homography, inlier pairs as K x 4)."""
        if len(q.kpts) < 8 or len(g.kpts) < 8:
            return 0, 0, None, None
        try:
            self.lightglue.setPairInfo(q.kpts, g.kpts, q.size, g.size)
            m = self.lightglue.match(q.desc, g.desc)
        except cv2.error:
            # A rare descriptor pair makes the ONNX matcher throw; count it as no evidence.
            return 0, 0, None, None
        if len(m) < 8:
            return len(m), 0, None, None
        p1 = q.kpts[[x.queryIdx for x in m]]
        p2 = g.kpts[[x.trainIdx for x in m]]
        H, mask = cv2.findHomography(p1, p2, cv2.USAC_MAGSAC, MAGSAC_PX)
        if H is None or mask is None:
            return len(m), 0, None, None
        keep = mask.ravel().astype(bool)
        return len(m), int(keep.sum()), H, np.hstack([p1[keep], p2[keep]])

    def decide(self, best: int, runner_up: int) -> str:
        if best >= self.t_match and best - runner_up >= self.t_new:
            return "match"
        if best <= self.t_new:
            return "new"
        return "ambiguous"


class Gallery:
    """Registry-side features, one entry per enrolled image."""

    def __init__(self):
        self.animal_ids: list[str] = []
        self.image_ids: list[str] = []
        self.features: list[Features] = []
        self._emb: np.ndarray | None = None

    def add(self, animal_id: str, image_id: str, f: Features) -> None:
        self.animal_ids.append(animal_id)
        self.image_ids.append(image_id)
        self.features.append(f)
        self._emb = None

    @property
    def embeddings(self) -> np.ndarray:
        if self._emb is None:
            self._emb = np.stack([f.embedding for f in self.features]) if self.features else np.zeros((0, 768), np.float32)
        return self._emb

    def shortlist(self, emb: np.ndarray, k: int = 8, exclude: set[str] | None = None) -> list[tuple[str, float, list[int]]]:
        """Top-k animals by best cosine over their images, with image indices."""
        if not self.features:
            return []
        sims = self.embeddings @ emb
        per_animal: dict[str, list[tuple[float, int]]] = {}
        for i, s in enumerate(sims):
            a = self.animal_ids[i]
            if exclude and a in exclude:
                continue
            per_animal.setdefault(a, []).append((float(s), i))
        ranked = sorted(per_animal.items(), key=lambda kv: max(s for s, _ in kv[1]), reverse=True)[:k]
        return [(a, max(s for s, _ in v), [i for _, i in sorted(v, reverse=True)[:3]]) for a, v in ranked]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for i, f in enumerate(self.features):
            arrays[f"k{i}"], arrays[f"d{i}"], arrays[f"e{i}"] = f.kpts, f.desc, f.embedding
            arrays[f"s{i}"] = np.array(f.size)
        np.savez_compressed(path, **arrays)
        path.with_suffix(".json").write_text(json.dumps({"animal_ids": self.animal_ids, "image_ids": self.image_ids}))

    @classmethod
    def load(cls, path: Path) -> "Gallery":
        g = cls()
        meta = json.loads(path.with_suffix(".json").read_text())
        z = np.load(path)
        for i, (a, im) in enumerate(zip(meta["animal_ids"], meta["image_ids"])):
            g.add(a, im, Features(z[f"k{i}"], z[f"d{i}"], tuple(int(x) for x in z[f"s{i}"]), z[f"e{i}"]))
        return g


def identify(engine: MuzzleEngine, gallery: Gallery, queries: list[Features], k: int = 8,
             verify_top: int | None = None, exclude: set[str] | None = None) -> IdentityResult:
    """Identify an animal from one or more query muzzle images.

    Geometric verification is the expensive step (one LightGlue pass each), so
    it is spent adaptively: the top `verify_top` shortlisted animals get one
    verification against their closest gallery image with the best query. Only
    a borderline result buys more verifications (more images, more queries).
    """
    verify_top = verify_top or engine.config.verify_top
    emb = np.mean([q.embedding for q in queries], axis=0)
    emb /= np.linalg.norm(emb) + 1e-8
    short = gallery.shortlist(emb, k=k, exclude=exclude)

    per_animal: dict[str, int] = {}
    ver: list[Verification] = []

    def run(animal_id: str, i: int, q: Features) -> None:
        m, inl, H, _ = engine.verify(q, gallery.features[i])
        ver.append(Verification(animal_id, gallery.image_ids[i], m, inl, inl / max(m, 1),
                                None if H is None else np.round(H, 5).tolist()))
        per_animal[animal_id] = max(per_animal.get(animal_id, 0), inl)

    for animal_id, _, idxs in short[:verify_top]:
        run(animal_id, idxs[0], queries[0])

    def borderline() -> bool:
        s = sorted(per_animal.values(), reverse=True) + [0, 0]
        return engine.decide(s[0], s[1]) == "ambiguous"

    if borderline():
        for animal_id, _, idxs in short[:verify_top]:
            for i in idxs[1:]:
                run(animal_id, i, queries[0])
            for q in queries[1:]:
                run(animal_id, idxs[0], q)

    ranked = sorted(per_animal.items(), key=lambda kv: kv[1], reverse=True)
    top_id, top = ranked[0] if ranked else (None, 0)
    second = ranked[1][1] if len(ranked) > 1 else 0
    decision = engine.decide(top, second)
    return IdentityResult(
        decision=decision,
        animal_id=top_id if decision != "new" else None,
        score=top,
        margin=top - second,
        shortlist=[{"animal_id": a, "cosine": round(c, 4), "inliers": per_animal.get(a, 0)} for a, c, _ in short],
        verifications=[asdict(v) for v in sorted(ver, key=lambda v: v.inliers, reverse=True)[:10]],
        thresholds={"t_match": engine.t_match, "t_new": engine.t_new},
    )


def draw_match(q_img: np.ndarray, g_img: np.ndarray, pairs: np.ndarray, max_lines: int = 80) -> np.ndarray:
    """Side-by-side evidence image: the inlier correspondences a human checks."""
    a, b = preprocess(q_img), preprocess(g_img)
    h = max(a.shape[0], b.shape[0])
    canvas = np.zeros((h, a.shape[1] + b.shape[1], 3), np.uint8)
    canvas[:a.shape[0], :a.shape[1]] = a
    canvas[:b.shape[0], a.shape[1]:] = b
    if pairs is not None and len(pairs):
        sel = pairs[np.linspace(0, len(pairs) - 1, min(max_lines, len(pairs))).astype(int)]
        for x1, y1, x2, y2 in sel:
            p, q = (int(x1), int(y1)), (int(x2) + a.shape[1], int(y2))
            cv2.line(canvas, p, q, (84, 196, 120), 1, cv2.LINE_AA)
            cv2.circle(canvas, p, 3, (40, 160, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, q, 3, (40, 160, 255), -1, cv2.LINE_AA)
    return canvas
