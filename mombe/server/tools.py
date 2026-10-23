"""Vision tools the agent calls. Each returns JSON-able evidence.

The same functions back the MCP server (deployed on Graviton with COOL) and the
in-process simulator used for evaluation, so the evaluated code is the shipped
code.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from mombe.agent import valuation
from mombe.server.store import LocalRegistry, LocalStore
from mombe.vision import liveness, quality
from mombe.vision.muzzle import Features, Gallery, MuzzleEngine, draw_match, identify

ROOT = Path(__file__).resolve().parents[2]
FEATURE_CACHE = ROOT / "data" / "features"
DATA_MUZZLE = ROOT / "data" / "muzzle"


def _cached_features(path: str) -> Features | None:
    p = Path(path)
    try:
        rel = p.resolve().relative_to(DATA_MUZZLE.resolve())
    except ValueError:
        return None
    npz = FEATURE_CACHE / rel.parent / (rel.stem + ".npz")
    if not npz.exists():
        return None
    z = np.load(npz)
    return Features(z["kpts"], z["desc"], tuple(int(x) for x in z["size"]), z["emb"])


def _load_npz(path: Path) -> Features:
    z = np.load(path)
    return Features(z["kpts"], z["desc"], tuple(int(x) for x in z["size"]), z["emb"])


class VisionTools:
    def __init__(self, store: LocalStore, registry: LocalRegistry, gallery: Gallery,
                 engine: MuzzleEngine | None = None, body=None, prices: valuation.Prices | None = None,
                 enrolled_dir: Path | None = None):
        self.store, self.registry, self.gallery = store, registry, gallery
        self.engine = engine or MuzzleEngine()
        self.body = body
        # Sourced quotes only: every value names its price, the price's source and its date.
        self.prices = prices or valuation.Prices()
        self.price = self.prices.low["usd_per_kg"]          # the floor price loans are sized on
        self.timings: list[dict] = []
        self._memo: dict[str, Features] = {}
        self._enrolled_src: dict[str, str] = {}      # image id -> capture frame, for enrolments of this process
        # Noseprints enrolled in the field join the gallery and survive a restart.
        self.enrolled_dir = Path(enrolled_dir) if enrolled_dir else None
        if self.enrolled_dir and self.enrolled_dir.exists():
            for p in sorted(self.enrolled_dir.glob("*/*.npz")):
                self.gallery.add(p.parent.name, p.stem, _load_npz(p))

    def _timed(self, name: str, fn):
        t = time.perf_counter()
        out = fn()
        self.timings.append({"tool": name, "ms": round((time.perf_counter() - t) * 1000, 1)})
        out["elapsed_ms"] = self.timings[-1]["ms"]
        return out

    def _features(self, path: str) -> Features:
        # identity, liveness and enrolment read the same frames: extract each once
        f = self._memo.get(path)
        if f is None:
            f = _cached_features(path) or self.engine.extract(cv2.imread(path))
            if len(self._memo) > 64:
                self._memo.pop(next(iter(self._memo)))
            self._memo[path] = f
        return f

    def _ranked_frames(self, c) -> list[str]:
        per = [quality.assess(cv2.imread(p)) for p in c.muzzle]
        order = sorted(range(len(per)), key=lambda i: (per[i].ok, per[i].score), reverse=True)
        return [c.muzzle[i] for i in order]

    # -- perception ---------------------------------------------------------
    def assess_capture(self, session_id: str) -> dict:
        def run():
            s = self.store.load(session_id)
            c = s.latest
            if c is None or not c.muzzle:
                return {"capture": None, "issues": ["NO_CAPTURE"],
                        "instructions": [{"code": "NO_CAPTURE", "message": "Photograph the nose, filling the frame."}]}
            per = [quality.assess(cv2.imread(p)) for p in c.muzzle]
            usable = [i for i, q in enumerate(per) if q.ok]
            best = max(range(len(per)), key=lambda i: (per[i].ok, per[i].score))
            issues = [] if usable else per[best].issues
            return {
                "capture": c.index,
                "frames": len(per),
                "usable_frames": usable,
                "best_frame": best,
                "issues": issues,
                "instructions": [] if usable else per[best].instructions(),
                "per_frame": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(q).items()} for q in per],
            }
        return self._timed("assess_capture", run)

    def identify_muzzle(self, session_id: str) -> dict:
        def run():
            s = self.store.load(session_id)
            c = s.latest
            per = [quality.assess(cv2.imread(p)) for p in c.muzzle]
            order = sorted(range(len(per)), key=lambda i: (per[i].ok, per[i].score), reverse=True)
            picks = [c.muzzle[i] for i in order[:2]]
            feats = [self._features(p) for p in picks]
            res = identify(self.engine, self.gallery, feats)
            out = res.to_dict()
            out["capture"] = c.index
            out["query_frames"] = picks
            if res.animal_id and res.verifications:
                top = res.verifications[0]
                gi = self.gallery.image_ids.index(top["image_id"])
                _, _, _, pairs = self.engine.verify(feats[0], self.gallery.features[gi])
                g_path = str(DATA_MUZZLE / "_".join(top["image_id"].split("_")[:2]) / (top["image_id"] + ".jpg"))
                if Path(g_path).exists():
                    ev = draw_match(cv2.imread(picks[0]), cv2.imread(g_path), pairs)
                    p = self.store.artifact_path(session_id, f"match_c{c.index:02d}.jpg")
                    cv2.imwrite(str(p), ev, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    out["evidence_image"] = str(p)
            return out
        return self._timed("identify_muzzle", run)

    def check_liveness(self, session_id: str) -> dict:
        def run():
            s = self.store.load(session_id)
            c = s.latest
            if c is None or not c.muzzle:
                return {"capture": None, "live": False, "score": 0.0, "failed_cue": "parallax",
                        "instructions": [{"code": "LIVE_RECAPTURE", "message": "Take two photos of the nose."}]}
            paths = c.muzzle[:4]                       # burst order as captured
            silent = ("glint", "range") if s.task == "claim" else ()   # a carcass nose is dry and dull
            res = liveness.assess([cv2.imread(p) for p in paths], [self._features(p) for p in paths], self.gallery,
                                  silent=silent, image_of=self._enrolled_image)
            out = res.to_dict()
            out["capture"] = c.index
            return out
        return self._timed("check_liveness", run)

    def _enrolled_image(self, animal_id: str, image_id: str) -> np.ndarray | None:
        """An enrolled photo for the pixel replay test: this process, field enrolments, then the seed dataset."""
        for p in (self._enrolled_src.get(image_id),
                  self.enrolled_dir / animal_id / f"{image_id}.jpg" if self.enrolled_dir else None,
                  DATA_MUZZLE / "_".join(image_id.split("_")[:2]) / f"{image_id}.jpg"):
            if p and Path(p).exists():
                return cv2.imread(str(p))
        return None

    def enrol_muzzle(self, session_id: str, animal_id: str) -> dict:
        """Add the capture's two best noseprints to the gallery under a newly enrolled animal."""
        def run():
            c = self.store.load(session_id).latest
            added = []
            for k, p in enumerate(self._ranked_frames(c)[:2]):
                f = self._features(p)
                image_id = f"{animal_id}_c{c.index:02d}_{k}"
                self.gallery.add(animal_id, image_id, f)
                self._enrolled_src[image_id] = p
                added.append(image_id)
                if self.enrolled_dir:
                    out = self.enrolled_dir / animal_id / f"{image_id}.npz"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(out, kpts=f.kpts, desc=f.desc, size=np.array(f.size), emb=f.embedding)
                    # the photo itself, at the pixel test's size, so a resent file is caught after a restart
                    img = cv2.imread(p)
                    sc = min(1.0, liveness.PIXEL_SIDE / max(img.shape[:2]))
                    img = cv2.resize(img, (round(img.shape[1] * sc), round(img.shape[0] * sc)), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(out.with_suffix(".jpg")), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return {"animal_id": animal_id, "added": added, "gallery_images": len(self.gallery.features)}
        return self._timed("enrol_muzzle", run)

    def lookup_animal(self, animal_id: str) -> dict:
        return self._timed("lookup_animal", lambda: {"animal_id": animal_id, "record": self.registry.get(animal_id)})

    # -- body, weight, value -----------------------------------------------
    def check_continuity(self, session_id: str) -> dict:
        return self._timed("check_continuity", lambda: self.body.continuity(self.store.load(session_id).latest))

    def calibrate_scale(self, session_id: str) -> dict:
        return self._timed("calibrate_scale", lambda: self.body.scale(self.store.load(session_id).latest))

    def measure_body(self, session_id: str) -> dict:
        return self._timed("measure_body", lambda: self.body.measure(self.store.load(session_id).latest))

    def estimate_weight(self, session_id: str) -> dict:
        return self._timed("estimate_weight", lambda: self.body.weight(self.store.load(session_id).latest))

    def value_animal(self, session_id: str, weight: dict) -> dict:
        """P10 value at the floor price, market value, the price's source and date, and the expiry."""
        return self._timed("value_animal", lambda: valuation.value_animal(weight, self.prices))

    # -- lender view ----------------------------------------------------------
    def herd_weights(self, owner: str | None = None, animal_ids: list[str] | None = None,
                     session_id: str | None = None) -> list[dict]:
        """Weighed animals of a herd: by owner or by id, plus a session's fresh weighing."""
        rows = {}
        for aid, rec in self.registry.animals.items():
            if (animal_ids and aid in animal_ids) or (owner and rec.get("owner") == owner and not animal_ids):
                if rec.get("weight_kg") and rec.get("weight_interval"):
                    lo, hi = rec["weight_interval"]
                    rows[aid] = {"animal_id": aid, "kg": rec["weight_kg"], "lo": lo, "hi": hi}
        if session_id:
            L = self.store.load(session_id).ledger or {}
            w, aid = L.get("weight"), (L.get("identity") or {}).get("animal_id")
            if w and w.get("kg") and aid:
                rows[aid] = {"animal_id": aid, "kg": w["kg"], "lo": w["lo"], "hi": w["hi"], "this_session": True}
        return [r for r in rows.values() if 0 < r["lo"] <= r["kg"] <= r["hi"]]

    def value_herd(self, owner: str | None = None, animal_ids: list[str] | None = None,
                   session_id: str | None = None, weights: list[dict] | None = None) -> dict:
        """Herd totals (independent and conservative P10) and shortfall at LTV 50, 60 and 70%."""
        def run():
            herd = weights or self.herd_weights(owner, animal_ids, session_id)
            if not herd:
                return {"animals": 0, "error": "no weighed animals in this herd"}
            out = valuation.herd_value(herd, self.prices)
            out["animal_ids"] = [w.get("animal_id") for w in herd]
            out["owner"] = owner
            return out
        return self._timed("value_herd", run)

    def price_book(self) -> dict:
        return self.prices.to_dict()
