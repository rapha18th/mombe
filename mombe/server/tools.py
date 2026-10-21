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
from mombe.vision import quality
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


class VisionTools:
    def __init__(self, store: LocalStore, registry: LocalRegistry, gallery: Gallery,
                 engine: MuzzleEngine | None = None, body=None, prices: valuation.Prices | None = None):
        self.store, self.registry, self.gallery = store, registry, gallery
        self.engine = engine or MuzzleEngine()
        self.body = body
        # Sourced quotes only: every value names its price, the price's source and its date.
        self.prices = prices or valuation.Prices()
        self.price = self.prices.low["usd_per_kg"]          # the floor price loans are sized on
        self.timings: list[dict] = []

    def _timed(self, name: str, fn):
        t = time.perf_counter()
        out = fn()
        self.timings.append({"tool": name, "ms": round((time.perf_counter() - t) * 1000, 1)})
        out["elapsed_ms"] = self.timings[-1]["ms"]
        return out

    def _features(self, path: str) -> Features:
        return _cached_features(path) or self.engine.extract(cv2.imread(path))

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
