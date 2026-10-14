"""Deterministic split of the public muzzle dataset into a registry and a field.

* 200 animals are registered: half of their cached images form the gallery,
  the rest are what the phone sees in the field.
* 68 animals are never registered: they play new animals at enrolment.
* A dev slice of registered animals calibrates the decision thresholds; the
  scenario suite runs on the remaining test animals only.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mombe.vision.muzzle import Features, Gallery

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "features"
IMGS = ROOT / "data" / "muzzle"

N_REGISTERED = int(os.environ.get("MOMBE_N_REGISTERED", 200))
N_DEV = int(os.environ.get("MOMBE_N_DEV", 60))


@dataclass
class Split:
    registered: list[str]
    unregistered: list[str]
    dev: list[str]
    test: list[str]
    gallery_imgs: dict[str, list[str]]
    field_imgs: dict[str, list[str]]


def cached(animal: str) -> list[str]:
    return sorted(p.stem for p in (FEAT / animal).glob("*.npz"))


def load_feat(animal: str, image_id: str) -> Features:
    z = np.load(FEAT / animal / f"{image_id}.npz")
    return Features(z["kpts"], z["desc"], tuple(int(x) for x in z["size"]), z["emb"])


def make_split(seed: int = 7) -> Split:
    animals = sorted(p.name for p in FEAT.iterdir() if p.is_dir() and len(cached(p.name)) >= 6)
    rng = random.Random(seed)
    rng.shuffle(animals)
    reg, unreg = sorted(animals[:N_REGISTERED]), sorted(animals[N_REGISTERED:])
    dev, test = reg[:N_DEV], reg[N_DEV:]
    gal, fld = {}, {}
    for a in reg:
        imgs = cached(a)
        gal[a] = imgs[0::2]          # alternate images: gallery and field share no image
        fld[a] = imgs[1::2]
    for a in unreg:
        fld[a] = cached(a)
    return Split(reg, unreg, dev, test, gal, fld)


def build_gallery(split: Split) -> Gallery:
    g = Gallery()
    for a in split.registered:
        for im in split.gallery_imgs[a]:
            g.add(a, im, load_feat(a, im))
    return g


def field_paths(split: Split) -> dict[str, list[str]]:
    return {a: [str(IMGS / a / f"{im}.jpg") for im in ims] for a, ims in split.field_imgs.items()}
