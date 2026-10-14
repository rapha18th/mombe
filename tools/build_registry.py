"""Seed a registry for the server from the evaluation split.

Writes var/gallery.npz (+ .json) with the registered animals' gallery features
and var/registry/animals.json with their owners.

Usage: python tools/build_registry.py [--out var]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.dataset import build_gallery, make_split  # noqa: E402
from eval.run_scenarios import owner_of  # noqa: E402
from mombe.server.store import LocalRegistry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "var"))
    a = ap.parse_args()
    out = Path(a.out)
    split = make_split()
    g = build_gallery(split)
    g.save(out / "gallery.npz")
    reg = LocalRegistry(out / "registry")
    for an in split.registered:
        reg.put(an, {"owner": owner_of(an), "enrolled": "seed"})
    print(f"gallery: {len(g.features)} images of {len(split.registered)} animals -> {out / 'gallery.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
