"""Fixed input sets for track 1d, built once locally and shipped to S3 (jobs/1d/data/).

  ops_images   150 Acme B4 side photos + 150 Zenodo muzzles (75 animals x 2,
               consecutive images are the same animal, for LightGlue/MAGSAC)
  probes       500 identify() probes: 400 field images of enrolled animals
               (never in the gallery) + 100 images of animals absent from it
  ess_fixed    pipeline matches from the stock sweep_21 dump: undistorted
               keypoint pairs per frame pair and the calibrated K

Usage: python tools/cool_equiv/make_sets.py --stock-dump PATH.pkl [--upload]
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import random
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "eval" / "out" / "cool_equiv" / "sets"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock-dump")
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    from eval.dataset import aliases
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(1001)
    alias = aliases()

    acme = sorted((ROOT / "data" / "acme" / "Pixel" / "B4" / "Side" / "images").glob("*.jpg"))
    acme_pick = sorted(rng.sample(acme, 150))
    gal = json.loads((ROOT / "var" / "gallery.json").read_text())
    gal_imgs = set(gal["image_ids"])
    gal_animals = set(gal["animal_ids"])
    by_animal: dict[str, list[Path]] = {}
    for p in sorted((ROOT / "data" / "muzzle").glob("*/*.jpg")):
        by_animal.setdefault(p.parent.name, []).append(p)
    # muzzles for the op diffs come from the gallery images every hub already holds (data/muzzle)
    gal_by: dict[str, list[Path]] = {}
    for p in sorted((ROOT / "data" / "muzzle").glob("*/*.jpg")):
        if p.stem in gal_imgs:
            gal_by.setdefault(p.parent.name, []).append(p)
    animals = sorted(a for a, v in gal_by.items() if len(v) >= 2)
    mz_animals = sorted(rng.sample(animals, 75))
    mz_pick = []
    for an in mz_animals:
        mz_pick += sorted(rng.sample(gal_by[an], 2))
    ops = [f"acme/{p.name}" for p in acme_pick] + [f"muzzle/{p.parent.name}/{p.name}" for p in mz_pick]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for p in acme_pick:
            z.write(p, f"acme/{p.name}")
    (OUT / "ops_images.zip").write_bytes(buf.getvalue())
    (OUT / "ops_images.json").write_text(json.dumps(ops, indent=0))

    # identify probes
    canon = lambda f: alias.get(f, f)  # noqa: E731
    enrolled = sorted(a for a in by_animal if canon(a) in {canon(x) for x in gal_animals})
    absent = sorted(a for a in by_animal if canon(a) not in {canon(x) for x in gal_animals})
    pool_in = [p for a in enrolled for p in by_animal[a] if p.stem not in gal_imgs]
    pool_out = [p for a in absent for p in by_animal[a]]
    rng2 = random.Random(2002)
    probes_in = sorted(rng2.sample(pool_in, 400))
    probes_out = sorted(rng2.sample(pool_out, min(100, len(pool_out))))
    probes = [{"path": f"probes/{p.parent.name}/{p.name}", "folder": p.parent.name, "canon": canon(p.parent.name),
               "expected": canon(p.parent.name) if p in probes_in else "new"} for p in probes_in + probes_out]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for p in probes_in + probes_out:
            z.write(p, f"probes/{p.parent.name}/{p.name}")
    (OUT / "probes.zip").write_bytes(buf.getvalue())
    (OUT / "probes.json").write_text(json.dumps(probes, indent=0))
    print("ops", len(ops), "probes", len(probes), "enrolled pool", len(pool_in), "absent pool", len(pool_out),
          "absent animals", len(absent))

    if a.stock_dump:
        d = pickle.loads(Path(a.stock_dump).read_bytes())
        K = np.asarray(d["calib"]["K"], np.float64)
        arrays = {"K": K}
        n = 0
        for (i, j), v in sorted(d["lg"].items(), key=lambda kv: kv[0]):
            if not isinstance(v, dict) or len(v["idx"]) < 16:
                continue
            arrays[f"p1_{n}"] = d["kpts"][i][v["idx"][:, 0]]
            arrays[f"p2_{n}"] = d["kpts"][j][v["idx"][:, 1]]
            arrays[f"pair_{n}"] = np.array([i, j])
            n += 1
        np.savez(OUT / "ess_fixed.npz", **arrays)
        print("essential fixed pairs", n)

    if a.upload:
        import boto3
        from bucket import bucket
        BUCKET = bucket()
        s3 = boto3.client("s3", region_name="eu-west-1")
        for f in ["ops_images.zip", "ops_images.json", "probes.zip", "probes.json", "ess_fixed.npz"]:
            if (OUT / f).exists():
                s3.upload_file(str(OUT / f), BUCKET, f"jobs/1d/data/{f}")
                print("uploaded", f, (OUT / f).stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
