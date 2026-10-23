"""Build the SYNTHETIC presentation-attack set for liveness evaluation.

Bona fide samples are real bursts from the public muzzle dataset: two field
photos of one animal taken moments apart (camera frame numbers at most 4
apart). Attack samples are rendered from real photos by mombe.sim.attacks:

  print   the photo on paper, held up as a card that moves between frames
  screen  the photo on a screen, with subpixel moire, colour shift and glare
  replay  two enrolled gallery images sent again, exact or lightly cropped

Identities follow eval/dataset.make_split and stay disjoint: dev animals (the
identity dev slice plus the first half of the never-registered animals, by
sorted id) and held-out animals (registered test animals plus the other half).
Replay needs an enrolled image, so it uses registered animals only. Every sample has
its own seed derived from the run seed and the sample id, so a rerun with the
same seed rebuilds the same images whatever the worker count.

Writes <out>/manifest.jsonl and the rendered frames under <out>/<group>/<kind>/.
Usage: python tools/synth_attacks.py [--out build/attacks] [--seed 2026] [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.dataset import IMGS, folder, make_split  # noqa: E402
from mombe.sim import attacks  # noqa: E402

MAX_GAP = 4          # camera frame numbers apart for two photos of one burst


def frame_no(image_id: str) -> tuple[str, int]:
    m = re.search(r"_([A-Z]+)_?(\d+)$", image_id)
    return (m.group(1), int(m.group(2))) if m else ("", -1)


def rel(p) -> str:
    """Repo-relative where possible, so the manifest survives a move of the checkout."""
    p = Path(p).absolute()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def image_path(image_id: str) -> str:
    return rel(IMGS / folder(image_id) / f"{image_id}.jpg")


def bursts(images: list[str], n: int) -> list[list[str]]:
    """Up to n non-overlapping consecutive pairs taken within one burst, evenly spaced."""
    cand = []
    for a, b in zip(images, images[1:]):
        (pa, na), (pb, nb) = frame_no(a), frame_no(b)
        if pa == pb and 0 < nb - na <= MAX_GAP:
            cand.append([a, b])
    picked, used = [], set()
    for i in np.linspace(0, len(cand) - 1, min(len(cand), 4 * n)).round().astype(int) if cand else []:
        a, b = cand[i]
        if a in used or b in used:
            continue
        picked.append(cand[i])
        used.update((a, b))
        if len(picked) == n:
            break
    return picked


def sample_seed(seed: int, sid: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(sid.encode())])


def plan(split, per_type: int, n_bona: int, limit: int = 0) -> list[dict]:
    # dev: the identity-threshold dev animals plus the first half of the never-registered
    # animals (all their photos are field photos, so they carry many live bursts)
    unreg = sorted(split.unregistered)
    half = len(unreg) // 2
    groups = [("dev", a, True) for a in split.dev] + [("dev", a, False) for a in unreg[:half]] + \
             [("heldout", a, True) for a in split.test] + [("heldout", a, False) for a in unreg[half:]]
    if limit:
        groups = groups[:limit]
    out = []
    for group, a, registered in groups:
        field_imgs = sorted(split.field_imgs[a])
        for k, pair in enumerate(bursts(field_imgs, n_bona)):
            out.append({"id": f"{a}-bona-{k}", "identity": a, "group": group, "kind": "bona_fide",
                        "registered": registered, "frames": [image_path(x) for x in pair], "sources": pair})
        srcs = [field_imgs[i] for i in np.linspace(0, len(field_imgs) - 1, 2 * per_type + 2).round().astype(int)[1:-1]]
        for k in range(per_type):
            for j, kind in enumerate(("print", "screen")):
                src = srcs[(2 * k + j) % len(srcs)]
                out.append({"id": f"{a}-{kind}-{k}", "identity": a, "group": group, "kind": kind,
                            "registered": registered, "sources": [src]})
        if registered:
            gal = sorted(split.gallery_imgs[a])
            for k in range(per_type):
                j = (k * 3) % max(1, len(gal) - 1)
                out.append({"id": f"{a}-replay-{k}", "identity": a, "group": group, "kind": "replay",
                            "registered": True, "sources": gal[j:j + 2]})
    return out


def render_one(args: tuple[dict, int, str]) -> dict:
    s, seed, out_root = args
    if s["kind"] == "bona_fide":
        return s
    rng = sample_seed(seed, s["id"])
    out_dir = Path(out_root) / s["group"] / s["kind"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if s["kind"] == "replay":
        enrolled = [cv2.imread(str(ROOT / image_path(x))) for x in s["sources"]]
        frames, meta = attacks.render_replay(enrolled, rng)
        paths = []
        for k, (f, src) in enumerate(zip(frames, s["sources"])):
            if f is None:
                paths.append(image_path(src))          # the enrolled file itself
            else:
                p = out_dir / f"{s['id']}_{k}.jpg"
                cv2.imwrite(str(p), f, [cv2.IMWRITE_JPEG_QUALITY, 100])
                paths.append(rel(p))
        return {**s, "frames": paths, "meta": meta}
    src = cv2.imread(str(ROOT / image_path(s["sources"][0])))
    frames, meta = attacks.render(s["kind"], src, 2, rng)
    paths = []
    for k, f in enumerate(frames):
        p = out_dir / f"{s['id']}_{k}.jpg"
        # frames already went through the simulated camera's JPEG; store them losslessly from here on
        cv2.imwrite(str(p), f, [cv2.IMWRITE_JPEG_QUALITY, 100])
        paths.append(rel(p))
    return {**s, "frames": paths, "meta": meta}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "build" / "attacks"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--per-type", type=int, default=1, help="attacks of each type per identity")
    ap.add_argument("--bona-fide", type=int, default=3, help="live bursts per identity")
    ap.add_argument("--limit", type=int, default=0, help="first N identities only (smoke runs)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true", help="keep samples already rendered under this seed")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    samples = plan(make_split(), a.per_type, a.bona_fide, a.limit)
    done = {}
    if a.resume and (out / "manifest.jsonl").exists():
        for x in (out / "manifest.jsonl").read_text().splitlines():
            r = json.loads(x)
            if r.get("seed") == a.seed and all((ROOT / p).exists() for p in r.get("frames", [])):
                done[r["id"]] = r
    t0, rows = time.time(), [{**done[s["id"]], "group": s["group"]} for s in samples if s["id"] in done]
    jobs = [(s, a.seed, str(out)) for s in samples if s["id"] not in done]
    it = map(render_one, jobs) if a.workers == 1 else ProcessPoolExecutor(a.workers).map(render_one, jobs, chunksize=2)
    for i, r in enumerate(it, 1):
        rows.append(r)
        if i % 100 == 0 or i == len(jobs):
            print(f"{i}/{len(jobs)} samples  {time.time() - t0:.0f}s", flush=True)
    order = {s["id"]: k for k, s in enumerate(samples)}
    rows.sort(key=lambda r: order[r["id"]])
    with (out / "manifest.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({**r, "synthetic": r["kind"] != "bona_fide", "seed": a.seed}) + "\n")
    kinds = {}
    for r in rows:
        kinds[(r["group"], r["kind"])] = kinds.get((r["group"], r["kind"]), 0) + 1
    print(json.dumps({f"{g}/{k}": n for (g, k), n in sorted(kinds.items())}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
