"""Fetch one batch of the Acme AI / bhalo / mPower cattle weight dataset (Kaggle, CC BY 4.0).

Side and rear photos with tape-formula weights in the file names, segmentation
masks (sticker, cattle, background) and COCO keypoints. Files are pulled one by
one in parallel so a 5 GB batch arrives without the 47.8 GB archive.

Kaggle rate-limits single-file downloads to about one a second; a batch takes hours.

Usage: python tools/fetch_acme.py --batch B4 [--workers 3]
"""

from __future__ import annotations

import argparse
import random
import time
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = "sadhliroomyprime/cattle-weight-detection-model-dataset-12k"
PREFIX = "www.acmeai.tech Dataset - BMGF-LivestockWeight-CV/"


def list_files(api) -> list[str]:
    names, tok = [], None
    while True:
        r = api.dataset_list_files(REF, page_token=tok, page_size=200)
        names += [f.name for f in r.files]
        tok = r.next_page_token
        if not tok:
            return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="B4")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    out = ROOT / "data" / "acme"
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "files.json"
    names = json.loads(cache.read_text()) if cache.exists() else list_files(api)
    cache.write_text(json.dumps(names))
    want = [n for n in names if n.startswith(f"{PREFIX}Pixel/{a.batch}/")
            or (n.startswith(f"{PREFIX}Vector/{a.batch}/") and n.endswith(".json"))
            or n.endswith("Readme.md")]
    todo = [n for n in want if not (out / n[len(PREFIX):]).exists()]
    # keypoints first, then whole animals (side and rear photos with their masks), so a
    # partial batch is already a usable set of complete animals
    def animal(n: str) -> str:
        return "_".join(n.rsplit("/", 1)[-1].split("_")[:2])
    def rank(n: str) -> int:
        if n.endswith((".json", ".md")):
            return 0
        if "/Side/annotations/" in n:
            return 1                     # the sticker scale: all the weight study needs per animal
        if "/Side/images/" in n:
            return 2
        return 3
    todo.sort(key=lambda n: (rank(n), animal(n), n))
    print(f"{len(want)} files in {a.batch}, {len(todo)} to fetch", flush=True)

    def fetch(name: str) -> str:
        dest = out / name[len(PREFIX):]
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.parent / f".dl_{abs(hash(name))}"          # one folder per file: threads never collide
        tmp.mkdir(exist_ok=True)
        for attempt in range(10):
            try:
                api.dataset_download_file(REF, name, path=str(tmp), quiet=True)
                got = sorted(tmp.iterdir(), key=lambda p: p.stat().st_mtime)[-1]
                if got.suffix == ".zip" and dest.suffix != ".zip":
                    with zipfile.ZipFile(got) as z:
                        dest.write_bytes(z.read(z.namelist()[0]))
                    got.unlink()
                else:
                    got.replace(dest)
                tmp.rmdir()
                return name
            except Exception as e:  # Kaggle answers 429 under load: back off and retry
                err = e
                time.sleep(min(120, 4 * 2 ** attempt) + random.random() * 3)
        for p in tmp.iterdir():
            p.unlink()
        tmp.rmdir()
        raise RuntimeError(f"{name}: {err}")

    done = failed = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(fetch, n) for n in todo]
        for f in as_completed(futs):
            try:
                f.result()
                done += 1
            except Exception as e:
                failed += 1
                print("FAIL", e, flush=True)
            if (done + failed) % 250 == 0:
                print(f"{done + failed}/{len(todo)} fetched, {failed} failed", flush=True)
    print(f"DONE {done} fetched, {failed} failed", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
