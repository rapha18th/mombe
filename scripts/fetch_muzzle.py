"""Fetch the public beef cattle muzzle dataset into data/muzzle/<animal>/<image>.jpg.

Source: Xiong, Li and Erickson (2022), Beef Cattle Muzzle/Noseprint database for
individual identification, Zenodo 6324361, CC BY 4.0. 4,923 images of 268 animals.
The archive (644 MB) arrives as parallel HTTP range requests that resume after a
dropped connection, is checked against the MD5 that Zenodo publishes, and is
unpacked with its top folder removed.

Usage: python scripts/fetch_muzzle.py [--dst data/muzzle] [--zip existing.zip] [--no-verify]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://zenodo.org/api/records/6324361/files/BeefCattle_Muzzle_database.zip/content"
MD5 = "f2b19d9bab2fc668460a796259944204"
SIZE = 643_745_309
PARTS = 8


def _part(z: Path, lo: int, hi: int, tries: int = 12) -> None:
    """Fetch bytes lo..hi into their place in z, resuming where a dropped connection stopped."""
    pos = lo
    for k in range(tries):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={pos}-{hi}", "User-Agent": "mombe-fetch"})
            with urllib.request.urlopen(req, timeout=60) as r, z.open("r+b") as f:
                f.seek(pos)
                while pos <= hi:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    pos += len(chunk)
            if pos > hi:
                return
        except OSError as e:
            print(f"  range {lo}-{hi}: retry {k + 1} at byte {pos} ({e})", flush=True)
        time.sleep(min(30, 2 * (k + 1)))
    raise SystemExit(f"download failed for bytes {pos}-{hi}; rerun to try again")


def download(z: Path) -> None:
    with z.open("wb") as f:
        f.truncate(SIZE)
    step = -(-SIZE // PARTS)
    spans = [(lo, min(lo + step, SIZE) - 1) for lo in range(0, SIZE, step)]
    with ThreadPoolExecutor(PARTS) as ex:
        list(ex.map(lambda s: _part(z, *s), spans))


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default=str(ROOT / "data" / "muzzle"))
    ap.add_argument("--zip", help="use this archive instead of downloading")
    ap.add_argument("--no-verify", action="store_true", help="skip the MD5 check")
    a = ap.parse_args()
    dst = Path(a.dst)
    if a.zip:
        z = Path(a.zip)
    else:
        z = ROOT / "data" / "raw" / "BeefCattle_Muzzle_database.zip"
        z.parent.mkdir(parents=True, exist_ok=True)
        if not (z.exists() and md5(z) == MD5):
            print(f"download {URL}\n      -> {z} (644 MB)", flush=True)
            download(z)
    if not a.no_verify:
        got = md5(z)
        if got != MD5:
            print(f"MD5 {got} differs from Zenodo's {MD5}; rerun, or pass --no-verify for a known mirror", file=sys.stderr)
            return 1
    n = 0
    with zipfile.ZipFile(z) as zf:
        for info in zf.infolist():
            parts = Path(info.filename).parts
            if info.is_dir() or not parts[-1].lower().endswith(".jpg"):
                continue
            animal, name = parts[-2], parts[-1]          # drop any top folder
            if not animal.startswith("cattle_"):
                continue
            out = dst / animal / name
            out.parent.mkdir(parents=True, exist_ok=True)
            if not (out.exists() and out.stat().st_size == info.file_size):
                out.write_bytes(zf.read(info))
            n += 1
    animals = sum(1 for p in dst.iterdir() if p.is_dir())
    print(f"{n} images of {animals} animals in {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
