"""Wait until every given S3 key under results/1d/ exists, then download a prefix.

Usage: python tools/cool_equiv/wait_s3.py --keys cool/bisect/x.json ... --get cool/bisect --to DIR [--timeout 3600]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import boto3

from bucket import bucket

BASE = "results/1d/"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="*", default=[])
    ap.add_argument("--get", nargs="*", default=[])
    ap.add_argument("--to", default=".")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--skip", default="", help="comma list of suffixes not to download, e.g. .pkl")
    a = ap.parse_args()
    BUCKET = bucket()
    s3 = boto3.client("s3", region_name="eu-west-1")
    t0 = time.time()
    for k in a.keys:
        while True:
            try:
                s3.head_object(Bucket=BUCKET, Key=BASE + k)
                break
            except Exception:
                if time.time() - t0 > a.timeout:
                    print("TIMEOUT waiting for", k)
                    return 1
                time.sleep(15)
        print("present", k, round(time.time() - t0), "s", flush=True)
    skip = [s for s in a.skip.split(",") if s]
    for g in a.get:
        pag = s3.get_paginator("list_objects_v2")
        for page in pag.paginate(Bucket=BUCKET, Prefix=BASE + g):
            for o in page.get("Contents", []):
                if any(o["Key"].endswith(s) for s in skip):
                    continue
                dst = Path(a.to) / o["Key"][len(BASE):]
                if dst.exists() and dst.stat().st_size == o["Size"]:
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(BUCKET, o["Key"], str(dst))
                print("got", dst, o["Size"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
