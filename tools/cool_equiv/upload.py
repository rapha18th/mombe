"""Ship the frozen code and the track 1d job scripts to S3 (jobs/1d/), nothing else.

code.zip is built from the committed tree (git archive p0-frozen), so edits other
agents make in the working tree never reach the hubs through this track.

Usage: python tools/cool_equiv/upload.py [--code] [--files a b ...]
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[2]
BUCKET = "mombe-967170427068-eu-west-1"
PREFIX = "jobs/1d"
REF = "p0-frozen"     # the frozen commit d697be9; never the moving HEAD


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", action="store_true", help="rebuild code.zip from the frozen tag")
    ap.add_argument("--files", nargs="*", default=None, help="extra local files to put under jobs/1d/<name>")
    a = ap.parse_args()
    s3 = boto3.client("s3", region_name="eu-west-1")
    if a.code:
        rev = subprocess.run(["git", "rev-parse", "--short", REF + "^{commit}"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        z = subprocess.run(["git", "archive", "--format=zip", REF, "mombe", "tools/eval_recon.py", "tools/synth_walk.py", "tools/sweep_body.py"], cwd=ROOT,
                           capture_output=True, check=True).stdout
        s3.put_object(Bucket=BUCKET, Key=f"{PREFIX}/code.zip", Body=z, Metadata={"rev": rev})
        print("code.zip from", rev, len(z), "bytes")
    for p in sorted((ROOT / "tools" / "cool_equiv").glob("*")):
        if p.suffix in (".py", ".sh", ".json") and p.is_file():
            body = p.read_bytes().replace(b"\r\n", b"\n")
            s3.put_object(Bucket=BUCKET, Key=f"{PREFIX}/tools/{p.name}", Body=body)
    print("scripts uploaded")
    for f in a.files or []:
        p = Path(f)
        s3.upload_file(str(p), BUCKET, f"{PREFIX}/{p.name}")
        print("uploaded", p.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
