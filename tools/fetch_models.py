"""Fetch the ONNX models Mombe runs through the OpenCV 5 DNN engine.

Every entry comes from OpenCV's own test-data registry
(opencv_extra/testdata/dnn/download_models.py, 5.x branch), so each file is one
the OpenCV 5 test suite already exercises. SHA-1 values are copied from there.

Usage: python tools/fetch_models.py [--only name ...] [--dst models]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS = {
    "aliked": (
        "https://raw.githubusercontent.com/YangGuanyuhan/lightglue_opencv_project/main/model/aliked-n16rot-top1k-640.onnx",
        "41faa7bf5d7eb68a2851471ba03aa20c9db30e4c",
        "aliked-n16rot-top1k-640.onnx",
    ),
    "aliked_lightglue": (
        "https://raw.githubusercontent.com/YangGuanyuhan/lightglue_opencv_project/main/model/aliked_lightglue.onnx",
        "02723aa521990e57fe33d90b67977590c460e351",
        "aliked_lightglue.onnx",
    ),
    "xfeat": (
        "https://raw.githubusercontent.com/YangGuanyuhan/lightglue_opencv_project/main/model/xfeat.onnx",
        "0d8bdfff4e123e5e0c04b9fe3b52e46555b8ba2c",
        "xfeat.onnx",
    ),
    "disk": (
        "https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v0.1.0/disk.onnx",
        "5f6a9069aed0af7302b67dcfb6d24b0d46707aec",
        "disk.onnx",
    ),
    "disk_lightglue": (
        "https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v0.1.0/disk_lightglue.onnx",
        "9c885c13df94a0cadd5fb577c278471b78cc7058",
        "disk_lightglue.onnx",
    ),
    "dinov2_small": (
        "https://huggingface.co/onnx-community/dinov2-small/resolve/main/onnx/model.onnx",
        "2c147e4cbd703f6ef2df3a737ace46b637af1ef3",
        "dinov2_small.onnx",
    ),
    "depth_anything_v2_small": (
        "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx",
        "034e45219f00397bb9691fa1f97cc7ef69f4dd7e",
        "depth_anything_v2_small.onnx",
    ),
}


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, dst: Path) -> bool:
    url, digest, filename = MODELS[name]
    target = dst / filename
    if target.exists() and sha1(target) == digest:
        print(f"ok    {name:24s} {target}")
        return True
    tmp = target.with_suffix(".part")
    print(f"fetch {name:24s} {url}")
    urllib.request.urlretrieve(url, tmp)
    got = sha1(tmp)
    if got != digest:
        tmp.unlink(missing_ok=True)
        print(f"FAIL  {name:24s} sha1 {got} != {digest}", file=sys.stderr)
        return False
    tmp.replace(target)
    print(f"ok    {name:24s} {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(MODELS))
    ap.add_argument("--dst", default=str(Path(__file__).resolve().parents[1] / "models"))
    args = ap.parse_args()
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    names = args.only or list(MODELS)
    return 0 if all(fetch(n, dst) for n in names) else 1


if __name__ == "__main__":
    raise SystemExit(main())
