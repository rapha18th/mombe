"""Turn a phone walk-around video into the frames the hub expects.

Samples the clip at the app's rate (2 frames a second, at most 80), scales the
long side to 1920 px, applies the phone's rotation, and writes frame_XX.jpg
into a folder that tools/live_value.py --walk and the body provider read.
The first frames are the nose close-up the protocol opens with.

Usage: python tools/video_frames.py walk.mp4 --out build/walks/cow1 [--fps 2] [--max 80]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max", type=int, default=80)
    ap.add_argument("--long-side", type=int, default=1920)
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.video}")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)          # Android stores rotation as metadata
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(src_fps / a.fps))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kept, i, size = 0, 0, None
    while kept < a.max:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            s = a.long_side / max(f.shape[:2])
            if s < 1:
                f = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / f"frame_{kept:02d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 92])
            size = [int(f.shape[1]), int(f.shape[0])]
            kept += 1
        i += 1
    info = {"video": str(a.video), "source_fps": round(src_fps, 2), "source_frames": total, "kept": kept,
            "size": size, "seconds_covered": round(i / src_fps, 1)}
    (out / "video.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
