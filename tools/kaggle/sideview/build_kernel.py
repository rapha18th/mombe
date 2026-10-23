"""Build a self-contained Kaggle script kernel for the SideViewCows2026 front-end evaluation.

Embeds the repo sources kernel_main.py imports and the shard settings, then writes
kernel.py + kernel-metadata.json into --out (default eval/out/sideview/push/<slug>).

  python tools/kaggle/sideview/build_kernel.py --shard 0 --nshards 4
  python tools/kaggle/sideview/build_kernel.py --probe              # 1 recording, 16 keyframes
  kaggle kernels push -p eval/out/sideview/push/<slug>
"""

from __future__ import annotations

import argparse
import base64
import json
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FILES = ["mombe/__init__.py", "mombe/body/__init__.py", "mombe/body/recon.py", "mombe/body/provider.py",
         "mombe/body/dense.py", "mombe/body/measure.py", "mombe/vision/__init__.py", "mombe/vision/quality.py",
         "mombe/vision/board.py", "eval/sideview_frontend.py", "tools/fetch_models.py"]
USER = "thetraveller"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--max-frames", type=int, default=0, help="override MAX_FRAMES (probe only)")
    ap.add_argument("--variants", nargs="+", default=["as_shipped", "hide_only"])
    ap.add_argument("--budget-h", type=float, default=11.0)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--slug", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.probe:
        a.only = a.only or ["594_51181673"]
        a.max_frames = a.max_frames or 16
        a.budget_h = min(a.budget_h, 2.0)
    slug = a.slug or ("mombe-sideview-probe" if a.probe else f"mombe-sideview-s{a.shard}of{a.nshards}")
    cfg = {"shard": a.shard, "nshards": a.nshards, "limit": a.limit, "only": a.only, "max_frames": a.max_frames,
           "variants": a.variants, "budget_h": a.budget_h, "probe": a.probe}
    src = {f: base64.b64encode(zlib.compress((ROOT / f).read_bytes(), 9)).decode() for f in FILES}
    tpl = (Path(__file__).parent / "kernel_main.py").read_text(encoding="utf-8")
    if "CONFIG = {}  # @@CONFIG@@" not in tpl or "SOURCES = {}  # @@SOURCES@@" not in tpl:
        raise SystemExit("template markers missing")
    body = (tpl.replace("CONFIG = {}  # @@CONFIG@@", "CONFIG = " + repr(cfg))
               .replace("SOURCES = {}  # @@SOURCES@@", "SOURCES = " + json.dumps(src, indent=0)))
    out = Path(a.out) if a.out else ROOT / "eval" / "out" / "sideview" / "push" / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernel.py").write_text(body, encoding="utf-8")
    meta = {"id": f"{USER}/{slug}", "title": slug, "code_file": "kernel.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
            "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(out, f"{len(body) / 1e3:.0f} kB", cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
