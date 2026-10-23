"""Kaggle CPU kernel: Mombe's front end on SideViewCows2026 barn recordings (one shard).

Self-contained: build_kernel.py embeds the repo sources this needs (mombe.body,
mombe.vision.quality/board, eval.sideview_frontend, tools.fetch_models) and the
shard settings. The kernel installs OpenCV 5.0.0, fetches the two ONNX models,
reads barn.zip straight from Zenodo with range requests (no full download), and
writes one JSON line per recording plus a few gallery JPEGs to /kaggle/working.
Dataset: Moeller, S. (2026). SideViewCows2026. Zenodo. doi:10.5281/zenodo.21605650 (CC BY 4.0).
"""

import base64
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zlib
from pathlib import Path

CONFIG = {}  # @@CONFIG@@
SOURCES = {}  # @@SOURCES@@

WORK = Path("/tmp/sv")
SRC = WORK / "src"
OUT = Path("/kaggle/working")
OPENCV = "opencv-contrib-python-headless==5.0.0.93"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sh(cmd: str) -> int:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    log(f"$ {cmd} -> {r.returncode}", *tail)
    return r.returncode


def setup() -> None:
    sh("pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless")
    if sh(f"pip install -q {OPENCV}") != 0:
        raise SystemExit("pip install of OpenCV 5 failed")
    for rel, blob in SOURCES.items():
        p = SRC / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(zlib.decompress(base64.b64decode(blob)))
    sys.path.insert(0, str(SRC))
    from tools import fetch_models
    (SRC / "models").mkdir(exist_ok=True)
    for name in ("aliked", "aliked_lightglue"):
        if not fetch_models.fetch(name, SRC / "models"):
            raise SystemExit(f"model {name} failed its SHA-1 check")


def sysinfo() -> dict:
    import cv2
    du = shutil.disk_usage("/tmp")
    mem = 0
    try:
        mem = int(open("/proc/meminfo").read().split()[1]) // 1024
    except OSError:
        pass
    return {"opencv": cv2.__version__, "cpus": os.cpu_count(), "cv2_threads": cv2.getNumThreads(),
            "mem_mb": mem, "tmp_free_gb": round(du.free / 1e9, 1), "python": sys.version.split()[0]}


def micro_timing(recon_mod, recon, frames) -> dict:
    import cv2
    import numpy as np
    K = np.array([[0.85 * 1920, 0, 960], [0, 0.85 * 1920, 540], [0, 0, 1.0]])
    t = time.time()
    feats = recon.features(frames[:3], K, np.zeros(5))
    tf = (time.time() - t) / 3
    tm = []
    for _ in range(3):
        t = time.time()
        recon.match(feats[0], feats[1], K)
        tm.append(time.time() - t)
    return {"aliked_s_per_frame": round(tf, 2), "match_s_per_pair": [round(x, 2) for x in tm],
            "kpts": [len(f.kpts) for f in feats]}


def main() -> None:
    t0 = time.time()
    shard, nshards = CONFIG["shard"], CONFIG["nshards"]
    log("config", CONFIG)
    setup()
    import cv2
    import numpy as np
    if not cv2.__version__.startswith("5."):
        raise SystemExit(f"expected OpenCV 5, got {cv2.__version__}")
    import mombe.body.dense  # noqa: F401  (PAIR_STEP)
    import mombe.body.recon as recon_mod
    from mombe.body import provider
    from mombe.vision import quality
    from eval import sideview_frontend as sv

    recon_mod.LOOP = False           # single flank: the walk-around loop closure does not apply
    max_frames = CONFIG.get("max_frames") or provider.MAX_FRAMES
    info = sysinfo()
    log("sysinfo", info)
    manifest = sv.http_get(sv.MANIFEST_URL).decode("utf-8")
    recs = sv.recordings(manifest)
    log("barn recordings", len(recs))
    only = CONFIG.get("only")
    if only:
        mine = [r for r in recs if r["id"] in only]
    else:
        mine = [r for k, r in enumerate(recs) if k % nshards == shard]
    if CONFIG.get("limit"):
        mine = mine[:CONFIG["limit"]]
    log("this shard", len(mine), [r["id"] for r in mine])
    rz = sv.RemoteZip()
    log("central directory", len(rz.info), "entries")
    recon = recon_mod.Reconstructor(SRC / "models")
    rows_f = OUT / f"rows_s{shard}.jsonl"
    gal_dir = OUT / "gallery"
    gal_dir.mkdir(exist_ok=True)
    gal_meta = OUT / f"gallery_s{shard}.jsonl"
    variants = tuple(CONFIG.get("variants", ("as_shipped", "hide_only")))
    done, failed, skipped = 0, 0, []
    timing = None
    provider_check = None
    from concurrent.futures import ThreadPoolExecutor

    def fetch(r):
        t = time.time()
        z = sv.RemoteZip.__new__(sv.RemoteZip)       # same directory, own byte counters
        z.__dict__.update({k: v for k, v in rz.__dict__.items()})
        z.bytes_fetched, z.span_requests = 0, 0
        data = z.fetch(r["images"] + r["masks"])
        return data, round(time.time() - t, 1), z.bytes_fetched, z.span_requests

    pool = ThreadPoolExecutor(1)                     # the next recording downloads while this one computes
    nxt = pool.submit(fetch, mine[0]) if mine else None
    total_bytes, total_req = 0, 0
    for k, r in enumerate(mine):
        fut, nxt = nxt, None
        if time.time() - t0 > CONFIG["budget_h"] * 3600:
            skipped.extend(x["id"] for x in mine[k:])
            break
        d = WORK / "rec" / r["id"]
        row: dict = {"id": r["id"], "individual": r["individual"], "frames": len(r["images"]), "shard": shard}
        try:
            data, row["download_s"], nb, nr = fut.result()
            if k + 1 < len(mine):
                nxt = pool.submit(fetch, mine[k + 1])
            total_bytes += nb
            total_req += nr
            row["download_mb"] = round(nb / 1e6, 1)
            d.mkdir(parents=True, exist_ok=True)
            paths = []
            for p in r["images"]:
                fp = d / Path(p).name
                fp.write_bytes(data[p])
                paths.append(str(fp))
            kidx = sv.select_keyframes(paths, max_frames)
            if provider_check is None and max_frames == provider.MAX_FRAMES:
                class Cap:
                    body = paths
                got = provider.WalkAroundBody._frames(None, Cap)
                provider_check = bool(len(got) == len(kidx) and all(
                    np.array_equal(a, cv2.imread(paths[i])) for a, i in zip(got, kidx)))
                log("keyframe selection equals provider._frames:", provider_check)
            sharp = sv.sharpness_all(paths)
            ks = sharp[kidx]
            row["keyframe_sharpness"] = {"all_median": round(float(np.median(sharp)), 1),
                                         "kept_median": round(float(np.median(ks)), 1),
                                         "kept_min": round(float(ks.min()), 1)}
            masks = [cv2.imdecode(np.frombuffer(data[r["masks"][i]], np.uint8), cv2.IMREAD_GRAYSCALE) > 127
                     for i in kidx]
            del data
            if timing is None:
                timing = micro_timing(recon_mod, recon, [cv2.imread(paths[i]) for i in kidx[:3]])
                log("micro timing", timing)
            g = sv.Shard()
            res = sv.evaluate_recording(paths, masks, kidx, recon_mod, recon, quality, shard=g, rec_id=r["id"],
                                        variants=variants)
            row.update(res)
            for gi in g.gallery:
                name = f"{gi['rec']}_{gi['variant']}_{gi['kind']}.jpg"
                (gal_dir / name).write_bytes(gi.pop("jpg"))
                gi["file"] = name
                with gal_meta.open("a") as fh:
                    fh.write(json.dumps(gi, default=float) + "\n")
            done += 1
            ho = res.get("hide_only", {}).get("sfm", {})
            log(f"[{k + 1}/{len(mine)}] {r['id']} kf {res['keyframes']} reg {res['sfm']['registered_frac']} "
                f"nb-inl {res['neighbour']['inliers_median']} on-cow {res['neighbour']['on_cow_median']} "
                f"| hide reg {ho.get('registered_frac')} | {res['seconds']} s, dl {row['download_s']} s")
        except Exception as e:
            if nxt is None and k + 1 < len(mine):
                nxt = pool.submit(fetch, mine[k + 1])
            failed += 1
            row["error"] = f"{type(e).__name__}: {e}"
            row["traceback"] = traceback.format_exc()[-2000:]
            log("FAILED", r["id"], row["error"])
        finally:
            shutil.rmtree(d, ignore_errors=True)
        with rows_f.open("a") as fh:
            fh.write(sv.to_json(row).replace("\n", "") + "\n")
    summary = {"shard": shard, "nshards": nshards, "attempted": len(mine), "measured": done, "failed": failed,
               "skipped_budget": skipped, "sysinfo": info, "micro_timing": timing,
               "provider_keyframe_check": provider_check, "bytes_fetched": total_bytes,
               "range_requests": total_req, "elapsed_h": round((time.time() - t0) / 3600, 2),
               "config": CONFIG}
    (OUT / f"summary_s{shard}.json").write_text(json.dumps(summary, indent=1))
    log("summary", summary)


if __name__ == "__main__":
    main()
