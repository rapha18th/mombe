"""Op-level diffs between two OpenCV builds on real images, on one machine.

Two ops_worker.py processes (one per build, started through run_env.sh) get
the same inputs. For chained ops each build starts from the same JPEG; for
isolated ops (board detection, cornerSubPix, LK, MAGSAC, DNN forward,
LightGlue) both builds get the SAME input array, taken from build A, so a
difference there belongs to that op alone.

Per op: images bit-identical, max abs diff, fraction of values differing, and
whether a downstream decision changes (quality issues, board ids, keypoint
sets, match sets, inlier thresholds). Hashes of every output on both sides go
to the JSON too, so runs on different machines can be compared later.

Usage: python ops_diff.py --a cool --b cvonly --images data/ops_images.json --root DIR
                          [--ess data/ess_fixed.npz] --out out/ops_cool_vs_cvonly.json [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T_MATCH, T_NEW, MIN_PAIR = 90, 20, 30


class Worker:
    def __init__(self, mode: str, log: Path):
        self.mode = mode
        # CE_LOCAL_PY runs both workers under one local python (smoke test of the harness only)
        local = os.environ.get("CE_LOCAL_PY")
        cmd = [local, str(HERE / "ops_worker.py")] if local else ["/opt/cool_equiv/run_env.sh", mode, str(HERE / "ops_worker.py")]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(log, "ab"))

    def __call__(self, *req):
        b = pickle.dumps(req, protocol=4)
        self.p.stdin.write(struct.pack("<Q", len(b)) + b)
        self.p.stdin.flush()
        hdr = self.p.stdout.read(8)
        if len(hdr) < 8:
            raise RuntimeError(f"{self.mode}: worker died on {req[0]} (exit {self.p.wait()}); see the .log file")
        n = struct.unpack("<Q", hdr)[0]
        r = pickle.loads(self.p.stdout.read(n))
        if isinstance(r, dict) and "error" in r and "trace" in r:
            raise RuntimeError(f"{self.mode}: {r['error']}\n{r['trace']}")
        return r

    def close(self):
        try:
            self("quit")
        except Exception:
            pass


def hsh(x) -> str | None:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:16]


def adiff(a, b) -> dict:
    if a is None or b is None:
        return {"identical": a is None and b is None, "missing": True}
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return {"identical": False, "shape": [list(a.shape), list(b.shape)]}
    if np.array_equal(a, b):
        return {"identical": True, "max_abs": 0.0, "frac": 0.0}
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    out = {"identical": False, "max_abs": float(d.max()), "frac": float((d > 0).mean()), "mean_abs": float(d.mean())}
    if a.dtype == np.uint8:
        out["frac_gt1"] = float((d > 1).mean())
    else:
        scale = float(np.abs(a).max()) or 1.0
        out["max_rel"] = float(d.max() / scale)
    return out


def nn(a: np.ndarray, b: np.ndarray, r: float = 0.05):
    from scipy.spatial import cKDTree
    a = np.asarray(a, np.float64).reshape(-1, 2)
    b = np.asarray(b, np.float64).reshape(-1, 2)
    if len(a) == 0 or len(b) == 0:
        return 0, 0.0
    ta, tb = cKDTree(a), cKDTree(b)
    dab, iab = tb.query(a)
    dba, iba = ta.query(b)
    ok = (dab <= r) & (iba[iab] == np.arange(len(a)))
    return int(ok.sum()), float(dab[ok].max()) if ok.any() else 0.0


def pts_diff(a, b) -> dict:
    a = np.zeros((0, 2)) if a is None else np.asarray(a).reshape(-1, 2)
    b = np.zeros((0, 2)) if b is None else np.asarray(b).reshape(-1, 2)
    if a.shape == b.shape and np.array_equal(a, b):
        return {"identical": True, "n_a": len(a), "n_b": len(b), "set_changed": False}
    m, mx = nn(a, b)
    return {"identical": False, "n_a": len(a), "n_b": len(b), "matched": m, "only_a": len(a) - m, "only_b": len(b) - m,
            "max_px_matched": mx, "set_changed": not (m == len(a) == len(b))}


class Agg:
    def __init__(self):
        self.ops: dict[str, list] = {}

    def add(self, op: str, rec: dict, img: str):
        self.ops.setdefault(op, []).append({**rec, "img": img})

    def summary(self) -> dict:
        out = {}
        for op, rows in self.ops.items():
            n = len(rows)
            ident = sum(1 for r in rows if r.get("identical"))
            s = {"n": n, "identical": ident, "identical_frac": round(ident / n, 4) if n else None}
            mx = [r["max_abs"] for r in rows if "max_abs" in r]
            if mx:
                s["max_abs"] = max(mx)
            fr = [r["frac"] for r in rows if "frac" in r]
            if fr:
                s["mean_frac_differing"] = float(np.mean(fr))
                s["max_frac_differing"] = float(max(fr))
            if any("max_rel" in r for r in rows):
                s["max_rel"] = max(r.get("max_rel", 0) for r in rows)
            if any("frac_gt1" in r for r in rows):
                s["max_frac_gt1"] = max(r.get("frac_gt1", 0) for r in rows)
            for k in ("decision_changed", "set_changed"):
                if any(k in r for r in rows):
                    s[k] = sum(1 for r in rows if r.get(k))
            if any("only_a" in r for r in rows):
                s["points_only_a"] = int(sum(r.get("only_a", 0) for r in rows))
                s["points_only_b"] = int(sum(r.get("only_b", 0) for r in rows))
                s["points_a"] = int(sum(r.get("n_a", 0) for r in rows))
                s["max_px_matched"] = max(r.get("max_px_matched", 0) for r in rows)
            for k in ("inliers_abs_diff", "cos_min", "rel_diff"):
                vals = [r[k] for r in rows if k in r]
                if vals:
                    s[k + ("_max" if k != "cos_min" else "")] = max(vals) if k != "cos_min" else min(vals)
            worst = sorted([r for r in rows if not r.get("identical")], key=lambda r: -(r.get("max_abs") or 0))[:3]
            s["worst"] = worst
            out[op] = s
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--ess")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--b-unoptimized", action="store_true", help="cv2.setUseOptimized(False) in worker B")
    a = ap.parse_args()
    imgs = json.loads(Path(a.images).read_text())
    if a.limit:
        imgs = imgs[: a.limit]
    log = Path(a.out).with_suffix(".log")
    A, B = Worker(a.a, log), Worker(a.b, log)
    info = {"a": A("info"), "b": B("info")}
    if a.b_unoptimized:
        info["b"]["useOptimized_set"] = B("set_optimized", False)
    agg = Agg()
    hashes = []
    t0 = time.time()
    feats_a: dict[int, dict] = {}
    for i, rel in enumerate(imgs):
        path = str(Path(a.root) / rel)
        ca, cb = A("chain", path), B("chain", path)
        rec = {"img": rel, "a": {}, "b": {}}
        for k in ca:
            va, vb = ca[k], cb[k]
            rec["a"][k], rec["b"][k] = hsh(va), hsh(vb)
            if k == "quality":
                agg.add("quality_assess", {"identical": va == vb, "decision_changed": va["issues"] != vb["issues"],
                                           "rel_diff": abs(va["sharpness"] - vb["sharpness"]) / max(va["sharpness"], 1e-9)}, rel)
            elif k == "laplacian_var":
                agg.add("laplacian_var", {"identical": va == vb, "rel_diff": abs(va - vb) / max(abs(va), 1e-12),
                                          "decision_changed": (va < 150) != (vb < 150)}, rel)
            elif k == "gftt":
                agg.add("goodFeaturesToTrack", pts_diff(va, vb), rel)
            else:
                agg.add(k, adiff(va, vb), rel)
        # isolated ops, same input on both builds (from build A)
        comp = A("composite", ca["imread"], i)
        compb = B("composite", ca["imread"], i)
        agg.add("board_composite_warp", adiff(comp, compb), rel)
        da, db = A("detect", comp), B("detect", comp)
        ida = None if da["ids"] is None else np.sort(np.asarray(da["ids"]).ravel())
        idb = None if db["ids"] is None else np.sort(np.asarray(db["ids"]).ravel())
        same_ids = (ida is None and idb is None) or (ida is not None and idb is not None and np.array_equal(ida, idb))
        r = {"identical": same_ids and hsh(da["corners"]) == hsh(db["corners"]), "decision_changed": not same_ids,
             "detected_a": ida is not None, "detected_b": idb is not None}
        if same_ids and ida is not None:
            oa, ob = np.argsort(np.asarray(da["ids"]).ravel()), np.argsort(np.asarray(db["ids"]).ravel())
            dd = np.abs(np.asarray(da["corners"])[oa] - np.asarray(db["corners"])[ob])
            r.update(max_abs=float(dd.max()), frac=float((dd > 0).any(1).mean()))
        agg.add("charuco_detect", r, rel)
        g640 = ca["gray640"]
        pts = ca["gftt"]
        if pts is not None and len(pts):
            agg.add("cornerSubPix", adiff(A("subpix", g640, pts), B("subpix", g640, pts)), rel)
            g1 = np.ascontiguousarray(np.roll(g640, 3, axis=1))
            la, lb = A("lk", g640, g1, pts), B("lk", g640, g1, pts)
            r = adiff(la["next"], lb["next"])
            r["decision_changed"] = not np.array_equal(la["status"], lb["status"])
            agg.add("calcOpticalFlowPyrLK", r, rel)
        pre = ca["preprocess"]
        ka, kb = A("aliked_kp", pre), B("aliked_kp", pre)
        r = pts_diff(ka["pts"], kb["pts"])
        if r["identical"]:
            r["identical"] = bool(np.array_equal(ka["desc"], kb["desc"]))
        agg.add("aliked_detect_same_input", r, rel)
        kbo = B("aliked_kp", cb["preprocess"])
        agg.add("aliked_detect_own_chain", pts_diff(ka["pts"], kbo["pts"]), rel)
        blob = np.ascontiguousarray(ca["blob_dinov2"])
        na, nb = A("dino", blob), B("dino", blob)
        r = adiff(na["tokens"], nb["tokens"])
        r["cos_min"] = float(np.dot(na["emb"], nb["emb"]))
        agg.add("dnn_dinov2_same_input", r, rel)
        nbo = B("dino", np.ascontiguousarray(cb["blob_dinov2"]))
        agg.add("dnn_dinov2_own_chain", {"identical": bool(np.array_equal(na["emb"], nbo["emb"])),
                                         "max_abs": float(np.abs(na["emb"] - nbo["emb"]).max()),
                                         "cos_min": float(np.dot(na["emb"], nbo["emb"]))}, rel)
        # raw ALIKED network on a 640 x 640 blob built from A's preprocessed image (letterboxed)
        canvas = np.zeros((640, 640, 3), np.uint8)
        canvas[: pre.shape[0], : pre.shape[1]] = pre
        ablob = np.ascontiguousarray(canvas[None].astype(np.float32).transpose(0, 3, 1, 2)[:, ::-1] / 255.0)
        oa, ob = A("aliked_net", ablob), B("aliked_net", ablob)
        for name in oa:
            agg.add(f"dnn_aliked_{name}_same_input", adiff(oa[name], ob[name]), rel)
        feats_a[i] = {"pts": ka["pts"], "desc": ka["desc"], "size": (pre.shape[1], pre.shape[0])}
        if rel.startswith("muzzle") and i % 2 == 1 and (i - 1) in feats_a and imgs[i - 1].startswith("muzzle"):
            f0, f1 = feats_a[i - 1], feats_a[i]
            la = A("lightglue", f0["pts"], f1["pts"], f0["desc"], f1["desc"], f0["size"], f1["size"])
            lb = B("lightglue", f0["pts"], f1["pts"], f0["desc"], f1["desc"], f0["size"], f1["size"])
            if "idx" in la and "idx" in lb:
                same = la["idx"].shape == lb["idx"].shape and np.array_equal(la["idx"], lb["idx"])
                sa = {tuple(x) for x in la["idx"].tolist()}
                sb = {tuple(x) for x in lb["idx"].tolist()}
                agg.add("lightglue_same_input", {"identical": same and np.array_equal(la["dist"], lb["dist"]),
                                                 "set_changed": sa != sb, "n_a": len(sa), "n_b": len(sb),
                                                 "max_abs": float(np.abs(la["dist"] - lb["dist"]).max()) if same and len(sa) else 0.0}, rel)
                if len(la["idx"]) >= 8:
                    p1 = f0["pts"][la["idx"][:, 0]]
                    p2 = f1["pts"][la["idx"][:, 1]]
                    ha, hb = A("homog", p1, p2), B("homog", p1, p2)
                    ia = 0 if ha["mask"] is None else int(ha["mask"].sum())
                    ib = 0 if hb["mask"] is None else int(hb["mask"].sum())
                    cat = lambda n: "match" if n >= T_MATCH else ("new" if n <= T_NEW else "ambiguous")  # noqa: E731
                    agg.add("findHomography_magsac_fixed", {"identical": hsh(ha["mask"]) == hsh(hb["mask"]) and hsh(ha["H"]) == hsh(hb["H"]),
                                                            "inliers_abs_diff": abs(ia - ib), "decision_changed": cat(ia) != cat(ib),
                                                            "set_changed": hsh(ha["mask"]) != hsh(hb["mask"]), "inliers_a": ia, "inliers_b": ib}, rel)
        hashes.append(rec)
        if i % 20 == 0:
            print(f"{i}/{len(imgs)} {time.time() - t0:.0f}s", flush=True)
    # essential matrices on fixed pipeline matches (sweep_21 pairs from the stock dump)
    if a.ess:
        z = np.load(a.ess)
        K = z["K"]
        for k in sorted({n.split("_")[1] for n in z.files if n.startswith("p1_")}, key=int):
            p1, p2 = z[f"p1_{k}"], z[f"p2_{k}"]
            ea, eb = A("ess", p1, p2, K), B("ess", p1, p2, K)
            ia = 0 if ea["mask"] is None else int(ea["mask"].sum())
            ib = 0 if eb["mask"] is None else int(eb["mask"].sum())
            agg.add("findEssentialMat_magsac_fixed", {"identical": hsh(ea["mask"]) == hsh(eb["mask"]) and hsh(ea["E"]) == hsh(eb["E"]),
                                                      "set_changed": hsh(ea["mask"]) != hsh(eb["mask"]), "inliers_abs_diff": abs(ia - ib),
                                                      "decision_changed": (ia >= MIN_PAIR) != (ib >= MIN_PAIR), "inliers_a": ia, "inliers_b": ib,
                                                      **({"max_abs": float(np.abs(ea["E"] - eb["E"]).max())} if ea["E"] is not None and eb["E"] is not None and ea["E"].shape == eb["E"].shape else {})}, f"pair{k}")
            fa, fb = A("fundamental", p1, p2), B("fundamental", p1, p2)
            agg.add("findFundamentalMat_magsac_fixed", {"identical": hsh(fa["mask"]) == hsh(fb["mask"]) and hsh(fa["F"]) == hsh(fb["F"]),
                                                        "set_changed": hsh(fa["mask"]) != hsh(fb["mask"]),
                                                        "inliers_abs_diff": abs(int(0 if fa["mask"] is None else fa["mask"].sum()) - int(0 if fb["mask"] is None else fb["mask"].sum()))}, f"pair{k}")
    A.close()
    B.close()
    out = {"a": a.a, "b": a.b, "info": info, "images": len(imgs), "seconds": round(time.time() - t0, 1), "ops": agg.summary()}
    Path(a.out).write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    Path(a.out).with_suffix(".hashes.json").write_text(json.dumps(hashes))
    for op, s in out["ops"].items():
        print(f"{op:40s} identical {s['identical']}/{s['n']}  max_abs {s.get('max_abs')}  decision_changed {s.get('decision_changed')}  set_changed {s.get('set_changed')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
