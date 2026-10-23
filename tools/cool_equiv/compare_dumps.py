"""Compare two per-stage dumps from bisect_take.py and name the first diverging stage.

Each stage is classed as
  identical   bit for bit
  noise       same discrete content (ids, counts, index sets) and continuous
              values within float noise (below the stage's tolerance)
  diverged    a discrete decision differs or values exceed the tolerance

Usage: python compare_dumps.py A.pkl B.pkl [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

TOL = {"board_px": 1e-3, "kpt_px": 1e-3, "desc": 1e-4, "K_rel": 1e-6}


def load(p):
    return pickle.loads(Path(p).read_bytes())


def nn_match(a: np.ndarray, b: np.ndarray, r: float = 0.05):
    """Mutual nearest neighbours within r pixels: index pairs and distances."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((0, 2), int), np.zeros(0)
    from scipy.spatial import cKDTree
    ta, tb = cKDTree(a), cKDTree(b)
    dab, iab = tb.query(a)
    dba, iba = ta.query(b)
    ok = (dab <= r) & (iba[iab] == np.arange(len(a)))
    return np.c_[np.where(ok)[0], iab[ok]], dab[ok]


def cmp_board(A, B):
    fa, fb = A["board"], B["board"]
    frames = sorted(set(fa) | set(fb))
    det_diff, id_diff, maxd, rows = [], [], 0.0, []
    exact = True
    for i in frames:
        ca, ia = fa.get(i, (None, None))
        cb, ib = fb.get(i, (None, None))
        if (ca is None) != (cb is None):
            det_diff.append(i)
            exact = False
            continue
        if ca is None:
            continue
        if len(ia) != len(ib) or not np.array_equal(np.sort(ia), np.sort(ib)):
            id_diff.append({"frame": i, "a_only": sorted(set(ia.tolist()) - set(ib.tolist())),
                            "b_only": sorted(set(ib.tolist()) - set(ia.tolist()))})
            exact = False
            continue
        oa, ob = np.argsort(ia), np.argsort(ib)
        d = np.abs(ca[oa] - cb[ob]).max()
        if d > 0:
            exact = False
        maxd = max(maxd, float(d))
        if d > 0:
            rows.append({"frame": i, "max_px": float(d), "corners_differing": int((np.abs(ca[oa] - cb[ob]).max(1) > 0).sum()),
                         "corners": len(ia)})
    cls = "identical" if exact else ("noise" if not det_diff and not id_diff and maxd < TOL["board_px"] else "diverged")
    return {"class": cls, "detected_a": sum(1 for c, _ in fa.values() if c is not None),
            "detected_b": sum(1 for c, _ in fb.values() if c is not None), "detection_differs": det_diff,
            "id_differs": id_diff, "max_corner_px": maxd, "frames_with_corner_diffs": len(rows),
            "worst_frames": sorted(rows, key=lambda r: -r["max_px"])[:6]}


def cmp_calib(A, B):
    Ka, Kb = np.asarray(A["calib"]["K"]), np.asarray(B["calib"]["K"])
    da, db = np.asarray(A["calib"]["dist"]).ravel(), np.asarray(B["calib"]["dist"]).ravel()
    rel = float(np.abs(Ka - Kb).max() / np.abs(Ka).max())
    exact = np.array_equal(Ka, Kb) and np.array_equal(da, db)
    return {"class": "identical" if exact else ("noise" if rel < TOL["K_rel"] else "diverged"),
            "fx_a": float(Ka[0, 0]), "fx_b": float(Kb[0, 0]), "K_rel_diff": rel, "dist_max_diff": float(np.abs(da - db).max()),
            "rms_a": A["calib"].get("rms"), "rms_b": B["calib"].get("rms"), "method": [A["calib"].get("method"), B["calib"].get("method")]}


def cmp_work(A, B):
    ha = [w["hash"] for w in A["work"]]
    hb = [w["hash"] for w in B["work"]]
    diff = [i for i, (x, y) in enumerate(zip(ha, hb)) if x != y]
    return {"class": "identical" if not diff and len(ha) == len(hb) else "diverged", "frames_differing": diff[:48],
            "n_differing": len(diff)}


def cmp_aliked(A, B):
    per = []
    exact = True
    for i, (a, b) in enumerate(zip(A["aliked"], B["aliked"])):
        pa, pb = a["pts"], b["pts"]
        same = len(pa) == len(pb) and np.array_equal(pa, pb) and np.array_equal(a["desc"], b["desc"])
        if same:
            per.append({"frame": i, "n_a": len(pa), "n_b": len(pb), "identical": True})
            continue
        exact = False
        m, dist = nn_match(pa, pb)
        dd = float(np.abs(a["desc"][m[:, 0]] - b["desc"][m[:, 1]]).max()) if len(m) else None
        rd = float(np.abs(a["resp"][m[:, 0]] - b["resp"][m[:, 1]]).max()) if len(m) else None
        order_same = bool(len(m) == len(pa) == len(pb) and np.array_equal(m[:, 0], m[:, 1]))
        per.append({"frame": i, "n_a": len(pa), "n_b": len(pb), "matched": len(m), "only_a": len(pa) - len(m),
                    "only_b": len(pb) - len(m), "max_px": float(dist.max()) if len(dist) else None,
                    "median_px": float(np.median(dist)) if len(dist) else None, "max_desc": dd, "max_resp": rd,
                    "same_order": order_same, "identical": False})
    if exact:
        cls = "identical"
    else:
        noisy = all(p["identical"] or (p["only_a"] == 0 and p["only_b"] == 0 and p["max_px"] < TOL["kpt_px"]
                                        and p["max_desc"] < TOL["desc"]) for p in per)
        cls = "noise" if noisy else "diverged"
    diffs = [p for p in per if not p["identical"]]
    agg = {}
    if diffs:
        agg = {"frames_differing": len(diffs), "kpts_only_a": int(sum(p["only_a"] for p in diffs)),
               "kpts_only_b": int(sum(p["only_b"] for p in diffs)),
               "kpts_total_a": int(sum(p["n_a"] for p in per)),
               "max_px_matched": max(p["max_px"] or 0 for p in diffs),
               "median_px_matched": float(np.median([p["median_px"] for p in diffs if p["median_px"] is not None])),
               "max_desc_matched": max(p["max_desc"] or 0 for p in diffs),
               "max_resp_matched": max(p["max_resp"] or 0 for p in diffs)}
    return {"class": cls, **agg, "per_frame": per}


def coord_pairs(A, p, idx):
    i, j = p
    return np.hstack([A["aliked"][i]["pts"][idx[:, 0]], A["aliked"][j]["pts"][idx[:, 1]]])


def pair_set(A, p, idx):
    return {tuple(np.round(r, 3)) for r in coord_pairs(A, p, idx)} if len(idx) else set()


def cmp_pairs(A, B, key):
    pa, pb = A[key], B[key]
    keys = sorted(set(pa) | set(pb), key=lambda k: (k[0] if k[0] is not None else -1, k[1] if k[1] is not None else -1))
    rows, exact = [], True
    for k in keys:
        a, b = pa.get(k), pb.get(k)
        ia = a["idx"] if isinstance(a, dict) else (a if isinstance(a, np.ndarray) else None)
        ib = b["idx"] if isinstance(b, dict) else (b if isinstance(b, np.ndarray) else None)
        if ia is None or ib is None:
            if (ia is None) != (ib is None) or str(a) != str(b):
                exact = False
                rows.append({"pair": list(k), "a": None if ia is None else len(ia), "b": None if ib is None else len(ib),
                             "note": "missing or error on one side"})
            continue
        if len(ia) == len(ib) and np.array_equal(ia, ib):
            continue
        exact = False
        sa, sb = pair_set(A, k, ia), pair_set(B, k, ib)
        inter = len(sa & sb)
        rows.append({"pair": list(k), "a": len(ia), "b": len(ib), "common": inter,
                     "jaccard": round(inter / max(len(sa | sb), 1), 4)})
    return {"class": "identical" if exact else "diverged", "pairs": len(keys), "pairs_differing": len(rows),
            "total_a": int(sum(len(v["idx"]) if isinstance(v, dict) else len(v) for v in pa.values() if not isinstance(v, str))),
            "total_b": int(sum(len(v["idx"]) if isinstance(v, dict) else len(v) for v in pb.values() if not isinstance(v, str))),
            "keys_only_a": [list(k) for k in set(pa) - set(pb)], "keys_only_b": [list(k) for k in set(pb) - set(pa)],
            "worst": sorted([r for r in rows if "jaccard" in r], key=lambda r: r["jaccard"])[:8],
            "rows": rows[:80]}


def cmp_sfm(A, B):
    ra = [s["new"] for s in A["sfm_steps"] if s["op"] == "tri" and s["new"]]
    rb = [s["new"] for s in B["sfm_steps"] if s["op"] == "tri" and s["new"]]
    first = next((n for n, (x, y) in enumerate(zip(ra, rb)) if x != y), None)
    pa = [p["inliers"] for p in A["pnp"]]
    pb = [p["inliers"] for p in B["pnp"]]
    ba_a = [s for s in A["sfm_steps"] if s["op"] == "ba"]
    ba_b = [s for s in B["sfm_steps"] if s["op"] == "ba"]
    return {"class": "identical" if ra == rb and pa == pb and A["rec"]["tracks_hash"] == B["rec"]["tracks_hash"]
            and np.array_equal(A["rec"]["points"], B["rec"]["points"]) else "diverged",
            "init_pair": [ra[0] if ra else None, rb[0] if rb else None], "order_a": ra, "order_b": rb,
            "first_order_difference_step": first, "pnp_inliers_a": pa, "pnp_inliers_b": pb,
            "tracks": [A["rec"]["n_tracks"], B["rec"]["n_tracks"]], "points": [len(A["rec"]["points"]), len(B["rec"]["points"])],
            "registered": [len(A["rec"]["R"]), len(B["rec"]["R"])],
            "final_ba_rms_px": [ba_a[-1]["rms_px"] if ba_a else None, ba_b[-1]["rms_px"] if ba_b else None],
            "final_focal": [float(A["rec"]["K"][0, 0]), float(B["rec"]["K"][0, 0])]}


def cmp_scale(A, B):
    sa, sb = A["scale"], B["scale"]
    return {"class": "identical" if sa["metres_per_unit"] == sb["metres_per_unit"] and sa["scale_rel_err"] == sb["scale_rel_err"] else "diverged",
            "claim": [sa["scale_rel_err"], sb["scale_rel_err"]],
            "per_placement_a": sa["detail"].get("per_placement"), "per_placement_b": sb["detail"].get("per_placement"),
            "between_a": sa["detail"].get("between_placements_rel"), "between_b": sb["detail"].get("between_placements_rel"),
            "true_scale_err": [A["score"]["animal_scale_err"], B["score"]["animal_scale_err"]],
            "orbit_scale_err": [A["score"]["orbit_scale_err"], B["score"]["orbit_scale_err"]],
            "focal_err": [A["score"]["focal_err"], B["score"]["focal_err"]],
            "dense": [A.get("dense"), B.get("dense")]}


def compare(A, B) -> dict:
    out = {"env_a": A.get("env", {}).get("opencv"), "env_b": B.get("env", {}).get("opencv")}
    st = {}
    st["frames"] = {"class": "identical" if A["frames_hash"] == B["frames_hash"] else "diverged"}
    ga, gb = A["gray"], B["gray"]
    gd = [i for i in ga if ga[i] != gb.get(i)]
    st["gray"] = {"class": "identical" if not gd else "diverged", "frames_differing": gd}
    st["board"] = cmp_board(A, B)
    st["calib"] = cmp_calib(A, B)
    st["work"] = cmp_work(A, B)
    st["aliked"] = cmp_aliked(A, B)
    ka, kb = A["kpts"], B["kpts"]
    if all(len(x) == len(y) for x, y in zip(ka, kb)):
        md = max(float(np.abs(x - y).max()) if len(x) else 0.0 for x, y in zip(ka, kb))
        st["kpts_undistorted"] = {"class": "identical" if md == 0 else ("noise" if md < TOL["kpt_px"] else "diverged"), "max_px": md}
    else:
        st["kpts_undistorted"] = {"class": "diverged", "note": "keypoint counts differ"}
    st["lightglue"] = cmp_pairs(A, B, "lg")
    st["magsac_essential"] = cmp_pairs(A, B, "ess")
    st["pair_graph"] = cmp_pairs(A, B, "matches")
    fa, fb = A.get("focal", {}), B.get("focal", {})
    st["focal_pairs"] = {"class": "identical" if fa == fb else "diverged", "a": fa, "b": fb}
    st["sfm"] = cmp_sfm(A, B)
    st["scale"] = cmp_scale(A, B)
    out["stages"] = st
    order = ["frames", "gray", "board", "calib", "work", "aliked", "kpts_undistorted", "lightglue", "magsac_essential",
             "pair_graph", "focal_pairs", "sfm", "scale"]
    out["first_not_identical"] = next((s for s in order if st[s]["class"] != "identical"), None)
    out["first_diverged"] = next((s for s in order if st[s]["class"] == "diverged"), None)
    out["table"] = [[s, st[s]["class"]] for s in order]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--json")
    a = ap.parse_args()
    r = compare(load(a.a), load(a.b))
    r["a"], r["b"] = Path(a.a).name, Path(a.b).name
    txt = json.dumps(r, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    if a.json:
        Path(a.json).write_text(txt)
    print(f"{r['a']}  vs  {r['b']}")
    for s, c in r["table"]:
        print(f"  {s:18s} {c}")
    print("first not identical:", r["first_not_identical"], "| first diverged:", r["first_diverged"])
    sc = r["stages"]["scale"]
    print("claim", sc["claim"], "true", sc["true_scale_err"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
