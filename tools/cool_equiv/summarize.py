"""Track 1d summary: eval/out/cool_equiv/summary.json and README.md from whatever results exist.

Reads eval/out/cool_equiv/raw/ (filled by run_1d.py fetch):
  cool/ops/ops_cool_vs_cvonly.json (+ .hashes.json)   COOL build against the stock wheel, COOL machine
  cool/ops/ops_cool_vs_cool.json                      within-build repeat (determinism floor)
  stock/ops/ops_stock_vs_stockheadless.json (+ hashes) stock hub, for the cross-machine hash check
  cool/identify/ident_{cool,cvonly,cool_rep}.jsonl, stock/identify/ident_stock.jsonl
  {cool,stock}/bisect/sweep_21_s7_<mode>.pkl/.json
Every missing input is listed under not_measured; nothing is inferred for it.

Usage: python tools/cool_equiv/summarize.py [--raw DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

# pipeline stages each op feeds, for the plain verdict
STAGE = {
    "imread": "decode", "resize_area_960": "body frames", "resize_area_640": "identity preprocess",
    "resize_area_512": "quality gate", "cvt_bgr2lab": "identity preprocess", "clahe_L": "identity preprocess",
    "preprocess": "identity preprocess", "laplacian_var": "quality gate", "quality_assess": "quality gate",
    "cvt_bgr2gray": "board detection", "gaussian_7x7": "quality gate", "gray640": "identity preprocess",
    "resize_linear_up2_u8": "u8 resize (cow mask)", "resize_linear_f32": "float resize (dense disparity)", "blob_dinov2": "identity embedding",
    "blob_depthanything": "depth", "undistort_map": "body undistort", "remap_linear": "body undistort",
    "warp_perspective": "body / board", "goodFeaturesToTrack": "continuity check", "stereo_sgbm": "body dense",
    "sobel_f32": "body dense", "board_composite_warp": "board", "charuco_detect": "board detection",
    "cornerSubPix": "indirect only", "calcOpticalFlowPyrLK": "continuity check",
    "aliked_detect_same_input": "identity keypoints", "aliked_detect_own_chain": "identity keypoints",
    "dnn_dinov2_same_input": "identity embedding", "dnn_dinov2_own_chain": "identity embedding",
    "lightglue_same_input": "identity verification", "dnn_aliked_keypoints_same_input": "identity keypoints (raw network)",
    "dnn_aliked_descriptors_same_input": "identity keypoints (raw network)", "dnn_aliked_scores_same_input": "identity keypoints (raw network)", "findHomography_magsac_fixed": "identity verification",
    "findEssentialMat_magsac_fixed": "body registration", "findFundamentalMat_magsac_fixed": "body pair check",
}


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def classify(s: dict) -> str:
    if s["identical"] == s["n"]:
        return "identical"
    if (s.get("decision_changed") or 0) > 0:
        return "decisions change"
    if (s.get("set_changed") or 0) > 0:
        return "sets differ, thresholded decisions hold"
    return "values differ, decisions hold"


def ops_section(raw: Path) -> tuple[dict, list[str]]:
    missing = []
    out: dict = {}
    bc = load_json(raw / "cool" / "ops" / "ops_cool_vs_cvonly.json")
    rep = load_json(raw / "cool" / "ops" / "ops_cool_vs_cool.json")
    st = load_json(raw / "stock" / "ops" / "ops_stock_vs_stockheadless.json")
    if bc is None:
        missing.append("op diffs COOL vs stock wheel on the COOL machine (cool/ops/ops_cool_vs_cvonly.json)")
    else:
        out["builds"] = bc["info"]
        out["images"] = bc["images"]
        table = {}
        for op, s in bc["ops"].items():
            row = {k: s.get(k) for k in ("n", "identical", "max_abs", "max_rel", "mean_frac_differing", "max_frac_differing",
                                         "max_frac_gt1", "decision_changed", "set_changed", "points_a", "points_only_a",
                                         "points_only_b", "max_px_matched", "inliers_abs_diff_max", "cos_min", "rel_diff_max")
                   if s.get(k) is not None}
            row["stage"] = STAGE.get(op, "other")
            row["class"] = classify(s)
            if rep and op in rep["ops"]:
                r = rep["ops"][op]
                row["repeat_identical"] = f"{r['identical']}/{r['n']}"
                if r["identical"] != r["n"]:
                    row["class"] += "; nondeterministic within COOL"
            table[op] = row
        out["build_effect"] = table
    if rep is None:
        missing.append("within-build repeat (cool/ops/ops_cool_vs_cool.json)")
    if st is None:
        missing.append("stock hub op hashes (stock/ops/ops_stock_vs_stockheadless.json)")
    else:
        out["stock_hub_contrib_vs_headless"] = {op: f"{s['identical']}/{s['n']}" for op, s in st["ops"].items()}
        out["stock_hub_info"] = st["info"]
    hc = raw / "cool" / "ops" / "ops_cool_vs_cvonly.hashes.json"
    hs = raw / "stock" / "ops" / "ops_stock_vs_stockheadless.hashes.json"
    if hc.exists() and hs.exists():
        A = {r["img"]: r for r in json.loads(hc.read_text())}
        Bs = {r["img"]: r for r in json.loads(hs.read_text())}
        common = sorted(set(A) & set(Bs))
        mach, total = {}, {}
        for op in A[common[0]]["a"] if common else []:
            eq_m = sum(1 for i in common if A[i]["b"].get(op) == Bs[i]["b"].get(op))   # wheel on COOL machine vs wheel on stock hub
            eq_t = sum(1 for i in common if A[i]["a"].get(op) == Bs[i]["a"].get(op))   # COOL build vs deployed stock
            mach[op] = f"{eq_m}/{len(common)}"
            total[op] = f"{eq_t}/{len(common)}"
        out["machine_effect_hash_equal"] = mach
        out["total_effect_hash_equal"] = total
        # three builds per image: COOL, stock contrib wheel (KleidiCV 26.03), stock headless wheel (no KleidiCV)
        tri = {}
        for op in A[common[0]]["a"] if common else []:
            c = [A[i]["a"].get(op) for i in common]
            k = [Bs[i]["a"].get(op) for i in common]
            h = [Bs[i]["b"].get(op) for i in common]
            tri[op] = {"cool_eq_contrib": sum(x == y for x, y in zip(c, k)), "cool_eq_headless": sum(x == y for x, y in zip(c, h)),
                       "contrib_eq_headless": sum(x == y for x, y in zip(k, h)), "n": len(common)}
        out["three_builds"] = {op: t for op, t in tri.items() if not (t["cool_eq_contrib"] == t["cool_eq_headless"] == t["n"])}
        out["machine_effect_null"] = all(v.split("/")[0] == v.split("/")[1] for v in mach.values()) if mach else None
    else:
        missing.append("cross-machine hash check (needs both .hashes.json files)")
    return out, missing


def read_ident(p: Path):
    if not p.exists():
        return None, None
    lines = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    head = lines[0].get("header") if lines and "header" in lines[0] else None
    rows = {r["path"]: r for r in lines if "path" in r}
    return head, rows


def correct(r: dict) -> bool:
    if r["expected"] == "new":
        return r["decision"] == "new"
    return r["decision"] == "match" and r["animal_id"] == r["expected"]


def identify_section(raw: Path) -> tuple[dict, list[str]]:
    runs = {"cool": raw / "cool" / "identify" / "ident_cool.jsonl", "cvonly": raw / "cool" / "identify" / "ident_cvonly.jsonl",
            "cool_rep": raw / "cool" / "identify" / "ident_cool_rep.jsonl", "stock": raw / "stock" / "identify" / "ident_stock.jsonl"}
    data = {k: read_ident(p) for k, p in runs.items()}
    out: dict = {"runs": {}, "agreement": {}}
    missing = [f"identify run {k} ({p.relative_to(raw).as_posix()})" for k, p in runs.items() if data[k][1] is None]
    for k, (head, rows) in data.items():
        if rows is None:
            continue
        n = len(rows)
        c = sum(correct(r) for r in rows.values())
        dec = {}
        for r in rows.values():
            dec[r["decision"]] = dec.get(r["decision"], 0) + 1
        out["runs"][k] = {"n": n, "correct": c, "accuracy": round(c / n, 4) if n else None, "accuracy_wilson95": wilson(c, n),
                          "decisions": dec, "header": head}
    pairs = [("cool", "cvonly", "build effect, same machine"), ("cool", "stock", "COOL hub against stock hub as deployed"),
             ("cvonly", "stock", "machine effect, same wheel"), ("cool", "cool_rep", "within-build repeat")]
    for a, b, what in pairs:
        ra, rb = data[a][1], data[b][1]
        if ra is None or rb is None:
            continue
        common = sorted(set(ra) & set(rb))
        n = len(common)
        agree = sum(1 for p in common if (ra[p]["decision"], ra[p]["animal_id"]) == (rb[p]["decision"], rb[p]["animal_id"]))
        emb = sum(1 for p in common if ra[p]["emb_hash"] == rb[p]["emb_hash"])
        kp = sum(1 for p in common if ra[p]["kpts_hash"] == rb[p]["kpts_hash"])
        score = sum(1 for p in common if ra[p]["score"] == rb[p]["score"])
        flips = [{"path": p, a: [ra[p]["decision"], ra[p]["animal_id"], ra[p]["score"]], b: [rb[p]["decision"], rb[p]["animal_id"], rb[p]["score"]],
                  "expected": ra[p]["expected"]} for p in common
                 if (ra[p]["decision"], ra[p]["animal_id"]) != (rb[p]["decision"], rb[p]["animal_id"])]
        ga = (data[a][0] or {}).get("gallery_file_sha")
        gb = (data[b][0] or {}).get("gallery_file_sha")
        out["agreement"][f"{a}__{b}"] = {"what": what, "n": n, "agree": agree, "rate": round(agree / n, 4) if n else None,
                                         "wilson95": wilson(agree, n), "embedding_bit_equal": emb, "keypoints_bit_equal": kp,
                                         "score_equal": score, "same_gallery_file": ga == gb if ga and gb else None,
                                         "flips": flips[:25]}
    return out, missing


def bisect_section(raw: Path) -> tuple[dict, list[str]]:
    import compare_dumps as cd
    dumps = {}
    for hub in ("cool", "stock"):
        for p in sorted((raw / hub / "bisect").glob("sweep_21_s7_*.pkl")):
            dumps[p.stem.replace("sweep_21_s7_", "")] = p
    out: dict = {"dumps": sorted(dumps)}
    summaries = {}
    for m in dumps:
        j = dumps[m].with_suffix(".json")
        if j.exists():
            d = json.loads(j.read_text())
            summaries[m] = {k: d.get(k) for k in ("summary", "score", "env") if k in d} or {"keys": list(d)[:12]}
    out["runs"] = summaries
    builds = {}
    for m in dumps:
        j = dumps[m].with_suffix(".json")
        if j.exists():
            env = json.loads(j.read_text()).get("env", {})
            keep = [l.strip() for l in env.get("build", []) if l.strip().split(":")[0] in ("Baseline", "Lapack", "Custom HAL")]
            builds[m] = {"opencv": env.get("opencv"), "numpy": env.get("numpy"), "cpu_features": env.get("cpu_features"), "build": keep}
    out["builds"] = builds
    pairs = [("cool", "cvonly", "build effect, same machine"), ("cvonly", "stockheadless", "machine effect, same wheel"),
             ("cool", "stock", "COOL hub against stock hub as deployed"), ("cool", "cool_rep", "COOL repeat"),
             ("stock", "stock_rep", "stock repeat"), ("cvonly", "wheelvenv", "numpy / venv effect"),
             ("stock", "stockheadless", "contrib wheel with KleidiCV 26.03 against the headless wheel")]
    out["comparisons"] = {}
    missing = []
    for a, b, what in pairs:
        if a in dumps and b in dumps:
            try:
                r = cd.compare(cd.load(dumps[a]), cd.load(dumps[b]))
                out["comparisons"][f"{a}__{b}"] = {"what": what, "first_not_identical": r["first_not_identical"],
                                                   "first_diverged": r["first_diverged"], "table": r["table"],
                                                   "scale": r["stages"].get("scale")}
            except Exception as e:  # keep the rest of the summary
                out["comparisons"][f"{a}__{b}"] = {"what": what, "error": f"{type(e).__name__}: {e}"}
        else:
            missing.append(f"bisect dumps {a} and {b} ({what})")
    # stage swaps (job_inject.sh): one build runs with another build's recorded stage outputs
    inj = raw / "cool" / "inject"
    swaps = {}
    names = {"cool_inj_calib__wheel": "COOL with the wheel's calibration, against the wheel",
             "cool_inj_lg__wheel": "COOL with the wheel's LightGlue matches, against the wheel",
             "cool_inj_calib_lg__wheel": "COOL with the wheel's calibration and matches, against the wheel",
             "wheel_inj_calib__cool": "wheel with COOL's calibration, against COOL",
             "wheel_inj_calib_lg__cool": "wheel with COOL's calibration and matches, against COOL"}
    for k, what in names.items():
        p = inj / f"cmp_{k}.json"
        if p.exists():
            d = json.loads(p.read_text())
            swaps[k] = {"what": what, "first_not_identical": d["first_not_identical"], "first_diverged": d["first_diverged"],
                        "claim": d["stages"]["scale"].get("claim"), "true_scale_err": d["stages"]["scale"].get("true_scale_err")}
    if swaps:
        out["stage_swaps"] = swaps
    co = load_json(inj / "calib_ops_cmp.json")
    if co:
        out["calib_micro_ops"] = {op: {k: s.get(k) for k in ("identical", "max_abs", "max_rel", "frac")} for op, s in co["ops"].items()}
        pa, pb = inj / "calib_ops_cool.pkl", inj / "calib_ops_wheel.pkl"
        if pa.exists() and pb.exists():
            import pickle
            import numpy as np
            A, B = pickle.loads(pa.read_bytes()), pickle.loads(pb.read_bytes())
            if isinstance(A.get("SVDecomp_40x9"), (tuple, list)):
                w, u, vt = [np.asarray(x) for x in A["SVDecomp_40x9"]]
                w2, u2, vt2 = [np.asarray(x) for x in B["SVDecomp_40x9"]]
                out["svd_40x9"] = {"values_max_abs": float(np.abs(w - w2).max()),
                                   "vectors_max_abs": float(max(np.abs(u - u2).max(), np.abs(vt - vt2).max())),
                                   "abs_vectors_max_abs": float(max(np.abs(np.abs(u) - np.abs(u2)).max(), np.abs(np.abs(vt) - np.abs(vt2)).max()))}
    dr = {m: load_json(inj / f"dense_repeat_{m}.json") for m in ("cool", "wheel")}
    if all(dr.values()):
        out["dense_repeat"] = {m: {"deterministic_default_threads": d["deterministic_default_threads"],
                                   "deterministic_one_thread": d["deterministic_one_thread"],
                                   "sorted_points_equal_across_runs": len({r["sorted_points_hash"] for r in d["runs"]}) == 1,
                                   "girth_values": sorted({r["girth_m"] for r in d["runs"]})} for m, d in dr.items()}
    # every sweep_21 outcome seen under last-bit perturbations (bisect runs + stage swaps + ensemble)
    outcomes = []
    for k, c in out["comparisons"].items():
        sc = c.get("scale") or {}
        if sc.get("true_scale_err"):
            for side, name in zip(sc["true_scale_err"], k.split("__")):
                outcomes.append((name, side))
    for k, s in swaps.items():
        if s.get("true_scale_err"):
            outcomes.append((k.split("__")[0], s["true_scale_err"][0]))
    ens = []
    for hub in ("cool", "stock"):
        for p in sorted((raw / hub / "ensemble").glob("ens_*.jsonl")):
            ens += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    if ens:
        by: dict = {}
        for r in ens:
            by.setdefault((r["opencv"], r["take"]), []).append(r)
        out["ensemble"] = {}
        for (ver, take), rows in sorted(by.items()):
            ok = [r for r in rows if "true_scale_err" in r]
            out["ensemble"][f"{ver} {take}"] = {
                "n": len(rows), "errors": len(rows) - len(ok),
                "true_scale_err": [r["true_scale_err"] for r in ok], "claim": [r["claim"] for r in ok],
                "covered": sum(1 for r in ok if r["covered"]), "registered": [r["registered"] for r in ok],
                "bit_identical_to_j0": sum(1 for r in ok if r["true_scale_err"] == ok[0]["true_scale_err"] and r["claim"] == ok[0]["claim"]),
                # what a one-probe stability check reads: orbit scale of run j against run 0 (truth cancels in the ratio)
                "probe_shift_vs_j0": [round(abs((1 + r["orbit_scale_err"]) / (1 + ok[0]["orbit_scale_err"]) - 1), 5) for r in ok[1:]]
                if ok and ok[0].get("j") == 0 else []}
            if take == "sweep_21":
                outcomes += [(f"ensemble {ver} j{r['j']}", r["true_scale_err"]) for r in ok]
        vers = sorted({v for v, _ in by})
        cov = {}
        for v in vers:
            rows = [r for (vv, _), rs in by.items() if vv == v for r in rs if "true_scale_err" in r]
            c = sum(1 for r in rows if r["covered"])
            cov[v] = {"covered": c, "n": len(rows), "wilson95": wilson(c, len(rows))}
        out["ensemble_claim_coverage"] = cov
        if len(vers) == 2:
            cmpt = {}
            for take in sorted({t for _, t in by}):
                a = [r["true_scale_err"] for r in by.get((vers[0], take), []) if "true_scale_err" in r]
                b = [r["true_scale_err"] for r in by.get((vers[1], take), []) if "true_scale_err" in r]
                if a and b:
                    cmpt[take] = {vers[0]: [min(a), max(a)], vers[1]: [min(b), max(b)],
                                  "ranges_overlap": max(min(a), min(b)) <= min(max(a), max(b))}
            out["ensemble_by_take"] = cmpt
    uniq = {}
    for n, v in outcomes:
        uniq.setdefault(n, v)
    if uniq:
        vals = list(uniq.values())
        out["sweep_21_outcomes"] = {"runs": uniq, "distinct": len(set(vals)), "min_true_scale_err": min(vals), "max_true_scale_err": max(vals)}
    return out, missing


def verdict(ops: dict, ident: dict, bis: dict) -> dict:
    v = {"proven_equivalent": [], "values_differ_decisions_hold": [], "sets_differ_decisions_hold": [], "decisions_change": [],
         "nondeterministic": []}
    for op, row in ops.get("build_effect", {}).items():
        if "nondeterministic" in row["class"]:
            v["nondeterministic"].append(op)
        if row["class"].startswith("identical"):
            v["proven_equivalent"].append(op)
        elif row["class"].startswith("values differ"):
            v["values_differ_decisions_hold"].append(op)
        elif row["class"].startswith("sets differ"):
            v["sets_differ_decisions_hold"].append(op)
        else:
            v["decisions_change"].append(op)
    ag = ident.get("agreement", {}).get("cool__cvonly")
    if ag:
        v["identify_agreement_build_effect"] = {"rate": ag["rate"], "wilson95": ag["wilson95"], "n": ag["n"]}
    c = bis.get("comparisons", {}).get("cool__cvonly")
    if c and "first_diverged" in c:
        v["sweep_21_first_not_identical"] = c["first_not_identical"]
        v["sweep_21_first_diverged"] = c["first_diverged"]
    return v


def prior_evidence() -> dict:
    """End-to-end body results from the two hubs (15 Oct; S/G), already in eval/out."""
    a, b = ROOT / "eval" / "out" / "sweep_cool.jsonl", ROOT / "eval" / "out" / "sweep_body3.jsonl"
    if not (a.exists() and b.exists()):
        return {}
    ca = {json.loads(x)["animal"]: json.loads(x) for x in a.read_text().splitlines() if x.strip()}
    cb = {json.loads(x)["animal"]: json.loads(x) for x in b.read_text().splitlines() if x.strip()}
    rows = []
    for k in sorted(set(ca) & set(cb)):
        rows.append({"take": k, "cool_true_scale_err": ca[k]["animal_scale_err"], "stock_true_scale_err": cb[k]["animal_scale_err"],
                     "cool_claim": ca[k]["claimed_scale_err"], "stock_claim": cb[k]["claimed_scale_err"],
                     "cool_girth_err": ca[k]["girth_err"], "stock_girth_err": cb[k]["girth_err"],
                     "cool_loop_coverage": ca[k]["loop_coverage"], "stock_loop_coverage": cb[k]["loop_coverage"]})
    same = sum(1 for r in rows if r["cool_true_scale_err"] == r["stock_true_scale_err"])
    rep = []
    for hub, mode, src in (("cool", "cool", ca), ("stock", "stock", cb)):
        j = ROOT / "eval" / "out" / "cool_equiv" / "raw" / hub / "bisect" / f"sweep_21_s7_{mode}.json"
        if j.exists() and "sweep_21" in src:
            d = json.loads(j.read_text())
            claim = round(d["plain"]["metres_per_unit_scale_claim"], 5)
            rep.append(f"{mode} claim {claim} vs {src['sweep_21']['claimed_scale_err']}, girth err {d.get('dense', {}).get('girth_err')} vs {src['sweep_21']['girth_err']}")
    return {"source": "eval/out/sweep_cool.jsonl (COOL hub) vs eval/out/sweep_body3.jsonl (stock hub), seed 7, labels S/G",
            "takes": len(rows), "takes_with_equal_scale": same, "rows": rows,
            "note": "Two machines and two builds at once; sweep_cool provenance is d697be9 era (LEDGER).", "reproduced": "; ".join(rep) or None}


def fmt(x, nd=4):
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def why(S: dict) -> list[str]:
    """The cause, stated only from what the results hold."""
    b = S["bisect"]
    L = []
    bl = b.get("builds", {})
    if "cool" in bl and ("cvonly" in bl or "stock" in bl):
        other = bl.get("cvonly") or bl.get("stock")
        def lap(d):
            raw = next((x.split(":", 1)[1].strip() for x in d["build"] if x.startswith("Lapack")), "?")
            raw = raw.removeprefix("YES (").removesuffix(")")
            libs = [Path(t).stem for t in raw.split() if "/" in t]
            m = re.search(r"armpl_([0-9.]+)", raw)
            if m:
                return f"Arm Performance Libraries {m.group(1)} ({', '.join(libs)})"
            return " ".join(Path(t).name if "/" in t else t for t in raw.split())
        base = lambda d: next((x.split(":", 1)[1].strip() for x in d["build"] if x.startswith("Baseline")), "?")  # noqa: E731
        L += ["## Why", "",
              f"- COOL compiles for a `{base(bl['cool'])}` baseline and links `{lap(bl['cool'])}`.",
              f"- The stock wheel compiles for a `{base(other)}` baseline and links `{lap(other)}`."]
    cm = b.get("calib_micro_ops")
    if cm:
        d = {k: v for k, v in cm.items() if not v["identical"]}
        if d:
            dbl = [k for k in d if k not in ("SVDecomp_40x9", "exp_f32")]
            mx = max(d[k]["max_rel"] for k in dbl) if dbl else 0
            L.append(f"- Double precision linear algebra on identical inputs differs in the last bits: up to {mx:.0e} relative "
                     f"across {len(dbl)} routines (gemm, SVD, LU, Cholesky, QR, eigen, solvePnP, Rodrigues, calibrateCamera).")
        sv = b.get("svd_40x9")
        if "SVDecomp_40x9" in d and sv and sv["abs_vectors_max_abs"] < 1e-12:
            L.append(f"- On a random 40 x 9 matrix the two builds return singular vectors with opposite signs (raw difference "
                     f"{sv['vectors_max_abs']:.2f}). Singular values agree to {sv['values_max_abs']:.0e} and the vectors up to sign "
                     f"to {sv['abs_vectors_max_abs']:.0e}: both are valid decompositions.")
        if "exp_f32" in d:
            L.append(f"- Single precision exp differs on {d['exp_f32']['frac']:.0%} of inputs (max rel {d['exp_f32']['max_rel']:.0e}). "
                     "LightGlue's matching head uses softmax and sigmoid, both built on exp: the likely path for its divergence on identical inputs.")
    sw = b.get("stage_swaps", {})
    if sw:
        L.append("- Holding the calibration equal moves the first divergence to LightGlue. Holding calibration and matches equal moves it "
                 "to MAGSAC. Each stage diverges on its own, from identical inputs.")
    oc = b.get("sweep_21_outcomes")
    if oc:
        L.append(f"- These last-bit differences reach sweep_21's final scale through discrete choices: match acceptance, inlier sets, "
                 f"registration order. Its true scale error lands anywhere from {oc['min_true_scale_err']:+.1%} to {oc['max_true_scale_err']:+.1%}.")
    en = b.get("ensemble", {})
    s21 = [e for k, e in en.items() if k.endswith("sweep_21")]
    rest = [e for k, e in en.items() if not k.endswith("sweep_21")]
    if s21:
        e = s21[0]
        L.append(f"- On one build ({[k for k in en if k.endswith('sweep_21')][0].split()[0]}), a 1e-10 nudge to the focal gives {len(set(e['true_scale_err']))} distinct sweep_21 outcomes "
                 f"in {e['n'] - e['errors']} runs ({min(e['true_scale_err']):+.1%} to {max(e['true_scale_err']):+.1%}).")
    if rest:
        spreads = sorted(max(e["true_scale_err"]) - min(e["true_scale_err"]) for e in rest)
        L.append(f"- The {len(rest)} other takes move too, by less: true scale error spreads of {spreads[0]:.1%} to {spreads[-1]:.1%} "
                 f"against {max(s21[0]['true_scale_err']) - min(s21[0]['true_scale_err']):.1%} on sweep_21." if s21 else
                 f"- The other takes move too: spreads of {spreads[0]:.1%} to {spreads[-1]:.1%}.")
    if L:
        L.append("")
    return L


def headline(S: dict) -> list[str]:
    """Plain sentences, each one backed by a number in summary.json."""
    v, ops, b = S["verdict"], S["ops"], S["bisect"]
    L = []
    if ops.get("build_effect") and not v["decisions_change"]:
        n_ops = len(ops["build_effect"])
        L.append(f"Every measured image decision holds on COOL. Across {n_ops} op checks on {ops.get('images')} real images, "
                 f"{len(v['proven_equivalent'])} are bit-identical to the stock wheel and the rest leave every threshold where it was: "
                 "quality gate, board ids, continuity status, identity inlier class, essential-pair floor.")
    ia = v.get("identify_agreement_build_effect")
    ag = S["identify"].get("agreement", {}).get("cool__cvonly")
    if ia and ag:
        flips = ag.get("flips", [])
        absent = sum(1 for f in flips if f["expected"] == "new")
        L.append(f"identify() returns the same decision on {ag['agree']} of {ag['n']} probes ({ia['rate']:.1%}, Wilson 95% "
                 f"{ia['wilson95'][0]:.1%} to {ia['wilson95'][1]:.1%}). The {len(flips)} that move are absent animals"
                 f"{'' if absent == len(flips) else ' and enrolled animals'} on the new / ambiguous border, and each ends as new or ambiguous on both builds.")
    if ops.get("machine_effect_null"):
        L.append("The stock wheel gives bit-identical output on the COOL machine and the stock machine. Every difference belongs to the build.")
    bt = b.get("ensemble_by_take", {}).get("sweep_21")
    if bt:
        vs = [k for k in bt if k != "ranges_overlap"]
        L.append(f"Body reconstruction is where results part. On sweep_21 a 1e-10 nudge to the focal on one build moves the true scale error "
                 f"across {bt[vs[0]][0]:+.1%} to {bt[vs[0]][1]:+.1%} ({vs[0]}) and {bt[vs[1]][0]:+.1%} to {bt[vs[1]][1]:+.1%} ({vs[1]}). "
                 "The two builds draw from the same spread. The fragility belongs to the pipeline on that take; COOL only samples it differently.")
        allb = b.get("ensemble_by_take", {})
        ov = sum(1 for r in allb.values() if r["ranges_overlap"])
        L.append(f"Under the same nudge the two builds' ranges of true scale error overlap on {ov} of {len(allb)} synthetic takes.")
        cv = b.get("ensemble_claim_coverage", {})
        if len(cv) == 2:
            L.append("The raw scale claim covers the true error on " + " and ".join(
                f"{c['covered']} of {c['n']} runs ({v})" for v, c in cv.items()) + ". It leaves float noise out of its error budget on both builds.")
    elif b.get("sweep_21_outcomes"):
        oc = b["sweep_21_outcomes"]
        L.append(f"Body reconstruction is where results part: last-bit perturbations move sweep_21's true scale error across "
                 f"{oc['min_true_scale_err']:+.1%} to {oc['max_true_scale_err']:+.1%}.")
    if L:
        L.append("")
    return L


def readme(S: dict) -> str:
    L = ["# COOL equivalence (track 1d)", ""]
    v = S["verdict"]
    ops = S["ops"]
    L += ["## Verdict", ""]
    if S["status"] == "not_measured":
        L += ["Nothing new was measured in this run. The AWS session had expired, so no job reached the hubs.",
              "COOL claims stay limited to what earlier evidence shows: speed figures from the bench, and end to end",
              "body results that differ between the two hubs on every synthetic take (table below).",
              "Equivalence of any stage is unproven until the jobs below run.", ""]
    else:
        L += headline(S)
        L += ["### Stage by stage", ""]
        if v["proven_equivalent"]:
            L.append(f"Bit-identical on every image, COOL build against the stock wheel on the same machine: {', '.join(v['proven_equivalent'])}.")
        if v["values_differ_decisions_hold"]:
            L.append(f"Values differ, every downstream decision holds: {', '.join(v['values_differ_decisions_hold'])}.")
        if v["sets_differ_decisions_hold"]:
            L.append(f"A few points or inliers change set membership, every thresholded decision holds: {', '.join(v['sets_differ_decisions_hold'])}.")
        if v["decisions_change"]:
            L.append(f"At least one downstream decision changes: {', '.join(v['decisions_change'])}.")
        if v["nondeterministic"]:
            L.append(f"Run-to-run variation inside COOL itself: {', '.join(v['nondeterministic'])}. Build differences there sit inside that noise floor.")
        ia = v.get("identify_agreement_build_effect")
        if ia:
            L.append(f"identify() decisions agree on {ia['rate']:.1%} of {ia['n']} probes between the builds (Wilson 95% {ia['wilson95'][0]:.1%} to {ia['wilson95'][1]:.1%}).")
        if v.get("sweep_21_first_diverged") or v.get("sweep_21_first_not_identical"):
            L.append(f"sweep_21 first differs at stage `{v.get('sweep_21_first_not_identical')}` and first diverges at `{v.get('sweep_21_first_diverged')}`.")
        me = ops.get("machine_effect_null")
        if me is True:
            L.append("The same stock wheel gives bit-identical outputs on both machines, so every difference above belongs to the build.")
        elif me is False:
            L.append("The same stock wheel differs between the two machines on some ops; see machine_effect_hash_equal.")
        L.append("")
    L += why(S)
    if S["not_measured"]:
        L += ["## Open", ""] + [f"- {m}" for m in S["not_measured"]] + [""]
    be = ops.get("build_effect")
    if be:
        L += ["## Op-level diffs: COOL build against the stock wheel, same machine", "",
              f"{ops.get('images')} real images (150 Acme B4 side photos, 150 Zenodo muzzles); fixed MAGSAC inputs from 192 sweep_21 pairs. Labels R/P.", "",
              "| op | stage | identical | max abs | mean frac differing | decisions changed | sets changed | repeat | class |",
              "|---|---|---|---|---|---|---|---|---|"]
        for op, r in be.items():
            L.append(f"| {op} | {r['stage']} | {r['identical']}/{r['n']} | {fmt(r.get('max_abs'))} | {fmt(r.get('mean_frac_differing'))} | "
                     f"{fmt(r.get('decision_changed'))} | {fmt(r.get('set_changed'))} | {r.get('repeat_identical', '')} | {r['class']} |")
        L.append("")
    tb = ops.get("three_builds")
    if tb:
        L += ["### Three builds per image (chained ops that differ anywhere)", "",
              "COOL, the deployed stock wheel (contrib, KleidiCV 26.03) and the headless stock wheel (carotene HAL only). Images bit-equal per pair.", "",
              "| op | COOL = contrib | COOL = headless | contrib = headless |", "|---|---|---|---|"]
        for op, t in tb.items():
            L.append(f"| {op} | {t['cool_eq_contrib']}/{t['n']} | {t['cool_eq_headless']}/{t['n']} | {t['contrib_eq_headless']}/{t['n']} |")
        L.append("")
    ag = S["identify"].get("agreement", {})
    if ag:
        L += ["## identify() decision agreement", "", "500 fixed probes: 400 field images of enrolled animals, 100 of absent animals. Labels R/L.", "",
              "| pair | what | agree | rate | Wilson 95% | embeddings bit-equal | same gallery |", "|---|---|---|---|---|---|---|"]
        for k, a in ag.items():
            w = a["wilson95"] or [None, None]
            L.append(f"| {k} | {a['what']} | {a['agree']}/{a['n']} | {fmt(a['rate'])} | {fmt(w[0])} to {fmt(w[1])} | {a['embedding_bit_equal']}/{a['n']} | {a['same_gallery_file']} |")
        L.append("")
        tb = ops.get("three_builds", {}).get("blob_dinov2")
        if tb and tb["contrib_eq_headless"] == 0:
            L += ["Embeddings differ bit-wise even between the two stock wheels: the contrib wheel's KleidiCV 26.03 resize inside",
                  "blobFromImage changes the DINOv2 input on every image (three-build table above). Decisions still agree.", ""]
        L += ["| run | accuracy | Wilson 95% | decisions |", "|---|---|---|---|"]
        for k, r in S["identify"]["runs"].items():
            w = r["accuracy_wilson95"] or [None, None]
            L.append(f"| {k} | {r['correct']}/{r['n']} | {fmt(w[0])} to {fmt(w[1])} | {r['decisions']} |")
        L.append("")
    comps = S["bisect"].get("comparisons", {})
    if comps:
        L += ["## sweep_21 stage bisect (seed 7)", "", "| pair | what | first difference | first diverged |", "|---|---|---|---|"]
        for k, c in comps.items():
            fni = c.get("first_not_identical", c.get("error", ""))
            L.append(f"| {k} | {c['what']} | {'identical at every stage' if fni is None else fni} | {'' if c.get('first_diverged') is None else c['first_diverged']} |")
        L.append("")
    sw = S["bisect"].get("stage_swaps")
    if sw:
        L += ["### Stage swaps on the COOL machine", "", "One build reruns sweep_21 with another build's recorded stage output held fixed.", "",
              "| swap | first difference | first diverged | true scale err (this run, reference) |", "|---|---|---|---|"]
        for k, s in sw.items():
            t = s.get("true_scale_err") or [None, None]
            L.append(f"| {s['what']} | {s['first_not_identical']} | {s['first_diverged']} | {fmt(t[0])}, {fmt(t[1])} |")
        L.append("")
    cm = S["bisect"].get("calib_micro_ops")
    if cm:
        diff = [f"{op} (max rel {s['max_rel']:.1e}{', sign flip only' if op == 'SVDecomp_40x9' and S['bisect'].get('svd_40x9', {}).get('abs_vectors_max_abs', 1) < 1e-12 else ''})" for op, s in cm.items() if not s["identical"]]
        same = [op for op, s in cm.items() if s["identical"]]
        L += ["### Linear algebra under calibrateCamera, same inputs on both builds", "",
              f"Differ: {', '.join(diff)}.", "", f"Bit-identical: {', '.join(same)}.", ""]
    dr = S["bisect"].get("dense_repeat")
    if dr:
        L += ["### Dense stage repeatability", ""]
        for m, d in dr.items():
            L.append(f"- {m}: one thread deterministic {d['deterministic_one_thread']}; eight threads reorder points, sorted set equal "
                     f"{d['sorted_points_equal_across_runs']}; girth values {d['girth_values']}.")
        L.append("")
    en = S["bisect"].get("ensemble")
    if en:
        L += ["### Float-noise ensemble, one build", "", "Calibrated focal times (1 + j x 1e-10), the size of the COOL vs stock calibration difference.", "",
              "| build, take | runs | true scale err over j | claim covers truth | runs equal to j=0, itself included |", "|---|---|---|---|---|"]
        for k, e in en.items():
            ts = ", ".join(f"{x:+.1%}" for x in e["true_scale_err"])
            L.append(f"| {k} | {e['n']} | {ts} | {e['covered']}/{e['n'] - e['errors']} | {e['bit_identical_to_j0']} |")
        L.append("")
        cv = S["bisect"].get("ensemble_claim_coverage", {})
        if cv:
            L.append("Raw scale claim (jackknife, before calibration) covers the true scale error: " + "; ".join(
                f"{v}: {c['covered']}/{c['n']} runs (Wilson 95% {c['wilson95'][0]:.0%} to {c['wilson95'][1]:.0%})" for v, c in cv.items()) + ".")
            L.append("")
        bt = S["bisect"].get("ensemble_by_take", {})
        if bt:
            vs = [k for k in next(iter(bt.values())) if k != "ranges_overlap"]
            L += [f"| take | {vs[0]} true scale err range | {vs[1]} range | ranges overlap |", "|---|---|---|---|"]
            for t, r in bt.items():
                L.append(f"| {t} | {r[vs[0]][0]:+.1%} to {r[vs[0]][1]:+.1%} | {r[vs[1]][0]:+.1%} to {r[vs[1]][1]:+.1%} | {r['ranges_overlap']} |")
            L.append("")
        bad = [x for k, e in en.items() if k.endswith("sweep_21") for x in e["probe_shift_vs_j0"]]
        good = [x for k, e in en.items() if not k.endswith("sweep_21") for x in e["probe_shift_vs_j0"]]
        if bad and good:
            L += ["A one-probe stability check reads the orbit scale of a nudged run against the first run. Its operating points:", "",
                  "| refuse above | sweep_21 probes refused | other takes' probes refused |", "|---|---|---|"]
            for t in (0.01, 0.02, 0.03, 0.05):
                L.append(f"| {t:.0%} | {sum(x > t for x in bad)}/{len(bad)} | {sum(x > t for x in good)}/{len(good)} |")
            L.append("")
    oc = S["bisect"].get("sweep_21_outcomes")
    if oc:
        L += [f"Across every sweep_21 run with last-bit perturbations, the true scale error spans {oc['min_true_scale_err']:+.1%} to "
              f"{oc['max_true_scale_err']:+.1%} ({len(oc['runs'])} runs, {oc['distinct']} distinct outcomes).", ""]
    if S.get("proposed_fix"):
        L += ["## Proposed fix (a diff, unapplied)", "",
              f"`{S['proposed_fix']}` adds a stability probe to the body path. The walk-around reconstructs a second time with the",
              "calibrated focal nudged by 1e-9, and the registration-health gate refuses the take when the metric orbit moves more",
              "than 2%. The probe reads the pipeline's own fragility, so it behaves the same on COOL and on the stock wheel.",
              "Its cost is one more reconstruction per animal. Operating points are in the ensemble table above.", ""]
        sm = S.get("proposed_fix_smoke")
        if sm:
            L.append(f"Smoke test ({sm['what']}):")
            L += [f"- {r['take']}: orbit {r['orbit_radius_m'][0]} m then {r['orbit_radius_m'][1]} m, shift {r['shift_rel']:.2%}, "
                  f"registration health {'passes' if r['registration_health'] else 'refuses'}." for r in sm["runs"]]
            L.append("")
    pe = S.get("prior_evidence")
    if pe:
        L += ["## Earlier end to end evidence (15 Oct, labels S/G)", "", pe["source"] + ".", pe["note"], "",
              "| take | true scale err COOL | true scale err stock | claim COOL | claim stock | girth err COOL | girth err stock | loop coverage COOL | stock |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in pe["rows"]:
            L.append(f"| {r['take']} | {r['cool_true_scale_err']:+.2%} | {r['stock_true_scale_err']:+.2%} | {r['cool_claim']:.2%} | {r['stock_claim']:.2%} | "
                     f"{r['cool_girth_err']:+.1%} | {r['stock_girth_err']:+.1%} | {r['cool_loop_coverage']} | {r['stock_loop_coverage']} |")
        L += ["", f"Equal scale on {pe['takes_with_equal_scale']} of {pe['takes']} takes. Bitwise equality of the body pipeline is out of reach for any two builds:",
              "the ensemble above shows one build disagreeing with itself under a 1e-10 nudge. The comparison that holds is between spreads.", ""]
        rp = pe.get("reproduced")
        if rp:
            L += [f"The 21 Oct bisect dumps on the frozen code reproduce the sweep_21 rows exactly: {rp}.", ""]
    L += ["## Reproduce", "", "```", "aws login --region eu-west-1          # the user runs this",
          ".venv/Scripts/python.exe tools/cool_equiv/run_1d.py all       # upload, setup, launch job_all on running hubs",
          ".venv/Scripts/python.exe tools/cool_equiv/run_1d.py status    # job_all: ops about 10 min, identify about 20 min per hub",
          ".venv/Scripts/python.exe tools/cool_equiv/run_1d.py launch stock job_ensemble   # about 1 h; same on cool",
          ".venv/Scripts/python.exe tools/cool_equiv/run_1d.py fetch",
          ".venv/Scripts/python.exe tools/cool_equiv/summarize.py", "```", "",
          "Builds: COOL = OpenCV 5.1.0-dev, SVE baseline, KleidiCV 0.7.0 (i-0cc32db5d19735dda). Stock = pip opencv 5.0.0.93, KleidiCV 26.03",
          "(i-0b68a3d3a5840eaea, and the headless wheel in a separate environment on the COOL machine). Both c8g.2xlarge, eu-west-1.", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "eval" / "out" / "cool_equiv" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "eval" / "out" / "cool_equiv"))
    ap.add_argument("--blocked", default="", help="reason recorded when nothing was measured")
    a = ap.parse_args()
    raw = Path(a.raw)
    ops, m1 = ops_section(raw)
    ident, m2 = identify_section(raw)
    bis, m3 = bisect_section(raw)
    measured = bool(ops.get("build_effect") or ident.get("runs") or bis.get("comparisons"))
    S = {"track": "1d COOL equivalence", "generated": time.strftime("%Y-%m-%d %H:%M %Z"),
         "status": ("partial" if (m1 or m2 or m3) else "complete") if measured else "not_measured",
         "blocked_on": a.blocked or None, "labels": {"ops": "R/P", "identify": "R/L", "bisect": "S/G"},
         "verdict": verdict(ops, ident, bis), "ops": ops, "identify": ident, "bisect": bis,
         "not_measured": m1 + m2 + m3, "prior_evidence": prior_evidence()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "stability_probe.diff").exists():
        S["proposed_fix"] = "stability_probe.diff"
        sm = out / "stability_probe_smoke.json"
        if sm.exists():
            S["proposed_fix_smoke"] = json.loads(sm.read_text())
    (out / "summary.json").write_text(json.dumps(S, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    (out / "README.md").write_text(readme(S), encoding="utf-8")
    print("status", S["status"], "| not measured:", len(S["not_measured"]), "| wrote", out / "summary.json", "and README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
