"""Track 1a-v2 local harness: measure() v1 and v2 on the cached Ruchay clouds.

The cache (eval/out/ruchay/cache/{cdb,cdb2}/<id>.npz) holds the frozen track 1a
preprocessing: fused cameras, levelled, floor z = 0, metres, cropped. The data
carry no licence file; the cache and every per-animal row stay gitignored.

Protocol (eval/out/ruchay/v2_split.json): every v2 rule is chosen on dev
(CowDatabase 1-30). Test (CowDatabase 32+) and external (CowDatabase2) are
scored once with the frozen version. Tape for test and external is read only
by the `final` command.

Usage:
  python eval/ruchay_v2.py v1check            cache reproduces v1 on dev
  python eval/ruchay_v2.py dev [--tag T]      v2 on dev, prints the table
  python eval/ruchay_v2.py final              v1 and v2 on dev, test, external
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import ruchay_bench as rb  # noqa: E402

OUT = ROOT / "eval" / "out" / "ruchay"
CACHE = OUT / "cache"
SPLIT = json.loads((OUT / "v2_split.json").read_text())
FRONT = {"cdb": "+x", "cdb2": "-x"}


def load(db: str, key: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(CACHE / db / f"{key}.npz")
    Q = z["xyz"].astype(np.float64) + z["off"]
    return Q, z["view"].astype(int)


def _read_csv(path: Path) -> dict[int, dict]:
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[int(r["cdb_id"])] = r
    return out


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def tape(db: str, ids) -> dict[int, dict]:
    """Tape truth for the given ids only."""
    rows = _read_csv(OUT / ("per_animal.csv" if db == "cdb" else "cdb2/per_animal.csv"))
    out = {}
    for i in ids:
        r = rows[int(i)]
        out[int(i)] = {k: _num(r.get(f"tape_{k}")) for k in ("girth_cm", "obl_cm", "withers_cm", "weight_kg")}
    return out


def front_board(Q: np.ndarray, db: str) -> dict:
    """The same pseudo-board track 1a used: front centre from find_front with the setup label."""
    from mombe.body.measure import _largest_cluster
    p = dict(rb.PARAMS, front_source=FRONT[db])
    Pc = _largest_cluster(Q[Q[:, 2] > 0.05])
    fr = rb.find_front(Pc, p)
    return fr


def run_v1(Q: np.ndarray, view: np.ndarray, db: str) -> dict:
    p = dict(rb.PARAMS, front_source=FRONT[db])
    return rb.run_variant(Q, view, "fused", p)


def pseudo_board(Q: np.ndarray, db: str) -> list[dict]:
    """Front label only: a point 3 m along the setup direction from the cloud centre (no "normal" key)."""
    sign = 1.0 if FRONT[db] == "+x" else -1.0
    c = Q.mean(0)
    return [{"centre": [float(c[0] + 3.0 * sign), float(c[1]), float(c[2])]}]


def run_v2(Q: np.ndarray, db: str, debug: bool = False) -> dict:
    import importlib
    import mombe.body.measure_v2 as mv2
    importlib.reload(mv2)
    try:
        return mv2.measure_v2(Q, pseudo_board(Q, db), debug=debug)
    except Exception as e:  # a crash is a failure, logged with its reason
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


def _job(args):
    db, key, which = args
    Q, v = load(db, key)
    if which == "v1":
        r = run_v1(Q, v, db)
        m = r.get("measure") or {}
        return key, {"ok": bool(r.get("ok")), "reason": r.get("reason", ""), "stage": r.get("stage", ""),
                     "girth_hull_cm": m.get("girth", {}).get("hull_m", np.nan) * 100 if r.get("ok") else np.nan,
                     "girth_ellipse_cm": m.get("heart_girth_m", np.nan) * 100 if r.get("ok") else np.nan,
                     "length_cm": m.get("body_length_m", np.nan) * 100 if r.get("ok") else np.nan,
                     "withers_cm": m.get("withers_height_m", np.nan) * 100 if r.get("ok") else np.nan,
                     "station_class": rb.station_class(r) if (r.get("front") or {}).get("ok") else f"failed_{r.get('stage')}",
                     "legs": m.get("legs"), "station_m": m.get("girth_station_m")}
    r = run_v2(Q, db)
    ok = bool(r.get("ok"))
    g = r.get("girth") or {}
    return key, {"ok": ok, "reason": r.get("reason", ""),
                 "girth_hull_cm": g.get("hull_m", np.nan) * 100 if ok else np.nan,
                 "girth_ellipse_cm": g.get("ellipse_m", np.nan) * 100 if ok else np.nan,
                 "length_cm": r.get("body_length_m", np.nan) * 100 if ok else np.nan,
                 "length_direct_cm": r.get("length_direct_m", np.nan) * 100 if ok else np.nan,
                 "withers_cm": r.get("withers_height_m", np.nan) * 100 if ok else np.nan,
                 "station_class": r.get("station_method", "failed") if ok else "failed",
                 "coverage": g.get("coverage"), "back_height_m": r.get("back_height_m"),
                 "trunk_m": r.get("trunk_m"), "station_m": r.get("girth_station_m"), "legs": r.get("legs"),
                 "hull_slices_m": g.get("hull_slices_m"),
                 "barrel_cm": r.get("barrel_girth_m", np.nan) * 100 if ok else np.nan,
                 "girth_barrel_cm": r.get("heart_girth_from_barrel_m", np.nan) * 100 if ok else np.nan,
                 "shoulder_s_m": r.get("shoulder_s_m"), "pin_s_m": r.get("pin_s_m"), "back_height_m": r.get("back_height_m")}


def run_many(db: str, ids, which: str, procs: int = 10) -> dict[int, dict]:
    from multiprocessing import Pool
    with Pool(procs) as pool:
        return dict(pool.map(_job, [(db, int(i), which) for i in ids]))


def _agree_line(meas, ref) -> dict:
    return rb.agreement(np.asarray(meas, float), np.asarray(ref, float))


def dev(tag: str, note: str = "") -> int:
    ids = SPLIT["dev"]["ids"]
    T = tape("cdb", ids)
    v1f = OUT / "v2_dev_v1_cache.json"
    if v1f.exists():
        v1 = {int(k): v for k, v in json.loads(v1f.read_text()).items()}
    else:
        v1 = run_many("cdb", ids, "v1")
        v1f.write_text(json.dumps(v1, default=float))
    v2 = run_many("cdb", ids, "v2")
    (OUT / f"v2_dev_{tag}.json").write_text(json.dumps(v2, default=float))
    lines = []
    for name, res in (("v1", v1), ("v2", v2)):
        ok = [i for i in ids if res[i]["ok"]]
        row = {"n_ok": len(ok)}
        for key, tk in (("girth_hull_cm", "girth_cm"), ("girth_ellipse_cm", "girth_cm"), ("length_cm", "obl_cm"),
                        ("withers_cm", "withers_cm"), ("girth_barrel_cm", "girth_cm")):
            a = _agree_line([res[i].get(key, np.nan) for i in ok], [T[i][tk] for i in ok])
            row[key] = a
        lines.append((name, row))
    print(f"{'id':>3} {'tapeG':>6} {'v1G':>6} {'v2G':>6} {'err':>6} {'v2cls':>15} {'tapeL':>6} {'v2L':>6} {'tapeW':>6} {'v2W':>6} cov")
    for i in ids:
        a = v2[i]
        print(f"{i:>3} {T[i]['girth_cm']:6.1f} {v1[i]['girth_hull_cm']:6.1f} {a['girth_ellipse_cm']:6.1f} "
              f"{a['girth_ellipse_cm'] - T[i]['girth_cm']:6.1f} {a['station_class']:>15} {T[i]['obl_cm']:6.1f} "
              f"{a['length_cm']:6.1f} {T[i]['withers_cm']:6.1f} {a['withers_cm']:6.1f} {a.get('coverage')} {a.get('reason', '')}")
    md = [f"\n## {tag}\n", note + "\n" if note else ""]
    md.append("| version | n ok | girth hull MAPE | hull bias (95% CI) | hull r | girth ellipse MAPE (bias, r) | from-barrel MAPE (r) | length vs OBL MAPE | length bias | withers MAPE | withers bias |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, row in lines:
        gh, ge, ln, wh, gb = row["girth_hull_cm"], row["girth_ellipse_cm"], row["length_cm"], row["withers_cm"], row["girth_barrel_cm"]
        md.append(f"| {name} | {row['n_ok']}/{len(ids)} | {gh.get('mape_pct')}% | {gh.get('bias')} ({gh.get('bias_ci95')}) | "
                  f"{gh.get('pearson_r')} | {ge.get('mape_pct')}% ({ge.get('bias')}, {ge.get('pearson_r')}) | "
                  f"{gb.get('mape_pct', '')}% ({gb.get('pearson_r', '')}) | {ln.get('mape_pct')}% | {ln.get('bias')} | "
                  f"{wh.get('mape_pct')}% | {wh.get('bias')} |")
    cls = {}
    for i in ids:
        cls[v2[i]["station_class"]] = cls.get(v2[i]["station_class"], 0) + 1
    md.append(f"\nv2 station methods: {cls}\n")
    txt = "\n".join(md)
    print(txt)
    log = OUT / "v2_dev_log.md"
    if not log.exists():
        log.write_text("# Track 1a-v2 dev log (CowDatabase dev animals 1-30 only)\n\n"
                       "Each entry is one v2 iteration scored on dev tape. Test and external stay unscored until the freeze.\n"
                       "v1 = measure() unchanged on the same cached clouds.\n")
    with open(log, "a") as f:
        f.write(txt + "\n")
    return 0


def dump(ids, dst: str) -> int:
    """Debug arrays of v2 for plotting (scratch only, never in the repo)."""
    Path(dst).mkdir(parents=True, exist_ok=True)
    for i in ids:
        Q, _ = load("cdb", i)
        r = run_v2(Q, "cdb", debug=True)
        d = r.pop("debug", None)
        if d is None:
            print(i, r)
            continue
        sub = np.arange(len(d["s"])) % 3 == 0
        np.savez_compressed(f"{dst}/{i}.npz", s=d["s"][sub], z=d["z"][sub], lat=d["lat"][sub],
                            sl_lat=d["lat"][d["slice"]], sl_z=d["z"][d["slice"]], stations=d["stations"],
                            ztop=d["ztop"], zlo=d["zlo"],
                            meta=json.dumps({k: v for k, v in r.items() if k != "girth"} | {"girth": r.get("girth"), "legs_dbg": d["legs"], "zcut": d["zcut"], "s_elbow": d["s_elbow"]}, default=float))
        print(i, "ok", r.get("station_method"))
    return 0


def _profile_one(args):
    db, i, dst = args
    import mombe.body.measure_v2 as mv2
    from mombe.body.measure import girth_of_slice
    Q, _ = load(db, i)
    r = run_v2(Q, db, debug=True)
    d = r.pop("debug", None)
    if d is None:
        return i, False
    s, z, lat = d["s"], d["z"], d["lat"]
    zb = r["back_height_m"]
    st = np.arange(np.percentile(s, 1), np.percentile(s, 99), 0.02)
    H = np.full(len(st), np.nan)
    E = np.full(len(st), np.nan)
    for k, x in enumerate(st):
        sl = (np.abs(s - x) < mv2.SLICE_HALF_M) & (z > mv2.LEG_CUT * zb)
        if sl.sum() < 30:
            continue
        kk = mv2._slice_keep(lat[sl], z[sl])
        if kk.sum() < 30:
            continue
        g = girth_of_slice(lat[sl][kk], z[sl][kk])
        H[k], E[k] = g.get("hull_m", np.nan), g.get("ellipse_m", np.nan)
    # leg-band occupancy per side, 2 cm bins along s, three height bands
    bands = {"hoof": (0.05, 0.18), "cannon": (0.18, 0.32), "forearm": (0.32, 0.42)}
    occ = {}
    for bn, (lo, hi) in bands.items():
        for side, m in (("L", lat > 0.02), ("R", lat < -0.02)):
            k = (z > lo * zb) & (z < hi * zb) & m
            occ[f"{bn}_{side}"] = np.histogram(s[k], bins=np.r_[st - 0.01, st[-1] + 0.01])[0]
    widths = {}
    for bn, (lo, hi) in {"w_low": (0.42, 0.55), "w_mid": (0.55, 0.75), "w_high": (0.75, 0.92)}.items():
        wv = np.full(len(st), np.nan)
        for k, x in enumerate(st):
            m_ = (np.abs(s - x) < 0.01) & (z > lo * zb) & (z < hi * zb)
            if m_.sum() >= 10:
                wv[k] = np.percentile(lat[m_], 98) - np.percentile(lat[m_], 2)
        widths[bn] = wv
    np.savez_compressed(f"{dst}/prof_{db}_{i}.npz", st=st, hull=H, ell=E, **widths, meta=json.dumps(
        {k: v for k, v in r.items() if k != "girth"}, default=float), ztop=np.interp(st, d["stations"], np.nan_to_num(d["ztop"])),
        zlo=np.interp(st, d["stations"], np.nan_to_num(d["zlo"])), **occ)
    return i, True


def profiles(db: str, ids, dst: str) -> int:
    from multiprocessing import Pool
    Path(dst).mkdir(parents=True, exist_ok=True)
    with Pool(10) as pool:
        res = pool.map(_profile_one, [(db, int(i), dst) for i in ids])
    print(res)
    return 0


def _loo_constant(ref: np.ndarray) -> dict:
    """Constant guess: each animal gets the mean of the others' tape values."""
    ref = np.asarray(ref, float)
    ref = ref[np.isfinite(ref)]
    n = len(ref)
    if n < 3:
        return {"n": n}
    pred = (ref.sum() - ref) / (n - 1)
    e = pred - ref
    return {"n": n, "mape_pct": round(float(np.mean(np.abs(e) / ref) * 100), 3), "mae": round(float(np.mean(np.abs(e))), 3),
            "tape_sd": round(float(np.std(ref, ddof=1)), 3)}


def _weights(G, L, W, Gt, Lt, folds_seed=0) -> dict:
    """Schaeffer and 10-fold log-log on measured dims, beside tape dims and the training-fold mean."""
    out = {}
    ok = np.isfinite(G) & np.isfinite(W) & (G > 0)
    hasL = L is not None
    if hasL:
        ok &= np.isfinite(L) & (L > 0)
    n = int(ok.sum())
    out["n"] = n
    if n < 20:
        return out
    if hasL:
        out["schaeffer_measured"] = rb.weight_scores(G[ok] ** 2 * L[ok] / 10838.0, W[ok])
    folds = rb.kfold_ids(n, 10, folds_seed)
    if hasL:
        out["cv10_loglog_measured_G_L"] = rb.weight_scores(rb.cv_loglog(G[ok], L[ok], W[ok], folds), W[ok])
    out["cv10_loglog_measured_G"] = rb.weight_scores(rb.cv_loglog(G[ok], None, W[ok], folds), W[ok])
    Wk = W[ok]
    const = np.full(n, np.nan)
    for f in np.unique(folds):
        const[folds == f] = Wk[folds != f].mean()
    out["cv10_training_fold_mean"] = rb.weight_scores(const, Wk)
    if Gt is not None:
        gt = Gt[ok]
        if Lt is not None and np.isfinite(Lt[ok]).all():
            lt = Lt[ok]
            out["schaeffer_tape"] = rb.weight_scores(gt ** 2 * lt / 10838.0, Wk)
            out["cv10_loglog_tape_G_L"] = rb.weight_scores(rb.cv_loglog(gt, lt, Wk, folds), Wk)
        out["cv10_loglog_tape_G"] = rb.weight_scores(rb.cv_loglog(gt, None, Wk, folds), Wk)
    out["folds"] = f"10 folds: numpy default_rng({folds_seed}).permutation over the {n} animals in id order, dealt round-robin"
    return out


def final() -> int:
    """Score v1 and the frozen v2 once on dev, test and external. Writes v2_summary.json and v2_per_animal.csv."""
    import hashlib
    sets = {"dev": ("cdb", SPLIT["dev"]["ids"]), "test": ("cdb", SPLIT["test"]["ids"]),
            "external": ("cdb2", SPLIT["external"]["ids"])}
    res = {}
    for nm, (db, ids) in sets.items():
        res[nm] = {"v1": run_many(db, ids, "v1"), "v2": run_many(db, ids, "v2")}
        print(nm, "done", flush=True)
    summary = {"what": "Track 1a-v2: measure() v1 and measure_v2() on cached Ruchay clouds vs tape; dev chose every v2 rule",
               "labels": {"measurements": "R/P: real capture, physical tape or stick truth (CowDatabase2 method unstated, P assumed)",
                          "weight": "R/L: real capture, live weight with weighing method unstated"},
               "measure_v2_sha256": hashlib.sha256((ROOT / "mombe/body/measure_v2.py").read_bytes()).hexdigest(),
               "measure_v1_sha256_worktree": hashlib.sha256((ROOT / "mombe/body/measure.py").read_bytes()).hexdigest(),
               "split": {k: {"source": SPLIT[k]["source"], "n": SPLIT[k]["n"]} for k in ("dev", "test", "external")},
               "published_baseline": {"bao2023_heart_girth_mape_pct": {"pose_normalised": 5.21, "raw": 5.59},
                                      "source": "Bao et al. 2023, ISPRS Annals X-1-W1-2023, all 103 CowDatabase animals"},
               "sets": {}}
    rows = []
    for nm, (db, ids) in sets.items():
        T = tape(db, ids)
        tg = np.array([T[i]["girth_cm"] for i in ids]); tl = np.array([T[i]["obl_cm"] for i in ids])
        tw = np.array([T[i]["withers_cm"] for i in ids]); tkg = np.array([T[i]["weight_kg"] for i in ids])
        has_obl = bool(np.isfinite(tl).any())
        S = {"n": len(ids), "constant_guess_loo": {"girth": _loo_constant(tg), "withers": _loo_constant(tw)}}
        if has_obl:
            S["constant_guess_loo"]["obl"] = _loo_constant(tl)
        for ver in ("v1", "v2"):
            R = res[nm][ver]
            col = lambda k: np.array([R[i].get(k, np.nan) if R[i]["ok"] else np.nan for i in ids], float)  # noqa: E731
            okm = np.array([R[i]["ok"] for i in ids])
            V = {"n_measured": int(okm.sum()), "failures": {}}
            for i in ids:
                if not R[i]["ok"]:
                    V["failures"][R[i]["reason"] or "?"] = V["failures"].get(R[i]["reason"] or "?", 0) + 1
            gh, ge, L, WH = col("girth_hull_cm"), col("girth_ellipse_cm"), col("length_cm"), col("withers_cm")
            V["girth_hull_vs_tape"] = rb.agreement(gh, tg)
            V["girth_ellipse_vs_tape"] = rb.agreement(ge, tg)
            if ver == "v2":
                gb = col("girth_barrel_cm")
                V["girth_from_barrel_vs_tape"] = rb.agreement(gb, tg)
                V["barrel_raw_vs_tape"] = rb.agreement(col("barrel_cm"), tg)
            if has_obl:
                V["length_vs_obl"] = rb.agreement(L, tl)
            V["withers_vs_tape"] = rb.agreement(WH, tw)
            Ghead = ge
            V["weight_headline_girth"] = "heart_girth_m (the ellipse girth in both versions, as in track 1a weight)"
            V["weight"] = _weights(Ghead, L, tkg, tg, tl if has_obl else None)
            if ver == "v2":
                V["weight_from_barrel"] = _weights(col("girth_barrel_cm"), L, tkg, tg, tl if has_obl else None)
            cls = {}
            for i in ids:
                c = R[i]["station_class"]
                cls.setdefault(c, []).append(i)
            tax = {}
            for c, members in sorted(cls.items()):
                mk = np.array([i in members for i in ids]) & okm
                key = "girth_ellipse_cm" if ver == "v2" else "girth_hull_cm"
                tax[c] = {"n": len(members), "girth_vs_tape": rb.agreement(col(key)[mk], tg[mk]) if mk.sum() >= 3 else {"n": int(mk.sum())}}
            V["station_taxonomy"] = tax
            S[ver] = V
        S["tape_weight_references"] = {"schaeffer_tape_all": rb.weight_scores(tg ** 2 * tl / 10838.0, tkg) if has_obl else None}
        summary["sets"][nm] = S
        for i in ids:
            r1, r2 = res[nm]["v1"][i], res[nm]["v2"][i]
            rows.append({"set": nm, "db": db, "id": i, **{f"tape_{k}": T[i][k] for k in T[i]},
                         **{f"v1_{k}": r1.get(k) for k in ("ok", "reason", "girth_hull_cm", "girth_ellipse_cm", "length_cm", "withers_cm", "station_class", "station_m")},
                         **{f"v2_{k}": r2.get(k) for k in ("ok", "reason", "girth_hull_cm", "girth_ellipse_cm", "girth_barrel_cm", "barrel_cm", "length_cm",
                                                           "withers_cm", "station_class", "station_m", "back_height_m", "coverage")}})
    # v1 on the cache against the original kernel run (CowDatabase, fused hull girth)
    orig = _read_csv(OUT / "per_animal.csv")
    d = [abs(r["v1_girth_hull_cm"] - _num(orig[r["id"]].get("fused_girth_hull_cm"))) for r in rows
         if r["db"] == "cdb" and r["v1_ok"] and np.isfinite(_num(orig[r["id"]].get("fused_girth_hull_cm")))]
    summary["v1_cache_vs_kernel_cdb"] = {"n_both": len(d), "median_abs_diff_cm": round(float(np.median(d)), 3),
                                         "max_abs_diff_cm": round(float(np.max(d)), 3), "n_over_1cm": int(sum(x > 1 for x in d))}
    (OUT / "v2_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    with open(OUT / "v2_per_animal.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    # Bland-Altman arrays for the figure
    ba = {}
    for nm in sets:
        ba[nm] = {}
        for ver, key in (("v1", "girth_hull_cm"), ("v2", "girth_ellipse_cm"), ("v2b", "girth_barrel_cm")):
            src = "v1" if ver == "v1" else "v2"
            rr = [r for r in rows if r["set"] == nm and r[f"{src}_ok"] and r.get(f"{src}_{key}") is not None
                  and np.isfinite(r[f"{src}_{key}"]) and np.isfinite(r["tape_girth_cm"])]
            m = np.array([r[f"{src}_{key}"] for r in rr]); t = np.array([r["tape_girth_cm"] for r in rr])
            ag = rb.agreement(m, t)
            ba[nm][ver] = {"mean": ((m + t) / 2).tolist(), "diff": (m - t).tolist(), "bias": ag.get("bias"), "loa95": ag.get("loa95"),
                           "mape_pct": ag.get("mape_pct"), "n": ag.get("n")}
    (OUT / "v2_bland_altman.json").write_text(json.dumps(ba))
    print(json.dumps({nm: {ver: {k: summary["sets"][nm][ver].get(k, {}).get("mape_pct") for k in
                                 ("girth_hull_vs_tape", "girth_ellipse_vs_tape", "girth_from_barrel_vs_tape", "length_vs_obl", "withers_vs_tape")}
                           for ver in ("v1", "v2")} for nm in sets}, indent=1))
    return 0


def v1check() -> int:
    rows = _read_csv(OUT / "per_animal.csv")
    for i in SPLIT["dev"]["ids"][:10]:
        Q, v = load("cdb", i)
        r = run_v1(Q, v, "cdb")
        m = r.get("measure") or {}
        g = (m.get("girth") or {}).get("hull_m", float("nan")) * 100 if r.get("ok") else float("nan")
        print(i, r.get("ok"), r.get("reason", ""), round(g, 2), rows[i].get("fused_girth_hull_cm"))
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["v1check", "dev", "dump", "profiles", "final"])
    ap.add_argument("--tag", default="iter")
    ap.add_argument("--note", default="")
    ap.add_argument("--ids", default="")
    ap.add_argument("--dst", default="")
    a = ap.parse_args()
    if a.cmd == "v1check":
        sys.exit(v1check())
    if a.cmd == "dump":
        sys.exit(dump([int(x) for x in a.ids.split(",")], a.dst))
    if a.cmd == "profiles":
        which = a.tag if a.tag in ("dev", "test", "external") else "dev"
        sys.exit(profiles("cdb2" if which == "external" else "cdb", SPLIT[which]["ids"], a.dst))
    if a.cmd == "final":
        sys.exit(final())
    sys.exit(dev(a.tag, a.note))
