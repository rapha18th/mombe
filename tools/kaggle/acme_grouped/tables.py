"""Markdown tables for eval/out/acme_grouped/README.md, read straight from the result files.

Usage: python tools/kaggle/acme_grouped/tables.py > eval/out/acme_grouped/tables.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[3] / "eval" / "out" / "acme_grouped"
METHODS = ("constant", "sex", "sticker_only", "physics_b1", "physics", "fitted", "fitted_noscale")
BANDS = ("under_150", "150_to_250", "over_250")


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def num(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def nulls():
    p = OUT / "b4_nulls.json"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    print(f"### B4 null baselines (N = {d['animals']}, acme_weight.py folds)\n")
    rows = []
    order = ["constant", "sex", "sticker_only", "physics_b1", "physics", "fitted", "fitted_noscale",
             "common.physics_b1", "common.physics", "common.fitted"]
    for m in order:
        v = d["methods"][m]
        rows.append([m, v["n_params"], pct(v["mape"]), pct(v["nrmse"]), num(v["mae_kg"]), pct(v["bias_pct"]),
                     pct(v["within_10pct"]), v["fit_on"]])
    print(table(["method", "params", "MAPE", "nRMSE", "MAE kg", "bias", "within 10%", "fitted on"], rows))
    print(f"\nacme_weight.cv_eval rerun today: {json.dumps(d['acme_weight_cv_eval'])}\n")
    print("B4 under the conformal pipeline: random 10-fold vs leave one sub-batch out (pooled over b4-1..b4-4)\n")
    r10 = d["b4_random10_same_pipeline"]["headline"]
    lso = d["b4_leave_one_subbatch_out"]["pooled"]["headline"]
    rows = [[m, pct(r10[m]["mape"]), pct(lso[m]["mape"]), pct(r10[m]["cov90"]), pct(lso[m]["cov90"])] for m in METHODS]
    print(table(["method", "MAPE random10", "MAPE leave-sub-batch-out", "cov90 random10", "cov90 leave-sub-batch-out"], rows))
    print()


def grouped():
    p = OUT / "summary.json"
    if not p.exists():
        return
    s = json.loads(p.read_text(encoding="utf-8"))
    acc = s["accounting"]
    print("### Accounting per batch\n")
    rows = []
    for b, a in acc.items():
        for proto in ("full", "common"):
            dr = ", ".join(f"{k} {v}" for k, v in sorted(a["drops"][proto].items(), key=lambda kv: -kv[1]))
            rows.append([b, proto, a["available_animals"], a["used"][proto], dr or "none"])
    print(table(["batch", "protocol", "available", "used", "dropped (reason count)"], rows))
    print()
    for b, a in acc.items():
        st = a["sticker"]
        print(f"- {b}: sticker found {st['found_exact']}/{st['masks_read']} by exact colour "
              f"(a tolerant match would rescue {st['exact_failures_a_tolerant_match_would_rescue']}), used {st['match_used']}; "
              f"mask vs keypoint image size {json.dumps(a['mask_vs_keypoint_image_size'])}; "
              f"side keypoint duplicates across files {a['side_keypoints']['duplicate_images_across_files']}")
        ob = a.get("keypoints_on_cattle_mask")
        if ob:
            print(f"  - alignment check: {ob['checked']} side files; mean share of the 6 outline keypoints on the "
                  f"cattle mask {pct(ob['mean'])}; all 6 on body {pct(ob['share_all_6_on_body'])}; "
                  f"at most 3 on body {pct(ob['share_at_most_3_on_body'])}")
    print()
    for proto, res in s["results"].items():
        print(f"## Protocol: {proto}\n")
        if "skipped" in res:
            print(f"Skipped: {res['skipped']} ({res['n_by_batch']})\n")
            continue
        print("Weight range per batch (kg):\n")
        rows = []
        for b, dg in res["diagnostics"].items():
            w = dg["weight_kg"]
            rows.append([b, dg["n"], w["min"], w["median"], w["max"], w["mean"],
                         " / ".join(str(w["n_by_band"][k]) for k in BANDS), json.dumps(dg["sex"])])
        print(table(["batch", "N", "min", "median", "max", "mean", "N <150 / 150-250 / >250", "sex"], rows))
        print()
        hl = res["headline"]
        for b in res["batches_used"]:
            h = hl[b]
            print(f"### Held out {b} (N = {h['n']}; trained on {', '.join(res['lobo']['per_heldout_batch'][b]['train_groups'])})\n")
            rows = []
            for m in METHODS:
                r, g = h["random10_within"][m], h["lobo_heldout"][m]
                rows.append([m, pct(r["mape"]), pct(g["mape"]), pct(g["nrmse"]), num(g["mae_kg"]), pct(g["bias_pct"]),
                             pct(g["cov80"]), pct(g["cov90"]), f"{num(g['half_width90_pct'])}%", pct(g["mondrian_cov90"])])
            print(table(["method", "MAPE random10 in batch", "MAPE held out", "nRMSE", "MAE kg", "bias", "cov80",
                         "cov90", "half-width90", "Mondrian cov90"], rows))
            print()
        h = hl["pooled"]
        print(f"### Pooled over held-out batches (N = {h['n']})\n")
        rows = []
        for m in METHODS:
            r, g = h["random10_pooled"][m], h["lobo_pooled"][m]
            rows.append([m, pct(r["mape"]), pct(g["mape"]), pct(h["lobo_macro_mape"][m]), pct(g["nrmse"]),
                         num(g["mae_kg"]), pct(g["bias_pct"]), pct(g["cov80"]), pct(g["cov90"]), pct(g["mondrian_cov90"])])
        print(table(["method", "MAPE random10 pooled", "MAPE LOBO pooled", "MAPE LOBO macro", "nRMSE", "MAE kg", "bias",
                     "cov80", "cov90", "Mondrian cov90"], rows))
        print()
        print("Coverage by TRUE weight band, 90% nominal (split conformal | Mondrian on predicted band):\n")
        rows = []
        for b, blk in list(res["lobo"]["per_heldout_batch"].items()) + [("pooled", res["lobo"]["pooled"])]:
            for m in ("physics", "fitted"):
                c = blk["metrics"][m]["conformal"]["90"]["by_true_band"]
                mo = blk["metrics"][m]["mondrian"]["90"]["by_true_band"]
                mp = blk["metrics"][m]["mondrian"]["90"]["by_pred_band"]
                cells = []
                for k in BANDS:
                    if c[k]["n"] == 0:
                        cells.append("n=0")
                    else:
                        cells.append(f"n={c[k]['n']}: {pct(c[k]['coverage'])} / {pct(mo[k]['coverage'])} "
                                     f"(pred-band {pct(mp[k].get('coverage'))})")
                rows.append([b, m] + cells)
        print(table(["held out", "method", "under 150", "150 to 250", "over 250"], rows))
        print()
        print("80% nominal, coverage by true band (split conformal):\n")
        rows = []
        for b, blk in list(res["lobo"]["per_heldout_batch"].items()) + [("pooled", res["lobo"]["pooled"])]:
            for m in ("physics", "fitted"):
                c = blk["metrics"][m]["conformal"]["80"]
                rows.append([b, m, pct(c["coverage"]), num(c["median_width_kg"])] +
                            [pct(c["by_true_band"][k].get("coverage")) for k in BANDS])
        print(table(["held out", "method", "cov80", "median width kg", "under 150", "150 to 250", "over 250"], rows))
        print()
        print("Seed range over 5 calibration splits (min to max):\n")
        rows = []
        for g, d in res["lobo"]["seed_range"].items():
            for m in ("physics", "fitted"):
                rows.append([g, m, f"{pct(d[m + '.mape'][0])} to {pct(d[m + '.mape'][1])}",
                             f"{pct(d[m + '.cov80'][0])} to {pct(d[m + '.cov80'][1])}",
                             f"{pct(d[m + '.cov90'][0])} to {pct(d[m + '.cov90'][1])}"])
        print(table(["held out", "method", "MAPE", "cov80", "cov90"], rows))
        print()
        if res.get("leave_subgroup_out"):
            for b, v in res["leave_subgroup_out"].items():
                h = res["headline"][b]
                print(f"Leave one sub-group out inside {b} ({v['groups']}):\n")
                rows = [[m, pct(h["random10_within"][m]["mape"]), pct(h["leave_subgroup_out"][m]["mape"]),
                         pct(h["leave_subgroup_out"][m]["cov90"])] for m in METHODS]
                print(table(["method", "MAPE random10", "MAPE leave-sub-group-out", "cov90"], rows))
                print()
        if res.get("local_calibration"):
            print("Extension: n labelled animals from the held-out batch (20 random draws, mean [min to max]); "
                  "model trained on all animals of the other batches; metrics on the rest of the held-out batch\n")
            rows = []
            for b, d in res["local_calibration"].items():
                for m, v in d.items():
                    rows.append([b, m, "0", pct(v["no_local"]["mape"]), "n/a", "n/a", "n/a"])
                    for n, blk in v.items():
                        if n == "no_local":
                            continue
                        f = lambda k: (f"{pct(blk[k]['mean'])} [{pct(blk[k]['min'])} to {pct(blk[k]['max'])}]"  # noqa: E731
                                       if k in blk else "n/a")
                        rows.append([b, m, n, f("local_shift_cal.mape"), f("local_shift_cal.cov80"),
                                     f("local_shift_cal.cov90"), f("local_cal.cov90")])
            print(table(["held out", "method", "n local", "MAPE (shift on n/2)", "cov80 (shift, cal on n/2)",
                         "cov90 (shift, cal on n/2)", "cov90 (no shift, cal on n)"], rows))
            print()
        print("Per-batch fit alone (diagnostic): physics a, b and physics_b1 a\n")
        rows = [[b, dg["fit_on_this_batch_alone"]["physics"]["beta"], dg["fit_on_this_batch_alone"]["physics_b1"]["a"],
                 json.dumps(dg["median"]), json.dumps(dg["median_ratio"])] for b, dg in res["diagnostics"].items()]
        print(table(["batch", "physics [a, b]", "physics_b1 a", "medians", "median ratios"], rows))
        print()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    nulls()
    grouped()
