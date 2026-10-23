"""Herd valuation on real cattle: herds resampled from the Acme set.

Input R: real side and rear photos (Acme AI / bhalo / mPower, CC BY 4.0).
Truth L: the set's tape-formula weight labels.
Herds RS: real animals grouped at random into herds of 5, 10 and 20.

Per animal: an out-of-fold weight and a 90% split-conformal interval from the
physics model of eval/acme_weight.py (log w = a + b log(G^2 L)), 10 folds.
Per herd: mombe.agent.valuation gives the independent and the conservative P10
at the floor price. Scored against the label weights at the same price:

  * P10 coverage: share of herds whose true value falls below the herd P10
    (nominal 10%), for both aggregations;
  * shortfall rate at LTV 50, 60 and 70%: share of herds whose true value falls
    below the loan (LTV x conservative herd P10), beside the probability the
    model predicts from the evidence (the price is the same on both sides).

Usage: python eval/value_herds.py [--batch B4] [--herds 2000] [--demo 20]
Writes eval/out/value_herds_<batch>.json; --demo also writes a replayed herd,
eval/out/value_herd_demo.json, for tools/seed_herd.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.acme_weight import build  # noqa: E402
from mombe.agent.valuation import LTVS, Prices, herd_value, weight_quantiles  # noqa: E402

SIZES = (5, 10, 20)
VALUED_ON = date(2026, 10, 21)       # fixed so the run reproduces


def out_of_fold(rows: list[dict], folds: int = 10, seed: int = 7, alpha: float = 0.10) -> list[dict]:
    """Physics model, 10-fold: held-out weight and its conformal half-width in log space."""
    rng = np.random.default_rng(seed)
    y = np.log([r["weight"] for r in rows])
    X = np.c_[np.ones(len(rows)), np.log([r["G"] ** 2 * r["L"] for r in rows])]
    fold = rng.permutation(len(rows)) % folds
    pred, half = np.zeros(len(rows)), np.zeros(len(rows))
    for f in range(folds):
        tr, te = fold != f, fold == f
        cal = tr & (rng.random(len(rows)) < 0.25)
        beta = np.linalg.lstsq(X[tr & ~cal], y[tr & ~cal], rcond=None)[0]
        res = np.abs(y[cal] - X[cal] @ beta)
        q = np.quantile(res, min(1, math.ceil((len(res) + 1) * (1 - alpha)) / len(res)))
        pred[te], half[te] = X[te] @ beta, q
    return [{"animal_id": f"acme_{r['animal']}", "kg": float(np.exp(p)), "lo": float(np.exp(p - h)),
             "hi": float(np.exp(p + h)), "true_kg": r["weight"]} for r, p, h in zip(rows, pred, half)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="B4")
    ap.add_argument("--herds", type=int, default=2000)
    ap.add_argument("--demo", type=int, default=0, help="also write a replayed herd of this size")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    animals = out_of_fold(build(a.batch))
    prices = Prices()
    floor = prices.low["usd_per_kg"]
    true_kg = np.array([x["true_kg"] for x in animals])
    p10_kg = np.array([weight_quantiles(x)["p10"] for x in animals])
    inside = np.array([x["lo"] <= x["true_kg"] <= x["hi"] for x in animals])
    summary = {
        "batch": a.batch, "animals": len(animals), "herds_per_size": a.herds, "valued_on": VALUED_ON.isoformat(),
        "labels": {"input": "R", "truth": "L", "herds": "RS"},
        "price": {"floor_usd_per_kg": floor, "source": prices.low["source"], "effective": prices.low["effective"]},
        "animal": {"interval_coverage": round(float(inside.mean()), 3),
                   "p10_coverage": round(float((true_kg < p10_kg).mean()), 3)},
        "herds": {},
    }
    rng = np.random.default_rng(a.seed)
    for size in SIZES:
        rows = []
        for _ in range(a.herds):
            idx = rng.choice(len(animals), size, replace=False)
            h = herd_value([animals[i] for i in idx], prices, on=VALUED_ON)
            truth = float(true_kg[idx].sum()) * floor
            rows.append((truth, h))
        truth = np.array([t for t, _ in rows])
        indep = np.array([h["p10_independent_usd"] for _, h in rows])
        cons = np.array([h["p10_conservative_usd"] for _, h in rows])
        central = np.array([h["usd"] for _, h in rows])
        s = {"p10_coverage_independent": round(float((truth < indep).mean()), 4),
             "p10_coverage_conservative": round(float((truth < cons).mean()), 4),
             "herd_ape_median": round(float(np.median(np.abs(central - truth) / truth)), 4),
             "conservative_p10_over_truth_median": round(float(np.median(cons / truth)), 3),
             "shortfall": []}
        for k, ltv in enumerate(LTVS):
            loans = np.array([h["shortfall"][k]["loan_usd"] for _, h in rows])
            pred_ind = np.array([h["shortfall"][k]["p_shortfall"]["evidence_independent"] for _, h in rows])
            pred_tog = np.array([h["shortfall"][k]["p_shortfall"]["evidence_together"] for _, h in rows])
            s["shortfall"].append({"ltv": ltv, "observed_rate": round(float((truth < loans).mean()), 4),
                                   "predicted_independent": round(float(pred_ind.mean()), 6),
                                   "predicted_together": round(float(pred_tog.mean()), 6)})
        summary["herds"][str(size)] = s
        print(f"herd {size:2d}: P10 coverage indep {s['p10_coverage_independent']:.3f} "
              f"cons {s['p10_coverage_conservative']:.3f}; herd APE {s['herd_ape_median']:.3f}; "
              + "; ".join(f"LTV {r['ltv']:.0%} observed {r['observed_rate']:.4f}" for r in s["shortfall"]), flush=True)

    out = ROOT / "eval" / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"value_herds_{a.batch}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "herds"}, indent=1))
    if a.demo:
        pick = np.random.default_rng(3).choice(len(animals), a.demo, replace=False)
        herd = [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in animals[i].items()} for i in pick]
        (out / "value_herd_demo.json").write_text(json.dumps(
            {"source": f"replayed Acme {a.batch} animals (CC BY 4.0): out-of-fold weights, label weights as true_kg",
             "labels": {"input": "R", "truth": "L"}, "animals": herd}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
