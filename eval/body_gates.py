"""Evidence behind the body-measurement gates: floors, the envelope and the scale claim.

Reads the body accuracy sweeps (tools/sweep_body.py) on both builds and the
Acme B4 keypoints, and prints one JSON summary:

  registration  registered fraction of keyframes and length agreement (fitted
                trunk over the direct cut) on every take
  coverage      girth-loop coverage on every take, and where a floor separates them
  envelope      body proportions on real cattle (Acme B4 keypoints, scale-free)
                and on the measured takes
  scale_claim   how often the claimed scale error covers the true one, before and
                after calibration; the calibration is fitted with each take (and
                each animal) held out in turn

Every take is a synthetic walk-around rendered with exact truth (input S, truth G).
Usage: python eval/body_gates.py [--sweeps eval/out] [--out eval/out/body_gates.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mombe.agent.policy import ENVELOPE, LENGTH_AGREEMENT, LOOP_COVERAGE_FLOOR  # noqa: E402
from mombe.body.calibration import Z90  # noqa: E402

TAKES = {"stock": "sweep_body3.jsonl", "cool": "sweep_cool.jsonl"}   # the current pipeline on each build
FRAMES_PER_WALK = 48
LOOP_FLOOR = LOOP_COVERAGE_FLOOR
ENVELOPE_GL = ENVELOPE["girth_length"][:2]


def load_takes(sweeps: Path) -> list[dict]:
    rows = []
    for build, name in TAKES.items():
        for line in (sweeps / name).read_text().splitlines():
            r = json.loads(line)
            if "girth_m" in r:
                rows.append({**r, "build": build})
    return rows


# -- scale claim calibration: sigma = sqrt(floor^2 + (mult * claimed)^2) ----------------
FLOORS = np.round(np.arange(0.0, 0.0201, 0.0001), 4)
MULTS = np.round(np.arange(1.0, 5.001, 0.01), 2)


def fit_claim(claimed: np.ndarray, err: np.ndarray) -> tuple[float, float]:
    """Narrowest (floor, multiplier) whose 90% band covers every training take.

    With n training takes, covering all n is the split-conformal rule for a 90%
    target while n < 20; the narrowest such pair minimises the mean log width.
    """
    f = FLOORS[:, None, None]
    k = MULTS[None, :, None]
    sig = np.sqrt(f ** 2 + (k * claimed[None, None, :]) ** 2)
    ok = np.all(np.abs(err)[None, None, :] <= Z90 * sig, axis=2)
    width = np.mean(np.log(np.maximum(sig, 1e-9)), axis=2)
    width[~ok] = np.inf
    i, j = np.unravel_index(int(np.argmin(width)), width.shape)
    return float(FLOORS[i]), float(MULTS[j])


def calibrated(claimed, floor, mult):
    return np.sqrt(floor ** 2 + (mult * np.asarray(claimed)) ** 2)


def scale_claim(rows: list[dict]) -> dict:
    c = np.array([r["claimed_scale_err"] for r in rows])
    e = np.array([r["animal_scale_err"] for r in rows])
    build = np.array([r["build"] for r in rows])
    animal = np.array([r["animal"] for r in rows])

    def cover(sig, z):
        return np.abs(e) <= z * sig

    before_1s, before_90 = cover(c, 1.0), cover(c, Z90)
    loto = np.zeros(len(rows), bool)
    loto_params = []
    for i in range(len(rows)):
        tr = np.arange(len(rows)) != i
        f, k = fit_claim(c[tr], e[tr])
        loto_params.append((f, k))
        loto[i] = abs(e[i]) <= Z90 * calibrated(c[i], f, k)
    loao = np.zeros(len(rows), bool)
    for a in sorted(set(animal)):
        te = animal == a
        f, k = fit_claim(c[~te], e[~te])
        loao[te] = cover(calibrated(c, f, k), Z90)[te]
    cross = {}
    for src, dst in (("stock", "cool"), ("cool", "stock")):
        f, k = fit_claim(c[build == src], e[build == src])
        cross[f"fit_{src}_test_{dst}"] = {"floor": f, "mult": k,
                                         "covered": int(cover(calibrated(c, f, k), Z90)[build == dst].sum()),
                                         "n": int((build == dst).sum())}
    # takes the other body gates accept: girth loop seen and proportions inside the envelope
    kept = np.array([r["loop_coverage"] >= LOOP_FLOOR and ENVELOPE_GL[0] <= r["girth_m"] / r["length_m"] <= ENVELOPE_GL[1]
                     for r in rows])
    fg, kg_ = fit_claim(c[kept], e[kept])
    gated = {"n": int(kept.sum()), "floor": fg, "mult": kg_,
             "leave_one_take_out": None}
    lg = []
    for i in np.where(kept)[0]:
        tr = kept & (np.arange(len(rows)) != i)
        f1, k1 = fit_claim(c[tr], e[tr])
        lg.append(abs(e[i]) <= Z90 * calibrated(c[i], f1, k1))
    gated["leave_one_take_out"] = f"{int(sum(lg))}/{len(lg)}"
    f, k = fit_claim(c, e)
    final = calibrated(c, f, k)

    def by_build(mask):
        return {b: f"{int(mask[build == b].sum())}/{int((build == b).sum())}" for b in ("stock", "cool")} | \
               {"all": f"{int(mask.sum())}/{len(mask)}"}
    per = [{"take": f"{r['build']}:{r['animal']}", "claimed": round(float(c[i]), 5), "true": round(float(e[i]), 5),
            "covered_1sigma": bool(before_1s[i]), "covered_90_raw": bool(before_90[i]),
            "loto_params": loto_params[i], "covered_90_loto": bool(loto[i]), "covered_90_loao": bool(loao[i]),
            "calibrated": round(float(final[i]), 5)} for i, r in enumerate(rows)]
    return {"model": "sigma = sqrt(floor^2 + (mult * claimed)^2); covered when |true| <= 1.645 sigma",
            "n_takes": len(rows), "n_animals": len(set(animal)),
            "before": {"covered_at_1_sigma": by_build(before_1s), "covered_at_90": by_build(before_90)},
            "after": {"leave_one_take_out": by_build(loto), "leave_one_animal_out": by_build(loao),
                      "transfer": cross, "fit_all": {"floor": f, "mult": k}, "gated_takes_only": gated,
                      "in_sample": by_build(cover(final, Z90))},
            "median_claim_raw": round(float(np.median(c)), 5), "median_claim_calibrated": round(float(np.median(final)), 5),
            "takes": per}


# -- envelope ---------------------------------------------------------------------------------
def acme_ratios(batch: str = "B4") -> dict:
    """Scale-free proportions from the Acme side-view keypoints (real cattle, human-annotated)."""
    from eval.acme_weight import animal_of, ellipse_perimeter, keypoints
    base = ROOT / "data" / "acme" / "Vector" / batch
    side = keypoints(next((base / "Side" / "data").glob("*.json")))
    rear = {animal_of(f)[0]: k for f, k in keypoints(next((base / "Rear" / "data").glob("*.json"))).items()}
    L, D, G, GL = [], [], [], []
    for f, k in side.items():
        a = animal_of(f)[0]
        h = np.linalg.norm(k[7] - k[8])                 # height, withers to ground
        if h <= 0:
            continue
        L.append(np.linalg.norm(k[2] - k[1]) / h)       # point of shoulder to pin bone
        D.append(np.linalg.norm(k[3] - k[4]) / h)       # chest depth at the girth
        if a in rear:
            r = rear[a]
            wr = np.linalg.norm(r[2] - r[3]) / max(np.linalg.norm(r[0] - r[1]), 1e-9)   # rear width in heights
            G.append(ellipse_perimeter(D[-1] / 2, wr / 2))
            GL.append(G[-1] / L[-1])
    return {"batch": batch, "length_withers": pct(L), "depth_withers": pct(D),
            "girth_proxy_withers": pct(G), "girth_proxy_length": pct(GL),
            "note": "girth proxy: ellipse through chest depth and rear width (via shared height); a rough stand-in for the tape"}


def pct(x) -> dict:
    x = np.asarray(x)
    return {p: round(float(np.percentile(x, p)), 3) for p in (0.5, 1, 5, 50, 95, 99, 99.5)} | {"n": int(len(x))}


def acme_weight_per_height(n: int, batch: str = "B4", seed: int = 7) -> dict:
    """Weight per cubic metre of withers height on a sample of Acme animals.

    The sticker gives metres per pixel up to its unpublished printed size, so the
    spread around the median is exact and the level is not. The level shown takes
    the sticker size that makes Schaeffer's formula on the girth proxy return the
    labels (tape-formula weights, truth L). Reads one mask per animal: slow.
    """
    import random
    from eval.acme_weight import animal_of, ellipse_perimeter, keypoints, sticker_px
    base = ROOT / "data" / "acme"
    side = keypoints(next((base / "Vector" / batch / "Side" / "data").glob("*.json")))
    rear = {animal_of(f)[0]: k for f, k in keypoints(next((base / "Vector" / batch / "Rear" / "data").glob("*.json"))).items()}
    files = sorted(side)
    random.Random(seed).shuffle(files)
    rows = []
    for f in files:
        if len(rows) >= n:
            break
        a, w, _ = animal_of(f)
        mask = base / "Pixel" / batch / "Side" / "annotations" / f"{f}___fuse.png"
        d = sticker_px(mask) if a in rear and mask.exists() else None
        if not d:
            continue
        k, r = side[f] / d, rear[a]
        L, D, H = np.linalg.norm(k[2] - k[1]), np.linalg.norm(k[3] - k[4]), np.linalg.norm(k[7] - k[8])
        wr = np.linalg.norm(r[2] - r[3]) / np.linalg.norm(r[0] - r[1]) * H
        rows.append((w, L, H, ellipse_perimeter(D / 2, wr / 2)))
    w, L, H, G = map(np.array, zip(*rows))
    x = w / H ** 3
    sticker_cm = float(np.median(np.cbrt(10838 * w / (G ** 2 * L))))
    return {"n": len(rows), "kg_per_h3_over_median": {p: round(v / float(np.median(x)), 3) for p, v in pct(x).items() if p != "n"},
            "sticker_cm_implied": round(sticker_cm, 2), "withers_cm": pct(H * sticker_cm),
            "kg_per_withers_m3": pct(w / (H * sticker_cm / 100) ** 3)}


def take_envelope(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        g, l = r["girth_m"], r["length_m"]
        ld = r["true_length_m"] * (1 + r["length_direct_err"])       # the direct cut; truth cancels in the ratio
        out.append({"take": f"{r['build']}:{r['animal']}", "girth_length": round(g / l, 3),
                    "true_girth_length": round(r["true_girth_m"] / r["true_length_m"], 3),
                    "length_fit_vs_cut": round(l / ld, 3), "girth_err": r["girth_err"], "length_err": r["length_err"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", default=str(ROOT / "eval" / "out"))
    ap.add_argument("--out", default=str(ROOT / "eval" / "out" / "body_gates.json"))
    ap.add_argument("--no-acme", action="store_true")
    ap.add_argument("--acme-weight", type=int, default=0, help="also sample this many Acme animals for weight per height")
    a = ap.parse_args()
    rows = load_takes(Path(a.sweeps))
    reg = []
    for r in rows:
        agree = r["length_m"] / (r["true_length_m"] * (1 + r["length_direct_err"]))   # fitted length / direct cut
        reg.append({"take": f"{r['build']}:{r['animal']}", "registered": r["registered"],
                    "fraction": round(r["registered"] / FRAMES_PER_WALK, 3), "length_agreement": round(agree, 3),
                    "refused": not LENGTH_AGREEMENT[0] <= agree <= LENGTH_AGREEMENT[1]})
    cov = sorted(({"take": f"{r['build']}:{r['animal']}", "girth_loop_coverage": r["loop_coverage"],
                   "girth_err": r["girth_err"], "length_err": r["length_err"]} for r in rows),
                 key=lambda x: x["girth_loop_coverage"])
    out = {"labels": "input S (synthetic walk-arounds), truth G (rendered geometry); Acme: input R, truth = keypoint annotation",
           "registration": {"takes": reg, "min_fraction": min(x["fraction"] for x in reg)},
           "coverage": {"takes": cov},
           "envelope": {"takes": take_envelope(rows)},
           "scale_claim": scale_claim(rows)}
    if not a.no_acme:
        out["envelope"]["acme"] = acme_ratios()
    if a.acme_weight:
        out["envelope"]["acme_weight_per_height"] = acme_weight_per_height(a.acme_weight)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "scale_claim"}, indent=1, default=float))
    sc = out["scale_claim"]
    print(json.dumps({k: sc[k] for k in ("model", "n_takes", "n_animals", "before", "after",
                                         "median_claim_raw", "median_claim_calibrated")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
