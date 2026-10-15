"""Weight from two photos on real cattle: the Acme AI / bhalo / mPower set (Bangladesh, CC BY 4.0).

Each animal has a side photo with a white disc sticker on the flank and a rear
photo, keypoints (withers, pin bone, point of shoulder, girth top and bottom,
height; rear top, bottom, left, right) and a weight label from the tape formula.

Scale: the sticker's major axis (a tilted disc keeps its major axis) gives
metres per pixel in the side photo. Its printed size is unpublished but equal
for every animal, so it folds into the fitted constant. The rear photo carries
no sticker; its width is scaled through the height both views share.

Models, compared by 10-fold cross-validation on animals:
  physics   log w = a + b log(G^2 L), G the ellipse through chest depth and rear width
  fitted    least squares on log measurements and sex
Intervals: split conformal on absolute log residuals, 90% target, coverage checked
on held-out folds.

Usage: python eval/acme_weight.py [--batch B4]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STICKER_BGR = (255, 117, 0)          # B3/B4 mask colour for the sticker (RGB 0,117,255)


def keypoints(path: Path) -> dict[str, np.ndarray]:
    c = json.loads(path.read_text())
    names = {i["id"]: i["file_name"].split("/")[-1] for i in c["images"]}
    out = {}
    for a in c["annotations"]:
        k = np.array(a["keypoints"], float).reshape(-1, 3)
        if (k[:, 2] > 0).all():
            out[names[a["image_id"]]] = k[:, :2]
    return out


def animal_of(fname: str) -> tuple[str, float, str]:
    p = fname.split("_")
    return f"{p[0]}_{p[1]}", float(p[3]), p[4].split(".")[0]


def sticker_px(mask_path: Path) -> float | None:
    m = cv2.imread(str(mask_path))
    if m is None:
        return None
    st = np.all(m == STICKER_BGR, axis=2).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(st)
    if n < 2:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cnt, _ = cv2.findContours((lab == big).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(cnt[0]) < 5:
        return None
    (_, _), (a, b), _ = cv2.fitEllipse(cnt[0])
    return float(max(a, b))


def ellipse_perimeter(a: float, b: float) -> float:
    h = ((a - b) / (a + b)) ** 2
    return math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))


def build(batch: str) -> list[dict]:
    base = ROOT / "data" / "acme"
    side = keypoints(next((base / "Vector" / batch / "Side" / "data").glob("*.json")))
    rear = {animal_of(f)[0]: k for f, k in keypoints(next((base / "Vector" / batch / "Rear" / "data").glob("*.json"))).items()}
    rows = []
    for f, k in side.items():
        a, w, sex = animal_of(f)
        if a not in rear:
            continue
        mask = base / "Pixel" / batch / "Side" / "annotations" / f"{f}___fuse.png"
        d = sticker_px(mask) if mask.exists() else None
        if not d:
            continue
        s = 1.0 / d                                          # sticker diameters per pixel
        L = np.linalg.norm(k[2] - k[1]) * s                  # point of shoulder to pin bone
        D = np.linalg.norm(k[3] - k[4]) * s                  # chest depth at the girth
        H = np.linalg.norm(k[7] - k[8]) * s                  # height
        r = rear[a]
        Wr = np.linalg.norm(r[2] - r[3]) / np.linalg.norm(r[0] - r[1]) * H   # rear width via shared height
        G = ellipse_perimeter(D / 2, Wr / 2)
        rows.append({"animal": a, "weight": w, "sex": sex, "L": L, "D": D, "H": H, "W": Wr, "G": G})
    return rows


def cv_eval(rows: list[dict], folds: int = 10, seed: int = 7, alpha: float = 0.10) -> dict:
    rng = np.random.default_rng(seed)
    y = np.log([r["weight"] for r in rows])
    X_phys = np.c_[np.ones(len(rows)), np.log([r["G"] ** 2 * r["L"] for r in rows])]
    X_fit = np.c_[np.ones(len(rows)), np.log([[r["G"], r["L"], r["D"], r["W"], r["H"]] for r in rows]),
                  [r["sex"] == "M" for r in rows]]
    fold = rng.permutation(len(rows)) % folds
    out = {}
    for name, X in (("physics", X_phys), ("fitted", X_fit)):
        pred = np.zeros(len(rows))
        lo = np.zeros(len(rows))
        hi = np.zeros(len(rows))
        for f in range(folds):
            tr, te = fold != f, fold == f
            # split the training folds again: fit on one part, calibrate the interval on the other
            cal = tr & (rng.random(len(rows)) < 0.25)
            fit = tr & ~cal
            beta = np.linalg.lstsq(X[fit], y[fit], rcond=None)[0]
            res = np.abs(y[cal] - X[cal] @ beta)
            q = np.quantile(res, min(1, math.ceil((len(res) + 1) * (1 - alpha)) / len(res)))
            beta = np.linalg.lstsq(X[tr & ~cal], y[tr & ~cal], rcond=None)[0]
            pred[te] = X[te] @ beta
            lo[te], hi[te] = pred[te] - q, pred[te] + q
        w, p = np.exp(y), np.exp(pred)
        ape = np.abs(p - w) / w
        out[name] = {"mape": round(float(ape.mean()), 4), "median_ape": round(float(np.median(ape)), 4),
                     "within_10pct": round(float((ape <= 0.10).mean()), 3),
                     "interval_coverage": round(float(((y >= lo) & (y <= hi)).mean()), 3),
                     "interval_half_width_pct": round(float((np.exp(np.median(hi - pred)) - 1) * 100), 1)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="B4")
    a = ap.parse_args()
    rows = build(a.batch)
    w = np.array([r["weight"] for r in rows])
    summary = {"batch": a.batch, "animals": len(rows),
               "weight_kg": {"min": float(w.min()), "median": float(np.median(w)), "max": float(w.max())} if len(w) else None}
    if len(rows) >= 40:
        summary.update(cv_eval(rows))
    out = ROOT / "eval" / "out" / f"acme_weight_{a.batch}.json"
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
