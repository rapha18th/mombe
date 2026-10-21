"""Price spread over a loan horizon, from a weekly liveweight price series.

Reads a saved price series and reports the spread of log price changes over
30, 60, 90 and 120 days: standard deviation, the 10th percentile and the worst
change. Entries above a cap are per-head prices posted under the per-kg series
and are dropped. The valuation's shortfall analysis uses the 90-day spread as
its price risk (mombe/agent/prices.py).

The series file is JSON with a "series" string of `YYYY-MM-DD:cents` tokens: the
weekly Commercial grade liveweight chart behind https://farmnport.com/prices/liveweight
(all buyers, US cents per kg), saved on 2026-10-21.

Usage: python tools/price_stats.py eval/out/price_series_commercial.json
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path

CAP_CENTS = 1000          # US$10 per kg liveweight: anything above is a per-head entry


def load(path: Path) -> list[tuple[date, float]]:
    d = json.loads(path.read_text())
    pts = []
    for tok in d["series"].split():
        day, v = tok.split(":")
        v = float(v)
        if 0 < v <= CAP_CENTS:
            pts.append((date.fromisoformat(day), v))
    return sorted(pts)


def horizon_changes(pts: list[tuple[date, float]], days: int, slack: int = 10) -> list[float]:
    """Log change from each point to the point nearest `days` later, within `slack` days."""
    out = []
    for i, (d0, v0) in enumerate(pts):
        best = None
        for d1, v1 in pts[i + 1:]:
            gap = (d1 - d0).days
            if gap > days + slack:
                break
            if abs(gap - days) <= slack and (best is None or abs(gap - days) < abs(best[0] - days)):
                best = (gap, v1)
        if best:
            out.append(math.log(best[1] / v0))
    return out


def summarise(ch: list[float]) -> dict:
    n = len(ch)
    m = sum(ch) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in ch) / (n - 1))
    s = sorted(ch)
    return {"n": n, "mean_log": round(m, 4), "sd_log": round(sd, 4),
            "p10_log": round(s[int(0.10 * (n - 1))], 4), "worst_log": round(s[0], 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("series", type=Path)
    a = ap.parse_args()
    pts = load(a.series)
    out = {"points": len(pts), "from": pts[0][0].isoformat(), "to": pts[-1][0].isoformat(),
           "horizons": {str(h): summarise(horizon_changes(pts, h)) for h in (30, 60, 90, 120)}}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
