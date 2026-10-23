"""Cattle prices in Zimbabwe, each with its source and date.

Every quote is a buyer's own price sheet as posted on FarmnPort (Marondera),
US dollars per kg liveweight. The auction quote is the market price. The
abattoir bid is the floor a lender can liquidate at, and the P10 value uses
the lowest quote of the grade. A JSON file named by MOMBE_PRICE_BOOK, with the
same shape, replaces this book without a code change.

Searched without finding a 2025 or 2026 per-kg liveweight figure: Livestock
and Meat Advisory Council, Cold Storage Company, The Herald, NewsDay, ZimStat.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

FARMNPORT = "https://farmnport.com/prices/liveweight"
_CC = {"source": "CC Sales livestock auction, price sheet on FarmnPort", "kind": "auction",
       "effective": "2026-10-14", "posted": "2026-10-19", "retrieved": "2026-10-21", "url": FARMNPORT}
_SURREY = {"source": "Surrey Group abattoir, price list on FarmnPort", "kind": "abattoir",
           "effective": "2026-08-25", "posted": "2026-09-02", "retrieved": "2026-10-21", "url": FARMNPORT}

PRICE_BOOK = {
    "unit": "USD per kg liveweight",
    "default_grade": "commercial",
    "grades": {
        "commercial": [
            {**_CC, "usd_per_kg": 2.13, "note": "Commercial grade average; mean animal 420 kg"},
            {**_SURREY, "usd_per_kg": 1.38, "note": "Commercial grade ex lairage, collected at the farm; delivered 1.43"},
        ],
        "economy": [
            {**_CC, "usd_per_kg": 1.97, "note": "Economy grade average; mean animal 324 kg"},
            {**_SURREY, "usd_per_kg": 1.21, "note": "Economy grade ex lairage, collected; delivered 1.26"},
        ],
        "manufacturing": [
            {**_CC, "usd_per_kg": 1.36, "note": "Manufacturing grade average; mean animal 285 kg"},
            {**_SURREY, "usd_per_kg": 0.55, "note": "Manufacturing grade ex lairage, collected; delivered 0.60"},
        ],
        "super": [
            {**_CC, "usd_per_kg": 2.51, "note": "Super grade average; mean animal 439 kg"},
        ],
    },
    # Spread of the price over a loan horizon: standard deviation of 90-day log
    # changes in the weekly series (tools/price_stats.py). Week-to-week sale mix
    # is part of the spread, which makes it conservative.
    "price_risk": {
        "sd_log": 0.18, "horizon_days": 90, "windows": 107, "from": "2024-01-10", "to": "2026-10-14",
        "series": "FarmnPort weekly Commercial grade liveweight prices, all buyers", "url": FARMNPORT,
    },
}


def load_book() -> dict:
    path = os.environ.get("MOMBE_PRICE_BOOK")
    if path and Path(path).exists():
        return json.loads(Path(path).read_text())
    return PRICE_BOOK
