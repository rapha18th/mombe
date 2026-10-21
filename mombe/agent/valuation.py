"""Lender valuation: a value a lender can lend against, and how long it holds.

A lender in Zimbabwe takes cattle as movable collateral. From one weighing it
needs a conservative value per animal, a total for the herd, the chance the
herd ends up worth less than the loan, and the date the valuation lapses.
The approval itself stays with a person.

Per animal
  The weight arrives as a median and a central 90% interval. Each half of the
  interval is read as a lognormal tail, so P10 weight = kg * (lo / kg) ** (1.2816 / 1.6449).
  On a symmetric interval this sits a little below the normal P10, on the
  lender's side. P10 value = P10 weight * the lowest sourced price for the grade.
  Market value = median weight * the latest auction price.

Herd
  Independent weighing errors partly cancel. The independent herd P10 comes from
  the distribution of the sum, matched to a lognormal (Fenton and Wilkinson).
  Errors that move together, such as one formula bias across a breed or a board
  printed at the wrong size, add up. The conservative herd P10 assumes every
  animal errs low at once: the sum of the animal P10s, exact for fully
  dependent errors. Loans are sized on the conservative total.

Shortfall
  The probability that the herd, sold at the floor price, fetches less than a
  loan of LTV * conservative herd P10, for LTV 50, 60 and 70%. Two sources of
  risk: the weight evidence, and the price moving while the valuation is valid
  (spread of 90-day changes in the weekly price series). Condition loss expected
  before expiry lowers the herd median.

Expiry
  A value runs for 90 days, a policy choice. From July to October, unsupplemented
  communal oxen lost 6 to 7% of their starting weight (Ndlovu, Francis and Hove
  1996). Each day in that window accrues expected loss at the upper rate; the
  value lapses once the accrued loss reaches 3.5%, half the documented loss,
  after about 62 days. A value also lapses 120 days after the oldest price quote
  behind it (policy choice).
"""

from __future__ import annotations

import math
import os
from datetime import date, timedelta
from statistics import NormalDist

from mombe.agent.prices import load_book

_N = NormalDist()
Z10 = -_N.inv_cdf(0.10)          # 1.2816
Z95 = _N.inv_cdf(0.95)           # 1.6449: the weight interval is a central 90% interval

EXPIRY_DAYS = 90                 # policy choice
LOSS_MONTHS = (7, 8, 9, 10)      # July to October, the window of the 1996 trial
LOSS_RATE_PER_DAY = 0.07 / 123   # upper end: 7% over July 1 to October 31
LOSS_BUDGET = 0.035              # policy choice: half the documented dry-season loss
PRICE_MAX_AGE_DAYS = 120         # policy choice
LTVS = (0.5, 0.6, 0.7)
LOSS_SOURCE = {
    "citation": "Ndlovu, Francis and Hove 1996, Tropical Animal Health and Production 28:298-306",
    "url": "https://link.springer.com/article/10.1007/BF02240822",
    "finding": "64 pairs of smallholder oxen; unsupplemented animals lost 6 to 7% of starting weight from July to October",
}


def today() -> date:
    d = os.environ.get("MOMBE_TODAY")
    return date.fromisoformat(d) if d else date.today()


class Prices:
    """The quotes for one grade: floor (lowest), market (latest auction) and high."""

    def __init__(self, book: dict | None = None, grade: str | None = None):
        self.book = book or load_book()
        self.grade = grade or self.book["default_grade"]
        self.quotes = self.book["grades"][self.grade]
        self.low = min(self.quotes, key=lambda q: q["usd_per_kg"])
        self.high = max(self.quotes, key=lambda q: q["usd_per_kg"])
        auctions = [q for q in self.quotes if q.get("kind") == "auction"]
        self.market = max(auctions, key=lambda q: q["effective"]) if auctions else self.high
        self.sd_log = float(self.book.get("price_risk", {}).get("sd_log", 0.0))

    def oldest(self) -> date:
        return min(date.fromisoformat(q["effective"]) for q in (self.low, self.market))

    def to_dict(self) -> dict:
        return {"grade": self.grade, "unit": self.book["unit"], "floor": self.low, "market": self.market,
                "range_usd_per_kg": [self.low["usd_per_kg"], self.high["usd_per_kg"]],
                "price_risk": self.book.get("price_risk")}


# -- per animal ---------------------------------------------------------------
def weight_quantiles(w: dict) -> dict:
    """P10, P50, P90 weight and the lower-tail log sigma from a central 90% interval."""
    kg, lo, hi = float(w["kg"]), float(w["lo"]), float(w["hi"])
    if not 0 < lo <= kg <= hi:
        raise ValueError(f"weight interval must satisfy 0 < lo <= kg <= hi, got {lo}, {kg}, {hi}")
    s_lo, s_hi = math.log(kg / lo) / Z95, math.log(hi / kg) / Z95
    return {"p10": kg * math.exp(-Z10 * s_lo), "p50": kg, "p90": kg * math.exp(Z10 * s_hi), "sigma_log": s_lo}


def _loss_between(start: date, days: int) -> float:
    return sum(LOSS_RATE_PER_DAY for d in range(max(days, 0)) if (start + timedelta(days=d)).month in LOSS_MONTHS)


def expiry(on: date, prices: Prices | None = None) -> dict:
    """The season rule, capped by the age of the price quotes."""
    loss, days = 0.0, 0
    while days < EXPIRY_DAYS and loss < LOSS_BUDGET - 1e-12:
        if (on + timedelta(days=days)).month in LOSS_MONTHS:
            loss += LOSS_RATE_PER_DAY
        days += 1
    reason = (f"expected dry-season condition loss reaches {LOSS_BUDGET:.1%}" if loss >= LOSS_BUDGET - 1e-12
              else f"{EXPIRY_DAYS}-day validity")
    expires = on + timedelta(days=days)
    if prices is not None:
        stale = prices.oldest() + timedelta(days=PRICE_MAX_AGE_DAYS)
        if stale < expires:
            expires = stale
            reason = f"price quote of {prices.oldest().isoformat()} turns {PRICE_MAX_AGE_DAYS} days old"
    valid = (expires - on).days
    return {"expires": expires.isoformat(), "valid_days": valid, "expiry_reason": reason,
            "expected_condition_loss": round(_loss_between(on, valid), 4), "loss_source": LOSS_SOURCE}


def is_current(value: dict | None, on: date | None = None) -> bool:
    if not value or not value.get("expires"):
        return bool(value)
    return (on or today()) < date.fromisoformat(value["expires"])


def value_animal(weight: dict, prices: Prices | None = None, on: date | None = None) -> dict:
    p, on = prices or Prices(), on or today()
    q = weight_quantiles(weight)
    floor, market = p.low["usd_per_kg"], p.market["usd_per_kg"]
    return {
        "p10_usd": round(q["p10"] * floor),
        # 90% range and median at the floor price: the lending basis the policy reads
        "usd": round(weight["kg"] * floor), "lo": round(weight["lo"] * floor), "hi": round(weight["hi"] * floor),
        "market_usd": round(weight["kg"] * market),
        "p10_kg": round(q["p10"], 1), "p50_kg": round(q["p50"], 1), "p90_kg": round(q["p90"], 1),
        "price_per_kg_usd": floor, "market_price_per_kg_usd": market,
        "price_source": f"{p.low['source']}, effective {p.low['effective']}",
        "price": p.to_dict(),
        "basis": "P10 weight x floor price",
        "valued_on": on.isoformat(),
        **expiry(on, p),
    }


# -- herd -----------------------------------------------------------------------
def _sum_lognormal(medians: list[float], sigmas: list[float], rho: float) -> tuple[float, float]:
    """Fenton-Wilkinson: (mu, s) of the lognormal matching the sum's mean and variance."""
    means = [m * math.exp(s * s / 2) for m, s in zip(medians, sigmas)]
    total = sum(means)
    var = 0.0
    for i, (ei, si) in enumerate(zip(means, sigmas)):
        for j, (ej, sj) in enumerate(zip(means, sigmas)):
            var += ei * ej * math.expm1(si * si if i == j else rho * si * sj)
    s2 = math.log1p(var / total ** 2)
    return math.log(total) - s2 / 2, math.sqrt(s2)


def p_below(loan: float, medians: list[float], sigmas: list[float], rho: float,
            price: float, price_sd: float = 0.0, loss: float = 0.0) -> float:
    """P(herd value < loan): weights (dependence rho), a lognormal price move, expected condition loss."""
    mu, s = _sum_lognormal(medians, sigmas, rho)
    centre = mu + math.log(price) + math.log1p(-loss)
    spread = math.sqrt(s * s + price_sd * price_sd)
    return _N.cdf((math.log(loan) - centre) / spread) if spread > 0 else float(loan > math.exp(centre))


def herd_value(weights: list[dict], prices: Prices | None = None, on: date | None = None,
               ltvs: tuple[float, ...] = LTVS) -> dict:
    """Totals and shortfall for a herd. `weights`: kg, lo, hi per animal (central 90% intervals)."""
    if not weights:
        raise ValueError("a herd needs at least one weighed animal")
    p, on = prices or Prices(), on or today()
    qs = [weight_quantiles(w) for w in weights]
    med, sig = [q["p50"] for q in qs], [q["sigma_log"] for q in qs]
    floor, market = p.low["usd_per_kg"], p.market["usd_per_kg"]
    mu, s = _sum_lognormal(med, sig, 0.0)
    p10_indep = math.exp(mu - Z10 * s) * floor
    p10_cons = sum(q["p10"] for q in qs) * floor
    exp = expiry(on, p)
    loss = exp["expected_condition_loss"]
    rows = []
    for ltv in ltvs:
        loan = ltv * p10_cons
        rows.append({"ltv": ltv, "loan_usd": round(loan),
                     "p_shortfall": {k: round(x, 6) for k, x in {
                         "evidence_independent": p_below(loan, med, sig, 0.0, floor),
                         "evidence_together": p_below(loan, med, sig, 1.0, floor),
                         "with_price_independent": p_below(loan, med, sig, 0.0, floor, p.sd_log, loss),
                         "with_price_together": p_below(loan, med, sig, 1.0, floor, p.sd_log, loss),
                     }.items()}})
    return {
        "animals": len(weights),
        "kg": round(sum(med), 1),
        "usd": round(sum(med) * floor),
        "market_usd": round(sum(med) * market),
        "p10_independent_usd": round(p10_indep),
        "p10_conservative_usd": round(p10_cons),
        "loan_basis": "LTV x conservative herd P10",
        "assumptions": {
            "independent": "weighing errors independent across animals; they partly cancel",
            "conservative": "every animal errs low together; herd P10 = sum of animal P10s",
            "price_sd_log": p.sd_log,
            "shortfall": "herd sold at the floor price; price spread from 90-day changes; "
                         "expected condition loss to expiry applied",
        },
        "shortfall": rows,
        "price_per_kg_usd": floor, "market_price_per_kg_usd": market,
        "price_source": f"{p.low['source']}, effective {p.low['effective']}",
        "price": p.to_dict(),
        "valued_on": on.isoformat(),
        **exp,
    }
