import math
from datetime import date

import numpy as np
import pytest

from mombe.agent import valuation as V
from mombe.agent.policy import Ledger, Task, evaluate

Z10 = 1.2815515655446004

BOOK = {
    "unit": "USD per kg liveweight", "default_grade": "commercial",
    "grades": {
        "commercial": [
            {"source": "Auction A", "kind": "auction", "usd_per_kg": 2.0, "effective": "2026-10-01", "url": "https://a.example"},
            {"source": "Abattoir B", "kind": "abattoir", "usd_per_kg": 1.5, "effective": "2026-09-20", "url": "https://b.example"},
        ],
        "super": [{"source": "Auction A", "kind": "auction", "usd_per_kg": 2.5, "effective": "2026-10-01", "url": "https://a.example"}],
    },
    "price_risk": {"sd_log": 0.18},
}
ON = date(2026, 11, 15)          # outside the dry-season loss window


def lognormal_weight(kg, s):
    return {"kg": kg, "lo": kg * math.exp(-1.6448536269514722 * s), "hi": kg * math.exp(1.6448536269514722 * s)}


# -- per animal -----------------------------------------------------------------
def test_p10_weight_is_exact_for_a_lognormal_interval():
    q = V.weight_quantiles(lognormal_weight(400, 0.1))
    assert q["p10"] == pytest.approx(400 * math.exp(-Z10 * 0.1), rel=1e-9)
    assert q["p90"] == pytest.approx(400 * math.exp(Z10 * 0.1), rel=1e-9)
    assert q["sigma_log"] == pytest.approx(0.1, rel=1e-9)


def test_p10_weight_sits_below_the_normal_p10_on_a_symmetric_interval():
    s = 0.05
    q = V.weight_quantiles({"kg": 400, "lo": 400 * (1 - 1.645 * s), "hi": 400 * (1 + 1.645 * s)})
    assert q["p10"] <= 400 * (1 - Z10 * s)
    assert q["p10"] > 400 * (1 - 1.645 * s)


def test_bad_interval_is_rejected():
    with pytest.raises(ValueError):
        V.weight_quantiles({"kg": 400, "lo": 420, "hi": 450})


def test_value_uses_the_floor_price_and_names_its_source_and_date():
    p = V.Prices(BOOK)
    w = lognormal_weight(400, 0.1)
    v = V.value_animal(w, p, on=ON)
    assert p.low["source"] == "Abattoir B" and p.market["source"] == "Auction A"
    assert v["p10_usd"] == round(400 * math.exp(-Z10 * 0.1) * 1.5)
    assert v["market_usd"] == 800 and v["price_per_kg_usd"] == 1.5
    assert "Abattoir B" in v["price_source"] and "2026-09-20" in v["price_source"]
    assert v["price"]["floor"]["url"] == "https://b.example"
    assert v["price"]["range_usd_per_kg"] == [1.5, 2.0]
    assert v["p10_usd"] < v["usd"] < v["market_usd"]
    assert v["expires"] and v["valued_on"] == ON.isoformat()


def test_single_quote_grade_uses_it_for_floor_and_market():
    p = V.Prices(BOOK, grade="super")
    assert p.low is p.market is p.high


def test_shipped_price_book_names_every_quote():
    p = V.Prices()
    for q in p.quotes:
        assert q["usd_per_kg"] > 0 and q["source"] and q["url"].startswith("https://")
        date.fromisoformat(q["effective"])
    assert p.low["usd_per_kg"] <= p.market["usd_per_kg"]


# -- herd -------------------------------------------------------------------------
HERD = [lognormal_weight(kg, s) for kg, s in
        [(320, 0.08), (410, 0.12), (365, 0.10), (500, 0.15), (280, 0.09),
         (450, 0.11), (390, 0.20), (300, 0.10), (420, 0.13), (350, 0.09)]]


def test_conservative_herd_p10_is_the_sum_of_animal_p10s_and_below_independent():
    p = V.Prices(BOOK)
    h = V.herd_value(HERD, p, on=ON)
    animal_p10 = sum(V.weight_quantiles(w)["p10"] for w in HERD) * 1.5
    assert h["p10_conservative_usd"] == round(animal_p10)
    assert h["p10_conservative_usd"] < h["p10_independent_usd"] < h["usd"]
    assert h["animals"] == 10 and h["kg"] == pytest.approx(sum(w["kg"] for w in HERD))


def test_one_animal_herd_matches_the_animal():
    p = V.Prices(BOOK)
    w = lognormal_weight(400, 0.1)
    h, v = V.herd_value([w], p, on=ON), V.value_animal(w, p, on=ON)
    assert h["p10_conservative_usd"] == v["p10_usd"]
    assert abs(h["p10_independent_usd"] - v["p10_usd"]) <= 1


def test_herd_p10_matches_monte_carlo():
    rng = np.random.default_rng(0)
    med = np.array([w["kg"] for w in HERD])
    sig = np.array([V.weight_quantiles(w)["sigma_log"] for w in HERD])
    indep = (med * np.exp(sig * rng.standard_normal((200_000, len(HERD))))).sum(1)
    mu, s = V._sum_lognormal(list(med), list(sig), 0.0)
    assert math.exp(mu - Z10 * s) == pytest.approx(np.quantile(indep, 0.10), rel=0.01)
    together = (med * np.exp(sig * rng.standard_normal((200_000, 1)))).sum(1)
    assert sum(V.weight_quantiles(w)["p10"] for w in HERD) == pytest.approx(np.quantile(together, 0.10), rel=0.005)


def test_shortfall_matches_monte_carlo_with_price_risk():
    rng = np.random.default_rng(1)
    med = np.array([w["kg"] for w in HERD])
    sig = np.array([V.weight_quantiles(w)["sigma_log"] for w in HERD])
    n = 400_000
    herd = (med * np.exp(sig * rng.standard_normal((n, len(HERD))))).sum(1)
    value = herd * 1.5 * np.exp(0.18 * rng.standard_normal(n)) * (1 - 0.02)
    loan = 0.7 * sum(V.weight_quantiles(w)["p10"] for w in HERD) * 1.5
    pred = V.p_below(loan, list(med), list(sig), 0.0, 1.5, 0.18, 0.02)
    assert pred == pytest.approx((value < loan).mean(), abs=0.003)


def test_shortfall_rises_with_ltv_price_risk_and_dependence():
    h = V.herd_value(HERD, V.Prices(BOOK), on=ON)
    rows = h["shortfall"]
    assert [r["ltv"] for r in rows] == [0.5, 0.6, 0.7]
    for key in ("evidence_independent", "evidence_together", "with_price_independent", "with_price_together"):
        ps = [r["p_shortfall"][key] for r in rows]
        assert ps == sorted(ps)
    for r in rows:
        p = r["p_shortfall"]
        assert p["evidence_independent"] <= p["evidence_together"] <= p["with_price_together"]
        assert p["evidence_independent"] <= p["with_price_independent"] <= p["with_price_together"]
        assert abs(r["loan_usd"] - r["ltv"] * h["p10_conservative_usd"]) <= 1


def test_empty_herd_is_rejected():
    with pytest.raises(ValueError):
        V.herd_value([], V.Prices(BOOK), on=ON)


# -- expiry -----------------------------------------------------------------------
def test_rainy_season_value_runs_ninety_days():
    e = V.expiry(date(2026, 11, 15))
    assert e["valid_days"] == 90 and e["expected_condition_loss"] == 0


def test_dry_season_value_lapses_when_condition_loss_reaches_budget():
    e = V.expiry(date(2026, 8, 1))
    assert 60 <= e["valid_days"] <= 63
    assert e["expected_condition_loss"] >= V.LOSS_BUDGET - 1e-9
    assert "dry-season" in e["expiry_reason"]
    assert "Ndlovu" in e["loss_source"]["citation"]


def test_value_made_before_the_window_accrues_loss_only_inside_it():
    e = V.expiry(date(2026, 6, 1))           # June has no accrual, July and August do
    assert e["valid_days"] == 90
    assert e["expected_condition_loss"] == pytest.approx(V.LOSS_RATE_PER_DAY * 60, abs=1e-3)


def test_old_price_quote_caps_the_expiry():
    p = V.Prices(BOOK)                        # oldest quote used: 2026-09-20
    e = V.expiry(ON, p)
    assert e["expires"] == "2027-01-18" and "price quote" in e["expiry_reason"]
    stale = V.expiry(date(2027, 3, 1), p)
    assert stale["valid_days"] <= 0
    assert not V.is_current({"expires": stale["expires"]}, on=date(2027, 3, 1))


def test_is_current():
    assert V.is_current({"expires": "2026-12-01"}, on=date(2026, 11, 30))
    assert not V.is_current({"expires": "2026-12-01"}, on=date(2026, 12, 1))
    assert V.is_current({"usd": 1}) and not V.is_current(None)


# -- policy ------------------------------------------------------------------------
CLEAN = {"issues": [], "instructions": []}
IDENT = {"decision": "match", "animal_id": "A1", "score": 120, "margin": 100, "thresholds": {"t_match": 90, "t_new": 20}}


def value_ledger(loan, value, weight=None):
    return Ledger(Task.VALUE, quality=CLEAN, identity=IDENT, registry={"owner": "Moyo"},
                  loan_threshold_usd=loan, ltv=0.5, continuity={"continuous": True, "breaks": 0},
                  scale={"board_frames": 12, "scale_rel_err": 0.004}, body={"coverage": 0.85},
                  weight=weight or {"kg": 400, "lo": 385, "hi": 418}, value=value)


def test_request_beyond_the_lendable_range_is_capped_at_p10():
    L = value_ledger(2000, {"p10_usd": 540, "usd": 552, "lo": 531, "hi": 577, "expires": "2999-01-01"})
    s = evaluate(L)
    assert s.required == "request_approval" and s.required_args["kind"] == "credit_offer"
    assert s.required_args["offer_cap_usd"] == 270


def test_loan_inside_p10_lendable_needs_no_second_pass():
    # 0.5 x lo = 265.5 < loan 268 <= 0.5 x P10 = 270: the P10 basis lends it outright
    L = value_ledger(268, {"p10_usd": 540, "usd": 552, "lo": 531, "hi": 577, "expires": "2999-01-01"})
    s = evaluate(L)
    assert s.required == "request_approval" and "offer_cap_usd" not in s.required_args


def test_reviewer_accepted_weight_still_reaches_the_person_capped_at_p10():
    # a wide interval the reviewer accepted; the request sits above the P10 lendable
    L = value_ledger(150, {"p10_usd": 215, "usd": 268, "lo": 203, "hi": 353, "expires": "2999-01-01"},
                     weight={"kg": 193.9, "lo": 146.8, "hi": 256.0})
    L.reviews["weight_interval"] = {"id": "r1", "status": "approved", "decided_by": "reviewer"}
    s = evaluate(L)
    assert s.required == "request_approval" and s.required_args["offer_cap_usd"] == 108


def test_lapsed_valuation_is_valued_again_before_approval():
    L = value_ledger(200, {"p10_usd": 540, "usd": 552, "lo": 531, "hi": 577, "expires": "2000-01-01"})
    s = evaluate(L)
    assert s.next_observation == "value_animal"
    assert "request_approval" not in s.allowed


def test_valuation_lapsed_on_the_day_it_was_made_stops_on_stale_prices():
    d = V.today().isoformat()
    L = value_ledger(200, {"p10_usd": 540, "usd": 552, "lo": 531, "hi": 577, "valued_on": d, "expires": d})
    s = evaluate(L)
    assert s.done and s.outcome == "price_stale"
    assert "request_approval" not in s.allowed


# -- controller: the approval request carries the valuation -------------------------
def test_approval_request_carries_p10_price_source_and_expiry(tmp_path):
    from mombe.agent.runtime import Controller
    from mombe.server.store import LocalRegistry, LocalStore

    class Tools:
        def __init__(self, store):
            self.store, self.prices = store, V.Prices(BOOK)

        def value_animal(self, sid, weight):
            return V.value_animal(weight, self.prices)

    store, reg = LocalStore(tmp_path / "s"), LocalRegistry(tmp_path / "r")
    reg.put("A1", {"owner": "Moyo"})
    s = store.create("value", claimed_owner="Moyo", loan_threshold_usd=200)
    L = value_ledger(200, None, weight=lognormal_weight(400, 0.03)).to_dict()
    s.ledger = L
    store.save(s)
    c = Controller(s.id, Tools(store), reg)
    assert c.state.next_observation == "value_animal"
    c.execute("value_animal")
    assert c.state.required == "request_approval"
    out = c.execute("request_approval", {"kind": "credit_offer", "note": "clean evidence"})
    summ = out["result"]["approval"]["summary"]
    val = summ["valuation"]
    assert val["p10_usd"] > 0 and val["current"] is True
    assert val["floor_quote"]["source"] == "Abattoir B" and val["floor_quote"]["effective"] == "2026-09-20"
    assert val["floor_quote"]["url"] == "https://b.example"
    assert val["expires"] and val["expiry_reason"]
    assert summ["loan"] == {"requested_usd": 200, "ltv": 0.5, "lendable_at_p10_usd": round(val["p10_usd"] * 0.5)}
    assert summ["owner"]["registered"] == "Moyo"
    assert {g["gate"] for g in summ["evidence"]} >= {"identity", "ownership", "weight_interval"}
    assert out["result"]["approval"]["status"] == "pending"      # the decision stays with a person
    assert "write_registry" not in c.state.allowed

    # a person approves; the record keeps the valuation for later claims
    reg.decide(out["result"]["approval"]["id"], "approved", by="lender")
    c.persist()
    c2 = Controller(s.id, Tools(store), reg)
    c2.execute("write_registry", dict(c2.state.required_args))
    assert reg.get("A1")["valuation"]["p10_usd"] == val["p10_usd"]
    assert reg.get("A1")["owner"] == "Moyo"

    # a claim on the same animal carries the valuation on record
    cl = store.create("claim", policy_animal="A1")
    cc = Controller(cl.id, Tools(store), reg)
    cc.L.quality, cc.L.identity, cc.L.registry = CLEAN, IDENT, reg.get("A1")
    cc.state = evaluate(cc.L)
    res = cc.execute("request_approval", {"kind": "claim_payout", "note": "carcass matches"})
    cv = res["result"]["approval"]["summary"]["valuation"]
    assert cv["p10_usd"] == val["p10_usd"] and cv["floor_quote"]["source"] == "Abattoir B"


def test_hub_herd_tool_reads_the_owner_herd(tmp_path):
    from mombe.server.store import LocalRegistry, LocalStore
    from mombe.server.tools import VisionTools

    store, reg = LocalStore(tmp_path / "s"), LocalRegistry(tmp_path / "r")
    for i, kg in enumerate((300, 350, 400, 450, 500)):
        w = lognormal_weight(kg, 0.1)
        reg.put(f"A{i}", {"owner": "Moyo", "weight_kg": w["kg"], "weight_interval": [w["lo"], w["hi"]]})
    reg.put("B0", {"owner": "Ncube", "weight_kg": 380, "weight_interval": [350, 410]})
    reg.put("A9", {"owner": "Moyo"})                        # never weighed: left out
    t = VisionTools(store, reg, gallery=None, engine=object(), prices=V.Prices(BOOK))
    h = t.value_herd(owner="Moyo")
    assert h["animals"] == 5 and set(h["animal_ids"]) == {f"A{i}" for i in range(5)}
    assert h["p10_conservative_usd"] < h["p10_independent_usd"]
    assert t.value_herd(owner="Nobody")["animals"] == 0
    assert t.price_book()["floor"]["source"] == "Abattoir B"
