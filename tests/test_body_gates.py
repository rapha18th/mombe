"""The body measurement gates against the takes their floors came from (input S, truth G)."""

import numpy as np

from mombe.agent.policy import Ledger, Task, evaluate
from mombe.body.calibration import Z90, scale_claim
from mombe.body.measure import unseen_arc
from tests.test_policy import BODY_OK, SCALE_OK, value_ledger

# eval/out sweep_body3.jsonl (stock) and sweep_cool.jsonl (COOL): girth-loop coverage,
# measured girth, fitted length and the direct length cut, metres
SWEEP_BODY = [
    ("stock:sweep_21", 0.806, 1.4899, 1.1781, 1.0400),
    ("stock:sweep_22", 0.764, 1.4456, 1.04, 1.0400),
    ("stock:sweep_23", 0.778, 1.6705, 1.3442, 1.1999),
    ("stock:sweep_24", 0.792, 1.8673, 1.4887, 1.4801),
    ("stock:sweep_25", 0.806, 1.8808, 1.5068, 1.5000),
    ("stock:sweep_26", 0.667, 1.2727, 1.06, 1.0600),
    ("stock:sweep_27", 0.847, 1.9562, 1.5476, 1.4999),
    ("cool:sweep_21", 0.542, 1.3206, 1.4754, 1.1000),
    ("cool:sweep_22", 0.722, 1.4955, 1.04, 1.0400),
    ("cool:sweep_23", 0.792, 1.665, 1.3185, 1.2400),
    ("cool:sweep_24", 0.75, 1.8578, 1.4537, 1.4601),
    ("cool:sweep_25", 0.861, 1.938, 1.5464, 1.5799),
    ("cool:sweep_26", 0.639, 1.302, 1.04, 1.0400),
    ("cool:sweep_27", 0.861, 1.944, 1.5465, 1.4799),
]
# claimed and true relative scale error per take
SWEEP_SCALE = [
    ("stock:sweep_21", 0.07063, -0.05918), ("stock:sweep_22", 0.0408, -0.03503),
    ("stock:sweep_23", 0.00622, 0.00736), ("stock:sweep_24", 0.01021, -0.01003),
    ("stock:sweep_25", 0.02586, -0.00839), ("stock:sweep_26", 0.00292, -0.00335),
    ("stock:sweep_27", 0.00064, -0.00459), ("cool:sweep_21", 0.04366, -0.08927),
    ("cool:sweep_22", 0.01581, -0.00533), ("cool:sweep_23", 0.00947, -0.00231),
    ("cool:sweep_24", 0.01078, -0.01072), ("cool:sweep_25", 0.0323, -0.01558),
    ("cool:sweep_26", 0.02765, 0.0022), ("cool:sweep_27", 0.00223, -0.00697),
]


def gates(L: Ledger) -> dict:
    return {g.name: g for g in evaluate(L).gates}


def test_registration_floor_refuses_a_half_orbit():
    s = evaluate(value_ledger(scale={**SCALE_OK, "registered_frames": 26}))
    assert s.required == "request_capture" and s.required_args["code"] == "REORBIT"
    assert gates(value_ledger())["registration_health"].passed is True


def test_failed_reconstruction_fails_registration():
    s = evaluate(value_ledger(scale={"board_frames": 0, "scale_rel_err": 1.0}))
    assert s.required_args.get("gate") == "registration_health"


def test_unseen_belly_asks_for_the_crouching_pass():
    body = {**BODY_OK, "girth_loop_coverage": 0.5, "girth_gap_deg": 180, "girth_gap_arc": "belly"}
    s = evaluate(value_ledger(body=body))
    assert s.required == "request_capture" and s.required_args["code"] == "LOW_CHEST"


def test_unseen_flank_names_the_side():
    body = {**BODY_OK, "girth_loop_coverage": 0.5, "girth_gap_deg": 170, "girth_gap_arc": "right"}
    s = evaluate(value_ledger(body=body))
    assert s.required_args["code"] == "SHOW_SIDE" and "right" in s.required_args["message"]


def test_implausible_girth_refuses_the_weight():
    body = {**BODY_OK, "girth_cm": 0.72 * BODY_OK["girth_cm"]}
    L = value_ledger(body=body, weight={"kg": 207, "lo": 199, "hi": 215}, loan_threshold_usd=300,
                     value={"lo": 398, "hi": 430})
    s = evaluate(L)
    assert s.required == "request_capture" and s.required_args["code"] == "RECHECK_BODY"
    assert "request_approval" not in s.allowed and "write_registry" not in s.allowed


def test_plausibility_waits_for_the_weight_then_checks_it():
    s = evaluate(value_ledger())
    assert s.next_observation == "estimate_weight"
    s = evaluate(value_ledger(weight={"kg": 900, "lo": 870, "hi": 930}))      # 430 kg per withers m^3
    assert s.required_args.get("code") == "RECHECK_BODY"


def test_failed_measurement_answers_without_a_weight():
    failed = {"coverage": 0.0, "missing_side": None, "girth_loop_coverage": 0.0, "reason": "front and rear legs not separated"}
    s = evaluate(value_ledger(body=failed, weight={"kg": None, "reason": failed["reason"]}))
    assert s.required_args.get("gate") == "body_coverage"
    assert gates(value_ledger(weight={"kg": None}))["weight_interval"].passed is False


def test_calibrated_claim_refuses_a_raw_claim_inside_the_old_gate():
    raw = 0.009
    s = evaluate(value_ledger(scale={**SCALE_OK, "scale_rel_err": scale_claim(raw), "scale_rel_err_raw": raw}, body=None))
    assert s.required_args.get("code") == "BOARD_CLOSER"


def test_sweep_takes_cool_21_refused_rest_accepted():
    for take, loop, g, ln, cut in SWEEP_BODY:
        body = {"coverage": 1.0, "girth_loop_coverage": loop, "girth_gap_deg": (1 - loop) * 360, "girth_gap_arc": "belly",
                "girth_cm": g * 100, "length_cm": ln * 100, "length_cut_cm": cut * 100}
        gs = gates(value_ledger(body=body))      # every take registered 48 of 48 keyframes
        if take == "cool:sweep_21":
            assert gs["registration_health"].passed is False, take
            assert gs["body_coverage"].passed is False and gs["plausibility"].passed is False, take
        else:
            assert gs["registration_health"].passed is True, take
            assert gs["body_coverage"].passed is True and gs["plausibility"].passed is not False, take


def test_full_registration_that_never_closed_asks_for_a_new_orbit():
    body = {**BODY_OK, "length_cut_cm": BODY_OK["length_cm"] / 1.34}       # COOL sweep_21's disagreement
    s = evaluate(value_ledger(body=body))
    assert s.required_args.get("gate") == "registration_health" and s.required_args["code"] == "REORBIT"


def test_scale_claim_coverage_before_and_after():
    c = np.array([x[1] for x in SWEEP_SCALE])
    e = np.abs([x[2] for x in SWEEP_SCALE])
    assert int((e <= c).sum()) == 9                                    # raw claim read as one sigma
    assert int((e <= Z90 * c).sum()) == 11                             # raw claim inside a 90% band
    assert int((e <= Z90 * np.array([scale_claim(x) for x in c])).sum()) == 14   # fitted on these takes


def test_unseen_arc_names_the_belly_and_flanks():
    nb = 72
    centres = (np.arange(nb) + 0.5) / nb * 2 * np.pi - np.pi
    deg = np.degrees(centres)
    belly = ~((deg > -150) & (deg < -30))
    assert unseen_arc(belly, centres)["gap_arc"] == "belly"
    right = ~((deg > 140) | (deg < -140))
    assert unseen_arc(right, centres)["gap_arc"] == "right"
    left = ~((deg > -40) & (deg < 40))
    out = unseen_arc(left, centres)
    assert out["gap_arc"] == "left" and 75 <= out["gap_deg"] <= 85
