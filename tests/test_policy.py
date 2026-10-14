from mombe.agent.policy import Ledger, Task, evaluate

CLEAN = {"issues": [], "instructions": []}
GLARE = {"issues": ["GLARE"], "instructions": [{"code": "GLARE", "message": "Shade the nose."}]}
TH = {"t_match": 60, "t_new": 18}


def ident(decision, aid="A1", score=120, margin=100):
    return {"decision": decision, "animal_id": aid, "score": score, "margin": margin, "thresholds": TH}


def test_glare_requires_recapture_then_escalates():
    L = Ledger(Task.VERIFY, quality=GLARE)
    s = evaluate(L)
    assert s.required == "request_capture" and s.required_args["code"] == "GLARE"
    L.recaptures["capture_quality"] = 2
    s = evaluate(L)
    assert s.required == "request_approval" and s.required_args["kind"] == "adjudicate"


def test_ambiguous_identity_asks_for_second_angle():
    s = evaluate(Ledger(Task.VERIFY, quality=CLEAN, identity=ident("ambiguous", score=35, margin=5)))
    assert s.required == "request_capture" and s.required_args["code"] == "SECOND_ANGLE"
    assert "write_registry" not in s.allowed


def test_duplicate_enrolment_is_flagged_then_adjudicated():
    L = Ledger(Task.ENROL, quality=CLEAN, identity=ident("match"), registry={"owner": "Moyo"})
    assert evaluate(L).required == "flag_fraud"
    L.fraud_flagged = True
    assert evaluate(L).required == "request_approval"


def test_clean_enrolment_writes_without_approval():
    L = Ledger(Task.ENROL, quality=CLEAN, identity=ident("new", aid=None, score=8))
    assert evaluate(L).required == "write_registry"
    L.written = True
    assert evaluate(L).outcome == "enrolled"


def test_claim_on_wrong_animal_is_blocked():
    L = Ledger(Task.CLAIM, policy_animal="A7", quality=CLEAN, identity=ident("match", aid="A9"))
    s = evaluate(L)
    assert s.required == "flag_fraud"
    assert "write_registry" not in s.allowed


def test_value_needs_scale_before_weight_matters():
    L = Ledger(Task.VALUE, quality=CLEAN, identity=ident("match"), registry={"owner": "Moyo"},
               continuity={"continuous": True, "breaks": 0},
               scale={"board_frames": 2, "scale_rel_err": 0.04})
    s = evaluate(L)
    assert s.required == "request_capture" and s.required_args["code"] == "BOARD_CLOSER"


def test_weight_interval_straddling_loan_requests_second_pass():
    L = Ledger(Task.VALUE, quality=CLEAN, identity=ident("match"), registry={"owner": "Moyo"},
               loan_threshold_usd=900, ltv=0.5,
               continuity={"continuous": True, "breaks": 0},
               scale={"board_frames": 12, "scale_rel_err": 0.004},
               body={"coverage": 0.85},
               weight={"kg": 400, "lo": 380, "hi": 425},
               value={"lo": 1700, "hi": 1950})
    s = evaluate(L)
    assert s.required == "request_capture" and s.required_args["code"] == "SECOND_PASS"


def test_value_goes_to_human_approval_then_write():
    L = Ledger(Task.VALUE, quality=CLEAN, identity=ident("match"), registry={"owner": "Moyo"},
               loan_threshold_usd=600, ltv=0.5,
               continuity={"continuous": True, "breaks": 0},
               scale={"board_frames": 12, "scale_rel_err": 0.004},
               body={"coverage": 0.85},
               weight={"kg": 400, "lo": 385, "hi": 418},
               value={"lo": 1700, "hi": 1850})
    s = evaluate(L)
    assert s.required == "request_approval" and s.required_args["kind"] == "credit_offer"
    assert "write_registry" not in s.allowed
    L.approval = {"id": "ap1", "status": "approved"}
    s = evaluate(L)
    assert s.required == "write_registry" and s.required_args["approval_id"] == "ap1"
