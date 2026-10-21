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


SCALE_OK = {"board_frames": 12, "scale_rel_err": 0.004, "registered_frames": 48, "frames": 48}
# a 400 kg animal with cattle proportions: girth 1.40 and length 1.05 times a 128 cm withers
BODY_OK = {"coverage": 0.85, "girth_loop_coverage": 0.8, "girth_gap_deg": 72, "girth_gap_arc": "belly",
           "girth_cm": 179.0, "length_cm": 134.5, "length_cut_cm": 128.0, "withers_cm": 128.0}


def value_ledger(**kw) -> Ledger:
    base = dict(quality=CLEAN, identity=ident("match"), registry={"owner": "Moyo"},
                continuity={"continuous": True, "breaks": 0}, scale=SCALE_OK, body=BODY_OK)
    return Ledger(Task.VALUE, **{**base, **kw})


def test_value_needs_scale_before_weight_matters():
    L = value_ledger(scale={**SCALE_OK, "board_frames": 2, "scale_rel_err": 0.04}, body=None)
    s = evaluate(L)
    assert s.required == "request_capture" and s.required_args["code"] == "BOARD_CLOSER"


def test_weight_interval_straddling_loan_requests_second_pass():
    L = value_ledger(loan_threshold_usd=900, ltv=0.5,
                     weight={"kg": 400, "lo": 380, "hi": 425},
                     value={"lo": 1700, "hi": 1950})
    s = evaluate(L)
    assert s.required == "request_capture" and s.required_args["code"] == "SECOND_PASS"


def test_value_goes_to_human_approval_then_write():
    L = value_ledger(loan_threshold_usd=600, ltv=0.5,
                     weight={"kg": 400, "lo": 385, "hi": 418},
                     value={"lo": 1700, "hi": 1850})
    s = evaluate(L)
    assert s.required == "request_approval" and s.required_args["kind"] == "credit_offer"
    assert "write_registry" not in s.allowed
    L.approval = {"id": "ap1", "status": "approved"}
    s = evaluate(L)
    assert s.required == "write_registry" and s.required_args["approval_id"] == "ap1"


def test_claim_write_keeps_registered_owner(tmp_path):
    from mombe.agent.runtime import Controller
    from mombe.server.store import LocalRegistry, LocalStore

    class Tools:
        def __init__(self, store):
            self.store = store

    store, reg = LocalStore(tmp_path / "s"), LocalRegistry(tmp_path / "r")
    reg.put("A7", {"owner": "Moyo"})
    s = store.create("claim", policy_animal="A7")
    c = Controller(s.id, Tools(store), reg)
    c.L.quality = CLEAN
    c.L.identity = ident("match", aid="A7")
    c.L.approval = {"id": "ap1", "status": "approved"}
    reg.approvals["ap1"] = {"id": "ap1", "status": "approved"}
    c.state = __import__("mombe.agent.policy", fromlist=["evaluate"]).evaluate(c.L)
    c.execute("write_registry", {"kind": "claim_payout", "approval_id": "ap1"})
    assert reg.get("A7")["owner"] == "Moyo"
