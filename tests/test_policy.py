from mombe.agent.policy import Ledger, Task, evaluate

CLEAN = {"issues": [], "instructions": []}
GLARE = {"issues": ["GLARE"], "instructions": [{"code": "GLARE", "message": "Shade the nose."}]}
TH = {"t_match": 60, "t_new": 18}
LIVE = {"live": True, "score": 0.4, "threshold": 0.01, "failed_cue": None}
FLAT = {"live": False, "score": 0.0, "threshold": 0.01, "failed_cue": "parallax"}


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
    s = evaluate(Ledger(Task.VERIFY, quality=CLEAN, liveness=LIVE, identity=ident("ambiguous", score=35, margin=5)))
    assert s.required == "request_capture" and s.required_args["code"] == "SECOND_ANGLE"
    assert "write_registry" not in s.allowed


def test_duplicate_enrolment_is_flagged_then_adjudicated():
    L = Ledger(Task.ENROL, quality=CLEAN, identity=ident("match"), registry={"owner": "Moyo"})
    assert evaluate(L).required == "flag_fraud"
    L.fraud_flagged = True
    assert evaluate(L).required == "request_approval"


def test_clean_enrolment_writes_without_approval():
    L = Ledger(Task.ENROL, quality=CLEAN, liveness=LIVE, identity=ident("new", aid=None, score=8))
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
    base = dict(quality=CLEAN, liveness=LIVE, identity=ident("match"), registry={"owner": "Moyo"},
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
    c.L.liveness = LIVE
    c.L.identity = ident("match", aid="A7")
    c.L.approval = {"id": "ap1", "status": "approved"}
    reg.approvals["ap1"] = {"id": "ap1", "status": "approved"}
    c.state = __import__("mombe.agent.policy", fromlist=["evaluate"]).evaluate(c.L)
    out = c.execute("write_registry", {"kind": "claim_payout", "approval_id": "ap1"})
    assert "error" not in out
    assert reg.get("A7")["owner"] == "Moyo"


# -- liveness and double enrolment ---------------------------------------------------
def test_liveness_is_measured_before_identity_matters():
    s = evaluate(Ledger(Task.VERIFY, quality=CLEAN, identity=ident("match"), registry={"owner": "Moyo"}))
    assert s.next_observation == "check_liveness" and not s.done


def test_failed_liveness_asks_once_then_escalates_as_fraud():
    L = Ledger(Task.VERIFY, claimed_owner="Moyo", quality=CLEAN, liveness=FLAT,
               identity=ident("match"), registry={"owner": "Moyo"})
    s = evaluate(L)
    assert s.required == "request_capture" and s.required_args["code"] == "LIVE_RECAPTURE"
    assert s.required_args["gate"] == "liveness"
    L.recaptures["liveness"] = 1
    s = evaluate(L)
    assert s.required == "flag_fraud" and "presentation attack" in s.required_args["reason"]
    L.fraud_flagged = True
    s = evaluate(L)
    assert s.required == "request_approval" and s.required_args["kind"] == "adjudicate"
    assert "gate" not in s.required_args          # a fraud review, never a quality waiver
    L.approval = {"id": "ap9", "status": "approved"}
    assert evaluate(L).outcome == "escalated"     # even an approving reviewer cannot pass a spoof in-session


def test_failed_liveness_blocks_enrolment_write():
    L = Ledger(Task.ENROL, claimed_owner="Moyo", quality=CLEAN, liveness=FLAT, identity=ident("new", aid=None, score=8))
    for _ in range(3):
        s = evaluate(L)
        assert "write_registry" not in s.allowed
        L.recaptures["liveness"] = L.recaptures.get("liveness", 0) + 1


def test_failed_liveness_blocks_credit_approval():
    L = Ledger(Task.VALUE, quality=CLEAN, liveness=FLAT, identity=ident("match"), registry={"owner": "Moyo"},
               loan_threshold_usd=600, continuity={"continuous": True, "breaks": 0},
               scale={"board_frames": 12, "scale_rel_err": 0.004}, body={"coverage": 0.85},
               weight={"kg": 400, "lo": 385, "hi": 418}, value={"lo": 1700, "hi": 1850})
    L.recaptures["liveness"] = 1
    s = evaluate(L)
    assert s.required == "flag_fraud" and "write_registry" not in s.allowed
    L.fraud_flagged = True
    s = evaluate(L)
    assert s.required_args.get("kind") == "adjudicate"


def test_liveness_from_another_capture_does_not_count():
    L = Ledger(Task.VERIFY, quality=CLEAN, liveness={**LIVE, "capture": 0},
               identity={**ident("match"), "capture": 1}, registry={"owner": "Moyo"})
    s = evaluate(L)
    assert s.required == "request_capture" and s.required_args["gate"] == "liveness"


def test_waived_quality_still_needs_liveness():
    L = Ledger(Task.VERIFY, quality=GLARE, liveness=FLAT, identity=ident("match"), registry={"owner": "Moyo"})
    L.recaptures = {"capture_quality": 2, "liveness": 1}
    L.reviews["capture_quality"] = {"id": "r1", "status": "approved"}
    assert evaluate(L).required == "flag_fraud"


class _Tools:
    """Stand-in vision tools: scripted identity, a gallery that remembers enrolments."""

    def __init__(self, store):
        self.store, self.enrolled = store, []

    def enrol_muzzle(self, sid, animal_id):
        self.enrolled.append(animal_id)
        return {"animal_id": animal_id, "added": [f"{animal_id}_c00_0"]}


def _controller(tmp_path, task, **kw):
    from mombe.agent.runtime import Controller
    from mombe.server.store import LocalRegistry, LocalStore
    store, reg = LocalStore(tmp_path / "s"), LocalRegistry(tmp_path / "r")
    s = store.create(task, **kw)
    tools = _Tools(store)
    return Controller(s.id, tools, reg), tools, reg


def test_enrolment_adds_the_noseprint_and_a_second_owner_is_refused(tmp_path):
    from mombe.agent.policy import evaluate as ev
    c, tools, reg = _controller(tmp_path, "enrol", claimed_owner="Moyo")
    c.L.quality, c.L.liveness, c.L.identity = CLEAN, LIVE, ident("new", aid=None, score=8)
    c.state = ev(c.L)
    out = c.execute("write_registry", {"kind": "enrolment"})
    aid = out["result"]["written"]
    assert tools.enrolled == [aid] and reg.get(aid)["owner"] == "Moyo"

    # the same nose comes back under another owner: identify-before-enrol finds it
    c2, _, _ = _controller(tmp_path, "enrol", claimed_owner="Stranger")
    c2.registry = reg
    c2.L.quality, c2.L.liveness, c2.L.identity = CLEAN, LIVE, ident("match", aid=aid)
    c2.L.registry = reg.get(aid)
    c2.state = ev(c2.L)
    assert c2.state.required == "flag_fraud"
    refused = c2.execute("write_registry", {"kind": "enrolment"})
    assert "error" in refused and reg.get(aid)["owner"] == "Moyo"


def test_driver_cannot_swap_a_fraud_review_for_a_credit_approval(tmp_path):
    from mombe.agent.policy import evaluate as ev
    c, _, reg = _controller(tmp_path, "verify", claimed_owner="Moyo")
    c.L.quality, c.L.liveness = CLEAN, FLAT
    c.L.recaptures["liveness"] = 1
    c.L.fraud_flagged = True
    c.state = ev(c.L)
    assert c.state.required == "request_approval"
    out = c.execute("request_approval", {"kind": "credit_offer", "note": "looks fine"})
    assert "error" in out and not reg.approvals
