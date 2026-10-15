"""Tuner: a developer agent that tunes the identity workload on COOL.

Loop: the model proposes a pipeline configuration and a hypothesis; the trial
runs on the COOL Graviton4 hub (tools/trial.py over SSM); the measured latency
and match quality come back and shape the next proposal. When a configuration
clears the acceptance gate, the agent deploys it to the hub, rebuilds the
gallery if extraction changed, and verifies a live identification through
CloudFront. The gate is code: the model cannot deploy a configuration that
loses accuracy or lets an impostor reach the match threshold.

Usage: python -m mombe.tuner.agent [--budget 8] [--model eu.anthropic.claude-sonnet-4-6]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import boto3
from strands import Agent, tool
from strands.models import BedrockModel

from mombe.vision.muzzle import PipelineConfig

ROOT = Path(__file__).resolve().parents[2]
STATE = json.loads((ROOT / "infra" / "state.json").read_text())
REGION = STATE["region"]
EXTRACTION_KEYS = {"work_side", "aliked_input", "aliked_engine", "aliked_fp16", "dino_engine", "dino_fp16"}

SYSTEM = """You are the Mombe tuner, a developer agent. Your job: make cattle noseprint identification faster on the
COOL (Cloud-Optimized OpenCV) Graviton4 hub without losing accuracy, then deploy the winner and prove it runs.

Knobs (PipelineConfig): work_side (preprocess long side), aliked_input (ALIKED network input), aliked_engine and
dino_engine ("new" graph engine or "classic"), aliked_fp16 / dino_fp16 / lightglue_fp16 (FP16 targets; Graviton4 runs
FP16 natively; the classic engine honours targets), threads (0 = all 8 cores), verify_top (shortlisted animals that get
the expensive LightGlue + MAGSAC check).

Method:
- Run the baseline first with an empty config. Read where the time goes (extract, verify, identify end to end).
- Each trial: state a one-sentence hypothesis tied to a measured number, change one or two knobs, run it.
- Treat errors as evidence: an engine that cannot load a model rules out that knob for that model.
- The acceptance gate is enforced in code. A configuration is deployable when it has no error, accuracy at least the
  baseline, impostor_max below the match threshold, separation at least 90% of the baseline, and end-to-end latency at
  least 5% lower than the baseline.
- When a trial clears the gate and further trials look unlikely to beat it, call deploy, then verify_live, then finish.
- Stay within the trial budget. Report numbers, never adjectives.
"""


class Tuner:
    def __init__(self, budget: int, run_dir: Path):
        self.ssm = boto3.client("ssm", region_name=REGION)
        self.iid = STATE["hubs"]["cool"]["instance_id"]
        self.budget, self.trials = budget, []
        self.baseline: dict | None = None
        self.deployed: dict | None = None
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self.trace = run_dir / "trace.jsonl"

    def log(self, **ev):
        ev["t"] = round(time.time(), 2)
        with self.trace.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ev, default=str) + "\n")

    def ssm_run(self, cmds: list[str], timeout: int = 1800) -> str:
        cid = self.ssm.send_command(InstanceIds=[self.iid], DocumentName="AWS-RunShellScript",
                                    Parameters={"commands": cmds, "executionTimeout": [str(timeout)]},
                                    TimeoutSeconds=timeout)["Command"]["CommandId"]
        while True:
            time.sleep(5)
            try:
                r = self.ssm.get_command_invocation(CommandId=cid, InstanceId=self.iid)
            except self.ssm.exceptions.InvocationDoesNotExist:
                continue
            if r["Status"] not in ("Pending", "InProgress", "Delayed"):
                return r["StandardOutputContent"] + r["StandardErrorContent"]

    def cool(self, cmd: str) -> list[str]:
        return ["cd /opt/mombe", f"bash -c 'source /opt/cool/venvs/python_3.12/bin/activate && {cmd}' 2>&1 | grep -v WARN"]

    # -- gate ------------------------------------------------------------------
    def gate(self, r: dict) -> tuple[bool, str]:
        b = self.baseline
        if b is None:
            return False, "no baseline yet"
        if "error" in r:
            return False, "trial failed"
        checks = [
            (r["accuracy"] >= b["accuracy"], f"accuracy {r['accuracy']} vs baseline {b['accuracy']}"),
            (r["impostor_max"] < r["thresholds"]["t_match"], f"impostor_max {r['impostor_max']} vs t_match {r['thresholds']['t_match']}"),
            (r["separation"] >= 0.9 * b["separation"], f"separation {r['separation']} vs 0.9 x {b['separation']}"),
            (r["latency_ms"]["identify_end_to_end"] <= 0.95 * b["latency_ms"]["identify_end_to_end"],
             f"identify {r['latency_ms']['identify_end_to_end']} ms vs 0.95 x {b['latency_ms']['identify_end_to_end']} ms"),
        ]
        failed = [why for ok, why in checks if not ok]
        return (not failed), ("passes: " + "; ".join(w for _, w in checks)) if not failed else ("fails: " + "; ".join(failed))

    # -- actions ---------------------------------------------------------------
    def run_trial(self, config: dict, hypothesis: str) -> dict:
        if len(self.trials) >= self.budget:
            return {"error": f"trial budget of {self.budget} spent; deploy the best gated trial or finish"}
        cfg = asdict(PipelineConfig(**config))
        t = time.time()
        out = self.ssm_run(self.cool(f"python tools/trial.py --config \"$(cat <<'J'\n{json.dumps(config)}\nJ\n)\""))
        line = next((l for l in out.splitlines() if l.startswith("TRIAL_RESULT ")), None)
        r = json.loads(line[len("TRIAL_RESULT "):]) if line else {"error": "no result", "raw": out[-800:]}
        r.pop("trace", None)
        r["config"], r["wall_s"] = cfg, round(time.time() - t, 1)
        if self.baseline is None and "error" not in r and config == {}:
            self.baseline = r
        ok, why = self.gate(r)
        r["gate"] = {"deployable": ok, "reason": why}
        r["trial"] = len(self.trials) + 1
        self.trials.append({"hypothesis": hypothesis, **r})
        prev_best = self.best(exclude_last=True)
        self.log(kind="trial", n=r["trial"], hypothesis=hypothesis, config=config, result=r,
                 new_best=bool(ok and (prev_best is None or r["latency_ms"]["identify_end_to_end"] < prev_best["latency_ms"]["identify_end_to_end"])))
        return r

    def best(self, exclude_last: bool = False) -> dict | None:
        pool = self.trials[:-1] if exclude_last else self.trials
        ok = [t for t in pool if t.get("gate", {}).get("deployable")]
        return min(ok, key=lambda t: t["latency_ms"]["identify_end_to_end"]) if ok else None

    def deploy(self, config: dict) -> dict:
        match = next((t for t in self.trials if t["config"] == asdict(PipelineConfig(**config))), None)
        if not match or not match["gate"]["deployable"]:
            why = "no trial of this exact configuration" if not match else match["gate"]["reason"]
            self.log(kind="refused", action="deploy", config=config, reason=why)
            return {"error": f"deploy refused by the acceptance gate: {why}"}
        cfg = asdict(PipelineConfig(**config))
        rebuild = any(cfg[k] != asdict(PipelineConfig())[k] for k in EXTRACTION_KEYS)
        cmds = ["cd /opt/mombe", f"cat > /opt/mombe/pipeline.json <<'J'\n{json.dumps(cfg)}\nJ"]
        if rebuild:
            cmds += self.cool("python tools/rebuild_gallery.py")[1:]
        cmds += ["systemctl restart mombe-hub", "sleep 8", "curl -s -m 5 localhost:8080/health"]
        out = self.ssm_run(cmds, timeout=3600)
        self.deployed = {"config": cfg, "gallery_rebuilt": rebuild, "hub": out.strip().splitlines()[-1][:300]}
        self.log(kind="deploy", **self.deployed)
        return self.deployed

    def verify_live(self) -> dict:
        """A real identification through CloudFront on the deployed configuration."""
        import cv2
        import httpx
        from eval.dataset import make_split
        from mombe.agent.remote_tools import RemoteVisionTools
        hub, key = f"https://{STATE['cdn_domain']}", STATE["hub_key"]
        split = make_split()
        animal = split.test[11]
        img = ROOT / "data" / "muzzle" / "_".join(split.field_imgs[animal][0].split("_")[:2]) / f"{split.field_imgs[animal][0]}.jpg"
        http = httpx.Client(base_url=hub, headers={"x-mombe-key": key}, timeout=120)
        sid = http.post("/api/sessions", json={"task": "verify", "claimed_owner": None}).json()["id"]
        data = cv2.imencode(".jpg", cv2.imread(str(img)))[1].tobytes()
        http.post(f"/api/sessions/{sid}/captures", files=[("muzzle_0", ("m.jpg", data, "image/jpeg"))])
        tools = RemoteVisionTools(f"{hub}/mcp", key=key)
        try:
            res = tools.identify_muzzle(sid)
        finally:
            tools.close()
        out = {"session": sid, "true_animal": animal, "decision": res.get("decision"), "animal_id": res.get("animal_id"),
               "score": res.get("score"), "server_ms": res.get("elapsed_ms"), "round_trip_ms": res.get("round_trip_ms"),
               "correct": res.get("animal_id") == animal}
        self.log(kind="verify_live", **out)
        return out


def build(t: Tuner, model_id: str) -> Agent:
    @tool
    def run_trial(config_json: str, hypothesis: str) -> str:
        """Run the identity workload on the COOL hub with this PipelineConfig (JSON object of changed knobs only)."""
        return json.dumps(t.run_trial(json.loads(config_json or "{}"), hypothesis))

    @tool
    def history() -> str:
        """All trials so far, with their gate verdicts, and the current best deployable trial."""
        rows = [{"trial": x["trial"], "hypothesis": x["hypothesis"], "config_changes": {k: v for k, v in x["config"].items() if v != asdict(PipelineConfig())[k]},
                 "error": x.get("error"), "accuracy": x.get("accuracy"), "separation": x.get("separation"),
                 "impostor_max": x.get("impostor_max"), "latency_ms": x.get("latency_ms"), "gate": x.get("gate")} for x in t.trials]
        b = t.best()
        return json.dumps({"trials": rows, "best": b and {"trial": b["trial"], "latency_ms": b["latency_ms"]}, "budget_left": t.budget - len(t.trials)})

    @tool
    def deploy(config_json: str) -> str:
        """Deploy a configuration to the COOL hub. Refused unless a trial of exactly this configuration passed the gate."""
        return json.dumps(t.deploy(json.loads(config_json)))

    @tool
    def verify_live() -> str:
        """Identify a held-out animal through CloudFront on the deployed hub and report correctness and latency."""
        return json.dumps(t.verify_live())

    @tool
    def finish(summary: str) -> str:
        """End the tuning run with a short numeric summary."""
        t.log(kind="finish", summary=summary)
        return "done"

    return Agent(model=BedrockModel(model_id=model_id, region_name=REGION, temperature=0.2, max_tokens=2000, streaming=False),
                 system_prompt=SYSTEM, tools=[run_trial, history, deploy, verify_live, finish], callback_handler=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--model", default=os.environ.get("MOMBE_MODEL_ID", "eu.anthropic.claude-sonnet-4-6"))
    ap.add_argument("--run", default=time.strftime("%m%d-%H%M"))
    a = ap.parse_args()
    t = Tuner(a.budget, ROOT / "eval" / "out" / "tuner" / a.run)
    agent = build(t, a.model)
    res = agent(f"Tune the identity workload on the COOL hub. Trial budget: {a.budget}. Start with the baseline.")
    for _ in range(3):
        if t.deployed and any(json.loads(l)["kind"] == "finish" for l in t.trace.read_text().splitlines()):
            break
        res = agent("Continue: finish the loop (deploy the best gated configuration, verify it live, then finish), "
                    f"or report why no configuration qualifies. Budget left: {t.budget - len(t.trials)}.")
    (t.run_dir / "summary.json").write_text(json.dumps({"final": str(res), "best": t.best(), "baseline": t.baseline,
                                                        "deployed": t.deployed, "trials": t.trials}, indent=1, default=str))
    print(str(res))


if __name__ == "__main__":
    main()
