"""Deploy the Mombe agent to AgentCore Runtime by direct code deployment.

Packages mombe/ with Linux arm64 wheels for requirements-agent.txt, uploads the
zip to the project bucket, creates (or updates) the execution role and the
runtime, and stores the runtime ARN in infra/state.json.

Usage: python infra/deploy_agent.py [--hub https://<cloudfront>] [--model eu.anthropic.claude-sonnet-5]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
from deploy import load, save, say  # noqa: E402

BUILD = ROOT / "build" / "agent"
NAME = "mombe_agent"
PY = "3.13"


OTEL_LAUNCHER = r"""#!/usr/bin/env python3
import re
import sys
from opentelemetry.instrumentation.auto_instrumentation import run
if __name__ == "__main__":
    sys.argv[0] = re.sub(r"(-script\.pyw|\.exe)?$", "", sys.argv[0])
    sys.exit(run())
"""


def package() -> Path:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)
    subprocess.run([sys.executable, "-m", "uv", "pip", "install", "--python-platform", "aarch64-manylinux2014",
                    "--python-version", PY, "--target", str(BUILD), "--only-binary=:all:", "-q",
                    "-r", str(ROOT / "requirements-agent.txt")], check=True)
    # uv on Windows writes .exe launchers; AgentCore runs Linux, so write the
    # OpenTelemetry launcher as a plain Python script instead.
    for exe in (BUILD / "bin").glob("*.exe"):
        exe.unlink()
    (BUILD / "bin" / "opentelemetry-instrument").write_bytes(OTEL_LAUNCHER.encode())
    for p in (ROOT / "mombe").rglob("*.py"):
        if "__pycache__" in p.parts or p.parts[-2] in ("vision", "sim"):
            continue
        dst = BUILD / p.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    (BUILD / "main.py").write_text("from mombe.agent.agent_app import app\n\nif __name__ == \"__main__\":\n    app.run()\n")
    out = ROOT / "build" / "agent.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in BUILD.rglob("*"):
            if "__pycache__" in f.parts:
                continue
            info = zipfile.ZipInfo.from_file(f, f.relative_to(BUILD).as_posix())
            exe = f.is_dir() or f.parent.name == "bin"
            info.external_attr = ((0o755 if exe else 0o644) | (stat.S_IFDIR if f.is_dir() else stat.S_IFREG)) << 16
            if f.is_dir():
                z.writestr(info, "")
            else:
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, f.read_bytes())
    say("packaged", out, f"{out.stat().st_size / 1e6:.1f} MB")
    return out


def role(s, st) -> str:
    iam = s.client("iam")
    rn, acct, region = "mombe-agent-runtime", st["account"], st["region"]
    trust = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Principal": {"Service": "bedrock-agentcore.amazonaws.com"}, "Action": "sts:AssumeRole",
        "Condition": {"StringEquals": {"aws:SourceAccount": acct},
                      "ArnLike": {"aws:SourceArn": f"arn:aws:bedrock-agentcore:{region}:{acct}:*"}}}]}
    try:
        arn = iam.get_role(RoleName=rn)["Role"]["Arn"]
    except ClientError:
        arn = iam.create_role(RoleName=rn, AssumeRolePolicyDocument=json.dumps(trust),
                              Description="Mombe agent on AgentCore Runtime")["Role"]["Arn"]
        say("created role", rn)
    pol = {"Version": "2012-10-17", "Statement": [
        {"Sid": "Models", "Effect": "Allow",
         "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream", "bedrock:Converse", "bedrock:ConverseStream"],
         "Resource": ["arn:aws:bedrock:*::foundation-model/anthropic.*", f"arn:aws:bedrock:*:{acct}:inference-profile/*"]},
        {"Sid": "Code", "Effect": "Allow", "Action": ["s3:GetObject"], "Resource": f"arn:aws:s3:::{st['bucket']}/agent/*"},
        {"Sid": "Logs", "Effect": "Allow",
         "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams", "logs:DescribeLogGroups"],
         "Resource": f"arn:aws:logs:{region}:{acct}:log-group:/aws/bedrock-agentcore/*"},
        {"Sid": "Traces", "Effect": "Allow",
         "Action": ["xray:PutTraceSegments", "xray:PutTelemetryRecords", "xray:GetSamplingRules", "xray:GetSamplingTargets"], "Resource": "*"},
        {"Sid": "Metrics", "Effect": "Allow", "Action": "cloudwatch:PutMetricData", "Resource": "*",
         "Condition": {"StringEquals": {"cloudwatch:namespace": "bedrock-agentcore"}}},
        {"Sid": "Identity", "Effect": "Allow",
         "Action": ["bedrock-agentcore:GetWorkloadAccessToken", "bedrock-agentcore:GetWorkloadAccessTokenForJWT",
                    "bedrock-agentcore:GetWorkloadAccessTokenForUserId"], "Resource": "*"}]}
    iam.put_role_policy(RoleName=rn, PolicyName="mombe-agent", PolicyDocument=json.dumps(pol))
    return arn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub")
    ap.add_argument("--model", default="eu.anthropic.claude-sonnet-4-6")
    ap.add_argument("--skip-package", action="store_true")
    a = ap.parse_args()
    st = load()
    s = boto3.Session(region_name=st["region"])
    st["account"] = s.client("sts").get_caller_identity()["Account"]
    zp = ROOT / "build" / "agent.zip" if a.skip_package else package()
    key = f"agent/{int(time.time())}/agent.zip"
    s.client("s3").upload_file(str(zp), st["bucket"], key, ExtraArgs={"ExpectedBucketOwner": st["account"]})
    role_arn = role(s, st)
    hub = a.hub or (f"https://{st['cdn_domain']}" if st.get("cdn_domain") else "http://unset")
    args = dict(
        agentRuntimeArtifact={"codeConfiguration": {
            "code": {"s3": {"bucket": st["bucket"], "prefix": key}},
            "runtime": "PYTHON_3_13", "entryPoint": ["opentelemetry-instrument", "main.py"]}},
        networkConfiguration={"networkMode": "PUBLIC"},
        roleArn=role_arn,
        environmentVariables={"MOMBE_HUB": hub, "MOMBE_KEY": st["hub_key"], "MOMBE_MODEL_ID": a.model,
                              "AWS_REGION": st["region"]},
        lifecycleConfiguration={"idleRuntimeSessionTimeout": 900, "maxLifetime": 3600},
    )
    cc = s.client("bedrock-agentcore-control")
    for attempt in range(6):
        try:
            if st.get("agent_runtime_id"):
                r = cc.update_agent_runtime(agentRuntimeId=st["agent_runtime_id"], **args)
            else:
                r = cc.create_agent_runtime(agentRuntimeName=NAME, **args)
            break
        except ClientError as e:
            if "role" in str(e).lower() and attempt < 5:   # a fresh IAM role takes a few seconds to propagate
                time.sleep(10)
                continue
            raise
    st.update(agent_runtime_id=r["agentRuntimeId"], agent_arn=r["agentRuntimeArn"], agent_model=a.model, agent_hub=hub)
    save(st)
    say("agent runtime", r["agentRuntimeArn"], r["status"])


if __name__ == "__main__":
    main()
