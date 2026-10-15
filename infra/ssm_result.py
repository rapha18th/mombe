"""Wait for an SSM command on every hub and print each one's output tail.

Usage: python infra/ssm_result.py <command-id>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import boto3

st = json.loads((Path(__file__).resolve().parent / "state.json").read_text())
ssm = boto3.client("ssm", region_name=st.get("region", "eu-west-1"))
cid = sys.argv[1]
for name, hub in st["hubs"].items():
    for _ in range(90):
        try:
            r = ssm.get_command_invocation(CommandId=cid, InstanceId=hub["instance_id"])
        except ssm.exceptions.InvocationDoesNotExist:
            r = {"Status": "Pending"}
        if r["Status"] not in ("Pending", "InProgress", "Delayed"):
            break
        time.sleep(5)
    out = (r.get("StandardOutputContent") or "")[-800:]
    err = (r.get("StandardErrorContent") or "")[-400:]
    # never echo the hub key
    key = st.get("hub_key", "")
    print(f"== {name} {r['Status']}\n{out.replace(key, '***') if key else out}")
    if err.strip():
        print("stderr:", err.replace(key, "***") if key else err)
