"""Run a shell script on one or both hubs through SSM and print the output.

The script text comes from a local file (or -c). The hub key never reaches
stdout: every output is scrubbed of it before printing.

Usage:
  python tools/cool_equiv/ssm.py cool  -f job.sh [--wait 600]
  python tools/cool_equiv/ssm.py both  -c "uname -a"
  python tools/cool_equiv/ssm.py get <command-id>          # read a finished command again
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[2]
ST = json.loads((ROOT / "infra" / "state.json").read_text())
KEY = ST.get("hub_key", "")
ssm = boto3.client("ssm", region_name=ST.get("region", "eu-west-1"))


def scrub(s: str) -> str:
    return s.replace(KEY, "***") if KEY else s


def wait(cid: str, iid: str, name: str, limit: int) -> None:
    t0 = time.time()
    while True:
        try:
            r = ssm.get_command_invocation(CommandId=cid, InstanceId=iid)
        except ssm.exceptions.InvocationDoesNotExist:
            r = {"Status": "Pending"}
        if r["Status"] not in ("Pending", "InProgress", "Delayed") or time.time() - t0 > limit:
            break
        time.sleep(4)
    print(f"== {name} {r['Status']} ({cid})")
    print(scrub(r.get("StandardOutputContent") or ""))
    err = r.get("StandardErrorContent") or ""
    if err.strip():
        print("-- stderr --")
        print(scrub(err))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=["cool", "stock", "both", "get"])
    ap.add_argument("cid", nargs="?")
    ap.add_argument("-f", "--file")
    ap.add_argument("-c", "--cmd")
    ap.add_argument("--wait", type=int, default=900)
    ap.add_argument("--timeout", type=int, default=7200, help="SSM execution timeout, seconds")
    a = ap.parse_args()
    hubs = ST["hubs"]
    if a.target == "get":
        targets = ssm.list_commands(CommandId=a.cid)["Commands"][0]["InstanceIds"]
        for n, h in hubs.items():
            if h["instance_id"] in targets:
                wait(a.cid, h["instance_id"], n, a.wait)
        return 0
    text = Path(a.file).read_text() if a.file else a.cmd
    names = ["cool", "stock"] if a.target == "both" else [a.target]
    ids = [hubs[n]["instance_id"] for n in names]
    cid = ssm.send_command(InstanceIds=ids, DocumentName="AWS-RunShellScript", TimeoutSeconds=60,
                           Parameters={"commands": [text], "executionTimeout": [str(a.timeout)]})["Command"]["CommandId"]
    print("command", cid, "->", names, flush=True)
    for n, i in zip(names, ids):
        wait(cid, i, n, a.wait)
    return 0


if __name__ == "__main__":
    sys.exit(main())
