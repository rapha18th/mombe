"""Track 1d driver, run locally once AWS credentials are fresh (`aws login --region eu-west-1`).

  python tools/cool_equiv/run_1d.py check            credentials + hub states (read only)
  python tools/cool_equiv/run_1d.py all              prepare, upload, setup, launch job_all on every RUNNING hub
  python tools/cool_equiv/run_1d.py status           result keys in S3 + the tail of each hub's job log
  python tools/cool_equiv/run_1d.py fetch            download results/1d/ to eval/out/cool_equiv/raw/
  python tools/cool_equiv/run_1d.py summarize        same as tools/cool_equiv/summarize.py

Single steps: prepare | upload | setup HUB | launch HUB [JOB]. HUB is cool or stock.
It never starts, stops or changes an instance and never touches the hub service:
jobs run under /opt/cool_equiv with nice 10. A stopped stock hub is reported and skipped.
The hub key in infra/state.json is never printed.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SETS = ROOT / "eval" / "out" / "cool_equiv" / "sets"
RAW = ROOT / "eval" / "out" / "cool_equiv" / "raw"
BUCKET = "mombe-967170427068-eu-west-1"
ST = json.loads((ROOT / "infra" / "state.json").read_text())
KEY = ST.get("hub_key", "")
REGION = ST.get("region", "eu-west-1")
DATA = ["ops_images.zip", "ops_images.json", "ops_muzzle.zip", "probes.zip", "probes.json", "ess_fixed.npz"]


def scrub(s: str) -> str:
    return s.replace(KEY, "***") if KEY else s


def clients():
    import boto3
    return boto3.client("ssm", region_name=REGION), boto3.client("s3", region_name=REGION), boto3.client("ec2", region_name=REGION)


def iid(hub: str) -> str:
    return ST["hubs"][hub]["instance_id"]


def check() -> dict:
    import boto3
    who = boto3.client("sts", region_name=REGION).get_caller_identity()
    print("account", who["Account"])
    _, _, ec2 = clients()
    states = {}
    for hub in ("cool", "stock"):
        r = ec2.describe_instances(InstanceIds=[iid(hub)])["Reservations"][0]["Instances"][0]
        states[hub] = r["State"]["Name"]
        print(hub, iid(hub), states[hub], r.get("InstanceType"))
    return states


def prepare() -> None:
    """ops_muzzle.zip: the 150 muzzle images of the op set, so the hubs need nothing of their own."""
    out = SETS / "ops_muzzle.zip"
    if out.exists():
        print("ops_muzzle.zip present", out.stat().st_size)
        return
    ops = json.loads((SETS / "ops_images.json").read_text())
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for rel in ops:
            if rel.startswith("muzzle/"):
                z.write(ROOT / "data" / rel, rel)
    out.write_bytes(buf.getvalue())
    print("built ops_muzzle.zip", out.stat().st_size)


def upload() -> None:
    subprocess.run([sys.executable, str(HERE / "upload.py"), "--code"], check=True, cwd=ROOT)
    _, s3, _ = clients()
    for f in DATA:
        p = SETS / f
        key = f"jobs/1d/data/{f}"
        try:
            if s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"] == p.stat().st_size:
                print("data present", f)
                continue
        except Exception:
            pass
        s3.upload_file(str(p), BUCKET, key)
        print("uploaded", f, p.stat().st_size)


def ssm_run(hub: str, text: str, wait: int = 900, timeout: int = 3600) -> tuple[str, str]:
    ssm, _, _ = clients()
    cid = ssm.send_command(InstanceIds=[iid(hub)], DocumentName="AWS-RunShellScript", TimeoutSeconds=60,
                           Parameters={"commands": [text], "executionTimeout": [str(timeout)]})["Command"]["CommandId"]
    t0 = time.time()
    while True:
        time.sleep(5)
        try:
            r = ssm.get_command_invocation(CommandId=cid, InstanceId=iid(hub))
        except ssm.exceptions.InvocationDoesNotExist:
            continue
        if r["Status"] not in ("Pending", "InProgress", "Delayed") or time.time() - t0 > wait:
            break
    out = scrub(r.get("StandardOutputContent") or "")
    err = scrub(r.get("StandardErrorContent") or "")
    print(f"== {hub} {r['Status']} ({cid})\n{out}")
    if err.strip():
        print("-- stderr --\n" + err[-3000:])
    return r["Status"], out


def setup(hub: str) -> bool:
    st, out = ssm_run(hub, (HERE / "setup_hub.sh").read_text().replace("\r\n", "\n"), wait=1500, timeout=1800)
    return "SETUP_OK" in out


def launch(hub: str, job: str = "job_all") -> None:
    ssm_run(hub, f"JOB={job}\n" + (HERE / "launch.sh").read_text().replace("\r\n", "\n"), wait=120, timeout=300)


def status() -> None:
    _, s3, _ = clients()
    pag = s3.get_paginator("list_objects_v2")
    for page in pag.paginate(Bucket=BUCKET, Prefix="results/1d/"):
        for o in page.get("Contents", []):
            print(o["LastModified"].strftime("%m-%d %H:%M"), o["Size"], o["Key"][len("results/1d/"):])
    for hub, state in check().items():
        if state == "running":
            ssm_run(hub, "cd /opt/cool_equiv/out 2>/dev/null && for f in $(ls -t *.log | head -2); do echo \"-- $f\"; tail -8 $f; done; "
                         "pgrep -af 'ops_diff|identify_probes|bisect_take' | cut -c1-160", wait=60, timeout=120)


def fetch() -> None:
    _, s3, _ = clients()
    pag = s3.get_paginator("list_objects_v2")
    for page in pag.paginate(Bucket=BUCKET, Prefix="results/1d/"):
        for o in page.get("Contents", []):
            dst = RAW / o["Key"][len("results/1d/"):]
            if dst.exists() and dst.stat().st_size == o["Size"]:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(BUCKET, o["Key"], str(dst))
            print("got", dst.relative_to(ROOT), o["Size"])


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "check":
        check()
    elif cmd == "prepare":
        prepare()
    elif cmd == "upload":
        upload()
    elif cmd == "setup":
        setup(args[0])
    elif cmd == "launch":
        launch(args[0], *(args[1:2]))
    elif cmd == "status":
        status()
    elif cmd == "fetch":
        fetch()
    elif cmd == "summarize":
        subprocess.run([sys.executable, str(HERE / "summarize.py")], check=True, cwd=ROOT)
    elif cmd == "all":
        states = check()
        prepare()
        upload()
        for hub in ("cool", "stock"):
            if states.get(hub) != "running":
                print(f"SKIP {hub}: instance is {states.get(hub)}; this driver never starts instances")
                continue
            if setup(hub):
                launch(hub)
            else:
                print(f"SETUP FAILED on {hub}; nothing launched there")
        print("launched. Poll with: run_1d.py status ; then run_1d.py fetch && run_1d.py summarize")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
