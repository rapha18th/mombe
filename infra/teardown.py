"""Inventory, pause, resize and retire every Mombe resource on AWS.

  inventory    every Mombe resource found by the IDs in infra/state.json, the project tag and the
               exact names the deploy scripts use, with its state and an estimated daily cost
  stop         stop a hub; its EBS volume keeps billing and its public IPv4 is released
  start        start a hub, record its new public DNS and point CloudFront at it; on a running hub
               it only checks the CloudFront origin and repairs it
  stop-stock   stop --flavour stock
  start-stock  start --flavour stock
  scale        check the new type, stop the hub, change its type, start it and point CloudFront at
               its new DNS; a failed change puts the old type back and starts the hub again
  destroy      delete everything in order; without --yes it prints the plan only

Usage:
  python infra/teardown.py                          # same as inventory
  python infra/teardown.py stop-stock | start-stock
  python infra/teardown.py start --flavour cool     # also repairs a stale CloudFront origin
  python infra/teardown.py scale --to c8g.xlarge [--flavour cool] [--dry-run] [--force]
  python infra/teardown.py destroy                  # the ordered plan, read only
  python infra/teardown.py destroy --yes [--keep-bucket] [--images] [--orphan-volumes] [--account-wide] [--no-wait-cdn]

scale to the current type starts a stopped hub and resyncs CloudFront, so it also recovers an
interrupted scale.

destroy order: AgentCore runtimes and endpoints, CloudFront (disable, wait, delete), instances
(terminate, wait), leftover volumes, security group, instance profile and roles, S3 bucket with
every object version, log groups and log deliveries, SSM documents and parameters, ECR
repositories. Every step tolerates resources that are already gone, so a rerun finishes the job.
A section destroy cannot read keeps its state.json entries, and destroy then exits non-zero.
--orphan-volumes also deletes unattached volumes made from a hub AMI; the stock AMI is a public
Ubuntu image, so other workloads in the account can own such volumes too. --account-wide also
switches X-Ray Transaction Search back to X-Ray, deletes aws/spans and /aws/application-signals/data,
and deletes AgentCore's runtime identity service-linked role once no runtime in the region remains;
those are account settings shared with any other workload in the region.

Daily costs use eu-west-1 on-demand list prices checked on 2026-10-21. Credits, Marketplace fees
and Bedrock tokens appear in Cost Explorer: python infra/budget.py credits.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError, WaiterError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
from deploy import COOL_PRODUCT_CODE, NAME, REGION, load, save, say  # noqa: E402
from deploy_agent import NAME as AGENT_NAME  # noqa: E402

# eu-west-1 on-demand list prices in USD, checked 2026-10-21.
# EC2, EBS, S3, CloudWatch: AWS public price maps at b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps.
# COOL software: aws.amazon.com/marketplace/pp/prodview-fdvbfiewzuehs (7 day free trial). The listing
# shows 10 of its 30 rates; a COOL hub on a type missing here prices as unknown.
# Public IPv4: aws.amazon.com/vpc/pricing. CloudFront stays inside its always-free tier
# (1 TB and 10M requests a month); AgentCore Runtime bills only while a session is active.
HOURLY = {"c8g.medium": 0.04264, "c8g.large": 0.08528, "c8g.xlarge": 0.17056, "c8g.2xlarge": 0.34112,
          "c8g.4xlarge": 0.68224, "m8g.large": 0.10004, "m8g.xlarge": 0.20008,
          "c7g.xlarge": 0.155, "c7g.2xlarge": 0.3101, "t4g.medium": 0.0368, "t4g.large": 0.0736, "t4g.xlarge": 0.1472}
COOL_HOURLY = {"c8g.large": 0.01, "c8g.xlarge": 0.01, "c8g.2xlarge": 0.02, "c8g.4xlarge": 0.04, "c8g.8xlarge": 0.08,
               "m8g.4xlarge": 0.04, "r8g.xlarge": 0.01}
GRAVITON4 = ("c8g.", "m8g.", "r8g.")
IPV4_HOURLY = 0.005
GB_MONTH = {"gp3": 0.088, "snapshot": 0.05, "s3": 0.023, "logs": 0.03}
DAYS = 730 / 24                     # AWS prices a month as 730 hours
ROLES = (f"{NAME}-hub", f"{NAME}-agent-runtime")
ORIGIN_ID = "hub"                   # deploy.py names the CloudFront origin "hub" and points it at the primary hub
FINAL_IMAGES = (f"{NAME}-cool-final", f"{NAME}-stock-final")   # the archive images the runbook makes
TRACE_LOGS = ("aws/spans", "/aws/application-signals/data")     # Transaction Search writes spans here
# Account-wide service-linked roles. AgentCore creates the first with any runtime made after 2025-10-13;
# the console's Transaction Search setup creates the second. Both are free.
RUNTIME_SLR = "AWSServiceRoleForBedrockAgentCoreRuntimeIdentity"
SIGNALS_SLR = "AWSServiceRoleForCloudWatchApplicationSignals"
GONE = {"ResourceNotFoundException", "NoSuchEntity", "NoSuchBucket", "NoSuchDistribution",
        "InvalidGroup.NotFound", "InvalidVolume.NotFound", "InvalidInstanceID.NotFound",
        "InvalidAMIID.NotFound", "InvalidAMIID.Unavailable", "InvalidSnapshot.NotFound",
        "RepositoryNotFoundException", "InvalidDocument", "ParameterNotFound"}
EXPIRED = {"ExpiredToken", "ExpiredTokenException", "InvalidClientTokenId", "UnrecognizedClientException",
           "RequestExpired", "InvalidToken"}
# The state.json keys each discovery section accounts for. A section that fails to read keeps its keys,
# so destroy never records a resource it could not see as destroyed.
STATE_KEYS = {"runtimes": ("agent_runtime_id", "agent_arn", "agent_hub", "agent_model"), "cdn": ("cdn_id", "cdn_domain"),
              "instances": ("hubs", "ami"), "groups": ("sg",), "iam": ("instance_profile",), "buckets": ("bucket",)}


def code(e: Exception) -> str:
    r = getattr(e, "response", None) or getattr(e, "last_response", None) or {}
    return r.get("Error", {}).get("Code", "")


def usd(x: float | None) -> str:
    return "price ?" if x is None else f"${x:.2f}/day"


def gb_day(gb: float, kind: str) -> float:
    return gb * GB_MONTH[kind] / DAYS


class Teardown:
    def __init__(self, region=REGION, dry=False):
        self.s = boto3.Session(region_name=region)
        self.region = region
        self.dry = dry
        self.acct = self.s.client("sts").get_caller_identity()["Account"]
        self.st = load()
        self.ec2 = self.s.client("ec2")
        self.cf = self.s.client("cloudfront")
        self.iam = self.s.client("iam")
        self.cc = self.s.client("bedrock-agentcore-control")
        self.logs = self.s.client("logs")

    def known(self, key: str) -> set:
        """A state.json ID, live or moved under destroyed by an earlier destroy pass."""
        return {v for v in (self.st.get(key), self.st.get("destroyed", {}).get(key)) if isinstance(v, str) and v}

    def hub_records(self) -> dict:
        """Instance ID to flavour for every hub state.json knows, destroyed ones included."""
        recs = {}
        for hubs in (self.st.get("destroyed", {}).get("hubs", {}), self.st.get("hubs", {})):
            recs.update({h["instance_id"]: fl for fl, h in hubs.items()})
        return recs

    # -- discovery -------------------------------------------------------------
    def instances(self) -> list[dict]:
        """Hubs by the IDs in state.json and by the project tag."""
        known = self.hub_records()
        raw = {}
        live = ["pending", "running", "stopping", "stopped", "shutting-down"]
        for page in self.ec2.get_paginator("describe_instances").paginate(Filters=[
                {"Name": "tag:project", "Values": [NAME]}, {"Name": "instance-state-name", "Values": live}]):
            for r in page["Reservations"]:
                for i in r["Instances"]:
                    raw[i["InstanceId"]] = i
        for iid in known:
            if iid in raw:
                continue
            try:
                i = self.ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]
                if i["State"]["Name"] != "terminated":
                    raw[iid] = i
            except (ClientError, IndexError):
                pass
        out = []
        for iid, i in raw.items():
            tags = {t["Key"]: t["Value"] for t in i.get("Tags", [])}
            out.append({
                "id": iid, "flavour": known.get(iid) or tags.get("Name", "?").rsplit("-", 1)[-1],
                "type": i["InstanceType"], "state": i["State"]["Name"], "ami": i.get("ImageId"),
                "dns": i.get("PublicDnsName") or "", "ip": i.get("PublicIpAddress"),
                "cool": known.get(iid) == "cool" or any(p.get("ProductCodeId") == COOL_PRODUCT_CODE
                                                       for p in i.get("ProductCodes", [])),
                "volumes": {m["Ebs"]["VolumeId"]: m["Ebs"].get("DeleteOnTermination", False)
                            for m in i.get("BlockDeviceMappings", []) if "Ebs" in m}})
        return out

    def volumes(self, insts: list[dict]) -> list[dict]:
        """Volumes attached to the hubs, plus any tagged for the project."""
        dot = {v: d for i in insts for v, d in i["volumes"].items()}
        found = {}
        for vid in dot:
            try:
                for v in self.ec2.describe_volumes(VolumeIds=[vid])["Volumes"]:
                    found[v["VolumeId"]] = v
            except ClientError as e:
                if code(e) not in GONE:
                    raise
        for page in self.ec2.get_paginator("describe_volumes").paginate(Filters=[{"Name": "tag:project", "Values": [NAME]}]):
            for v in page["Volumes"]:
                found[v["VolumeId"]] = v
        return [{"id": v["VolumeId"], "size": v["Size"], "type": v["VolumeType"], "state": v["State"],
                 "instance": (v.get("Attachments") or [{}])[0].get("InstanceId"),
                 "delete_on_termination": dot.get(v["VolumeId"], False)} for v in found.values()]

    def orphans(self, insts: list[dict], attached: list[dict]) -> list[dict]:
        """Unattached volumes made from a hub AMI's snapshots.

        deploy.py tags instances only, so a root volume that outlived an earlier hub carries no tag.
        """
        amis = {i["ami"] for i in insts} | {h.get("ami") for h in self.st.get("hubs", {}).values()}
        snaps = set()
        for ami in sorted(a for a in amis if a):
            try:
                for im in self.ec2.describe_images(ImageIds=[ami])["Images"]:
                    snaps |= {b["Ebs"]["SnapshotId"] for b in im.get("BlockDeviceMappings", [])
                              if b.get("Ebs", {}).get("SnapshotId")}
            except ClientError as e:
                if code(e) not in GONE:
                    raise
        if not snaps:
            return []
        seen = {v["id"] for v in attached}
        out = []
        for page in self.ec2.get_paginator("describe_volumes").paginate(Filters=[
                {"Name": "snapshot-id", "Values": sorted(snaps)}, {"Name": "status", "Values": ["available"]}]):
            for v in page["Volumes"]:
                if v["VolumeId"] not in seen:
                    out.append({"id": v["VolumeId"], "size": v["Size"], "type": v["VolumeType"],
                                "snapshot": v.get("SnapshotId"), "created": f"{v['CreateTime']:%Y-%m-%d}"
                                if v.get("CreateTime") else "?"})
        return out

    def images(self) -> list[dict]:
        """AMIs this account owns with the project tag or a runbook archive name."""
        found = {}
        for flt in ([{"Name": "tag:project", "Values": [NAME]}], [{"Name": "name", "Values": list(FINAL_IMAGES)}]):
            for im in self.ec2.describe_images(Owners=["self"], Filters=flt)["Images"]:
                found[im["ImageId"]] = im
        return [{"id": im["ImageId"], "name": im.get("Name", ""),
                 "snapshots": [(b["Ebs"]["SnapshotId"], b["Ebs"].get("VolumeSize", 0))
                               for b in im.get("BlockDeviceMappings", []) if b.get("Ebs", {}).get("SnapshotId")]}
                for im in found.values()]

    def groups(self) -> list[dict]:
        found = {}
        for sg in self.known("sg"):
            try:
                for g in self.ec2.describe_security_groups(GroupIds=[sg])["SecurityGroups"]:
                    found[g["GroupId"]] = g
            except ClientError as e:
                if code(e) not in GONE:
                    raise
        for g in self.ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [f"{NAME}-hub"]}])["SecurityGroups"]:
            found[g["GroupId"]] = g
        return [{"id": g["GroupId"], "name": g["GroupName"], "vpc": g.get("VpcId")} for g in found.values()]

    def identities(self) -> dict:
        """The two roles and the instance profile the deploy scripts create, by exact name."""
        out = []
        for rn in ROLES:
            try:
                self.iam.get_role(RoleName=rn)
            except ClientError as e:
                if code(e) in GONE:
                    continue
                raise
            att = [p["PolicyArn"] for p in self.iam.list_attached_role_policies(RoleName=rn)["AttachedPolicies"]]
            inl = self.iam.list_role_policies(RoleName=rn)["PolicyNames"]
            out.append({"name": rn, "attached": att, "inline": inl})
        profiles = {}
        for pn in sorted({f"{NAME}-hub"} | self.known("instance_profile")):
            try:
                p = self.iam.get_instance_profile(InstanceProfileName=pn)["InstanceProfile"]
            except ClientError as e:
                if code(e) in GONE:
                    continue
                raise
            profiles[pn] = [r["RoleName"] for r in p.get("Roles", [])]
        return {"roles": out, "profiles": profiles}

    def buckets(self) -> list[dict]:
        """The project bucket: the name in state.json and the name deploy.py derives."""
        s3 = self.s.client("s3")
        names = self.known("bucket") | {f"{NAME}-{self.acct}-{self.region}"}
        out = []
        for b in sorted(names):
            try:
                s3.head_bucket(Bucket=b, ExpectedBucketOwner=self.acct)
            except ClientError:
                continue
            n = markers = size = 0
            for page in s3.get_paginator("list_object_versions").paginate(Bucket=b, ExpectedBucketOwner=self.acct):
                for v in page.get("Versions", []):
                    n += 1
                    size += v.get("Size", 0)
                markers += len(page.get("DeleteMarkers", []))
            ver = s3.get_bucket_versioning(Bucket=b, ExpectedBucketOwner=self.acct).get("Status", "Off")
            out.append({"name": b, "versions": n, "markers": markers, "bytes": size, "versioning": ver})
        return out

    def distributions(self, insts: list[dict]) -> list[dict]:
        dns = {i["dns"] for i in insts if i["dns"]} | {h.get("public_dns") for h in self.st.get("hubs", {}).values()}
        ids = self.known("cdn_id")
        out = []
        for page in self.cf.get_paginator("list_distributions").paginate():
            for d in page.get("DistributionList", {}).get("Items", []):
                origins = [o["DomainName"] for o in d.get("Origins", {}).get("Items", [])]
                if d["Id"] in ids or d.get("Comment") == "Mombe hub" or dns & set(origins):
                    out.append({"id": d["Id"], "domain": d["DomainName"], "enabled": d["Enabled"],
                                "status": d["Status"], "origins": origins})
        return out

    def runtimes(self) -> list[dict]:
        """The AgentCore runtime named by deploy_agent.py or recorded in state.json, its endpoints and identity.

        GetAgentRuntime returns the environment, which holds the hub key; only named fields are kept.
        """
        ids = self.known("agent_runtime_id")
        rts = [x for p in self.cc.get_paginator("list_agent_runtimes").paginate() for x in p["agentRuntimes"]
               if x["agentRuntimeName"] == AGENT_NAME or x["agentRuntimeId"] in ids]
        out = []
        for x in rts:
            g = self.cc.get_agent_runtime(agentRuntimeId=x["agentRuntimeId"])
            eps = [e for p in self.cc.get_paginator("list_agent_runtime_endpoints").paginate(agentRuntimeId=x["agentRuntimeId"])
                   for e in p["runtimeEndpoints"]]
            out.append({"id": x["agentRuntimeId"], "arn": x["agentRuntimeArn"], "name": x["agentRuntimeName"],
                        "status": g["status"], "version": g["agentRuntimeVersion"], "role": g["roleArn"],
                        "identity": (g.get("workloadIdentityDetails") or {}).get("workloadIdentityArn"),
                        "endpoints": [{"name": e["name"], "status": e["status"]} for e in eps]})
        return out

    def runtime_ids(self, rts: list[dict]) -> set:
        return {r["id"] for r in rts} | self.known("agent_runtime_id")

    def log_groups(self, rts: list[dict]) -> list[dict]:
        """Log groups whose name holds a Mombe runtime ID: the runtime logs and any vended logs."""
        found = {}
        for rid in sorted(self.runtime_ids(rts)):
            for page in self.logs.get_paginator("describe_log_groups").paginate(logGroupNamePattern=rid):
                for g in page["logGroups"]:
                    found[g["logGroupName"]] = g
        return [{"name": g["logGroupName"], "bytes": g.get("storedBytes", 0), "retention": g.get("retentionInDays")}
                for g in found.values()]

    def deliveries(self, rts: list[dict]) -> dict:
        """CloudWatch log delivery sources tied to the runtime, their deliveries and the destinations only they use."""
        ids = self.runtime_ids(rts)

        def ours(name, res=()):
            text = " ".join([name or "", *res])
            return any(i in text for i in ids) or f":runtime/{AGENT_NAME}-" in text
        src = [s["name"] for p in self.logs.get_paginator("describe_delivery_sources").paginate()
               for s in p.get("deliverySources", []) if ours(s.get("name"), s.get("resourceArns", []))]
        alld = [d for p in self.logs.get_paginator("describe_deliveries").paginate() for d in p.get("deliveries", [])]
        mine = [d for d in alld if d.get("deliverySourceName") in src]
        shared = {d.get("deliveryDestinationArn") for d in alld if d.get("deliverySourceName") not in src}
        want = {d.get("deliveryDestinationArn") for d in mine} - shared
        dst = [d["name"] for p in self.logs.get_paginator("describe_delivery_destinations").paginate()
               for d in p.get("deliveryDestinations", [])
               if d.get("arn") in want or (ours(d.get("name")) and d.get("arn") not in shared)]
        return {"deliveries": [d["id"] for d in mine], "sources": src, "destinations": dst}

    def ssm_items(self) -> dict:
        """SSM items in the project namespace: parameters under /mombe/, documents named mombe*. None exist today."""
        ssm = self.s.client("ssm")
        docs = [d["Name"] for p in ssm.get_paginator("list_documents").paginate(Filters=[{"Key": "Owner", "Values": ["Self"]}])
                for d in p["DocumentIdentifiers"] if d["Name"].lower().startswith(NAME)]
        params = [x["Name"] for p in ssm.get_paginator("describe_parameters").paginate(
                  ParameterFilters=[{"Key": "Name", "Option": "BeginsWith", "Values": [f"/{NAME}/"]}]) for x in p["Parameters"]]
        return {"documents": docs, "parameters": params}

    def repositories(self) -> list[str]:
        """ECR repositories named mombe*. None exist today: deploy_agent.py deploys code from S3."""
        ecr = self.s.client("ecr")
        return [r["repositoryName"] for p in ecr.get_paginator("describe_repositories").paginate()
                for r in p["repositories"] if r["repositoryName"].lower().startswith(NAME)]

    def tracing(self) -> dict:
        """Account settings that AgentCore Observability relies on; destroy touches them only with --account-wide."""
        d = self.s.client("xray").get_trace_segment_destination()
        stored = {}
        for name in TRACE_LOGS:
            for g in self.logs.describe_log_groups(logGroupNamePrefix=name).get("logGroups", []):
                if g["logGroupName"] == name:
                    stored[name] = g.get("storedBytes", 0)
        pols = [p["policyName"] for p in self.logs.describe_resource_policies().get("resourcePolicies", [])
                if "aws/spans" in p.get("policyDocument", "")]
        roles = []
        for rn in (RUNTIME_SLR, SIGNALS_SLR):
            try:
                self.iam.get_role(RoleName=rn)
                roles.append(rn)
            except ClientError as e:
                if code(e) not in GONE:
                    raise
        return {"destination": d.get("Destination"), "status": d.get("Status"), "spans_bytes": stored.get("aws/spans"),
                "logs": stored, "policies": pols, "roles": roles}

    def find(self) -> dict:
        """Every Mombe resource, section by section; f["unread"] maps each section this caller cannot read to its error."""
        f = {"unread": {}}

        def grab(key, fn, default):
            try:
                f[key] = fn()
            except ClientError as e:
                if code(e) in EXPIRED:
                    raise
                f[key] = default
                f["unread"][key] = code(e) or type(e).__name__
        grab("instances", self.instances, [])
        grab("volumes", lambda: self.volumes(f["instances"]), [])
        grab("orphans", lambda: self.orphans(f["instances"], f["volumes"]), [])
        grab("images", self.images, [])
        grab("groups", self.groups, [])
        grab("iam", self.identities, {"roles": [], "profiles": {}})
        grab("buckets", self.buckets, [])
        grab("cdn", lambda: self.distributions(f["instances"]), [])
        grab("runtimes", self.runtimes, [])
        grab("log_groups", lambda: self.log_groups(f["runtimes"]), [])
        grab("deliveries", lambda: self.deliveries(f["runtimes"]), {"deliveries": [], "sources": [], "destinations": []})
        grab("ssm", self.ssm_items, {"documents": [], "parameters": []})
        grab("ecr", self.repositories, [])
        grab("tracing", self.tracing, {})
        return f

    # -- costs -----------------------------------------------------------------
    def instance_day(self, i: dict) -> float | None:
        if i["state"] not in ("running", "pending"):
            return 0.0
        hr = HOURLY.get(i["type"])
        sw = COOL_HOURLY.get(i["type"]) if i["cool"] else 0.0
        if hr is None or sw is None:
            return None
        return (hr + sw + (IPV4_HOURLY if i["ip"] else 0.0)) * 24

    # -- inventory -------------------------------------------------------------
    def inventory(self, f: dict | None = None) -> dict:
        f = f or self.find()
        origins = {o for d in f["cdn"] for o in d["origins"]}
        total, unknown = 0.0, []
        say("account", self.acct, "region", self.region)
        say("EC2 hubs")
        for i in f["instances"]:
            c = self.instance_day(i)
            total += c or 0.0
            if c is None:
                unknown.append(i["id"])
            tag = "  CloudFront origin" if i["dns"] and i["dns"] in origins else ""
            sw = " + COOL software" if i["cool"] and i["state"] == "running" else ""
            say(f"  {i['id']}  {i['flavour']:5}  {i['type']:12} {i['state']:9} {i['ami']}  {usd(c)}{sw}{tag}")
        say("EBS volumes")
        for v in f["volumes"]:
            c = gb_day(v["size"], "gp3")
            total += c
            dot = "deletes with instance" if v["delete_on_termination"] else "survives termination"
            say(f"  {v['id']}  {v['size']} GB {v['type']}  {v['state']}  {v['instance'] or '-'}  {dot}  {usd(c)}")
        if f["orphans"]:
            say("  unattached volumes from a hub AMI (possible leftovers; destroy deletes them only with --orphan-volumes)")
        for v in f["orphans"]:
            c = gb_day(v["size"], "gp3")
            total += c
            say(f"  {v['id']}  {v['size']} GB {v['type']}  available  from {v['snapshot']}  created {v['created']}  {usd(c)}")
        if f["images"]:
            say("AMIs and snapshots")
        for im in f["images"]:
            gb = sum(s for _, s in im["snapshots"])
            total += gb_day(gb, "snapshot")
            say(f"  AMI {im['id']} {im['name']}  snapshots {[s for s, _ in im['snapshots']]}  <= {usd(gb_day(gb, 'snapshot'))}")
        say("security groups", [g["id"] + " " + g["name"] for g in f["groups"]] or "none")
        say("IAM")
        for r in f["iam"]["roles"]:
            say(f"  role {r['name']}  managed {[a.rsplit('/', 1)[-1] for a in r['attached']]}  inline {r['inline']}")
        for p, rs in f["iam"]["profiles"].items():
            say(f"  instance profile {p}  roles {rs}")
        say("S3")
        for b in f["buckets"]:
            c = gb_day(b["bytes"] / 1e9, "s3")
            total += c
            say(f"  {b['name']}  {b['versions']} object versions, {b['markers']} delete markers, "
                f"{b['bytes'] / 1e9:.2f} GB, versioning {b['versioning']}  {usd(c)}")
        say("CloudFront")
        for d in f["cdn"]:
            on = "enabled" if d["enabled"] else "disabled"
            say(f"  {d['id']}  {d['domain']}  {on} {d['status']}  origin {d['origins']}  $0.00/day inside the free tier")
        live = {i["dns"] for i in f["instances"] if i["dns"]}
        for d in f["cdn"]:
            dead = [o for o in d["origins"] if o.endswith(".compute.amazonaws.com") and o not in live]
            if dead:
                say(f"  WARNING: origin {dead} matches no running hub; run: "
                    f"python infra/teardown.py start --flavour {self.primary()}")
        say("AgentCore")
        for r in f["runtimes"]:
            say(f"  runtime {r['id']}  {r['status']} v{r['version']}  role {r['role'].rsplit('/', 1)[-1]}  $0.00/day idle")
            for e in r["endpoints"]:
                say(f"    endpoint {e['name']}  {e['status']}")
            if r["identity"]:
                say(f"    workload identity {r['identity'].rsplit('/', 1)[-1]}")
        say("CloudWatch Logs")
        for g in f["log_groups"]:
            c = gb_day(g["bytes"] / 1e9, "logs")
            total += c
            say(f"  {g['name']}  {g['bytes'] / 1e6:.1f} MB  retention {g['retention'] or 'forever'}  {usd(c)}")
        dl = f["deliveries"]
        if any(dl.values()):
            say(f"  log deliveries {dl['deliveries']}  sources {dl['sources']}  destinations {dl['destinations']}")
        t = f.get("tracing") or {}
        if t:
            sb = "absent" if t["spans_bytes"] is None else f"{t['spans_bytes'] / 1e6:.1f} MB"
            say(f"  account: Transaction Search destination {t['destination']} ({t['status']}), aws/spans {sb}, "
                f"logs resource policies {t['policies']}")
            for name, b in t.get("logs", {}).items():
                total += gb_day(b / 1e9, "logs")
                if name != "aws/spans":
                    say(f"  account: {name}  {b / 1e6:.1f} MB")
            for rn in t.get("roles", []):
                say(f"  account: service-linked role {rn}, free" +
                    ("; destroy --account-wide removes it" if rn == RUNTIME_SLR else "; destroy leaves it"))
        if f["ssm"]["documents"] or f["ssm"]["parameters"]:
            say("SSM", f["ssm"])
        if f["ecr"]:
            say("ECR", f["ecr"])
        unread = f.get("unread", {})
        recorded = {"cdn": ("cdn_id", {d["id"] for d in f["cdn"]}),
                    "runtimes": ("agent_runtime_id", {r["id"] for r in f["runtimes"]}),
                    "groups": ("sg", {g["id"] for g in f["groups"]}), "buckets": ("bucket", {b["name"] for b in f["buckets"]})}
        for sec, (k, ids) in recorded.items():
            if self.st.get(k) and self.st[k] not in ids and sec not in unread:
                say(f"state.json {k} {self.st[k]} is gone on AWS")
        for sec, why in unread.items():
            say("could not read", sec, f"({why}); its resources are missing from this list")
        if unknown:
            say("no list price on file for", unknown, "; the total leaves them out")
        say(f"estimated fixed cost {usd(total)} at list price, before credits; Bedrock tokens and "
            "AgentCore sessions add per use")
        say("Marketplace subscriptions (COOL AMI, Claude) stay after destroy and bill only with use")
        return f

    # -- stop, start, scale ----------------------------------------------------
    def hub(self, flavour: str) -> dict:
        h = self.st.get("hubs", {}).get(flavour)
        if not h:
            sys.exit(f"no {flavour} hub in infra/state.json")
        return h

    def primary(self) -> str:
        """The flavour deploy.py puts behind CloudFront."""
        return "cool" if "cool" in self.st.get("hubs", {}) else "stock"

    def describe(self, iid: str) -> dict:
        return self.ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]

    def cdn_origins(self) -> list[str]:
        if not self.st.get("cdn_id"):
            return []
        cfg = self.cf.get_distribution_config(Id=self.st["cdn_id"])["DistributionConfig"]
        return [o["DomainName"] for o in cfg["Origins"]["Items"]]

    def other_dns(self, flavour: str) -> set:
        """Stored and live public DNS of the other hubs, which an origin repair must leave alone."""
        out = set()
        for fl, h in self.st.get("hubs", {}).items():
            if fl == flavour:
                continue
            out.add(h.get("public_dns"))
            try:
                out.add(self.describe(h["instance_id"]).get("PublicDnsName"))
            except (ClientError, IndexError):
                pass
        return {d for d in out if d}

    def sync_origin(self, flavour: str, olds: set, new: str):
        """Point the CloudFront origin that served this hub at its current public DNS.

        An origin matches by the stored or the live DNS from before the change. For the primary hub,
        deploy.py's origin "hub" also matches when its DNS belongs to no other hub. Exits non-zero
        when the primary hub's origin stays behind.
        """
        cid = self.st.get("cdn_id")
        if not cid or not new:
            return
        c = self.cf.get_distribution_config(Id=cid)
        cfg, etag = c["DistributionConfig"], c["ETag"]
        items = cfg["Origins"]["Items"]
        if any(o["DomainName"] == new for o in items):
            return say("CloudFront origin already", new)
        olds = {d for d in olds if d and d != new}
        hit = [o for o in items if o["DomainName"] in olds]
        primary = flavour == self.primary()
        if not hit and primary:
            others = self.other_dns(flavour)
            hit = [o for o in items if o["Id"] == ORIGIN_ID and o["DomainName"].endswith(".compute.amazonaws.com")
                   and o["DomainName"] not in others]
            for o in hit:
                say("CloudFront origin", o["DomainName"], "matches neither the stored nor the live DNS;",
                    f"it is deploy.py's '{ORIGIN_ID}' origin, so it follows the {flavour} hub")
        if not hit:
            if primary:
                sys.exit(f"WARNING: CloudFront {cid} still points at {[o['DomainName'] for o in items]}; the {flavour} "
                         f"hub answers at {new}. Set the origin in the CloudFront console, then check /health")
            return
        for o in hit:
            o["DomainName"] = new
        self.cf.update_distribution(Id=cid, IfMatch=etag, DistributionConfig=cfg)
        say("CloudFront origin now", new, "(propagates in about 5 min)")

    def stop(self, flavour: str, force=False):
        h = self.hub(flavour)
        i = self.describe(h["instance_id"])
        state = i["State"]["Name"]
        if state in ("stopped", "stopping"):
            return say(flavour, "hub already", state)
        if i.get("PublicDnsName") in self.cdn_origins() and not force:
            sys.exit(f"the {flavour} hub serves CloudFront; stopping it takes the app offline (--force to stop it)")
        ids = [m["Ebs"]["VolumeId"] for m in i.get("BlockDeviceMappings", []) if "Ebs" in m]
        gb = sum(v["Size"] for v in self.ec2.describe_volumes(VolumeIds=ids)["Volumes"]) if ids else 0
        if self.dry:
            return say("would stop", flavour, "hub", h["instance_id"], f"({gb} GB EBS keeps billing {usd(gb_day(gb, 'gp3'))})")
        self.ec2.stop_instances(InstanceIds=[h["instance_id"]])
        say("stopping", flavour, "hub", h["instance_id"])
        self.ec2.get_waiter("instance_stopped").wait(InstanceIds=[h["instance_id"]])
        say(flavour, "hub stopped;", gb, "GB EBS keeps billing", usd(gb_day(gb, "gp3")))

    def start(self, flavour: str):
        """Start a hub and point CloudFront at it; on a running hub, check and repair the origin only."""
        h = self.hub(flavour)
        iid = h["instance_id"]
        i = self.describe(iid)
        state, live = i["State"]["Name"], i.get("PublicDnsName", "")
        olds = {h.get("public_dns", ""), live}
        if state in ("running", "pending"):
            say(flavour, "hub already", state, "at", live)
            if not self.dry:
                if live and live != h.get("public_dns"):
                    h["public_dns"] = live
                    save(self.st)
                self.sync_origin(flavour, olds, live)
            return
        if self.dry:
            return say("would start", flavour, "hub", iid, i["InstanceType"])
        if state == "stopping":
            self.ec2.get_waiter("instance_stopped").wait(InstanceIds=[iid])
        self.ec2.start_instances(InstanceIds=[iid])
        say("starting", flavour, "hub", iid)
        self.ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
        dns = self.describe(iid).get("PublicDnsName", "")
        h["public_dns"] = dns
        save(self.st)
        say(flavour, "hub running at", dns, "; the hub service starts on boot")
        self.sync_origin(flavour, olds, dns)

    def check_type(self, i: dict, itype: str) -> list[str]:
        """Read-only checks that the hub can run as itype: offered in its zone, same CPU architecture."""
        az = i["Placement"]["AvailabilityZone"]
        try:
            offers = self.ec2.describe_instance_type_offerings(LocationType="availability-zone", Filters=[
                {"Name": "location", "Values": [az]}, {"Name": "instance-type", "Values": [itype]}])["InstanceTypeOfferings"]
            if not offers:
                return [f"EC2 offers no {itype} in {az}; check the spelling"]
            t = self.ec2.describe_instance_types(InstanceTypes=[itype])["InstanceTypes"][0]
        except ClientError as e:
            if code(e) in EXPIRED:
                raise
            return [f"could not check {itype} ({code(e)})"]
        arch = i.get("Architecture")
        if arch and arch not in t["ProcessorInfo"]["SupportedArchitectures"]:
            return [f"{itype} runs {t['ProcessorInfo']['SupportedArchitectures']}; the hub AMI is {arch}"]
        return []

    def scale(self, flavour: str, itype: str, force=False):
        h = self.hub(flavour)
        iid = h["instance_id"]
        i = self.describe(iid)
        cur, state = i["InstanceType"], i["State"]["Name"]
        if cur == itype:
            say(flavour, "hub is already", itype)
            return self.start(flavour)
        if flavour == "cool" and not itype.startswith(GRAVITON4) and not force:
            sys.exit("the COOL AMI targets Graviton4 (c8g, m8g, r8g); --force to use another family")
        probs = self.check_type(i, itype)
        for p in probs:
            say("check:", p)
        if probs and not force:
            sys.exit(f"scale refused before any change; the {flavour} hub stays {state} as {cur} (--force to try anyway)")
        cool = flavour == "cool"
        if itype not in HOURLY or (cool and itype not in COOL_HOURLY):
            say("no list price on file for", itype, "; check it before relying on the estimate")
        before = self.instance_day({"state": "running", "type": cur, "cool": cool, "ip": True})
        after = self.instance_day({"state": "running", "type": itype, "cool": cool, "ip": True})
        say(f"scale {flavour} hub {iid}: {cur} -> {itype}, {usd(before)} -> {usd(after)} while running; "
            "a few minutes offline; CloudFront follows the new public DNS")
        if self.dry:
            return
        olds = {h.get("public_dns", ""), i.get("PublicDnsName", "")}
        was_up = state in ("running", "pending")
        if state != "stopped":
            self.ec2.stop_instances(InstanceIds=[iid])
            self.ec2.get_waiter("instance_stopped").wait(InstanceIds=[iid])
            say("stopped", iid)
        try:
            self.ec2.modify_instance_attribute(InstanceId=iid, InstanceType={"Value": itype})
        except ClientError as e:
            if code(e) in EXPIRED:
                raise
            return self.recover(flavour, cur, olds, was_up, f"EC2 refused {itype} ({code(e)})")
        h["instance_type"] = itype
        save(self.st)
        say("instance type now", itype)
        if not was_up:
            return say(flavour, "hub was stopped before; it stays stopped")
        try:
            self.ec2.start_instances(InstanceIds=[iid])
            self.ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
        except (ClientError, WaiterError) as e:
            if code(e) in EXPIRED:
                raise
            return self.recover(flavour, cur, olds, was_up, f"start as {itype} failed ({code(e) or e})")
        dns = self.describe(iid).get("PublicDnsName", "")
        h["public_dns"] = dns
        save(self.st)
        say(flavour, "hub running as", itype, "at", dns)
        self.sync_origin(flavour, olds, dns)
        if self.st.get("cdn_domain"):
            say(f"check: curl -s https://{self.st['cdn_domain']}/health")

    def recover(self, flavour: str, cur: str, olds: set, was_up: bool, why: str):
        """After a failed scale: put the old type back, start the hub if it ran before, resync CloudFront, exit non-zero."""
        h = self.hub(flavour)
        iid = h["instance_id"]
        say(f"scale failed: {why}; restoring {cur}")
        try:
            i = self.describe(iid)
            if i["State"]["Name"] in ("pending", "stopping"):
                self.ec2.get_waiter("instance_stopped" if i["State"]["Name"] == "stopping" else "instance_running").wait(
                    InstanceIds=[iid])
                i = self.describe(iid)
            if i["State"]["Name"] == "stopped":
                if i["InstanceType"] != cur:
                    self.ec2.modify_instance_attribute(InstanceId=iid, InstanceType={"Value": cur})
                    say("instance type back to", cur)
                if was_up:
                    self.ec2.start_instances(InstanceIds=[iid])
                    self.ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
            i = self.describe(iid)
        except (ClientError, WaiterError) as e:
            sys.exit(f"recovery failed too ({code(e) or e}). The {flavour} hub {iid} may be stopped. When EC2 "
                     f"allows it: python infra/teardown.py scale --to {cur} --flavour {flavour}")
        h["instance_type"] = i["InstanceType"]
        h["public_dns"] = i.get("PublicDnsName", "") or h.get("public_dns", "")
        save(self.st)
        say(flavour, "hub", i["State"]["Name"], "as", i["InstanceType"], "at", i.get("PublicDnsName") or "-")
        if i["State"]["Name"] == "running":
            self.sync_origin(flavour, olds, i.get("PublicDnsName", ""))
        sys.exit(f"scale failed ({why}); the {flavour} hub is {i['State']['Name']} as {i['InstanceType']}")

    # -- destroy ---------------------------------------------------------------
    def plan(self, f: dict, a) -> None:
        say("destroy plan for account", self.acct, "region", self.region)
        steps = [
            ("AgentCore runtimes, endpoints, workload identities, log deliveries",
             [r["id"] for r in f["runtimes"]] + f["deliveries"]["sources"]),
            ("CloudFront: disable, wait, delete", [d["id"] + " " + d["domain"] for d in f["cdn"]]),
            ("EC2: terminate and wait", [i["id"] + " " + i["flavour"] for i in f["instances"]]),
            ("EBS volumes left after termination", [v["id"] for v in f["volumes"]]),
            ("unattached volumes from a hub AMI" + ("" if a.orphan_volumes else " (kept: pass --orphan-volumes)"),
             [f"{v['id']} ({v['size']} GB, created {v['created']})" for v in f["orphans"]]),
            ("security groups", [g["id"] for g in f["groups"]]),
            ("IAM instance profiles and roles",
             [f"profile {p}" for p in f["iam"]["profiles"]] + [f"role {r['name']}" for r in f["iam"]["roles"]]),
            ("S3 buckets with every object version" + (" (kept: --keep-bucket)" if a.keep_bucket else ""),
             [f"{b['name']} ({b['versions']} versions)" for b in f["buckets"]]),
            ("CloudWatch log groups", [g["name"] for g in f["log_groups"]]),
            ("SSM documents and parameters", f["ssm"]["documents"] + f["ssm"]["parameters"]),
            ("ECR repositories", f["ecr"]),
            ("AMIs and snapshots" + ("" if a.images else " (kept: pass --images)"), [im["id"] for im in f["images"]]),
        ]
        if a.account_wide:
            t = f.get("tracing") or {}
            steps.append(("account-wide: Transaction Search back to X-Ray, span log groups, their logs resource policy, "
                          "AgentCore's runtime identity service-linked role",
                          list(t.get("logs", {})) + t.get("policies", []) + [r for r in t.get("roles", []) if r == RUNTIME_SLR]))
        for n, (what, items) in enumerate(steps, 1):
            say(f"{n:2}. {what}: {items or 'none found'}")
        for sec, why in self.blind(f, a).items():
            say("could not read", sec, f"({why}); destroy skips what it cannot see and keeps its state.json entries")

    def blind(self, f: dict, a) -> dict:
        """Unread sections that this destroy run would act on; the read-only extras count only when their flag is set."""
        extra = {"images": a.images, "orphans": a.orphan_volumes, "tracing": a.account_wide}
        return {s: why for s, why in f.get("unread", {}).items() if extra.get(s, True)}

    def destroy(self, a):
        f = self.find()
        self.plan(f, a)
        if not a.yes:
            return say("dry run; pass --yes to delete the resources above")
        done = set()
        if self.drop_runtimes(f["runtimes"], f["deliveries"]):
            done |= {"agent_runtime_id", "agent_arn", "agent_hub", "agent_model"}
        cdn = [self.drop_cdn(d, wait=not a.no_wait_cdn) for d in f["cdn"]]
        if all(cdn):
            done |= {"cdn_id", "cdn_domain"}
        self.drop_instances(f["instances"])
        self.drop_volumes([v["id"] for v in f["volumes"]])
        if a.orphan_volumes:
            self.drop_volumes([v["id"] for v in f["orphans"]])
        done |= {"hubs", "ami"}
        if self.drop_groups(f["groups"]):
            done.add("sg")
        self.drop_iam(f["iam"])
        done.add("instance_profile")
        if not a.keep_bucket:
            for b in f["buckets"]:
                self.drop_bucket(b["name"])
            done.add("bucket")
        self.drop_logs(f["log_groups"])
        self.drop_ssm(f["ssm"])
        self.drop_ecr(f["ecr"])
        if a.images:
            self.drop_images(f["images"])
        if a.account_wide:
            self.drop_tracing(f.get("tracing") or {})
        blind = self.blind(f, a)
        kept = sorted(k for s in blind for k in STATE_KEYS.get(s, ()) if k in done)
        self.forget(done - set(kept))
        if blind:
            keep = f"; state.json keeps {kept}" if kept else ""
            sys.exit(f"destroy incomplete: could not read {sorted(blind)}{keep}. Fix access, then rerun destroy --yes")
        say("destroy finished; rerun inventory to confirm nothing is left")

    def drop_runtimes(self, rts: list[dict], dl: dict) -> bool:
        clean = True
        for r in rts:
            for e in r["endpoints"]:
                if e["name"] == "DEFAULT":
                    continue
                try:
                    self.cc.delete_agent_runtime_endpoint(agentRuntimeId=r["id"], endpointName=e["name"])
                    say("deleting endpoint", e["name"], "of", r["id"])
                except ClientError as x:
                    if code(x) not in GONE:
                        raise
            for attempt in range(30):
                try:
                    self.cc.delete_agent_runtime(agentRuntimeId=r["id"])
                    say("deleting AgentCore runtime", r["id"])
                    break
                except ClientError as x:
                    if code(x) in GONE:
                        say("AgentCore runtime", r["id"], "already gone")
                        break
                    if code(x) == "ConflictException" and attempt < 29:   # endpoints still deleting
                        time.sleep(10)
                        continue
                    raise
            clean &= self.wait_gone(lambda: self.cc.get_agent_runtime(agentRuntimeId=r["id"]), f"runtime {r['id']}")
            if r["identity"]:
                name = r["identity"].rsplit("/", 1)[-1]
                try:
                    self.cc.delete_workload_identity(name=name)
                    say("deleted workload identity", name)
                except ClientError as x:
                    say("workload identity", name, "already gone" if code(x) in GONE else f"left in place ({code(x)})")
        for d in dl["deliveries"]:
            self._quiet(lambda: self.logs.delete_delivery(id=d), "deleted log delivery", d)
        for s in dl["sources"]:
            self._quiet(lambda: self.logs.delete_delivery_source(name=s), "deleted delivery source", s)
        for d in dl["destinations"]:
            self._quiet(lambda: self.logs.delete_delivery_destination(name=d), "deleted delivery destination", d)
        return clean

    def wait_gone(self, probe, what: str, timeout=600) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                probe()
            except ClientError as x:
                if code(x) in GONE:
                    say(what, "deleted")
                    return True
                raise
            time.sleep(10)
        say(what, f"still deleting after {timeout} s; rerun destroy to finish")
        return False

    def drop_cdn(self, d: dict, wait=True) -> bool:
        try:
            c = self.cf.get_distribution_config(Id=d["id"])
        except ClientError as x:
            if code(x) in GONE:
                say("CloudFront", d["id"], "already gone")
                return True
            raise
        cfg, etag = c["DistributionConfig"], c["ETag"]
        if cfg["Enabled"]:
            cfg["Enabled"] = False
            self.cf.update_distribution(Id=d["id"], IfMatch=etag, DistributionConfig=cfg)
            say("disabled CloudFront", d["id"], d["domain"])
        if self.cf.get_distribution(Id=d["id"])["Distribution"]["Status"] != "Deployed":
            if not wait:
                say("CloudFront", d["id"], "is disabling; rerun destroy in about 15 min to delete it")
                return False
            say("waiting for CloudFront", d["id"], "to finish disabling (up to 35 min)")
            self.cf.get_waiter("distribution_deployed").wait(Id=d["id"], WaiterConfig={"Delay": 30, "MaxAttempts": 70})
        etag = self.cf.get_distribution(Id=d["id"])["ETag"]
        self.cf.delete_distribution(Id=d["id"], IfMatch=etag)
        say("deleted CloudFront", d["id"], d["domain"])
        return True

    def drop_instances(self, insts: list[dict]):
        ids = [i["id"] for i in insts]
        if not ids:
            return say("no hub instances")
        self.ec2.terminate_instances(InstanceIds=ids)
        say("terminating", ids)
        self.ec2.get_waiter("instance_terminated").wait(InstanceIds=ids, WaiterConfig={"Delay": 15, "MaxAttempts": 60})
        say("terminated", ids)

    def drop_volumes(self, ids: list[str]):
        for v in ids:
            try:
                for _ in range(12):            # a volume that outlives its instance detaches within a minute
                    st = self.ec2.describe_volumes(VolumeIds=[v])["Volumes"][0]["State"]
                    if st != "in-use":
                        break
                    time.sleep(5)
            except ClientError as x:
                if code(x) in GONE:
                    say("volume", v, "went with its instance")
                    continue
                raise
            if st == "available":
                self.ec2.delete_volume(VolumeId=v)
                say("deleted volume", v)
            elif st in ("deleting", "deleted"):
                say("volume", v, "went with its instance")
            else:
                say("volume", v, "is", st, "; left in place")

    def drop_groups(self, groups: list[dict]) -> bool:
        clean = True
        for g in groups:
            for attempt in range(18):          # network interfaces take a minute or two to release
                try:
                    self.ec2.delete_security_group(GroupId=g["id"])
                    say("deleted security group", g["id"])
                    break
                except ClientError as x:
                    if code(x) in GONE:
                        say("security group", g["id"], "already gone")
                        break
                    if code(x) == "DependencyViolation" and attempt < 17:
                        time.sleep(10)
                        continue
                    say("security group", g["id"], "left in place:", code(x))
                    clean = False
                    break
        return clean

    def drop_iam(self, found: dict):
        iam = self.iam
        for p, roles in found["profiles"].items():
            for r in roles:
                self._quiet(lambda: iam.remove_role_from_instance_profile(InstanceProfileName=p, RoleName=r),
                            "removed role", r, "from instance profile", p)
            self._quiet(lambda: iam.delete_instance_profile(InstanceProfileName=p), "deleted instance profile", p)
        for r in found["roles"]:
            rn = r["name"]
            try:
                for p in iam.list_attached_role_policies(RoleName=rn)["AttachedPolicies"]:
                    iam.detach_role_policy(RoleName=rn, PolicyArn=p["PolicyArn"])
                    say("detached", p["PolicyName"], "from", rn)
                for p in iam.list_role_policies(RoleName=rn)["PolicyNames"]:
                    iam.delete_role_policy(RoleName=rn, PolicyName=p)
                    say("deleted inline policy", p, "of", rn)
                for ip in iam.list_instance_profiles_for_role(RoleName=rn)["InstanceProfiles"]:
                    iam.remove_role_from_instance_profile(InstanceProfileName=ip["InstanceProfileName"], RoleName=rn)
                    say("removed", rn, "from instance profile", ip["InstanceProfileName"])
                iam.delete_role(RoleName=rn)
                say("deleted role", rn)
            except ClientError as x:
                if code(x) not in GONE:
                    raise
                say("role", rn, "already gone")

    def drop_bucket(self, b: str):
        s3 = self.s.client("s3")
        own = {"ExpectedBucketOwner": self.acct}
        n = 0
        try:
            while True:
                r = s3.list_object_versions(Bucket=b, MaxKeys=1000, **own)
                objs = [{"Key": o["Key"], "VersionId": o["VersionId"]}
                        for o in r.get("Versions", []) + r.get("DeleteMarkers", [])]
                if not objs:
                    break
                res = s3.delete_objects(Bucket=b, Delete={"Objects": objs, "Quiet": True}, **own)
                if res.get("Errors"):
                    e = res["Errors"][0]
                    sys.exit(f"S3 refused to delete {len(res['Errors'])} objects in {b}, first {e['Key']}: {e['Code']}")
                n += len(objs)
            for page in s3.get_paginator("list_multipart_uploads").paginate(Bucket=b, **own):
                for u in page.get("Uploads", []):
                    s3.abort_multipart_upload(Bucket=b, Key=u["Key"], UploadId=u["UploadId"], **own)
            s3.delete_bucket(Bucket=b, **own)
            say("deleted bucket", b, "and", n, "object versions")
        except ClientError as x:
            if code(x) not in GONE:
                raise
            say("bucket", b, "already gone")

    def drop_logs(self, groups: list[dict]):
        for g in groups:
            self._quiet(lambda: self.logs.delete_log_group(logGroupName=g["name"]), "deleted log group", g["name"])

    def drop_ssm(self, items: dict):
        ssm = self.s.client("ssm")
        for d in items["documents"]:
            self._quiet(lambda: ssm.delete_document(Name=d), "deleted SSM document", d)
        ps = items["parameters"]
        for k in range(0, len(ps), 10):
            ssm.delete_parameters(Names=ps[k:k + 10])
            say("deleted SSM parameters", ps[k:k + 10])

    def drop_ecr(self, repos: list[str]):
        ecr = self.s.client("ecr")
        for r in repos:
            self._quiet(lambda: ecr.delete_repository(repositoryName=r, force=True), "deleted ECR repository", r)

    def drop_images(self, images: list[dict]):
        for im in images:
            self._quiet(lambda: self.ec2.deregister_image(ImageId=im["id"]), "deregistered AMI", im["id"])
            for sid, _ in im["snapshots"]:
                self._quiet(lambda: self.ec2.delete_snapshot(SnapshotId=sid), "deleted snapshot", sid)

    def drop_tracing(self, t: dict):
        """Account-wide: send spans back to X-Ray, remove the span log groups and their policy, then the runtime SLR."""
        if t.get("destination") == "CloudWatchLogs":
            self.s.client("xray").update_trace_segment_destination(Destination="XRay")
            say("Transaction Search off: trace segments go to X-Ray")
        for name in t.get("logs", {}):
            self._quiet(lambda: self.logs.delete_log_group(logGroupName=name), "deleted log group", name)
        for p in t.get("policies", []):
            self._quiet(lambda: self.logs.delete_resource_policy(policyName=p), "deleted logs resource policy", p)
        if SIGNALS_SLR in t.get("roles", []):
            say("kept", SIGNALS_SLR, "; it belongs to CloudWatch Application Signals and costs nothing")
        if RUNTIME_SLR not in t.get("roles", []):
            return
        left = [x["agentRuntimeId"] for p in self.cc.get_paginator("list_agent_runtimes").paginate() for x in p["agentRuntimes"]]
        if left:
            return say("kept", RUNTIME_SLR, "; runtimes remain in", self.region, left)
        try:
            task = self.iam.delete_service_linked_role(RoleName=RUNTIME_SLR)["DeletionTaskId"]
            say("deleting", RUNTIME_SLR, "; IAM keeps it if a runtime in another region still uses it; task", task)
        except ClientError as x:
            say(RUNTIME_SLR, "already gone" if code(x) in GONE else f"left in place ({code(x)})")

    def _quiet(self, fn, verb: str, *what):
        """Run one delete now; a resource that is already gone counts as done."""
        try:
            fn()
            say(verb, *what)
        except ClientError as x:
            if code(x) not in GONE:
                raise
            say(*what, "already gone")

    def forget(self, keys: set):
        """Move deleted resources out of state.json so a later deploy starts clean; earlier passes stay on record."""
        st = load()
        gone = {k: st.pop(k) for k in sorted(keys) if k in st}
        if gone:
            st["destroyed"] = {**st.get("destroyed", {}), **gone,
                               "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
            save(st)
            say("state.json: moved", sorted(gone), "under destroyed")


def main():
    ap = argparse.ArgumentParser(description="Inventory, pause, resize and retire Mombe on AWS.",
                                 epilog="Run with no command for the inventory.")
    ap.add_argument("cmd", nargs="?", default="inventory",
                    choices=["inventory", "stop", "start", "stop-stock", "start-stock", "scale", "destroy"])
    ap.add_argument("--to", metavar="TYPE", help="scale: the new instance type, e.g. c8g.xlarge")
    ap.add_argument("--flavour", choices=["cool", "stock"], default="cool",
                    help="stop, start, scale: which hub (default cool)")
    ap.add_argument("--dry-run", action="store_true", help="stop, start, scale: print the change only")
    ap.add_argument("--force", action="store_true",
                    help="stop a hub that serves CloudFront, or scale past a failed type check or outside Graviton4")
    ap.add_argument("--yes", action="store_true", help="destroy: delete for real")
    ap.add_argument("--keep-bucket", action="store_true", help="destroy: keep the S3 bucket and its objects")
    ap.add_argument("--images", action="store_true", help="destroy: also deregister mombe AMIs and delete their snapshots")
    ap.add_argument("--orphan-volumes", action="store_true",
                    help="destroy: also delete unattached volumes made from a hub AMI (check the plan first)")
    ap.add_argument("--account-wide", action="store_true",
                    help="destroy: also turn Transaction Search off, delete the span log groups and AgentCore's "
                         "runtime identity service-linked role (shared account settings)")
    ap.add_argument("--no-wait-cdn", action="store_true", help="destroy: disable CloudFront and delete it on a later run")
    a = ap.parse_args()
    if a.cmd == "scale" and not a.to:
        ap.error("scale needs --to TYPE")
    try:
        t = Teardown(dry=a.dry_run)
        if a.cmd == "inventory":
            t.inventory()
        elif a.cmd in ("stop", "stop-stock"):
            t.stop("stock" if a.cmd == "stop-stock" else a.flavour, force=a.force)
        elif a.cmd in ("start", "start-stock"):
            t.start("stock" if a.cmd == "start-stock" else a.flavour)
        elif a.cmd == "scale":
            t.scale(a.flavour, a.to, force=a.force)
        else:
            t.destroy(a)
    except ClientError as e:
        if code(e) in EXPIRED:
            sys.exit(f"AWS credentials expired ({code(e)}). Run: aws login --region {REGION}")
        raise
    except WaiterError as e:                # a BotoCoreError, so it must come first
        sys.exit(f"AWS took too long or reported a failure state while {a.cmd} waited ({e}). Check with: "
                 f"python infra/teardown.py inventory, then rerun {a.cmd}; finished steps are skipped")
    except BotoCoreError as e:
        sys.exit(f"AWS call failed before reaching the service ({type(e).__name__}: {e}). "
                 f"If credentials expired, run: aws login --region {REGION}")


if __name__ == "__main__":
    main()
