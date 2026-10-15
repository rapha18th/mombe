"""Deploy Mombe to AWS, idempotently. State lands in infra/state.json.

  bucket     S3 bucket for the code bundle, models, gallery and dataset images
  role       EC2 instance role: S3 read, SSM, InvokeAgentRuntime
  sg         vision-hub security group: port 8080 from CloudFront only
  instance   EC2 Graviton4 (c8g) from the COOL AMI, vision hub as a systemd unit
  cdn        CloudFront in front of the hub: HTTPS for the phone app, the API and MCP
  agent      AgentCore Runtime is deployed with the starter toolkit (see README)

Usage:
  python infra/deploy.py plan
  python infra/deploy.py up [--instance-type c8g.2xlarge]
  python infra/deploy.py hub-env --agent-arn <arn>      # wire the hub to the agent
  python infra/deploy.py push-code                       # ship code, install new requirements, restart
  python infra/deploy.py hub-set MOMBE_BODY=real         # set hub.env lines, keep the rest, restart
  python infra/deploy.py down
"""

from __future__ import annotations

import argparse
import io
import json
import secrets
import sys
import time
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "infra" / "state.json"
REGION = "eu-west-1"
NAME = "mombe"
COOL_PRODUCT = "prodview-fdvbfiewzuehs"   # Cloud Optimized OpenCV for AWS Graviton4
COOL_PRODUCT_CODE = "aajkmdd4qo3r7yhqg61a7aah9"   # the subscription's AMI product code (v3.x is built on OpenCV 5)
CF_CACHING_DISABLED = "4135ea2d-6df8-44a3-9df3-4b5a84be39ad"
CF_ALL_VIEWER_EXCEPT_HOST = "b689b0a8-53d0-40ab-baf2-68738e2966ac"


def load() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save(st: dict) -> None:
    STATE.write_text(json.dumps(st, indent=1))


def say(*a):
    print("·", *a, flush=True)


class Deploy:
    def __init__(self, region=REGION):
        self.s = boto3.Session(region_name=region)
        self.region = region
        self.acct = self.s.client("sts").get_caller_identity()["Account"]
        self.st = load()
        self.st.setdefault("region", region)
        self.st.setdefault("hub_key", secrets.token_urlsafe(24))

    # -- storage --------------------------------------------------------------
    def bucket(self):
        b = self.st.get("bucket") or f"{NAME}-{self.acct}-{self.region}"
        s3 = self.s.client("s3")
        try:
            s3.head_bucket(Bucket=b)
        except ClientError:
            s3.create_bucket(Bucket=b, CreateBucketConfiguration={"LocationConstraint": self.region})
            s3.put_public_access_block(Bucket=b, PublicAccessBlockConfiguration={
                "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
            s3.put_bucket_encryption(Bucket=b, ServerSideEncryptionConfiguration={
                "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]})
            say("created bucket", b)
        self.st["bucket"] = b
        save(self.st)
        return b

    def _have(self, key: str, size: int) -> bool:
        try:
            return self.s.client("s3").head_object(Bucket=self.st["bucket"], Key=key)["ContentLength"] == size
        except ClientError:
            return False

    def bundle(self):
        """Code, models, gallery and the gallery images the evidence drawings need."""
        b, s3 = self.st["bucket"], self.s.client("s3")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for p in list((ROOT / "mombe").rglob("*.py")) + list((ROOT / "web").glob("*")) + \
                     [ROOT / "requirements-server.txt"] + list((ROOT / "eval").glob("*.py")) + list((ROOT / "tools").glob("*.py")):
                if "__pycache__" not in p.parts:
                    z.write(p, p.relative_to(ROOT))
            th = ROOT / "eval" / "out" / "thresholds.json"
            if th.exists():
                z.write(th, th.relative_to(ROOT))
        s3.put_object(Bucket=b, Key="bundle/code.zip", Body=buf.getvalue())
        for f in ["aliked-n16rot-top1k-640.onnx", "aliked_lightglue.onnx", "dinov2_small.onnx",
                  "depth_anything_v2_small.onnx", "xfeat.onnx"]:
            p = ROOT / "models" / f
            if not self._have(f"bundle/models/{f}", p.stat().st_size):
                s3.upload_file(str(p), b, f"bundle/models/{f}")
                say("uploaded", f)
        for f in (ROOT / "var").glob("gallery.*"):
            if not self._have(f"bundle/var/{f.name}", f.stat().st_size):
                s3.upload_file(str(f), b, f"bundle/var/{f.name}")
        s3.upload_file(str(ROOT / "var" / "registry" / "animals.json"), b, "bundle/var/registry/animals.json")
        meta = json.loads((ROOT / "var" / "gallery.json").read_text())
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
            for a, im in zip(meta["animal_ids"], meta["image_ids"]):
                f = "_".join(im.split("_")[:2])
                p = ROOT / "data" / "muzzle" / f / f"{im}.jpg"
                if p.exists():
                    z.write(p, f"data/muzzle/{f}/{im}.jpg")
        if not self._have("bundle/gallery_images.zip", len(buf.getvalue())):
            s3.put_object(Bucket=b, Key="bundle/gallery_images.zip", Body=buf.getvalue())
        say("uploaded bundle, models, gallery and", len(meta["image_ids"]), "gallery images")

    # -- identity and network --------------------------------------------------
    def role(self):
        iam = self.s.client("iam")
        rn = f"{NAME}-hub"
        trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
        try:
            iam.get_role(RoleName=rn)
        except ClientError:
            iam.create_role(RoleName=rn, AssumeRolePolicyDocument=json.dumps(trust), Description="Mombe vision hub")
            iam.attach_role_policy(RoleName=rn, PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore")
            iam.attach_role_policy(RoleName=rn, PolicyArn="arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy")
            say("created role", rn)
        pol = {"Version": "2012-10-17", "Statement": [
            {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
             "Resource": [f"arn:aws:s3:::{self.st['bucket']}", f"arn:aws:s3:::{self.st['bucket']}/*"]},
            {"Effect": "Allow", "Action": ["bedrock-agentcore:InvokeAgentRuntime"], "Resource": "*"}]}
        iam.put_role_policy(RoleName=rn, PolicyName="mombe-hub", PolicyDocument=json.dumps(pol))
        try:
            iam.get_instance_profile(InstanceProfileName=rn)
        except ClientError:
            iam.create_instance_profile(InstanceProfileName=rn)
            iam.add_role_to_instance_profile(InstanceProfileName=rn, RoleName=rn)
            time.sleep(10)
        self.st["instance_profile"] = rn
        save(self.st)

    def sg(self):
        ec2 = self.s.client("ec2")
        vpc = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
        found = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [f"{NAME}-hub"]}, {"Name": "vpc-id", "Values": [vpc]}])["SecurityGroups"]
        if found:
            gid = found[0]["GroupId"]
        else:
            gid = ec2.create_security_group(GroupName=f"{NAME}-hub", Description="Mombe hub: CloudFront only", VpcId=vpc)["GroupId"]
            pl = ec2.describe_managed_prefix_lists(Filters=[{"Name": "prefix-list-name", "Values": ["com.amazonaws.global.cloudfront.origin-facing"]}])["PrefixLists"][0]["PrefixListId"]
            ec2.authorize_security_group_ingress(GroupId=gid, IpPermissions=[
                {"IpProtocol": "tcp", "FromPort": 8080, "ToPort": 8080, "PrefixListIds": [{"PrefixListId": pl, "Description": "CloudFront"}]}])
            say("created security group", gid)
        self.st.update(sg=gid, vpc=vpc)
        save(self.st)

    # -- compute ---------------------------------------------------------------
    def cool_ami(self) -> str:
        ec2 = self.s.client("ec2")
        imgs = ec2.describe_images(Owners=["aws-marketplace"], Filters=[
            {"Name": "product-code", "Values": [COOL_PRODUCT_CODE]}, {"Name": "architecture", "Values": ["arm64"]}])["Images"]
        if not imgs:
            sys.exit("COOL AMI not visible. Subscribe in AWS Marketplace first: "
                     f"https://aws.amazon.com/marketplace/pp/{COOL_PRODUCT}")
        img = sorted(imgs, key=lambda i: i["CreationDate"])[-1]
        say("COOL AMI", img["ImageId"], img.get("Name"))
        self.st["ami"] = img["ImageId"]
        return img["ImageId"]

    def stock_ami(self) -> str:
        ec2 = self.s.client("ec2")
        imgs = ec2.describe_images(Owners=["099720109477"], Filters=[
            {"Name": "name", "Values": ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-arm64-server-*"]}])["Images"]
        img = sorted(imgs, key=lambda i: i["CreationDate"])[-1]
        say("stock Ubuntu 24.04 arm64 AMI", img["ImageId"])
        return img["ImageId"]

    def user_data(self, flavour: str = "cool") -> str:
        """Boot script: pull the bundle with the instance role, install, start the hub."""
        b = self.st["bucket"]
        env = "/opt/cool/venvs/python_3.12" if flavour == "cool" else "/opt/stockcv"
        make_env = ("true" if flavour == "cool" else
                    "apt-get install -y python3.12-venv libgl1 libglib2.0-0t64 && python3.12 -m venv /opt/stockcv && "
                    "/opt/stockcv/bin/pip install opencv-contrib-python==5.0.0.93 numpy")
        # COOL keeps cv2 in a central store and exports these in its activate script.
        cool_env = ("PYTHONPATH=/opt/cool/python_3.12/site-packages/cv2/python-3.12\n"
                    "Environment=LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib") if flavour == "cool" else "MOMBE_FLAVOUR=stock"
        return f"""#!/bin/bash
set -euxo pipefail
exec > /var/log/mombe-setup.log 2>&1
apt-get update -y && apt-get install -y unzip
snap install aws-cli --classic || true
export PATH=$PATH:/snap/bin
mkdir -p /opt/mombe && cd /opt/mombe
aws s3 cp s3://{b}/bundle/ /opt/mombe/s3/ --recursive --region {self.region}
unzip -o s3/code.zip && unzip -o -q s3/gallery_images.zip
mkdir -p models var/registry && cp s3/models/* models/ && cp s3/var/gallery.* var/ && cp s3/var/registry/animals.json var/registry/
rm -rf s3
{make_env}
{env}/bin/pip install -r requirements-server.txt
{"export PYTHONPATH=/opt/cool/python_3.12/site-packages/cv2/python-3.12 LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib" if flavour == "cool" else "true"}
{env}/bin/python -c "import cv2; print('OPENCV', cv2.__version__, cv2.__file__)"
cat > /etc/systemd/system/mombe-hub.service <<UNIT
[Unit]
Description=Mombe vision hub (OpenCV 5, {flavour})
After=network-online.target
[Service]
WorkingDirectory=/opt/mombe
Environment=MOMBE_KEY={self.st['hub_key']}
Environment=MOMBE_HOME=/opt/mombe
Environment=AWS_REGION={self.region}
Environment=AWS_DEFAULT_REGION={self.region}
Environment={cool_env}
EnvironmentFile=-/opt/mombe/hub.env
ExecStart={env}/bin/python -m mombe.server.mcp_app --host 0.0.0.0 --port 8080
Restart=always
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload && systemctl enable --now mombe-hub
"""

    def instance(self, itype: str, flavour: str = "cool"):
        ec2 = self.s.client("ec2")
        hubs = self.st.setdefault("hubs", {})
        h = hubs.get(flavour)
        if h:
            r = ec2.describe_instances(InstanceIds=[h["instance_id"]])["Reservations"]
            if r and r[0]["Instances"][0]["State"]["Name"] not in ("terminated", "shutting-down"):
                say(flavour, "hub exists", h["instance_id"])
                return
        ami = self.cool_ami() if flavour == "cool" else self.stock_ami()
        img = ec2.describe_images(ImageIds=[ami])["Images"][0]
        root = img["RootDeviceName"]
        snap = next((b["Ebs"].get("VolumeSize", 8) for b in img.get("BlockDeviceMappings", [])
                     if b.get("DeviceName") == root and "Ebs" in b), 8)
        r = ec2.run_instances(
            ImageId=ami, InstanceType=itype, MinCount=1, MaxCount=1,
            IamInstanceProfile={"Name": self.st["instance_profile"]},
            SecurityGroupIds=[self.st["sg"]], UserData=self.user_data(flavour),
            MetadataOptions={"HttpTokens": "required"},
            BlockDeviceMappings=[{"DeviceName": root, "Ebs": {"VolumeSize": max(40, snap), "VolumeType": "gp3", "Encrypted": True}}],
            TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": f"{NAME}-hub-{flavour}"}, {"Key": "project", "Value": NAME}]}])
        iid = r["Instances"][0]["InstanceId"]
        say("launched", flavour, iid, itype, "from", ami)
        ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
        d = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]
        hubs[flavour] = {"instance_id": iid, "instance_type": itype, "public_dns": d["PublicDnsName"], "ami": ami}
        save(self.st)
        say(flavour, "running at", d["PublicDnsName"])

    def primary(self) -> dict:
        hubs = self.st.get("hubs", {})
        return hubs.get("cool") or hubs["stock"]

    def cdn(self):
        cf = self.s.client("cloudfront")
        if self.st.get("cdn_id"):
            say("distribution exists", self.st["cdn_domain"])
            return
        cfg = {
            "CallerReference": f"{NAME}-{int(time.time())}",
            "Comment": "Mombe hub",
            "Enabled": True,
            "Origins": {"Quantity": 1, "Items": [{
                "Id": "hub", "DomainName": self.primary()["public_dns"],
                "CustomOriginConfig": {"HTTPPort": 8080, "HTTPSPort": 443, "OriginProtocolPolicy": "http-only",
                                       "OriginReadTimeout": 60, "OriginKeepaliveTimeout": 30}}]},
            "DefaultCacheBehavior": {
                "TargetOriginId": "hub", "ViewerProtocolPolicy": "redirect-to-https",
                "AllowedMethods": {"Quantity": 7, "Items": ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"],
                                   "CachedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]}},
                "CachePolicyId": CF_CACHING_DISABLED, "OriginRequestPolicyId": CF_ALL_VIEWER_EXCEPT_HOST,
                "Compress": True},
            "PriceClass": "PriceClass_All",
            "HttpVersion": "http2and3",
        }
        d = cf.create_distribution(DistributionConfig=cfg)["Distribution"]
        self.st.update(cdn_id=d["Id"], cdn_domain=d["DomainName"])
        save(self.st)
        say("CloudFront", d["DomainName"], "(deploying, ~5 min)")

    def push_code(self):
        """Ship a new code.zip to every hub and restart it, keeping models and registry."""
        self.bundle_code_only()
        cmds = ["export PATH=$PATH:/snap/bin", "cd /opt/mombe",
                f"aws s3 cp s3://{self.st['bucket']}/bundle/code.zip /tmp/code.zip --region {self.region}",
                "unzip -o -q /tmp/code.zip -d /opt/mombe",
                # new requirements land in whichever venv this hub runs (COOL or stock)
                "for e in /opt/cool/venvs/python_3.12 /opt/stockcv; do [ -x $e/bin/pip ] && $e/bin/pip install -q -r requirements-server.txt; done; true",
                "systemctl restart mombe-hub", "sleep 6",
                "curl -s -m 5 localhost:8080/health"]
        ids = [h["instance_id"] for h in self.st["hubs"].values()]
        cid = self.s.client("ssm").send_command(InstanceIds=ids, DocumentName="AWS-RunShellScript",
                                                Parameters={"commands": cmds})["Command"]["CommandId"]
        say("pushed code to", ids, "command", cid)

    def bundle_code_only(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for p in list((ROOT / "mombe").rglob("*.py")) + list((ROOT / "web").glob("*")) +                      [ROOT / "requirements-server.txt"] + list((ROOT / "eval").glob("*.py")) + list((ROOT / "tools").glob("*.py")):
                if "__pycache__" not in p.parts:
                    z.write(p, p.relative_to(ROOT))
            th = ROOT / "eval" / "out" / "thresholds.json"
            if th.exists():
                z.write(th, th.relative_to(ROOT))
        self.s.client("s3").put_object(Bucket=self.st["bucket"], Key="bundle/code.zip", Body=buf.getvalue())

    def hub_env(self, agent_arn: str):
        ssm = self.s.client("ssm")
        env = f"MOMBE_AGENT_ARN={agent_arn}\\nMOMBE_DRIVER=claude\\nMOMBE_HUB=https://{self.st['cdn_domain']}\\n"
        cmd = ssm.send_command(InstanceIds=[h["instance_id"] for h in self.st["hubs"].values()], DocumentName="AWS-RunShellScript",
                               Parameters={"commands": [f"printf '{env}' > /opt/mombe/hub.env", "systemctl restart mombe-hub"]})
        self.st["agent_arn"] = agent_arn
        save(self.st)
        say("hub wired to agent", agent_arn, "command", cmd["Command"]["CommandId"])

    def hub_set(self, pairs: list[str]):
        """Set KEY=VALUE lines in every hub's hub.env, keeping the others, and restart."""
        cmds = ["touch /opt/mombe/hub.env"]
        for kv in pairs:
            k, v = kv.split("=", 1)
            cmds += [f"sed -i '/^{k}=/d' /opt/mombe/hub.env", f"echo '{k}={v}' >> /opt/mombe/hub.env"]
        cmds += ["systemctl restart mombe-hub", "sleep 6", "curl -s -m 5 localhost:8080/health",
                 "grep -v KEY /opt/mombe/hub.env"]
        cid = self.s.client("ssm").send_command(InstanceIds=[h["instance_id"] for h in self.st["hubs"].values()],
                                                DocumentName="AWS-RunShellScript",
                                                Parameters={"commands": cmds})["Command"]["CommandId"]
        say("set", pairs, "command", cid)

    def down(self):
        ec2, cf = self.s.client("ec2"), self.s.client("cloudfront")
        ids = [h["instance_id"] for h in self.st.get("hubs", {}).values()]
        if ids:
            ec2.terminate_instances(InstanceIds=ids)
            say("terminating", ids)
        if self.st.get("cdn_id"):
            c = cf.get_distribution_config(Id=self.st["cdn_id"])
            cfg, etag = c["DistributionConfig"], c["ETag"]
            if cfg["Enabled"]:
                cfg["Enabled"] = False
                cf.update_distribution(Id=self.st["cdn_id"], IfMatch=etag, DistributionConfig=cfg)
                say("disabled CloudFront; delete it once deployed state settles")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "up", "hub-env", "hub-set", "push-code", "down"])
    ap.add_argument("pairs", nargs="*", help="hub-set: KEY=VALUE ...")
    ap.add_argument("--instance-type", default="c8g.2xlarge")
    ap.add_argument("--agent-arn")
    ap.add_argument("--flavour", choices=["cool", "stock", "both"], default="both")
    a = ap.parse_args()
    d = Deploy()
    if a.cmd == "plan":
        say("account", d.acct, "region", d.region)
        say("will create: bucket, role, security group, COOL instance", a.instance_type, "and a CloudFront distribution")
        d.cool_ami()
    elif a.cmd == "up":
        d.bucket(); d.bundle(); d.role(); d.sg()
        for fl in (["cool", "stock"] if a.flavour == "both" else [a.flavour]):
            d.instance(a.instance_type, fl)
        d.cdn()
        say("hub key stored in infra/state.json")
    elif a.cmd == "hub-env":
        d.hub_env(a.agent_arn)
    elif a.cmd == "push-code":
        d.push_code()
    elif a.cmd == "hub-set":
        d.hub_set(a.pairs)
    else:
        d.down()


if __name__ == "__main__":
    main()
