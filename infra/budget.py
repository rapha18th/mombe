"""Watch Mombe spend: an AWS Budget with email alerts, and a credits report.

  create   a monthly cost budget with an actual-cost alert and a forecast alert to one email
  credits  the promotional credits on the account with their sharing settings, and month-to-date
           cost by record type, by service and by seller (AWS or AWS Marketplace)

Usage:
  python infra/budget.py create --email you@example.com --limit 300 [--actual 50] [--forecast 100]
                                [--basis gross|net] [--scope account|org] [--name NAME] [--yes]
  python infra/budget.py credits [--scope account|org]

create prints the request and stops; --yes sends it. Thresholds are percentages of --limit,
or dollars with --absolute. The gross basis counts usage before credits, so it tracks the burn
rate. The net basis counts what credits leave over, so it trips when cash spending starts. The
account scope filters to the caller's own account, which matters in an organization's
management account, where an unfiltered budget covers every member account.

Budgets, Billing and Cost Explorer answer in us-east-1. Each Cost Explorer request costs $0.01.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
from deploy import NAME, REGION, say  # noqa: E402

BILLING_REGION = "us-east-1"
SERVICES = ("Bedrock", "Marketplace", "Elastic Compute Cloud", "EC2", "CloudFront", "CloudWatch", "AgentCore",
            "Simple Storage Service")


def code(e: ClientError) -> str:
    return e.response.get("Error", {}).get("Code", "")


def money(a: dict | None) -> str:
    return "?" if not a else f"{float(a['currencyAmount']):,.2f} {a['currencyCode']}"


def sharing(c: dict, acct: str) -> str:
    """The credit's sharing fields and what they mean for acct.

    Accounts must have credit sharing activated to give or receive shared credits, and the pool holds
    only credits of accounts that were members on the first day of the month (Applying AWS credits).
    """
    on, kind, accts = c.get("accountHasCreditSharingEnabled"), c.get("creditSharingType"), c.get("shareableAccounts")
    rule = f", rule {c['ruleName']}" if c.get("ruleName") else ""
    head = (f"sharing: owner sharing {'on' if on else 'off' if on is False else '?'}, type {kind or '?'}{rule}, "
            f"shareable accounts {', '.join(accts) if accts else 'none listed' if accts is not None else 'not returned'}")
    if c["accountId"] == acct:
        verdict = f"owned by {acct}, so it covers {acct} first"
    elif on is False or kind == "DISABLED":
        verdict = f"stays with {c['accountId']}; {acct} gets none of it"
    elif accts is not None and acct not in accts:
        verdict = f"{acct} is outside the shareable accounts"
    elif accts and acct in accts:
        verdict = (f"{acct} is listed; the shared pool holds this credit from the first month "
                   f"{c['accountId']} starts as a member")
    else:
        verdict = "unclear; check Billing preferences, Credit sharing preferences"
    return f"{head}; {verdict}"


def request(acct: str, a) -> dict:
    """The CreateBudget request: FilterExpression and Metrics, the current Budgets API form.

    Metrics takes the Budgets enum spelling (UnblendedCost). botocore skips enum checks, so a wrong
    spelling passes the dry run and fails only at AWS.
    """
    parts = []
    if a.scope == "account":
        parts.append({"Dimensions": {"Key": "LINKED_ACCOUNT", "Values": [acct]}})
    if a.basis == "gross":
        parts.append({"Not": {"Dimensions": {"Key": "RECORD_TYPE", "Values": ["Credit"]}}})
    budget = {"BudgetName": a.name, "BudgetType": "COST", "TimeUnit": "MONTHLY",
              "BudgetLimit": {"Amount": f"{a.limit:.2f}", "Unit": "USD"}, "Metrics": ["UnblendedCost"]}
    if len(parts) == 1:
        budget["FilterExpression"] = parts[0]
    elif parts:
        budget["FilterExpression"] = {"And": parts}
    kind = "ABSOLUTE_VALUE" if a.absolute else "PERCENTAGE"
    subs = [{"SubscriptionType": "EMAIL", "Address": a.email}]
    notes = [{"Notification": {"NotificationType": t, "ComparisonOperator": "GREATER_THAN",
                               "Threshold": float(v), "ThresholdType": kind}, "Subscribers": subs}
             for t, v in (("ACTUAL", a.actual), ("FORECASTED", a.forecast))]
    return {"AccountId": acct, "Budget": budget, "NotificationsWithSubscribers": notes,
            "ResourceTags": [{"Key": "project", "Value": NAME}]}


def create(s, acct: str, a):
    req = request(acct, a)
    unit = "USD" if a.absolute else "% of the limit"
    say(f"budget {a.name}: ${a.limit:.2f} a month, {a.basis} basis, {a.scope} scope; "
        f"alerts at actual > {a.actual:g} {unit} and forecast > {a.forecast:g} {unit}")
    print(json.dumps(req, indent=1))
    if not a.yes:
        return say("dry run; pass --yes to create this budget")
    b = s.client("budgets", region_name=BILLING_REGION)
    try:
        b.create_budget(**req)
    except ClientError as e:
        if code(e) != "DuplicateRecordException":
            raise
        if not a.replace:
            sys.exit(f"budget {a.name} exists; pass --replace to recreate it or choose another --name")
        b.delete_budget(AccountId=acct, BudgetName=a.name)
        b.create_budget(**req)
        say("replaced budget", a.name)
    say("created budget", a.name, "; AWS emails", a.email, "when a threshold is crossed")


def credits(s, acct: str, a):
    bill = s.client("billing", region_name=BILLING_REGION)
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=360)
    cr = None
    for payer in (True, False):            # the payer view covers the whole organization when allowed
        try:
            cr = bill.get_credits(accountId=acct, startDate=since, payerAccountFlag=payer)["credits"]
            say("credits", "across the billing family" if payer else "owned by", acct)
            break
        except ClientError as e:
            say("get-credits", "payer view" if payer else "account view", "refused:", code(e))
    for c in cr or []:
        prods = c.get("applicableProductNames") or []
        hits = [k for k in SERVICES if any(k.lower() in p.lower() for p in prods)]
        end = f"{c['endDate']:%Y-%m-%d}" if c.get("endDate") else "?"
        say(f"  {c['description']}  [{c['creditType']} {c.get('creditStatus', '')}]  owner {c['accountId']}")
        say(f"    remaining {money(c['remainingAmount'])} (estimate {money(c.get('estimatedAmount'))}) "
            f"of {money(c['initialAmount'])}, expires {end}")
        say(f"    applies to {len(prods)} products; Mombe services named: {', '.join(hits) or 'none'}")
        say("   ", sharing(c, acct))
        if a.verbose:
            say("    ", prods)
    if cr == []:
        say("  no credits found")
    if cr:
        say("  per-account activation lives in the console: Billing preferences, Credit sharing preferences; "
            "the preference active on the month's last day sets that month's bill")
    try:
        p = s.client("freetier", region_name=BILLING_REGION).get_account_plan_state()
        left = p.get("accountPlanRemainingCredits") or {}
        say("free tier plan", p["accountPlanType"], p["accountPlanStatus"], "credits left", left.get("amount"), left.get("unit"))
    except ClientError as e:
        say("free tier plan: none readable", f"({code(e)})")

    ce = s.client("ce", region_name=BILLING_REGION)
    today = dt.date.today()
    period = {"Start": today.replace(day=1).isoformat(), "End": (today + dt.timedelta(days=1)).isoformat()}
    base = {"Dimensions": {"Key": "LINKED_ACCOUNT", "Values": [acct]}} if a.scope == "account" else None

    def q(group: str, record: str | None = None) -> dict:
        flt = [x for x in (base, record and {"Dimensions": {"Key": "RECORD_TYPE", "Values": [record]}}) if x]
        args = dict(TimePeriod=period, Granularity="MONTHLY", Metrics=["UnblendedCost"],
                    GroupBy=[{"Type": "DIMENSION", "Key": group}])
        if len(flt) == 1:
            args["Filter"] = flt[0]
        elif flt:
            args["Filter"] = {"And": flt}
        rows = {}
        while True:
            r = ce.get_cost_and_usage(**args)
            for t in r["ResultsByTime"]:
                for g in t["Groups"]:
                    rows[g["Keys"][0]] = rows.get(g["Keys"][0], 0.0) + float(g["Metrics"]["UnblendedCost"]["Amount"])
            if not r.get("NextPageToken"):
                return rows
            args["NextPageToken"] = r["NextPageToken"]

    def show(title: str, rows: dict):
        say(title)
        for k, v in sorted(rows.items(), key=lambda kv: -abs(kv[1])):
            if abs(v) >= 0.005:
                say(f"  {v:10.2f}  {k}")

    scope = acct if a.scope == "account" else "the organization"
    say(f"Cost Explorer {period['Start']} to {today} for {scope} (4 requests, $0.04)")
    rec = q("RECORD_TYPE")
    show("by record type (USD)", rec)
    usage = sum(v for k, v in rec.items() if k not in ("Credit", "Refund", "Tax"))
    credit = -rec.get("Credit", 0.0)
    say(f"  usage {usage:.2f}, credits applied {credit:.2f}, net {sum(rec.values()):.2f}"
        + (f", credits cover {100 * credit / usage:.0f}% of usage" if usage > 0 else ""))
    show("usage by service", q("SERVICE", "Usage"))
    show("credits by service (negative is covered)", q("SERVICE", "Credit"))
    show("usage by seller (AWS Marketplace lines fall outside most promotional credits)", q("BILLING_ENTITY", "Usage"))
    say("Cost Explorer lags usage by up to 24 hours; credits on the open month are estimates until the invoice")


def main():
    ap = argparse.ArgumentParser(description="AWS Budget alerts and a credits report for Mombe.")
    ap.add_argument("cmd", choices=["create", "credits"])
    ap.add_argument("--email", help="create: who receives the alerts")
    ap.add_argument("--limit", type=float, help="create: monthly budget in USD")
    ap.add_argument("--actual", type=float, default=80, help="create: actual-cost alert threshold (default 80)")
    ap.add_argument("--forecast", type=float, default=100, help="create: forecast alert threshold (default 100)")
    ap.add_argument("--absolute", action="store_true", help="create: thresholds are USD amounts")
    ap.add_argument("--basis", choices=["gross", "net"], default="gross",
                    help="create: gross counts usage before credits; net counts what credits leave over")
    ap.add_argument("--scope", choices=["account", "org"], default="account",
                    help="the caller's account only (default) or the whole organization")
    ap.add_argument("--name", help="create: budget name (default mombe-<basis>)")
    ap.add_argument("--replace", action="store_true", help="create: recreate a budget with the same name")
    ap.add_argument("--yes", action="store_true", help="create: send the request")
    ap.add_argument("--verbose", action="store_true", help="credits: list every applicable product")
    a = ap.parse_args()
    if a.cmd == "create":
        if not a.email or not a.limit:
            ap.error("create needs --email and --limit")
        a.name = a.name or f"{NAME}-{a.basis}"
    try:
        s = boto3.Session(region_name=REGION)
        acct = s.client("sts").get_caller_identity()["Account"]
        if a.cmd == "create":
            create(s, acct, a)
        else:
            credits(s, acct, a)
    except (BotoCoreError, ClientError) as e:
        sys.exit(f"AWS call failed ({type(e).__name__}: {e}). If credentials expired, run: aws login --region {REGION}")


if __name__ == "__main__":
    main()
