"""A shareable sample of one session's decision trace, and its README excerpt.

Reads GET <hub>/api/sessions/<id>/trace, keeps the decision fields, drops evidence
images, file paths and host details, and writes the sample as JSON. It prints a
six-row Markdown excerpt: the capture check whose result changed the next action,
the retake it asked for, the check on the new capture, liveness, identity and the
finish. With --readme the excerpt replaces the block between the trace-excerpt markers.

The key travels in the X-Mombe-Key header and is never printed. It comes from
MOMBE_API_KEY, or from --key-file (a JSON file) and --key-field.

Usage:
  python scripts/trace_sample.py <session-id> --hub https://<cloudfront-domain> \\
      --key-file infra/demo_access.json --out docs/samples/trace_verify.json --readme README.md
  python scripts/trace_sample.py --from docs/samples/trace_verify.json      # excerpt from a saved sample
  ... --steps 3,4,6,9,10,12                                                 # choose the rows by step
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

START, END = "<!-- trace-excerpt:start -->", "<!-- trace-excerpt:end -->"
DROP_KEYS = {"evidence_image", "image", "path", "file", "client", "host", "hub"}
HOSTISH = re.compile(r"(https?://(?!farmnport\.com)\S+|[\w.-]+\.(amazonaws\.com|cloudfront\.net)\b|"
                     r"(^|\s)/(opt|tmp|home|var|srv|root)/\S*|[A-Za-z]:\\\S*|\bvar[/\\]store\S*)")
DECISIONS = ("observation", "action", "refused")


def fetch(hub: str, sid: str, key: str) -> list[dict]:
    req = urllib.request.Request(f"{hub.rstrip('/')}/api/sessions/{sid}/trace", headers={"X-Mombe-Key": key})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def clean(x):
    """Drop evidence images, paths and host details, keep everything a decision rests on."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()
                if k not in DROP_KEYS and not k.endswith(("_image", "_path", "_url", "_dir"))}
    if isinstance(x, list):
        return [clean(v) for v in x]
    if isinstance(x, str) and HOSTISH.search(x):
        return HOSTISH.sub(" <removed>", x).strip()
    return x


def evidence(e: dict) -> str:
    r, n = e.get("result") or {}, e.get("name")
    if e.get("kind") == "refused":
        return f"refused: {e.get('reason', '')}"
    if n == "assess_capture":
        return "issues: " + ", ".join(r["issues"]) if r.get("issues") else "clear"
    if n == "check_liveness":
        s = f"live, score {r.get('score')}" if r.get("live") else f"held back: {r.get('failed_cue') or 'liveness'}"
        return s + (f" (threshold {r['threshold']})" if r.get("threshold") is not None else "")
    if n == "identify_muzzle":
        return f"{r.get('decision')}, {r.get('animal_id') or 'no animal'}, {r.get('score')} verified points"
    if n == "request_capture":
        msg = str((r.get("instruction") or {}).get("message") or (e.get("args") or {}).get("message", ""))
        msg = msg.split(" / ")[0]   # the English half; the sample keeps the Shona
        return "officer asked: " + (msg if len(msg) <= 110 else msg[:107].rsplit(" ", 1)[0] + " ...")
    if n == "finish":
        return f"outcome {r.get('outcome')}"
    if n == "lookup_animal":
        return "register read"
    return ", ".join(f"{k} {v}" for k, v in r.items() if not isinstance(v, (dict, list)))[:120]


def pick(events: list[dict]) -> list[dict]:
    ev = [e for e in events if e.get("kind") in DECISIONS]

    def after(i: int, test) -> int | None:
        return next((j for j in range(i + 1, len(ev)) if test(ev[j])), None)

    first = next((j for j, e in enumerate(ev) if e.get("name") == "assess_capture" and e.get("changed_next_action")
                  and e.get("required_after") == "request_capture"), None)
    rows = [first]
    for name in ("request_capture", "assess_capture", "check_liveness", "identify_muzzle"):
        rows.append(None if rows[-1] is None else after(rows[-1], lambda e, n=name: e.get("name") == n))
    fin = [j for j, e in enumerate(ev) if e.get("name") == "finish"]
    rows.append(fin[-1] if fin else None)
    if any(r is None for r in rows):
        return ev[:6]
    return [ev[j] for j in rows]


def excerpt(events: list[dict], sid: str, steps: list[int] | None, sample: str | None) -> str:
    rows = [e for e in events if e.get("step") in steps] if steps else pick(events)
    src = f" ([`{sample}`]({sample}))" if sample else ""
    out = [f"Trace excerpt from live verify session `{sid}`{src}:", "",
           "| Step | Call | Evidence | Next required | Changed next action |", "|---|---|---|---|---|"]
    for e in rows:
        nxt = e.get("required_after") or "none"
        gate = (e.get("required_args_after") or {}).get("gate") or (e.get("required_args_after") or {}).get("outcome")
        nxt += f" ({gate})" if gate else ""
        cell = evidence(e).replace("|", "/")
        out.append(f"| {e.get('step')} | `{e.get('name')}` ({e.get('actor', '')}) | {cell} | `{nxt}` | "
                   f"{'yes' if e.get('changed_next_action') else 'no'} |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="?")
    ap.add_argument("--hub")
    ap.add_argument("--key-file")
    ap.add_argument("--key-field", default="officer_passcode")
    ap.add_argument("--from", dest="src", help="a saved sample instead of the hub")
    ap.add_argument("--out")
    ap.add_argument("--readme")
    ap.add_argument("--steps", help="comma list of trace steps for the excerpt")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.src:
        events = [clean(e) for e in json.loads(Path(a.src).read_text(encoding="utf-8"))]
        sid = a.session or next((e.get("session") for e in events if e.get("session")), "?")
    else:
        if not (a.session and a.hub):
            ap.error("give a session id and --hub, or --from a saved sample")
        key = os.environ.get("MOMBE_API_KEY", "")
        if a.key_file:
            key = json.loads(Path(a.key_file).read_text(encoding="utf-8")).get(a.key_field, "")
        if not key:
            sys.exit("no key: set MOMBE_API_KEY or pass --key-file")
        sid = a.session
        events = [clean(e) for e in fetch(a.hub, sid, key)]
        if key and key in json.dumps(events):
            sys.exit("the key appears in the trace; nothing written")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(events, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {a.out}: {len(events)} events")
    steps = [int(s) for s in a.steps.split(",")] if a.steps else None
    sample = (a.out or a.src or "").replace("\\", "/") or None
    block = excerpt(events, sid, steps, sample)
    print(block)
    if a.readme:
        p = Path(a.readme)
        text = p.read_text(encoding="utf-8")
        new, n = re.subn(re.escape(START) + r".*?" + re.escape(END), lambda m: f"{START}\n{block}\n{END}", text, flags=re.S)
        if n != 1:
            sys.exit(f"{p}: expected one {START} ... {END} block, found {n}")
        p.write_text(new, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
