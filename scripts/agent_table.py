"""Agent suite table for the README, computed from scenario runs (eval/run_scenarios.py).

Reads <run>/results.jsonl for each run given. Per run and driver: sessions correct,
re-captures, attacks accepted (synthetic presentation attacks and double enrolments
that got through) and the sessions missed.

A row that carries an `error` or has no outcome is an errored session: the run broke
before the driver decided (for example bedrock-runtime unreachable). Errored rows are
counted and listed by error type, and kept out of the correct, attack and missed
columns, so an infrastructure failure never reads as an agent result.

For a driver present in several complete runs of the same scenarios it adds pass^k:
the share of scenarios the driver got right in every run, over the scenarios that ran
without error in all of them.

Usage:
  python scripts/agent_table.py eval/out/final-r1 eval/out/final-r2 eval/out/final-r3
  python scripts/agent_table.py eval/out/final-r1 eval/out/final-r2 eval/out/final-r3 --readme README.md
  python scripts/agent_table.py ... --strict     # exit 1 when any run has errored rows
With --readme the block between the agent-table markers is replaced in place.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

START, END = "<!-- agent-table:start -->", "<!-- agent-table:end -->"
NAMES = {"claude": "Claude Sonnet 4.6 (Strands, Bedrock)", "autopilot": "Autopilot (policy alone)",
         "open_loop": "Open loop (one pass, top identity)"}


def load(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines() if line.strip()]


def errored(r: dict) -> bool:
    return bool(r.get("error")) or r.get("outcome") is None


def error_type(r: dict) -> str:
    e = str(r.get("error") or "no outcome")
    return e.split(":", 1)[0].strip() or "error"


def table(runs: list[Path]) -> tuple[str, int]:
    out = ["| Run | Driver | Correct | Re-captures | Attacks accepted | Missed sessions |", "|---|---|---|---|---|---|"]
    per_driver: dict[str, list[tuple[set, dict[str, bool]]]] = defaultdict(list)
    notes, n_err = [], 0
    for run in runs:
        rows = load(run)
        by: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by[r["driver"]].append(r)
        full = max(len(v) for v in by.values())
        for d in ("claude", "autopilot", "open_loop"):
            rs_all = by.get(d)
            if not rs_all:
                continue
            bad = [r for r in rs_all if errored(r)]
            rs = [r for r in rs_all if not errored(r)]
            n_err += len(bad)
            ok = sum(bool(r["correct"]) for r in rs)
            atk = [r for r in rs if "fraud_accepted" in r]
            acc = sum(bool(r["fraud_accepted"]) for r in atk)
            missed = [r for r in rs if not r["correct"]]
            if len(missed) > 4:
                order = [r["kind"] for r in rs]
                kinds = sorted({r["kind"] for r in missed}, key=order.index)
                miss = f"{len(missed)} sessions in {len(kinds)} kinds: " + ", ".join(kinds)
            else:
                miss = ", ".join(f"{r['scenario']} ({r['outcome']})" for r in missed) or "none"
            done = (run / "summary.json").exists()   # run_scenarios writes it when the run ends
            partial = (f" (partial, run of {full})" if len(rs_all) < full
                       else "" if done else " (run in progress)")
            err = f"; {len(bad)} errored, excluded" if bad else ""
            out.append(f"| {run.name} | {NAMES.get(d, d)} | {ok}/{len(rs)}{partial}{err} | "
                       f"{sum(r['recaptures'] for r in rs)} | {f'{acc} of {len(atk)}' if atk else 'n/a'} | {miss} |")
            if bad:
                types = Counter(error_type(r) for r in bad)
                atk_bad = sum(bool("fraud_accepted" in r or r.get("synthetic_attack")) for r in bad)
                notes.append(f"{run.name}, {NAMES.get(d, d)}: {len(bad)} of {len(rs_all)} sessions errored before a decision "
                             f"({', '.join(f'{k} {v}' for k, v in types.most_common())}), {atk_bad} of them attack sessions. "
                             "They are excluded above and need a rerun.")
            complete = done and len(rs_all) == full
            per_driver[d].append(({r["scenario"] for r in rs_all} if complete else set(),
                                  {r["scenario"]: bool(r["correct"]) for r in rs}))
    for d, runs_ok in per_driver.items():
        if len(runs_ok) < 2:
            continue
        sets = [s for s, _ in runs_ok]
        if not sets[0] or any(s != sets[0] for s in sets):
            continue   # pass^k only over complete repeats of one scenario set
        k = len(runs_ok)
        common = set.intersection(*(set(x) for _, x in runs_ok))
        if not common:
            continue
        all_ok = sum(all(x[s] for _, x in runs_ok) for s in common)
        mean1 = sum(sum(x[s] for s in common) for _, x in runs_ok) / (k * len(common))
        scope = (f"the same {len(common)} scenarios" if len(common) == len(sets[0])
                 else f"the {len(common)} of {len(sets[0])} scenarios that ran without error in all {k} runs")
        notes.append(f"{NAMES.get(d, d)}: pass^{k} {all_ok}/{len(common)} ({all_ok / len(common):.1%}); "
                     f"mean pass@1 {mean1:.1%} over {k} runs of {scope}.")
    return "\n".join(out + ([""] + notes if notes else [])), n_err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--readme")
    ap.add_argument("--strict", action="store_true", help="exit 1, writing nothing, when any run has errored rows")
    a = ap.parse_args()
    block, n_err = table([Path(r) for r in a.runs])
    print(block)
    if a.strict and n_err:
        print(f"\n{n_err} errored rows: rerun those scenarios first (--strict).")
        return 1
    if a.readme:
        p = Path(a.readme)
        text = p.read_text(encoding="utf-8")
        new, n = re.subn(re.escape(START) + r".*?" + re.escape(END), f"{START}\n{block}\n{END}", text, flags=re.S)
        if n != 1:
            raise SystemExit(f"{p}: expected one {START} ... {END} block, found {n}")
        p.write_text(new, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
