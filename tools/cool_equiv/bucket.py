"""The project's S3 bucket for the COOL equivalence jobs.

MOMBE_BUCKET wins. Otherwise the bucket infra/deploy.py recorded in infra/state.json.
The shell jobs on the hubs read MOMBE_BUCKET, which run_1d.py and ssm.py export first.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

STATE = Path(__file__).resolve().parents[2] / "infra" / "state.json"


def bucket() -> str:
    name = os.environ.get("MOMBE_BUCKET")
    if not name and STATE.exists():
        name = json.loads(STATE.read_text()).get("bucket")
    if not name:
        raise SystemExit("set MOMBE_BUCKET, or run infra/deploy.py up so infra/state.json names the bucket")
    return name
