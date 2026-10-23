#!/usr/bin/env bash
# Unit tests: evidence policy, body gates, liveness cues, lender valuation, hub API guards.
# Needs only the Python packages in requirements-lock.txt plus pytest; no data, models or AWS.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
"$PY" -m pytest -q "$@"
