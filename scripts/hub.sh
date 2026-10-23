#!/usr/bin/env bash
# Run the vision hub on this machine: MCP at /mcp, REST at /api, phone app at /app/, /health.
# Needs scripts/setup.sh first (models, gallery, thresholds). Set MOMBE_KEY to a secret of your own.
# /mcp answers to MOMBE_KEY alone (header X-Mombe-Key); /api also takes MOMBE_OFFICER_KEYS.
#
#   MOMBE_KEY=change-me PORT=8080 bash scripts/hub.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
mkdir -p eval/out
[ -f eval/out/thresholds.json ] || cp config/thresholds.json eval/out/thresholds.json
export MOMBE_KEY="${MOMBE_KEY:-dev-key}"
exec "$PY" -m mombe.server.mcp_app --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}"
