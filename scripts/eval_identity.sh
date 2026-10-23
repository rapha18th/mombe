#!/usr/bin/env bash
# Held-out identity: the BC et al. 2026 leakage-controlled open-set protocol, 5 identity-disjoint
# partitions, on the Zenodo muzzle set (input R, truth L). Runs the frozen plan in
# config/identity_heldout/plan.json on this machine, then writes the report.
#
#   PY=python WORKERS=6 bash scripts/eval_identity.sh
#   LIMIT=20 bash scripts/eval_identity.sh      # smoke run on the first 20 probes (metrics meaningless);
#                                               # it still extracts the 711 enrolment images first
#   USE_CACHE=1 ...                             # reuse data/features where present (the published run extracted every image)
#
# Needs data/muzzle and models (scripts/setup.sh). A full run extracts about 2,470 images and
# verifies about 21,000 LightGlue pairs: about 2 to 2.5 hours on a c8g.2xlarge, many hours on a
# laptop. The published run used 9 Kaggle CPU shards (eval/identity_heldout.py kaggle --kind pairs).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
OUT=eval/out/identity_heldout
mkdir -p "$OUT"
[ -f "$OUT/plan.json" ] || cp config/identity_heldout/plan.json "$OUT/plan.json"
[ -f "$OUT/duplicates_reconciled.json" ] || cp config/identity_heldout/duplicates_reconciled.json "$OUT/duplicates_reconciled.json"

"$PY" eval/identity_heldout.py pairs --plan "$OUT/plan.json" --out "$OUT/pairs_local" \
  --workers "${WORKERS:-4}" ${LIMIT:+--limit "$LIMIT"} ${USE_CACHE:+--use-cache}
"$PY" eval/identity_heldout.py report --plan "$OUT/plan.json" --pairs "$OUT/pairs_local" --summary "$OUT/summary_local.json"
"$PY" eval/identity_heldout.py readme --summary "$OUT/summary_local.json" --out "$OUT/README_local.md"
echo "report: $OUT/README_local.md"
