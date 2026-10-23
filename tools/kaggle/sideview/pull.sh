#!/usr/bin/env bash
# Pull SideViewCows shard outputs from Kaggle with retries (the API drops connections).
# usage: bash tools/kaggle/sideview/pull.sh <slug> <shard-index> <outdir> [extra file-pattern ...]
# Pulls the shard's summary, rows and gallery index; extra patterns (e.g. one gallery JPEG) follow.
slug="$1"; n="$2"; out="$3"; shift 3
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
mkdir -p "$out"
for pat in "summary_s$n.json" "rows_s$n.jsonl" "gallery_s$n.jsonl" "$@"; do
  for i in 1 2 3 4 5 6; do
    if kaggle kernels output "thetraveller/$slug" -p "$out" --file-pattern "$pat" > "$out/.pull.txt" 2>&1 \
       && ! grep -q "Connection aborted\|Error" "$out/.pull.txt"; then
      echo "$slug $pat ok (try $i)"; break
    fi
    echo "$slug $pat retry $i: $(tail -1 "$out/.pull.txt" | cut -c1-120)"
  done
done
rm -f "$out/.pull.txt"
