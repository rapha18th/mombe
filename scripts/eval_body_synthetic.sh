#!/usr/bin/env bash
# Synthetic body sweep (input S, truth G): seven rendered walk-arounds, small heifer to large ox,
# each reconstructed (ALIKED + LightGlue SfM, ChArUco scale), densified (stereo + TSDF) and
# measured against the renderer's exact girth, length and scale.
#
#   PY=python bash scripts/eval_body_synthetic.sh
#
# Needs the models (scripts/setup.sh); the muzzle data is not used. Rendering writes build/synth/.
# One take is reconstruction plus dense fusion: about 30 s on a c8g.2xlarge; one take took 21 min on a
# loaded 12-thread x86 laptop. Numbers differ by build: sweep_21 is the fragile take (see README, COOL).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
OUT="${OUT:-eval/out/sweep_body_local.jsonl}"
rm -f "$OUT"
"$PY" tools/sweep_body.py --animals 7 --seeds 7 --out "$OUT"

# The gate evidence (eval/body_gates.py) reads one sweep per build: eval/out/sweep_body3.jsonl
# (stock OpenCV 5.0.0 hub) and eval/out/sweep_cool.jsonl (COOL hub). Run it when both exist.
if [ -f eval/out/sweep_body3.jsonl ] && [ -f eval/out/sweep_cool.jsonl ]; then
  if [ -d data/acme ]; then "$PY" eval/body_gates.py; else "$PY" eval/body_gates.py --no-acme; fi
fi
