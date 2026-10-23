#!/usr/bin/env bash
# Ruchay real-cattle benchmark for the body back end (input R; truth P: tape, and live
# weight, which the source lists with the manual measurements; its weighing method is unstated).
# measure() runs unchanged on fused Kinect clouds of 103 Hereford cattle and is scored
# against manual tape measurements and live weights.
#
#   PY=python bash scripts/eval_ruchay.sh
#
# Data: Ruchay et al., github.com/ruchaya/CowDatabase (clouds) and github.com/ruchaya/CowDB
# (Manual_measurements.xlsx). The repositories carry no licence file: research use only.
# Keep the clouds and per-animal rows out of any public material; publish aggregates.
# Download: about 4.8 GB into data/ruchay (gitignored). Needs git 2.25 or newer.
# The published run used the Kaggle kernel instead: python tools/kaggle/ruchay/build_kernel.py,
# then kaggle kernels push -p tools/kaggle/ruchay (edit the id in kernel-metadata.json first).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
D="${RUCHAY_DATA:-data/ruchay}"
CDB_COMMIT=b1bb2d27907851fbcbf6d5363a3e20c26d3dde64     # CowDatabase commit of the published run
COWDB_COMMIT=270f2908de9a6931789400fbea122fe5e8df35b6   # CowDB commit of the published run
mkdir -p "$D"

if [ ! -d "$D/CowDatabase/.git" ]; then
  git clone --filter=blob:none --sparse https://github.com/ruchaya/CowDatabase "$D/CowDatabase"
  git -C "$D/CowDatabase" sparse-checkout set AlignedClouds TopClouds Matrices
fi
git -C "$D/CowDatabase" checkout -q "$CDB_COMMIT"
if [ ! -d "$D/CowDB/.git" ]; then
  git clone --filter=blob:none --no-checkout https://github.com/ruchaya/CowDB "$D/CowDB"
fi
git -C "$D/CowDB" ls-tree -r --name-only "$COWDB_COMMIT" > "$D/cowdb_names.txt"
git -C "$D/CowDB" checkout -q "$COWDB_COMMIT" -- Manual_measurements.xlsx

# Headline run: head direction from the recording-setup label (+x), every eligible animal.
"$PY" eval/ruchay_bench.py --cdb "$D/CowDatabase" --cowdb-xlsx "$D/CowDB/Manual_measurements.xlsx" \
  --cowdb-names "$D/cowdb_names.txt" --out eval/out/ruchay_local --ids 1-103 --source aligned --front=+x
echo "results: eval/out/ruchay_local (gitignored; aggregates only in public material)"
