#!/usr/bin/env bash
# Local setup for the hub and the evaluations: models, the muzzle dataset, features,
# the demo registry and the frozen thresholds. Safe to rerun; finished steps are skipped.
#
#   PY=python WORKERS=4 bash scripts/setup.sh
#
# Downloads: about 240 MB of ONNX models (SHA-1 checked) and the 644 MB Zenodo
# muzzle archive (MD5 checked). Feature extraction is the slow step.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
WORKERS="${WORKERS:-4}"

"$PY" tools/fetch_models.py --only aliked aliked_lightglue dinov2_small depth_anything_v2_small
"$PY" scripts/fetch_muzzle.py

# Frozen identity thresholds (t_new 20, t_match 90) and the duplicate-identity merge list the
# split reads. A file already in eval/out is kept, so a local calibration survives a rerun.
mkdir -p eval/out
[ -f eval/out/thresholds.json ] || cp config/thresholds.json eval/out/thresholds.json
[ -f eval/out/duplicates.json ] || cp config/duplicates.json eval/out/duplicates.json

# Features (data/features/<animal>/<image>.npz) for the image set the published split, gallery and
# scenario runs used: the first 499 images in path order, then 10 evenly spaced images per animal
# (2,708 images of 268 animals). The split reads whatever features exist, so keep this set.
"$PY" tools/build_features.py --workers "$WORKERS" --limit 499
"$PY" tools/build_features.py --workers "$WORKERS" --per-animal 10
"$PY" tools/build_registry.py --out var                # var/gallery.npz and var/registry/animals.json
echo "setup done: start the hub with  make hub  (or bash scripts/hub.sh)"
