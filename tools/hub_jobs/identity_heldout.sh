#!/usr/bin/env bash
# Held-out identity (track 1e) on the Graviton4 COOL hub: full BC et al. protocol plus latency.
#
# What it runs: eval/identity_heldout.py pairs (every probe of plan.json, one shard, 8 workers)
# and eval/identity_heldout.py bench (per-stage latency on this hub). Results go to
# s3://$BUCKET/results/identity_heldout/$LABEL/ with a DONE marker at the end.
#
# Operator steps (run from the repo root on the PC; AWS credentials via `aws login --region eu-west-1`):
#   1. .venv/Scripts/python.exe infra/deploy.py push-code
#        ships eval/identity_heldout.py to /opt/mombe on both hubs
#   2. aws s3 cp eval/out/identity_heldout/plan.json s3://mombe-967170427068-eu-west-1/jobs/identity_heldout/plan.json --region eu-west-1
#      aws s3 cp tools/hub_jobs/identity_heldout.sh s3://mombe-967170427068-eu-west-1/jobs/identity_heldout/identity_heldout.sh --region eu-west-1
#   3. aws ssm send-command --region eu-west-1 --instance-ids i-0cc32db5d19735dda --document-name AWS-RunShellScript \
#        --parameters 'commands=["aws s3 cp s3://mombe-967170427068-eu-west-1/jobs/identity_heldout/identity_heldout.sh /tmp/ih.sh --region eu-west-1","nohup bash /tmp/ih.sh > /tmp/ih.log 2>&1 &"]'
#   4. Poll: aws s3 ls s3://mombe-967170427068-eu-west-1/results/identity_heldout/cool/ --region eu-west-1   (DONE appears at the end)
#      Live log: aws ssm send-command ... --parameters 'commands=["tail -5 /tmp/ih.log"]', then infra/ssm_result.py <command-id>
#   5. aws s3 cp s3://mombe-967170427068-eu-west-1/results/identity_heldout/cool/ eval/out/identity_heldout/pairs_hub/ --recursive --region eu-west-1
#      .venv/Scripts/python.exe eval/identity_heldout.py report --pairs eval/out/identity_heldout/pairs_hub
#   6. Clean up: the job leaves /tmp/ih_* behind until reboot; delete them with
#        aws ssm send-command ... --parameters 'commands=["rm -rf /tmp/ih_data /tmp/ih_work /tmp/ih.sh"]'
#      Stop the stock hub if it was started only for this run (it is unused here).
#
# Expected size (plan.json of 2026-10-21: 237 identities after duplicate removal, 711 enrolment
# images, 1,756 probes): about 2,470 feature extractions and about 21,000 LightGlue pairs, from a
# proxy count of 12.0 pairs per probe on cached embeddings. At the hub's measured 350 ms per pair
# (8 threads, eval/out/bench/cool.json) that is roughly 2 to 2.5 hours on c8g.2xlarge
# ($0.34112/h + COOL $0.02/h, infra/teardown.py prices checked 2026-10-21), so about $0.80.
# The same plan also runs as 8 Kaggle shards (eval/identity_heldout.py kaggle --kind pairs);
# a hub run replicates those scores on Graviton4 and adds hub latency.
# The live hub keeps serving, more slowly, while it runs.
set -euo pipefail

BUCKET="${BUCKET:-mombe-967170427068-eu-west-1}"
REGION="${REGION:-eu-west-1}"
LABEL="${LABEL:-cool}"
APP=/opt/mombe
DATA=/tmp/ih_data
WORK=/tmp/ih_work
OUT=/tmp/ih_work/out
export PATH="$PATH:/snap/bin"

if [ -x /opt/cool/venvs/python_3.12/bin/python ]; then
  PY=/opt/cool/venvs/python_3.12/bin/python
  export PYTHONPATH=/opt/cool/python_3.12/site-packages/cv2/python-3.12
  export LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib
else
  PY=/opt/stockcv/bin/python
  LABEL="${LABEL/cool/stock}"
fi
echo "python $PY label $LABEL"
"$PY" -c "import cv2, numpy; print('cv2', cv2.__version__, 'numpy', numpy.__version__)"

mkdir -p "$DATA" "$WORK" "$OUT"
aws s3 cp "s3://$BUCKET/jobs/identity_heldout/plan.json" "$WORK/plan.json" --region "$REGION"

# Zenodo 6324361 (CC BY 4.0), md5 f2b19d9bab2fc668460a796259944204
if [ ! -f "$DATA/.unzipped" ]; then
  curl -sSL -o "$DATA/muzzle.zip" "https://zenodo.org/api/records/6324361/files/BeefCattle_Muzzle_database.zip/content"
  echo "f2b19d9bab2fc668460a796259944204  $DATA/muzzle.zip" | md5sum -c -
  "$PY" -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$DATA/muzzle.zip" "$DATA"
  rm -f "$DATA/muzzle.zip"
  touch "$DATA/.unzipped"
fi
echo "jpgs: $(find "$DATA" -name 'cattle_*.jpg' | wc -l)"

cd "$APP"
NPROC="$(nproc)"
"$PY" eval/identity_heldout.py pairs --plan "$WORK/plan.json" --images "$DATA" --models "$APP/models" \
  --out "$OUT" --work "$WORK" --shard 0 --shards 1 --workers "$NPROC" --threads 1 --tag "$LABEL"
aws s3 cp "$OUT/" "s3://$BUCKET/results/identity_heldout/$LABEL/" --recursive --region "$REGION"

# per-stage latency with the server's thread setting (all cores); bench reads data/muzzle
mkdir -p "$APP/data"
FIRST="$(find "$DATA" -name 'cattle_0100_*.jpg' | head -1)"
[ -e "$APP/data/muzzle" ] || ln -s "$(dirname "$(dirname "$FIRST")")" "$APP/data/muzzle"
"$PY" eval/identity_heldout.py bench --n 12 --label "${LABEL}_hub" --note "c8g.2xlarge hub, measured right after the pairs job"
aws s3 cp "$APP/eval/out/identity_heldout/bench_${LABEL}_hub.json" "s3://$BUCKET/results/identity_heldout/$LABEL/" --region "$REGION"
date -u > "$WORK/DONE"
aws s3 cp "$WORK/DONE" "s3://$BUCKET/results/identity_heldout/$LABEL/DONE" --region "$REGION"
echo "done"
