#!/bin/bash
# Track 1d on one hub, in deliverable order, one job at a time (the COOL hub serves the live demo).
# Started in the background by launch.sh with JOB=job_all. Each step uploads its own results.
export PATH=$PATH:/snap/bin
cd /opt/cool_equiv
B=s3://${MOMBE_BUCKET:?set MOMBE_BUCKET}
if [ -d /opt/cool ]; then HUB=cool; else HUB=stock; fi
J=code/tools/cool_equiv
echo "== job_all on $HUB $(date)"
bash $J/job_ops.sh
bash $J/job_identify.sh
if aws s3 ls $B/results/1d/$HUB/bisect/ --region eu-west-1 | grep -q "sweep_21_s7_.*json"; then
  echo "bisect results already in S3; skipping"
else
  bash $J/job_bisect.sh
fi
echo ALL_DONE
date > out/ALL_DONE && aws s3 cp out/ALL_DONE $B/results/1d/$HUB/ALL_DONE --region eu-west-1 --only-show-errors
