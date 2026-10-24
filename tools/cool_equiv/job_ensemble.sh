#!/bin/bash
# Float-noise sensitivity on ONE build (stock hub, stock wheel): the calibrated focal times (1 + j * 1e-10),
# the size of the COOL vs stock calibrateCamera difference. sweep_21 gets 10 perturbations, the other
# six original takes get 4 each. Spread over j = the pipeline's own sensitivity, with the build held fixed.
export PATH=$PATH:/snap/bin
cd /opt/cool_equiv
B=s3://${MOMBE_BUCKET:?set MOMBE_BUCKET}
R=/opt/cool_equiv/run_env.sh
E=code/tools/cool_equiv/ensemble.py
if [ -d /opt/cool ]; then HUB=cool; M=cool; else HUB=stock; M=stock; fi
O=out/ensemble
mkdir -p $O
$R $M $E --takes takes/sweep_21 --n 10 --dense-first 0 --out $O/ens_${M}_sweep21.jsonl 2>&1 | grep -v WARN | tail -12
aws s3 cp $O $B/results/1d/$HUB/ensemble/ --recursive --region eu-west-1 --only-show-errors
$R $M $E --takes takes/sweep_22 takes/sweep_23 takes/sweep_24 takes/sweep_25 takes/sweep_26 takes/sweep_27 --n 4 --dense-first 0 \
  --out $O/ens_${M}_others.jsonl 2>&1 | grep -v WARN | tail -26
aws s3 cp $O $B/results/1d/$HUB/ensemble/ --recursive --region eu-west-1 --only-show-errors
echo ENSEMBLE_DONE
date > $O/ENSEMBLE_DONE && aws s3 cp $O/ENSEMBLE_DONE $B/results/1d/$HUB/ensemble/ENSEMBLE_DONE --region eu-west-1 --only-show-errors
