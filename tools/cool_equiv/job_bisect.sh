#!/bin/bash
# Bisect job: sweep_21, seed 7, per-stage dumps in every environment on this hub.
export PATH=$PATH:/snap/bin
cd /opt/cool_equiv
TAKE=${TAKE:-sweep_21}
SEED=${SEED:-7}
O=out/bisect
mkdir -p $O
R=/opt/cool_equiv/run_env.sh
J=code/tools/cool_equiv/bisect_take.py
if [ -d /opt/cool ]; then HUB=cool; MODES="cool cvonly wheelvenv"; else HUB=stock; MODES="stock stockheadless"; fi
for m in $MODES; do
  $R $m $J --take takes/$TAKE --out $O/${TAKE}_s${SEED}_$m --seed $SEED --plain --dense
done
# within-build repeat (determinism)
first=$(echo $MODES | cut -d' ' -f1)
$R $first $J --take takes/$TAKE --out $O/${TAKE}_s${SEED}_${first}_rep --seed $SEED
aws s3 cp $O s3://mombe-967170427068-eu-west-1/results/1d/$HUB/bisect/ --recursive --region eu-west-1 --only-show-errors
echo BISECT_DONE
