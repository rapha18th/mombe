#!/bin/bash
# Op-level diffs on the 300 fixed real images (track 1d, deliverables 1 and 3).
#   COOL hub:  cool vs cvonly     COOL build against the stock wheel, same machine, same python + numpy
#              cool vs cool       within-build repeat on 40 images (determinism floor)
#   stock hub: stock vs stockheadless   stock hub hashes for the cross-machine check (same wheel as cvonly)
# Workers run under nice 10; the hub service is never touched.
export PATH=$PATH:/snap/bin
cd /opt/cool_equiv
B=s3://${MOMBE_BUCKET:?set MOMBE_BUCKET}
R=/opt/cool_equiv/run_env.sh
J=code/tools/cool_equiv/ops_diff.py
D=/opt/cool_equiv/data
mkdir -p $D out/ops
for f in ops_images.zip ops_images.json ops_muzzle.zip ess_fixed.npz; do
  [ -f $D/$f ] || aws s3 cp $B/jobs/1d/data/$f $D/$f --region eu-west-1 --only-show-errors
done
[ -d $D/acme ] || unzip -o -q $D/ops_images.zip -d $D
[ -d $D/muzzle ] || unzip -o -q $D/ops_muzzle.zip -d $D
echo "images: acme $(ls $D/acme | wc -l), muzzle animals $(ls $D/muzzle | wc -l)"
if [ -d /opt/cool ]; then HUB=cool; else HUB=stock; fi
run() {  # name a b [extra args]
  n=$1; a=$2; b=$3; shift 3
  echo "== $n $(date +%T)"
  $R $a $J --a $a --b $b --images $D/ops_images.json --root $D --ess $D/ess_fixed.npz --out out/ops/$n.json "$@" 2>&1 | tail -80
  aws s3 cp out/ops/ $B/results/1d/$HUB/ops/ --recursive --region eu-west-1 --only-show-errors
}
if [ $HUB = cool ]; then
  run ops_cool_vs_cvonly cool cvonly
  run ops_cool_vs_cool cool cool --limit 40
else
  run ops_stock_vs_stockheadless stock stockheadless
fi
echo OPS_DONE
date > out/ops/OPS_DONE && aws s3 cp out/ops/OPS_DONE $B/results/1d/$HUB/ops/OPS_DONE --region eu-west-1 --only-show-errors
