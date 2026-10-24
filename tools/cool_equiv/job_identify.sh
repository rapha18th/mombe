#!/bin/bash
# identify() decisions on the 500 fixed probes (track 1d, deliverable 2).
#   COOL hub:  cool, then cvonly (stock wheel on the COOL machine), then cool again on 60 probes (repeat)
#   stock hub: stock
# Same shipped config (identify_config.json) and thresholds (90/20) everywhere; gallery hash in each header.
export PATH=$PATH:/snap/bin
cd /opt/cool_equiv
B=s3://${MOMBE_BUCKET:?set MOMBE_BUCKET}
R=/opt/cool_equiv/run_env.sh
J=code/tools/cool_equiv/identify_probes.py
D=/opt/cool_equiv/data
mkdir -p $D out/identify
for f in probes.zip probes.json; do
  [ -f $D/$f ] || aws s3 cp $B/jobs/1d/data/$f $D/$f --region eu-west-1 --only-show-errors
done
[ -d $D/probes ] || unzip -o -q $D/probes.zip -d $D
echo "probe folders $(ls $D/probes | wc -l)"
if [ -d /opt/cool ]; then HUB=cool; MODES="cool cvonly"; else HUB=stock; MODES="stock"; fi
for m in $MODES; do
  echo "== identify $m $(date +%T)"
  $R $m $J --probes $D/probes.json --root $D --out out/identify/ident_$m.jsonl 2>&1 | tail -5
  aws s3 cp out/identify/ $B/results/1d/$HUB/identify/ --recursive --region eu-west-1 --only-show-errors
done
first=$(echo $MODES | cut -d' ' -f1)
echo "== identify ${first} repeat $(date +%T)"
$R $first $J --probes $D/probes.json --root $D --out out/identify/ident_${first}_rep.jsonl --limit 60 2>&1 | tail -3
aws s3 cp out/identify/ $B/results/1d/$HUB/identify/ --recursive --region eu-west-1 --only-show-errors
echo IDENTIFY_DONE
date > out/identify/IDENTIFY_DONE && aws s3 cp out/identify/IDENTIFY_DONE $B/results/1d/$HUB/identify/IDENTIFY_DONE --region eu-west-1 --only-show-errors
