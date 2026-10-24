#!/bin/bash
# Render extra synthetic takes with the frozen code (stock hub), check that the
# renderer reproduces sweep_21 byte for byte, and ship frames + truth to S3.
export PATH=$PATH:/snap/bin
cd /opt/cool_equiv
R=/opt/cool_equiv/run_env.sh
S=code/build/synth
B=s3://${MOMBE_BUCKET:?set MOMBE_BUCKET}/jobs/1d/takes
$R stock code/tools/synth_walk.py --name sweep_21_check --seed 21 --scale 0.85
echo "render check: $(cat $S/sweep_21_check/frame_*.jpg | md5sum | cut -c1-12) vs $(cat takes/sweep_21/frame_*.jpg | md5sum | cut -c1-12)"
for sp in 28:0.8 29:0.88 30:0.93 31:0.97 32:1.03 33:1.1 34:1.18 35:0.82; do
  s=${sp%%:*}; k=${sp##*:}; n=sweep_$s
  $R stock code/tools/synth_walk.py --name $n --seed $s --scale $k
  mkdir -p takes/$n && cp $S/$n/frame_*.jpg $S/$n/truth.json takes/$n/
  aws s3 cp takes/$n $B/$n/ --recursive --region eu-west-1 --only-show-errors
done
echo RENDER_DONE
echo done | aws s3 cp - $B/RENDER_DONE --region eu-west-1
