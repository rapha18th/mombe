#!/bin/bash
# Stage swaps between COOL and the stock wheel on the COOL machine, linear-algebra
# micro-ops under calibrateCamera, and dense repeatability. COOL hub only.
export PATH=$PATH:/snap/bin
cd /opt/cool_equiv
O=out/inject
B=s3://${MOMBE_BUCKET:?set MOMBE_BUCKET}
mkdir -p $O
R=/opt/cool_equiv/run_env.sh
J=code/tools/cool_equiv/bisect_take.py
C=code/tools/cool_equiv/compare_dumps.py
T=takes/sweep_21
COOL=out/bisect/sweep_21_s7_cool.pkl
WHEEL=out/bisect/sweep_21_s7_cvonly.pkl
$R cool $J --take $T --out $O/cool_inj_calib --inject calib --src $WHEEL
$R cool $J --take $T --out $O/cool_inj_lg --inject lg --src $WHEEL
$R cool $J --take $T --out $O/cool_inj_calib_lg --inject calib,lg --src $WHEEL
$R cvonly $J --take $T --out $O/wheel_inj_calib_lg --inject calib,lg --src $COOL
$R cvonly $J --take $T --out $O/wheel_inj_calib --inject calib --src $COOL
for x in cool_inj_calib cool_inj_lg cool_inj_calib_lg; do
  $R cvonly $C $O/$x.pkl $WHEEL --json $O/cmp_${x}__wheel.json
done
for x in wheel_inj_calib_lg wheel_inj_calib; do
  $R cvonly $C $O/$x.pkl $COOL --json $O/cmp_${x}__cool.json
done
# calibrateCamera and the linear algebra under it, same inputs on both builds
$R cool code/tools/cool_equiv/calib_ops.py run --dump $WHEEL --out $O/calib_ops_cool.pkl
$R cvonly code/tools/cool_equiv/calib_ops.py run --dump $WHEEL --out $O/calib_ops_wheel.pkl
$R cvonly code/tools/cool_equiv/calib_ops.py compare $O/calib_ops_cool.pkl $O/calib_ops_wheel.pkl $O/calib_ops_cmp.json
# dense repeatability on both builds
$R cool code/tools/cool_equiv/dense_repeat.py --take $T --out $O/dense_repeat_cool.json
$R cvonly code/tools/cool_equiv/dense_repeat.py --take $T --out $O/dense_repeat_wheel.json
aws s3 cp $O $B/results/1d/cool/inject/ --recursive --region eu-west-1 --only-show-errors --exclude "*.pkl" --include "calib_ops_*.pkl"
echo INJECT_DONE
touch $O/INJECT_DONE && aws s3 cp $O/INJECT_DONE $B/results/1d/cool/inject/INJECT_DONE --region eu-west-1 --only-show-errors
