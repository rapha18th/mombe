# One-time setup of /opt/cool_equiv on a hub (idempotent). Touches nothing the hub service uses.
export PATH=$PATH:/snap/bin
B=s3://mombe-967170427068-eu-west-1/jobs/1d
set -e
mkdir -p /opt/cool_equiv/code /opt/cool_equiv/takes /opt/cool_equiv/out
cd /opt/cool_equiv
aws s3 cp $B/code.zip /tmp/ce_code.zip --region eu-west-1 --only-show-errors
rm -rf code/mombe && unzip -o -q /tmp/ce_code.zip -d code
aws s3 cp $B/tools/ code/tools/cool_equiv/ --recursive --region eu-west-1 --only-show-errors
cp code/tools/cool_equiv/run_env.sh /opt/cool_equiv/run_env.sh && chmod +x /opt/cool_equiv/run_env.sh
ln -sfn /opt/mombe/models code/models
ln -sfn /opt/mombe/var code/var
for d in /opt/mombe/build/synth/*; do
  n=$(basename $d); mkdir -p takes/$n
  [ -f takes/$n/truth.json ] || cp $d/frame_*.jpg $d/truth.json takes/$n/
done
md5sum code/models/*.onnx | cut -c1-12
if [ -d /opt/cool ]; then
  if [ ! -x /opt/cool_equiv/stockvenv/bin/python ]; then
    python3.12 -m venv /opt/cool_equiv/stockvenv
    /opt/cool_equiv/stockvenv/bin/pip install -q opencv-contrib-python-headless==5.0.0.93 numpy==2.5.4 scipy==1.18.1
  fi
  [ -d /opt/cool_equiv/site_wheel/cv2 ] || /opt/cool/venvs/python_3.12/bin/pip install -q --no-deps --target /opt/cool_equiv/site_wheel opencv-contrib-python-headless==5.0.0.93
  for m in cool wheelvenv cvonly; do echo "-- $m"; /opt/cool_equiv/run_env.sh $m -c "import cv2, numpy, scipy; print(cv2.__version__, cv2.__file__, numpy.__version__, scipy.__version__)"; done
else
  [ -d /opt/cool_equiv/site_wheel/cv2 ] || /opt/stockcv/bin/pip install -q --no-deps --target /opt/cool_equiv/site_wheel opencv-contrib-python-headless==5.0.0.93
  for m in stock stockheadless; do echo "-- $m"; /opt/cool_equiv/run_env.sh $m -c "import cv2, numpy, scipy; print(cv2.__version__, cv2.__file__, numpy.__version__, scipy.__version__)"; done
fi
ls takes
echo SETUP_OK
