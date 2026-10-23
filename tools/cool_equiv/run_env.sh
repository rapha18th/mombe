#!/bin/bash
# Run a Python job under one of the OpenCV environments on a hub.
#   cool       COOL hub: COOL venv + OpenCV 5.1.0-dev (SVE, KleidiCV 0.7.0), numpy 2.5.1
#   wheelvenv  COOL hub: fresh venv with the stock wheel (headless 5.0.0.93), numpy 2.5.4, scipy 1.18.1
#   cvonly     COOL hub: COOL venv python and numpy, cv2 swapped for the stock wheel (headless 5.0.0.93)
#   stock      stock hub: /opt/stockcv (opencv-contrib-python 5.0.0.93), numpy 2.5.4
#   stockheadless  stock hub: /opt/stockcv python with the headless wheel in front
mode=$1; shift
case $mode in
  cool) PY=/opt/cool/venvs/python_3.12/bin/python
        export PYTHONPATH=/opt/cool/python_3.12/site-packages/cv2/python-3.12
        export LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib ;;
  wheelvenv) PY=/opt/cool_equiv/stockvenv/bin/python; unset PYTHONPATH LD_LIBRARY_PATH ;;
  cvonly) PY=/opt/cool/venvs/python_3.12/bin/python; export PYTHONPATH=/opt/cool_equiv/site_wheel; unset LD_LIBRARY_PATH ;;
  stock) PY=/opt/stockcv/bin/python; unset PYTHONPATH LD_LIBRARY_PATH ;;
  stockheadless) PY=/opt/stockcv/bin/python; export PYTHONPATH=/opt/cool_equiv/site_wheel; unset LD_LIBRARY_PATH ;;
  *) echo "unknown mode $mode"; exit 2 ;;
esac
export MOMBE_CODE=/opt/cool_equiv/code
cd /opt/cool_equiv
exec nice -n 10 $PY "$@"
