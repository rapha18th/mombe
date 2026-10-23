export PATH=$PATH:/snap/bin
echo "== host"; uname -m; nproc; df -h / | tail -1; free -g | head -2; uptime
echo "== python envs"; ls -d /opt/cool/venvs/python_3.12 /opt/stockcv 2>/dev/null
for e in /opt/cool/venvs/python_3.12 /opt/stockcv; do
  [ -x $e/bin/python ] || continue
  echo "-- $e"
  if [ $e = /opt/cool/venvs/python_3.12 ]; then
    export PYTHONPATH=/opt/cool/python_3.12/site-packages/cv2/python-3.12 LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib
  else
    unset PYTHONPATH LD_LIBRARY_PATH
  fi
  $e/bin/python -c "
import cv2, numpy, scipy, sys
print('python', sys.version.split()[0], 'cv2', cv2.__version__, cv2.__file__)
print('numpy', numpy.__version__, 'scipy', scipy.__version__)
bi = cv2.getBuildInformation()
for l in bi.splitlines():
    s = l.strip()
    if any(k in s for k in ('Baseline', 'Dispatched', 'KleidiCV', 'HAL', 'JPEG', 'PNG', 'Parallel', 'Lapack', 'Eigen', 'IPP', 'carotene', 'Carotene', 'NEON', 'SVE', 'Version control', 'C++ flags (Release)', 'Extra modules', 'Python 3:', 'CPU/HW')):
        print('   ', s[:200])
print('useOptimized', cv2.useOptimized(), 'threads', cv2.getNumThreads(), 'cpus', cv2.getNumberOfCPUs())
print('cpu features', cv2.getCPUFeaturesLine())
"
  $e/bin/pip list 2>/dev/null | grep -i -E "numpy|scipy|opencv"
done
unset PYTHONPATH LD_LIBRARY_PATH
echo "== mombe dir"; ls /opt/mombe; ls /opt/mombe/build/synth 2>/dev/null; ls /opt/mombe/models
for d in /opt/mombe/build/synth/*; do echo "$d $(ls $d | wc -l) $(ls $d | grep -c sfm_cache)"; done 2>/dev/null
echo "== synth hashes sweep_21 frame_00/10"; md5sum /opt/mombe/build/synth/sweep_21/frame_00.jpg /opt/mombe/build/synth/sweep_21/frame_10.jpg /opt/mombe/build/synth/sweep_21/truth.json 2>/dev/null
ls -la /opt/mombe/build/synth/sweep_21 2>/dev/null | head -5
echo "== libs"; ldd /opt/cool/python_3.12/site-packages/cv2/python-3.12/cv2*.so 2>/dev/null | grep -i -E "jpeg|kleidi|opencv_core|png|gomp|tbb|blas" | head
ls /opt/cool/cpp_sdk/lib 2>/dev/null | head -50
python3 -c "import sys; print(sys.version)"; dpkg -l | grep -E "python3.12-venv|libgl1 |libglib2.0-0" | awk '{print $2, $3}'
echo "== services"; systemctl is-active mombe-hub; cat /opt/mombe/hub.env 2>/dev/null | grep -v -i key
