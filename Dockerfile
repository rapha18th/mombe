# syntax=docker/dockerfile:1
# Mombe vision hub on stock OpenCV 5.0.0 from PyPI: the baseline path, linux/arm64 and linux/amd64.
# The deployed hub runs on the COOL AMI instead (OpenCV 5.1.0-dev with SVE, KleidiCV 0.7.0); see README.
#
#   docker buildx build --platform linux/arm64,linux/amd64 -t mombe-hub:stock .
#   docker run --rm -p 8080:8080 -e MOMBE_KEY=change-me -v "$PWD/var:/opt/mombe/var" mombe-hub:stock
#
# The image holds code, the four ONNX models the hub runs (fetched and SHA-1 checked at build time)
# and the frozen identity thresholds. The registry gallery lives in /opt/mombe/var: mount the one
# scripts/setup.sh builds, or start empty and enrol. Mount data/muzzle read-only at
# /opt/mombe/data/muzzle to draw the matched gallery image in identity evidence.
#
# OPENCV_REQ picks the wheel. The default is the headless wheel of the stock release.
# The deployed stock hub ran opencv-contrib-python==5.0.0.93; pass that to rebuild it exactly.
FROM python:3.12-slim-bookworm

ARG OPENCV_REQ=opencv-python-headless==5.0.0.93
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MOMBE_HOME=/opt/mombe
WORKDIR /opt/mombe

COPY requirements-lock.txt ./
RUN set -eux; \
    pip install -r requirements-lock.txt; \
    case "$OPENCV_REQ" in \
      opencv-python-headless==5.0.0.93) ;; \
      *headless*) pip uninstall -y opencv-python-headless; pip install "$OPENCV_REQ" ;; \
      *) apt-get update; apt-get install -y --no-install-recommends libgl1 libglib2.0-0; \
         rm -rf /var/lib/apt/lists/*; pip uninstall -y opencv-python-headless; pip install "$OPENCV_REQ" ;; \
    esac; \
    python -c "import cv2; print('OpenCV', cv2.__version__); assert hasattr(cv2, 'ALIKED_create') and hasattr(cv2, 'LightGlueMatcher_create')"

COPY tools/fetch_models.py tools/fetch_models.py
RUN python tools/fetch_models.py --only aliked aliked_lightglue dinov2_small depth_anything_v2_small

COPY mombe/ mombe/
COPY web/ web/
COPY config/thresholds.json eval/out/thresholds.json

RUN useradd --system --user-group --uid 10001 --home-dir /opt/mombe mombe \
    && mkdir -p var/store var/registry var/enrolled \
    && chown -R mombe:mombe var
USER mombe
VOLUME ["/opt/mombe/var"]
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=6s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=5)"
# The hub listens on every interface here, so it starts only with a key of your own:
# MOMBE_KEY guards /mcp and /api; MOMBE_OFFICER_KEYS adds passcodes for /api.
CMD ["sh", "-c", "if [ -z \"$MOMBE_KEY\" ] || [ \"$MOMBE_KEY\" = dev-key ]; then echo 'set MOMBE_KEY to a secret of your own' >&2; exit 64; fi; exec python -m mombe.server.mcp_app --host 0.0.0.0 --port 8080"]
