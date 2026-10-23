# One-command targets. Each target calls a script in scripts/, so the same steps run without make:
#   bash scripts/<name>.sh
# PY selects the interpreter, e.g.  make test PY=.venv/bin/python
PY ?= python
export PY

.PHONY: help setup test hub eval-identity eval-body-synthetic eval-ruchay eval-agent docker lock

help:
	@echo "setup                models, muzzle data, features, registry, frozen thresholds"
	@echo "test                 unit tests (no data, models or AWS)"
	@echo "hub                  vision hub on http://127.0.0.1:8080 (MCP /mcp, app /app/)"
	@echo "eval-identity        held-out identity, BC et al. 2026 protocol (long; LIMIT=20 for a smoke run)"
	@echo "eval-body-synthetic  seven synthetic walk-arounds through the body pipeline"
	@echo "eval-ruchay          real Hereford clouds against tape (downloads 4.8 GB; research use only)"
	@echo "eval-agent           scenario suite; DRIVERS=\"claude autopilot open_loop\" adds Claude on Bedrock"
	@echo "docker               stock OpenCV 5.0.0 hub image for linux/arm64 and linux/amd64"
	@echo "lock                 regenerate requirements-lock.txt with uv"

setup:
	bash scripts/setup.sh

test:
	bash scripts/test.sh

hub:
	bash scripts/hub.sh

eval-identity:
	bash scripts/eval_identity.sh

eval-body-synthetic:
	bash scripts/eval_body_synthetic.sh

eval-ruchay:
	bash scripts/eval_ruchay.sh

eval-agent:
	bash scripts/eval_agent.sh

docker:
	docker buildx build --platform linux/arm64,linux/amd64 -t mombe-hub:stock .

lock:
	uv pip compile --universal --python-version 3.12 requirements-hub.in -o requirements-lock.txt
