#!/usr/bin/env bash
# Agent scenario suite: 23 session kinds (verify, enrol, custody, value, body gates, synthetic
# presentation attacks, double enrolment), each on held-out animals, scored against the
# scripted expected outcome (input S over real noses, truth G).
#
#   PY=python bash scripts/eval_agent.sh                                  # autopilot and open loop, no AWS
#   DRIVERS="claude autopilot open_loop" RUN=mine bash scripts/eval_agent.sh   # adds Claude on Bedrock
#
# Needs scripts/setup.sh (features, thresholds, duplicate list). The claude driver needs the
# packages in requirements-agent.txt and calls
# Claude Sonnet 4.6 through Amazon Bedrock (eu.anthropic.claude-sonnet-4-6 by default; set
# MOMBE_MODEL_ID and AWS_REGION to change it) with the AWS credentials of this shell.
# Each run directory is replaced, so give every run its own RUN name.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
RUN="${RUN:-local-$(date +%m%d-%H%M)}"
mkdir -p eval/out
[ -f eval/out/thresholds.json ] || cp config/thresholds.json eval/out/thresholds.json
[ -f eval/out/duplicates.json ] || cp config/duplicates.json eval/out/duplicates.json
# shellcheck disable=SC2086
"$PY" eval/run_scenarios.py --drivers ${DRIVERS:-autopilot open_loop} --per-kind "${PER_KIND:-2}" --run "$RUN"
"$PY" scripts/agent_table.py "eval/out/$RUN"
