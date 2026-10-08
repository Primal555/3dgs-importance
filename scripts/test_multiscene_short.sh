#!/usr/bin/env bash
set -euo pipefail
export BOOTSTRAP_STEPS="${BOOTSTRAP_STEPS:-100}"
export RENDER_STEPS="${RENDER_STEPS:-100}"
export ALLOCATION_STEPS="${ALLOCATION_STEPS:-100}"
export JOINT_STEPS="${JOINT_STEPS:-50}"
export VALIDATE_EVERY="${VALIDATE_EVERY:-50}"
export TEST_VIEWS="${TEST_VIEWS:-4}"
export TEST_TRIALS="${TEST_TRIALS:-1}"
bash "$(dirname "$0")/train_multiscene_full.sh" "$@"
