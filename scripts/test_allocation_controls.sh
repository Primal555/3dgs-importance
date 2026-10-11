#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
cd "$(dirname "$0")/.."
RUN="${1:?Supply the completed multiscene run directory}"
OUT="${2:?Supply a NEW output directory}"
shift 2
PYTHON_BIN="${PYTHON_BIN:-python}"
ARGS=(--run "$RUN" --out "$OUT" --allocation-steps "${ALLOCATION_STEPS:-1000}"
      --validate-every "${VALIDATE_EVERY:-250}" --test-views "${TEST_VIEWS:-0}"
      --trials "${TEST_TRIALS:-3}" --image-views "${IMAGE_VIEWS:-2}")
if [[ -n "${MANIFEST:-}" ]]; then ARGS+=(--manifest "$MANIFEST"); fi
if [[ -n "${SCENES:-}" ]]; then
  read -r -a SELECTED <<< "$SCENES"
  ARGS+=(--scenes "${SELECTED[@]}")
fi
"$PYTHON_BIN" -u -m gaussian_jscc.allocation_controls "${ARGS[@]}" "$@"
