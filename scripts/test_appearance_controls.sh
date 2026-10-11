#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available physical GPU}"
cd "$(dirname "$0")/.."
RUN="${1:?Usage: test_appearance_controls.sh SOURCE_RUN NEW_OUT [extra options]}"
OUT="${2:?Supply a NEW common output directory}"
shift 2
PYTHON_BIN="${PYTHON_BIN:-python}"
ARGS=(--run "$RUN" --out "$OUT" --suite "${SUITE:-all}"
      --color-steps "${COLOR_STEPS:-2000}" --bootstrap-steps "${BOOTSTRAP_STEPS:-2000}"
      --render-steps "${RENDER_STEPS:-1000}" --validate-every "${VALIDATE_EVERY:-250}"
      --lr "${LR:-1e-4}" --test-views "${TEST_VIEWS:-0}" --trials "${TEST_TRIALS:-2}"
      --image-views "${IMAGE_VIEWS:-2}")
if [[ -n "${MANIFEST:-}" ]]; then ARGS+=(--manifest "$MANIFEST"); fi
if [[ -n "${SCENES:-}" ]]; then
  read -r -a SELECTED <<< "$SCENES"
  ARGS+=(--scenes "${SELECTED[@]}")
fi
if [[ -n "${COLOR_ROIS:-}" ]]; then ARGS+=(--color-rois "$COLOR_ROIS"); fi
exec "$PYTHON_BIN" -u -m gaussian_jscc.appearance_controls "${ARGS[@]}" "$@"
