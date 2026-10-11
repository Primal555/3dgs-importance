#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available physical GPU}"
cd "$(dirname "$0")/.."
SOURCE_RUN="${1:?Supply the source multiscene run path; define RUN in THIS terminal first}"
CONTROL_OUT="${2:?Supply a NEW common output directory}"
shift 2
PYTHON_BIN="${PYTHON_BIN:-python}"
ARGS=(--run "$SOURCE_RUN" --out "$CONTROL_OUT"
      --bootstrap-steps "${BOOTSTRAP_STEPS:-2000}" --render-steps "${RENDER_STEPS:-1000}"
      --validate-every "${VALIDATE_EVERY:-250}" --lr "${LR:-1e-4}"
      --test-views "${TEST_VIEWS:-0}" --trials "${TEST_TRIALS:-2}" --image-views "${IMAGE_VIEWS:-2}")
read -r -a CONTROL_MODES <<< "${DECODER_GROUPS:-direct residual}"
ARGS+=(--groups "${CONTROL_MODES[@]}")
if [[ -n "${MANIFEST:-}" ]]; then ARGS+=(--manifest "$MANIFEST"); fi
if [[ -n "${COLOR_ROIS:-}" ]]; then ARGS+=(--color-rois "$COLOR_ROIS"); fi
exec "$PYTHON_BIN" -u -m gaussian_jscc.residual_controls "${ARGS[@]}" "$@"
