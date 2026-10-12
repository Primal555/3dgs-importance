#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available physical GPU}"
cd "$(dirname "$0")/.."
SOURCE_RUN="${1:?Usage: test_residual_freeze_controls.sh RESIDUAL_EXPERIMENT_RUN NEW_OUT [extra options]}"
CONTROL_OUT="${2:?Supply a NEW output directory outside the source run}"
shift 2
PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a PHASE_STEPS <<< "${STAGE_STEPS:-300 300 300}"
ARGS=(--run "$SOURCE_RUN" --out "$CONTROL_OUT" --stage-steps "${PHASE_STEPS[@]}"
      --lr "${LR:-1e-5}" --validate-every "${VALIDATE_EVERY:-100}"
      --views-per-step "${VIEWS_PER_STEP:-2}" --heldout-views "${HELDOUT_VIEWS:-4}"
      --validation-trials "${VALIDATION_TRIALS:-2}" --image-views "${IMAGE_VIEWS:-1}"
      --audit-blocks "${AUDIT_BLOCKS:-4}" --seed "${SEED:-42}")
if [[ -n "${MANIFEST:-}" ]]; then ARGS+=(--manifest "$MANIFEST"); fi
if [[ -n "${CHECKPOINT:-}" ]]; then ARGS+=(--checkpoint "$CHECKPOINT"); fi
if [[ -n "${BLOCKS_PER_BATCH:-}" ]]; then ARGS+=(--blocks-per-batch "$BLOCKS_PER_BATCH"); fi
exec "$PYTHON_BIN" -u -m gaussian_jscc.residual_freeze_controls "${ARGS[@]}" "$@"
