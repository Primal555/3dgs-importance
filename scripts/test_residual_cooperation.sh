#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available physical GPU}"
cd "$(dirname "$0")/.."
SOURCE_RUN="${1:?Usage: test_residual_cooperation.sh RESIDUAL_EXPERIMENT_RUN NEW_OUT [extra options]}"
CONTROL_OUT="${2:?Supply a NEW output directory outside the source run}"
shift 2
PYTHON_BIN="${PYTHON_BIN:-python}"
ARGS=(--run "$SOURCE_RUN" --out "$CONTROL_OUT"
      --probes "${PROBES:-8}" --views-per-probe "${VIEWS_PER_PROBE:-2}"
      --heldout-views "${HELDOUT_VIEWS:-4}" --image-views "${IMAGE_VIEWS:-1}"
      --seed "${SEED:-42}")
read -r -a PROBE_LRS <<< "${PROBE_LRS:-1e-4 1e-5}"
ARGS+=(--lrs "${PROBE_LRS[@]}")
if [[ -n "${MANIFEST:-}" ]]; then ARGS+=(--manifest "$MANIFEST"); fi
if [[ -n "${CHECKPOINT:-}" ]]; then ARGS+=(--checkpoint "$CHECKPOINT"); fi
if [[ -n "${BLOCKS_PER_BATCH:-}" ]]; then ARGS+=(--blocks-per-batch "$BLOCKS_PER_BATCH"); fi
exec "$PYTHON_BIN" -u -m gaussian_jscc.residual_cooperation "${ARGS[@]}" "$@"
