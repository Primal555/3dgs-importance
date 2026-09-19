#!/usr/bin/env bash
# Random initialization -> isolated response pretraining -> scene render MSE.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU before running}"
export INITIALIZATION=random
export BOOTSTRAP_OBJECTIVE=local-response
export POSITION_DELIVERY="${POSITION_DELIVERY:-quantized}"
[[ "$POSITION_DELIVERY" != learned ]] || { echo 'Local response requires explicit XYZ delivery.' >&2; exit 1; }
export POSITION_BITS="${POSITION_BITS:-12}"
export BOOTSTRAP_STEPS="${BOOTSTRAP_STEPS:-2000}"
export RENDER_STEPS="${RENDER_STEPS:-300}"
export JOINT_STEPS=0
export LR="${LR:-0.0002}"
export RENDER_LR="${RENDER_LR:-0.0002}"
export LR_SCHEDULE="${LR_SCHEDULE:-constant}"
export TRAIN_VIEWS="${TRAIN_VIEWS:-12}"
export VIEWS_PER_STEP="${VIEWS_PER_STEP:-2}"
export VALIDATE_EVERY="${VALIDATE_EVERY:-100}"
export VALIDATION_VIEWS="${VALIDATION_VIEWS:-4}"
export VALIDATION_TRIALS="${VALIDATION_TRIALS:-2}"
export SAVE_EVERY="${SAVE_EVERY:-500}"
export PATIENCE=0
OUT="${1:-$PROJECT/output/truck_local_response_$(date +%Y%m%d_%H%M%S)}"
exec bash "$PROJECT/scripts/train_codec_learned.sh" "$OUT"
