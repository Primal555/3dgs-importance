#!/usr/bin/env bash
# Controlled XYZ-delivery experiment. Same weights, draws, payload and cameras;
# side-stream cost is EXTRA, so this is not an equal-total-rate comparison.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
OUT="${1:-$PROJECT/output/truck_position_delivery_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
command -v "$PYTHON_BIN" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN.' >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Output exists: $OUT" >&2; exit 1; }
# Do not guess which shared GPU is free. Respect the caller's explicit selection.
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo 'Set CUDA_VISIBLE_DEVICES to an available GPU.' >&2; exit 1; }
export INITIALIZATION=random BOOTSTRAP_STEPS=0 JOINT_STEPS=0
export RENDER_STEPS="${RENDER_STEPS:-300}" TRAIN_VIEWS="${TRAIN_VIEWS:-12}"
export VIEWS_PER_STEP="${VIEWS_PER_STEP:-2}" BLOCKS_PER_BATCH="${BLOCKS_PER_BATCH:-32}"
export VALIDATE_EVERY="${VALIDATE_EVERY:-25}" VALIDATION_VIEWS="${VALIDATION_VIEWS:-4}"
export VALIDATION_TRIALS="${VALIDATION_TRIALS:-2}" PATIENCE=0 SAVE_EVERY="${SAVE_EVERY:-100}"
export POSITION_BITS="${POSITION_BITS:-12}" POSITION_NET_BITS_PER_USE="${POSITION_NET_BITS_PER_USE:-2}"
mkdir -p "$OUT"
for mode in learned float32 quantized; do
  export POSITION_DELIVERY="$mode"
  printf '\nStarting %s; GPU=%s; log=%s/%s.log\n' "$mode" "$CUDA_VISIBLE_DEVICES" "$OUT" "$mode"
  bash "$PROJECT/scripts/train_codec_learned.sh" "$OUT/$mode" 2>&1 | tee "$OUT/$mode.log"
done
"$PYTHON_BIN" -u "$PROJECT/scripts/summarize_position_delivery.py" "$OUT"
printf 'Done: %s\nInspect summary.json, validation_comparison.csv and each run/validation_images/.\n' "$OUT"
