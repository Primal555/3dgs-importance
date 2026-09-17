#!/usr/bin/env bash
# Render-first mainline. No checkpoint is silently selected.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="${1:-$PROJECT/output/truck_learned_$(date +%Y%m%d_%H%M%S)}"
PLY="${PLY:-$PROJECT/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply}"
SCENE="${SCENE:-$PROJECT/data/tandt_db/tandt/truck}"
[[ -f "$PLY" ]] || { echo "Missing PLY: $PLY" >&2; exit 1; }
[[ -d "$SCENE/sparse" || -f "$SCENE/transforms_train.json" ]] || { echo "Missing scene: $SCENE" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Output exists: $OUT" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN to its absolute python path.' >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
EXTRA=()
if [[ -n "${INIT:-}" ]]; then
  [[ -f "$INIT" ]] || { echo "Missing initializer: $INIT" >&2; exit 1; }
  EXTRA+=(--init "$INIT")
  DEFAULT_BOOTSTRAP=0
else
  DEFAULT_BOOTSTRAP=500
fi
printf 'Architecture: learned_joint\nGPU: %s\nOutput: %s\n' "$CUDA_VISIBLE_DEVICES" "$OUT"
exec "$PYTHON_BIN" -u -m gaussian_jscc train-learned \
  --ply "$PLY" --source "$SCENE" --out "$OUT" "${EXTRA[@]}" \
  --device cuda --snr 10 --channel awgn \
  --bootstrap-steps "${BOOTSTRAP_STEPS:-${STEPS:-$DEFAULT_BOOTSTRAP}}" --render-steps "${RENDER_STEPS:-1000}" --joint-steps "${JOINT_STEPS:-0}" \
  --block-size 256 --decoder-window 32 --blocks-per-batch "${BLOCKS_PER_BATCH:-32}" \
  --rates 0 8 16 32 --lr "${LR:-0.0001}" --render-lr "${RENDER_LR:-0.00001}" \
  --clip-mode "${CLIP_MODE:-none}" --clip-norm "${CLIP_NORM:-10}" \
  --render-backward replay --training-data-device "${TRAINING_DATA_DEVICE:-cpu}" --resolution "${RESOLUTION:-2}" \
  --train-views "${TRAIN_VIEWS:-0}" --views-per-step "${VIEWS_PER_STEP:-2}" \
  --validate-every "${VALIDATE_EVERY:-100}" --validation-views "${VALIDATION_VIEWS:-4}" \
  --validation-trials "${VALIDATION_TRIALS:-2}" --patience "${PATIENCE:-8}" --save-every "${SAVE_EVERY:-500}"
