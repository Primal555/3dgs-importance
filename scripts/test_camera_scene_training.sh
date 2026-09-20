#!/usr/bin/env bash
# Random-start learned XYZ; teacher coordinates are TRAINING labels only.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PLY="${PLY:-$PROJECT/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply}"
SCENE="${SCENE:-$PROJECT/data/tandt_db/tandt/truck}"
OUT="${1:-$PROJECT/output/truck_camera_scene_$(date +%Y%m%d_%H%M%S)}"
[[ -f "$PLY" ]] || { echo "Missing PLY: $PLY" >&2; exit 1; }
[[ -d "$SCENE/sparse" || -f "$SCENE/transforms_train.json" ]] || { echo "Missing scene: $SCENE" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Output already exists: $OUT" >&2; exit 1; }
echo "GPU: $CUDA_VISIBLE_DEVICES; output: $OUT"
echo 'Random weights; learned XYZ payload; camera initialization -> complete decoded scene rendering.'
exec "$PYTHON_BIN" -u -m gaussian_jscc train-learned \
  --ply "$PLY" --source "$SCENE" --out "$OUT" --device cuda \
  --architecture learned_split_logcov --position-delivery learned \
  --bootstrap-steps 0 --camera-init-steps "${CAMERA_INIT_STEPS:-1000}" \
  --render-steps "${RENDER_STEPS:-1000}" --joint-steps 0 \
  --camera-geometry-weight "${CAMERA_GEOMETRY_WEIGHT:-0.01}" \
  --camera-pixel-scale "${CAMERA_PIXEL_SCALE:-2}" --camera-depth-scale "${CAMERA_DEPTH_SCALE:-0.1}" \
  --rates 0 8 16 32 --snr "${SNR:-10}" --channel "${CHANNEL:-awgn}" \
  --lr "${LR:-2e-4}" --render-lr "${LR:-2e-4}" --lr-schedule constant \
  --clip-mode none --drop 0 --blocks-per-batch "${BLOCKS_PER_BATCH:-32}" \
  --render-backward replay --training-data-device "${TRAINING_DATA_DEVICE:-cpu}" \
  --resolution "${RESOLUTION:-4}" --views-per-step "${VIEWS_PER_STEP:-2}" \
  --validation-views "${VALIDATION_VIEWS:-4}" --validation-trials "${VALIDATION_TRIALS:-2}" \
  --validate-every "${VALIDATE_EVERY:-500}" --save-every "${SAVE_EVERY:-500}" \
  --patience 0 --seed "${SEED:-42}"
