#!/usr/bin/env bash
# Explicit head migration; no installation and no overwrite of the initializer.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
PYTHON_BIN="${PYTHON_BIN:-python}"
INIT="${1:?Usage: bash scripts/train_codec_position_v3.sh /path/to/codec.pt [new_output]}"
OUT="${2:-$PROJECT/output/truck_position_v3_$(date +%Y%m%d_%H%M%S)}"
PLY="${PLY:-$PROJECT/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply}"
SCENE="${SCENE:-$PROJECT/data/tandt_db/tandt/truck}"
command -v "$PYTHON_BIN" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN.' >&2; exit 1; }
[[ -f "$INIT" && -f "$PLY" ]] || { echo "Missing checkpoint/PLY: $INIT | $PLY" >&2; exit 1; }
[[ -d "$SCENE/sparse" || -f "$SCENE/transforms_train.json" ]] || { echo "Missing scene: $SCENE" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Output exists: $OUT" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
printf 'Checkpoint: %s\nOutput: %s\nGPU: %s\n' "$INIT" "$OUT" "$CUDA_VISIBLE_DEVICES"
echo 'Explicit position_v3 upgrade: old sigmoid XYZ head is RESET, other weights are retained.'
echo 'A v3 initializer keeps its trained affine head. Adam state starts fresh in either case.'
exec "$PYTHON_BIN" -u -m gaussian_jscc train \
  --ply "$PLY" --init "$INIT" --source "$SCENE" --out "$OUT" \
  --loss-profile position_v3 --position-head normalized_affine_v3 --upgrade-position-head \
  --steps "${POSITION_STEPS:-3000}" --render-steps "${RENDER_STEPS:-300}" \
  --lr 5e-5 --render-lr 1e-5 --attr-weight 1 --render-target source \
  --tier-training all --attribute-drop 0 --clip-mode branch --clip-norm 1 \
  --position-eval-every 500 --position-eval-blocks 8 --position-eval-snrs 0 10 20 \
  --snr-range 0 20 --channel awgn --blocks-per-batch 32 --render-backward replay \
  --training-data-device cuda --resolution 2 --save-every 500 --profile-every 10 --device cuda
