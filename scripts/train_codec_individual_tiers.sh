#!/usr/bin/env bash
# Explicit wire/loss upgrade. CPU diagnostics do not establish render quality.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
PYTHON_BIN="${PYTHON_BIN:-python}"
INIT="${1:?Usage: bash scripts/train_codec_individual_tiers.sh /path/to/codec.pt [new_output]}"
OUT="${2:-$PROJECT/output/truck_individual_tiers_$(date +%Y%m%d_%H%M%S)}"
PLY="${PLY:-$PROJECT/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply}"
SCENE="${SCENE:-$PROJECT/data/tandt_db/tandt/truck}"
[[ -f "$INIT" && -f "$PLY" ]] || { echo 'Missing checkpoint or PLY.' >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Output exists: $OUT" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
printf 'Initializer: %s\nOutput: %s\nGPU: %s\n' "$INIT" "$OUT" "$CUDA_VISIBLE_DEVICES"
exec "$PYTHON_BIN" -u -m gaussian_jscc train \
  --ply "$PLY" --source "$SCENE" --init "$INIT" --out "$OUT" \
  --position-head reference_v6 --upgrade-position-head --individual-tiers --loss-profile robust_v4 \
  --fixed-snr 10 --channel awgn --tier-training all --attribute-drop 0 \
  --steps "${STEPS:-3000}" --render-steps "${RENDER_STEPS:-0}" --lr "${LR:-0.00001}" \
  --clip-mode branch --clip-norm 1 --blocks-per-batch 32 --render-backward replay \
  --device cuda --training-data-device cuda --resolution 2 --save-every 500 --profile-every 10
