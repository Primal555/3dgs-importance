#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Select an available GPU}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PLY="${PLY:-$PROJECT/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply}"
SCENE="${SCENE:-$PROJECT/data/tandt_db/tandt/truck}"
OUT="${1:-$PROJECT/output/truck_center_self_vs_light_$(date +%Y%m%d_%H%M%S)}"
[[ -f "$PLY" && -d "$SCENE" ]] || { echo 'Missing PLY or scene directory' >&2; exit 1; }
exec "$PYTHON_BIN" -u scripts/compare_center_decoders.py \
  --comparison self_light --readout-norm affine \
  --ply "$PLY" --source "$SCENE" --out "$OUT" --device cuda \
  --steps "${STEPS:-5000}" --blocks-per-batch "${BLOCKS_PER_BATCH:-32}" \
  --lr "${LR:-0.0002}" --seed "${SEED:-42}" --resolution "${RESOLUTION:-2}" \
  --validate-every 500 --render-every 500
