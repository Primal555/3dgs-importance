#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Select an available GPU}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PLY="${PLY:-$PROJECT/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply}"
SCENE="${SCENE:-$PROJECT/data/tandt_db/tandt/truck}"
OUT="${1:-$PROJECT/output/truck_center_affine_$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-5000}"
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || { echo 'STEPS must be a positive integer' >&2; exit 1; }
[[ -f "$PLY" && -d "$SCENE" ]] || { echo 'Missing PLY or scene directory' >&2; exit 1; }
exec "$PYTHON_BIN" -u -m gaussian_jscc train-center-attributes \
  --ply "$PLY" --source "$SCENE" --out "$OUT" --device cuda \
  --center-decoder-kind transformer --center-readout-norm "${READOUT_NORM:-affine}" \
  --center-steps "$STEPS" --min-center-steps "$((STEPS+1))" \
  --attribute-steps 0 --joint-steps 0 --center-probe-blocks 32 \
  --center-lr "${LR:-0.0002}" --clip-norm 0 \
  --blocks-per-batch "${BLOCKS_PER_BATCH:-32}" --render-blocks-per-batch 64 \
  --hidden 96 --block-size 256 --validation-region-size 512 \
  --seed "${SEED:-42}" --resolution "${RESOLUTION:-2}" \
  --validate-every 500 --render-every 500 --save-every 500 --profile-every 50
