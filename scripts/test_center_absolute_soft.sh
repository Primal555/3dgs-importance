#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Select an available GPU}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PLY="${PLY:-$PROJECT/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply}"
SCENE="${SCENE:-$PROJECT/data/tandt_db/tandt/truck}"
OUT="${1:-$PROJECT/output/truck_center_absolute_soft_$(date +%Y%m%d_%H%M%S)}"
[[ -f "$PLY" && -d "$SCENE" ]] || { echo 'Missing PLY or scene directory' >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Choose a new output directory: $OUT" >&2; exit 1; }
mkdir -p "$OUT"
exec > >(tee -a "$OUT/console.log") 2>&1
echo "GPU: $CUDA_VISIBLE_DEVICES; RANDOM start; per-point absolute XYZ; no block centroid"
echo "Baseline: 52edc1c; self-only decoder; affine readout; 32 center latent dimensions"
echo "5000 center steps; continuous Adam; no loss approval or momentum restart"
echo "Microbatches/update: ${CENTER_ACCUMULATION_STEPS:-1}; 32 blocks each; update count remains 5000"
echo "LR: 1e-4 for first half, cosine to 2e-5; proposal-size protection: 100-step median x3"
exec "$PYTHON_BIN" -u -m gaussian_jscc train-center-attributes \
  --ply "$PLY" --source "$SCENE" --out "$OUT" --device cuda \
  --center-decoder-kind transformer --center-readout-norm affine --center-attention-scope self \
  --center-steps 5000 --min-center-steps 5001 --attribute-steps 0 --joint-steps 0 \
  --center-lr 0.0001 --clip-norm 0 --center-probe-blocks 32 --center-drift-every 50 \
  --latent-dim 64 --center-latent-dim 32 --decoder-depth 4 --attention-heads 4 \
  --blocks-per-batch 32 --render-blocks-per-batch 64 --hidden 96 --block-size 256 \
  --validation-region-size 512 --seed 42 --resolution "${RESOLUTION:-2}" \
  --validate-every 500 --render-every 500 --save-every 500 --profile-every 50 \
  --center-update-policy soft --center-update-window 100 --center-update-warmup 100 \
  --center-update-multiplier 3 --center-lr-schedule late_cosine \
  --center-lr-decay-start 0.5 --center-lr-end-ratio 0.2 \
  --center-accumulation-steps "${CENTER_ACCUMULATION_STEPS:-1}"
