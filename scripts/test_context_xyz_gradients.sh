#!/usr/bin/env bash
# No installation, no source checkpoint overwrite, one GPU, arms run sequentially.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
PYTHON_BIN="${PYTHON_BIN:-python}"
INIT="${1:?Usage: bash scripts/test_context_xyz_gradients.sh /path/to/codec.pt [new_output]}"
OUT="${2:-$PROJECT/output/context_xyz_gradients_$(date +%Y%m%d_%H%M%S)}"
PLY="${PLY:-$PROJECT/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply}"
SCENE="${SCENE:-$PROJECT/data/tandt_db/tandt/truck}"
command -v "$PYTHON_BIN" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN.' >&2; exit 1; }
[[ -f "$INIT" && -f "$PLY" ]] || { echo "Missing checkpoint/PLY: $INIT | $PLY" >&2; exit 1; }
[[ -d "$SCENE/sparse" || -f "$SCENE/transforms_train.json" ]] || { echo "Missing scene: $SCENE" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Output exists: $OUT" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
printf 'Checkpoint: %s\nOutput: %s\nGPU: %s\n' "$INIT" "$OUT" "$CUDA_VISIBLE_DEVICES"
exec "$PYTHON_BIN" -u diagnose_geometry_gradients.py \
  --ply "$PLY" --checkpoint "$INIT" --source "$SCENE" --out "$OUT" \
  --steps "${DIAG_STEPS:-200}" --render-steps "${DIAG_RENDER_STEPS:-100}" \
  --lr 5e-5 --render-lr 1e-5 --attr-weight 1 --clip-norm 1 \
  --probe-every 50 --eval-every 100 --eval-blocks 8 --eval-snrs 0 10 20 \
  --snr-range 0 20 --blocks-per-batch 32 --training-data-device cuda \
  --test-views 2 --render-eval-tier 2 --render-eval-snr 10 --resolution 2 --device cuda
