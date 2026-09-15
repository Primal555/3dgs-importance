#!/usr/bin/env bash
# Run from an already configured maskgs environment. No installs or file deletion.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
PYTHON_BIN="${PYTHON_BIN:-python}"
PLY="${PLY:-$PROJECT/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply}"
SCENE="${SCENE:-$PROJECT/data/tandt_db/tandt/truck}"
INIT="${1:-${INIT:-$PROJECT/output/truck_geometry_first_20260915_095729/codec.pt}}"
OUT="${2:-${OUT:-$PROJECT/output/truck_codec_balanced_$(date +%Y%m%d_%H%M%S)}}"
command -v "$PYTHON_BIN" >/dev/null || { echo "Python not found; activate maskgs or set PYTHON_BIN." >&2; exit 1; }
[[ -f "$PLY" && -f "$INIT" ]] || { echo "Missing PLY or codec: $PLY | $INIT" >&2; exit 1; }
[[ -d "$SCENE/sparse" || -f "$SCENE/transforms_train.json" ]] || { echo "Missing scene: $SCENE" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Output already exists: $OUT" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
printf 'Initializer: %s\nOutput: %s\nVisible GPU(s): %s\n' "$INIT" "$OUT" "$CUDA_VISIBLE_DEVICES"
exec "$PYTHON_BIN" -u -m gaussian_jscc train \
  --ply "$PLY" --source "$SCENE" --init "$INIT" --out "$OUT" \
  --loss-profile balanced_v2 --steps "${REPAIR_STEPS:-3000}" --render-steps "${RENDER_STEPS:-1000}" \
  --lr 5e-5 --render-lr 1e-5 --attr-weight 1 \
  --snr-range 0 20 --channel awgn --attribute-drop 0 \
  --render-target source --blocks-per-batch 32 --render-backward replay \
  --training-data-device cuda --resolution 2 --device cuda \
  --save-every 500 --profile-every 10
