#!/usr/bin/env bash
# Fixed 10 dB geometry optimization. Does not launch rendering automatically.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
PYTHON_BIN="${PYTHON_BIN:-python}"
INIT="${1:?Usage: bash scripts/train_codec_fixed_geometry.sh /path/to/codec.pt [new_output]}"
OUT="${2:-$PROJECT/output/truck_reference_v6_$(date +%Y%m%d_%H%M%S)}"
PLY="${PLY:-$PROJECT/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply}"
command -v "$PYTHON_BIN" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN.' >&2; exit 1; }
[[ -f "$INIT" && -f "$PLY" ]] || { echo "Missing checkpoint/PLY: $INIT | $PLY" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Output exists: $OUT" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
printf 'Initializer: %s\nOutput: %s\nGPU: %s\n' "$INIT" "$OUT" "$CUDA_VISIBLE_DEVICES"
echo 'Fixed AWGN 10 dB; random noise each step; q1/q2/q3; ONLY geometric payload weights train.'
echo 'V6 resets geometry on migration; trains shared reference, local detail and mixed/drop layouts.'
exec "$PYTHON_BIN" -u -m gaussian_jscc train \
  --ply "$PLY" --init "$INIT" --out "$OUT" \
  --position-head reference_v6 --upgrade-position-head --loss-profile position_v3 \
  --geometry-group-size "${GEOMETRY_GROUP_SIZE:-256}" \
  --geometry-only --fixed-snr 10 --channel awgn \
  --steps "${POSITION_STEPS:-3000}" --render-steps 0 --lr "${POSITION_LR:-0.0001}" \
  --tier-training all --attribute-drop 0 --clip-mode branch --clip-norm 1 \
  --position-eval-every "${EVAL_EVERY:-100}" --position-eval-blocks "${EVAL_BLOCKS:-16}" \
  --position-patience "${POSITION_PATIENCE:-8}" --position-min-delta 0 \
  --blocks-per-batch 32 --training-data-device cuda --device cuda \
  --save-every 500 --profile-every 10
