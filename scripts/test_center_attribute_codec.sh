#!/usr/bin/env bash
# All phases are clean representation learning. No JSCC noise or rate claims.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Select an available GPU before launching}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PLY="${PLY:-$PROJECT/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply}"
SCENE="${SCENE:-$PROJECT/data/tandt_db/tandt/truck}"
OUT="${1:-$PROJECT/output/truck_center_attribute_$(date +%Y%m%d_%H%M%S)}"
[[ -z "${INIT:-}" ]] || { echo 'Random initialization only; unset INIT. Use --resume for exact recovery.' >&2; exit 1; }
[[ -f "$PLY" && -d "$SCENE" ]] || { echo 'Missing PLY or scene directory.' >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN.' >&2; exit 1; }
echo "GPU=$CUDA_VISIBLE_DEVICES; center -> minibatch attributes (no rasterizer) -> joint render replay; no communication"
exec "$PYTHON_BIN" -u -m gaussian_jscc train-center-attributes \
  --ply "$PLY" --source "$SCENE" --out "$OUT" --device cuda \
  --center-steps "${CENTER_STEPS:-5000}" --attribute-steps "${ATTRIBUTE_STEPS:-1000}" \
  --joint-steps "${JOINT_STEPS:-1000}" \
  --min-center-steps "${MIN_CENTER_STEPS:-500}" --min-attribute-steps "${MIN_ATTRIBUTE_STEPS:-200}" \
  --min-joint-steps "${MIN_JOINT_STEPS:-200}" \
  --center-lr "${CENTER_LR:-0.0002}" --attribute-lr "${ATTRIBUTE_LR:-0.0002}" \
  --joint-center-lr "${JOINT_CENTER_LR:-0.0002}" --joint-attribute-lr "${JOINT_ATTRIBUTE_LR:-0.0002}" \
  --center-smoothing "${CENTER_SMOOTHING:-0.001}" --center-max-gap-db "${CENTER_MAX_GAP_DB:-3}" \
  --attribute-min-improvement "${ATTRIBUTE_MIN_IMPROVEMENT:-0.05}" \
  --attribute-views "${ATTRIBUTE_VIEWS:-4}" \
  --transition-patience "${TRANSITION_PATIENCE:-3}" --stop-patience "${STOP_PATIENCE:-5}" \
  --latent-dim "${LATENT_DIM:-64}" --center-latent-dim "${CENTER_LATENT_DIM:-32}" \
  --block-size "${BLOCK_SIZE:-256}" --blocks-per-batch "${BLOCKS_PER_BATCH:-32}" \
  --render-blocks-per-batch "${RENDER_BLOCKS_PER_BATCH:-64}" --render-backward replay \
  --validate-every "${VALIDATE_EVERY:-100}" --render-every "${RENDER_EVERY:-500}" \
  --save-every 500 --profile-every 10 --resolution "${RESOLUTION:-2}" --seed "${SEED:-42}"
