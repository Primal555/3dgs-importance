#!/usr/bin/env bash
# Random representation training by default. All artifacts live in OUT.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Select an available GPU before launching}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PLY="${PLY:-$PROJECT/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply}"
SCENE="${SCENE:-$PROJECT/data/tandt_db/tandt/truck}"
OUT="${1:-$PROJECT/output/truck_representation_$(date +%Y%m%d_%H%M%S)}"
[[ -f "$PLY" ]] || { echo "Missing PLY: $PLY" >&2; exit 1; }
[[ -d "$SCENE" ]] || { echo "Missing scene: $SCENE" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN.' >&2; exit 1; }
EXTRA=()
if [[ -n "${AXIS_FLOOR_WORLD:-}" ]]; then
  EXTRA+=(--axis-floor-world "$AXIS_FLOOR_WORLD")
fi
if [[ -n "${INIT:-}" ]]; then
  [[ -f "$INIT" ]] || { echo "Missing INIT: $INIT" >&2; exit 1; }
  EXTRA+=(--init "$INIT")
fi
if (( ${ADAPTER_STEPS:-0} > 0 || ${JOINT_STEPS:-0} > 0 )); then
  : "${MIN_CLEAN_PSNR:?Set an explicit source-render PSNR gate before enabling communication training}"
  EXTRA+=(--min-clean-psnr "$MIN_CLEAN_PSNR")
fi
if (( ${JOINT_STEPS:-0} > 0 )); then
  : "${MAX_ADAPTER_PSNR_DROP:?Set the allowed clean-to-communication PSNR drop before joint tuning}"
  EXTRA+=(--max-adapter-psnr-drop "$MAX_ADAPTER_PSNR_DROP")
fi
echo "GPU=$CUDA_VISIBLE_DEVICES; representation=${REPRESENTATION_STEPS:-5000}, adapters=${ADAPTER_STEPS:-0}, joint=${JOINT_STEPS:-0}"
echo 'Fixed q3; clean path bypasses ALL channel transforms. Scene renders are validation only.'
exec "$PYTHON_BIN" -u -m gaussian_jscc train-representation \
  --ply "$PLY" --source "$SCENE" --out "$OUT" --device cuda \
  --representation-steps "${REPRESENTATION_STEPS:-5000}" \
  --adapter-steps "${ADAPTER_STEPS:-0}" --joint-steps "${JOINT_STEPS:-0}" \
  --latent-dim "${LATENT_DIM:-64}" --block-size "${BLOCK_SIZE:-256}" \
  --blocks-per-batch "${BLOCKS_PER_BATCH:-32}" --validation-region-size 512 \
  --lr "${LR:-0.0002}" --clean-weight "${CLEAN_WEIGHT:-1}" \
  --position-objective "${POSITION_OBJECTIVE:-scene-scale}" \
  --axis-floor-percentile "${AXIS_FLOOR_PERCENTILE:-1}" \
  --channel "${CHANNEL:-none}" --snr "${SNR:-10}" \
  --validate-every "${VALIDATE_EVERY:-100}" --render-every "${RENDER_EVERY:-500}" \
  --save-every 500 --profile-every 10 --resolution "${RESOLUTION:-2}" \
  --validation-blocks 16 --validation-views 4 --seed "${SEED:-42}" "${EXTRA[@]}"
