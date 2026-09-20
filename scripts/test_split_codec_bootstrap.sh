#!/usr/bin/env bash
# New random-start architecture. No old checkpoint or XYZ side channel.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
case "${SPLIT_REPRESENTATION:-scale_rotation}" in
  scale_rotation) export ARCHITECTURE=learned_split ;;
  logcov) export ARCHITECTURE=learned_split_logcov ;;
  *) echo 'Unknown SPLIT_REPRESENTATION' >&2; exit 1 ;;
esac
export BOOTSTRAP_STEPS="${BOOTSTRAP_STEPS:-5000}"
export SAVE_EVERY="${SAVE_EVERY:-500}"
export PLY="${PLY:-$PROJECT/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply}"
OUT="${1:-$PROJECT/output/truck_split_codec_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCENE="${SCENE:-$PROJECT/data/tandt_db/tandt/truck}"
RENDER_HISTORY="${RENDER_HISTORY:-1}"
if [[ "$RENDER_HISTORY" == 1 ]]; then
  [[ -d "$SCENE/sparse" || -f "$SCENE/transforms_train.json" ]] || {
    echo "Missing scene: $SCENE; set SCENE or RENDER_HISTORY=0." >&2; exit 1;
  }
  [[ "$BOOTSTRAP_STEPS" =~ ^[1-9][0-9]*$ && "$SAVE_EVERY" =~ ^[1-9][0-9]*$ ]] || exit 1
  (( BOOTSTRAP_STEPS >= SAVE_EVERY && BOOTSTRAP_STEPS % SAVE_EVERY == 0 )) || {
    echo 'History requires BOOTSTRAP_STEPS to be a positive multiple of SAVE_EVERY.' >&2; exit 1;
  }
  [[ ! -e "$OUT/render_history" ]] || { echo 'Render history output exists.' >&2; exit 1; }
fi
echo "$ARCHITECTURE: geometry/appearance streams, sender-relative local attention, ONE shared JSCC payload."
if [[ "$ARCHITECTURE" == learned_split_logcov ]]; then
  echo 'spatial_logcov_v1: logcov head and physical Frobenius shape loss replace scale/quaternion and native overlap.'
else
  echo 'Loss, rates, LR and SNR remain matched to spatial_response_v3; this isolates the architecture change.'
fi
bash "$PROJECT/scripts/test_learned_xyz_bootstrap.sh" "$OUT"
if [[ "$RENDER_HISTORY" == 1 ]]; then
  "$PYTHON_BIN" -u evaluate_bootstrap_history.py \
    --training "$OUT" --ply "$PLY" --source "$SCENE" --out "$OUT/render_history" \
    --start "$SAVE_EVERY" --stop "$BOOTSTRAP_STEPS" --every "$SAVE_EVERY" \
    --snr "${SNR:-10}" --channel "${CHANNEL:-awgn}" --trials "${VALIDATION_TRIALS:-2}" \
    --views "${RENDER_VIEWS:-8}" --resolution "${RESOLUTION:-4}" \
    --blocks-per-batch "${BLOCKS_PER_BATCH:-32}" --device "${DEVICE:-cuda}"
fi
