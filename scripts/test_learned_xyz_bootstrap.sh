#!/usr/bin/env bash
# Isolated experiment: random learned XYZ + attributes, bootstrap only, no XYZ side stream.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PLY="${PLY:-$PROJECT/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply}"
OUT="${1:-$PROJECT/output/truck_learned_xyz_v3_$(date +%Y%m%d_%H%M%S)}"
[[ -f "$PLY" ]] || { echo "Missing PLY: $PLY" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Output exists: $OUT" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN.' >&2; exit 1; }
echo 'Random weights; position capture + fine XYZ + shape + centered RGB. No XYZ side stream or render training; exact objective is logged in training.json.'
# Deliberately do not inherit INIT, POSITION_DELIVERY, RENDER_STEPS, JOINT_STEPS or LR_SCHEDULE.
TIER_ARGS=()
if [[ -n "${BOOTSTRAP_TIER:-}" ]]; then
  TIER_ARGS=(--bootstrap-tier "$BOOTSTRAP_TIER")
fi
exec "$PYTHON_BIN" -u -m gaussian_jscc train-learned \
  "${TIER_ARGS[@]}" \
  --ply "$PLY" --out "$OUT" --device "${DEVICE:-cuda}" \
  --architecture "${ARCHITECTURE:-learned_joint}" \
  --context-mode "${CONTEXT_MODE:-window}" \
  --xyz-decoder "${XYZ_DECODER:-additive}" \
  --decoder-attention "${DECODER_ATTENTION:-window}" --decoder-neighbors "${DECODER_NEIGHBORS:-16}" \
  --decoder-depth "${DECODER_DEPTH:-4}" \
  --decoder-localization "${DECODER_LOCALIZATION:-none}" \
  --encoder-attention "${ENCODER_ATTENTION:-window}" --encoder-neighbors "${ENCODER_NEIGHBORS:-16}" \
  --position-delivery learned --bootstrap-objective spatial-response \
  --bootstrap-steps "${BOOTSTRAP_STEPS:-5000}" --render-steps 0 --joint-steps 0 \
  --snr "${SNR:-10}" --channel "${CHANNEL:-awgn}" --rates 0 8 16 32 \
  --block-size 256 --decoder-window 32 --blocks-per-batch "${BLOCKS_PER_BATCH:-32}" \
  --local-response-views "${LOCAL_RESPONSE_VIEWS:-4}" \
  --spatial-fine-weight "${SPATIAL_FINE_WEIGHT:-1}" \
  --lr "${LR:-0.0002}" --lr-schedule constant --clip-mode none \
  --training-data-device cpu --validate-every "${VALIDATE_EVERY:-100}" \
  --validation-blocks "${VALIDATION_BLOCKS:-16}" --validation-trials "${VALIDATION_TRIALS:-2}" \
  --save-every "${SAVE_EVERY:-500}" --seed "${SEED:-42}"
