#!/usr/bin/env bash
# One shared codec, eight learned positive prefixes; all results in OUT.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="${1:-$PROJECT/output/truck_progressive16_rate_sweep_$(date +%Y%m%d_%H%M%S)}"
PLY="${PLY:-$PROJECT/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply}"
SCENE="${SCENE:-$PROJECT/data/tandt_db/tandt/truck}"
[[ -f "$PLY" ]] || { echo "Missing PLY: $PLY" >&2; exit 1; }
[[ -d "$SCENE/sparse" || -f "$SCENE/transforms_train.json" ]] || { echo "Missing scene: $SCENE" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Output exists: $OUT" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN.' >&2; exit 1; }
[[ -z "${INIT:-}" ]] || echo 'Random initialization: ignoring inherited INIT. Three-tier weights are not reused.'
echo 'Eight cumulative prefixes: 4 8 12 16 20 24 28 32 COMPLEX symbols per Gaussian.'
echo 'One codeword; eight independently normalized 4-symbol layers. XYZ: fixed 16-bit delta_zlib.'
echo 'One update each for q1..q8, then one per-Gaussian mixed update; no mask training.'
echo 'Defaults: 11250 + 11250 updates = 1250 complete layout cycles per phase.'
echo 'This preserves uniform updates/tier versus the old 5000-step, four-layout schedule, not equal wall time.'
echo 'Override BOOTSTRAP_STEPS and RENDER_STEPS for a shorter scan; raw logs record actual exposure.'
printf 'GPU: %s\nOutput: %s\n' "$CUDA_VISIBLE_DEVICES" "$OUT"
exec "$PYTHON_BIN" -u -m gaussian_jscc train-learned \
  --ply "$PLY" --source "$SCENE" --out "$OUT" \
  --device cuda --snr 10 --channel awgn --prefix-mode progressive \
  --position-delivery quantized --position-bits 16 --position-compression delta_zlib \
  --position-net-bits-per-use 2 \
  --rates 0 4 8 12 16 20 24 28 32 \
  --bootstrap-steps "${BOOTSTRAP_STEPS:-11250}" --bootstrap-objective local-response --local-response-views 4 \
  --render-steps "${RENDER_STEPS:-11250}" --joint-steps 0 \
  --block-size 256 --decoder-window 32 --blocks-per-batch "${BLOCKS_PER_BATCH:-64}" \
  --lr "${LR:-0.0001}" --render-lr "${RENDER_LR:-0.0001}" --drop 0 --clip-mode none \
  --render-backward replay --training-data-device cpu --resolution 2 \
  --train-views 0 --views-per-step 2 \
  --validate-every "${VALIDATE_EVERY:-500}" --validation-views "${VALIDATION_VIEWS:-4}" \
  --validation-trials "${VALIDATION_TRIALS:-2}" \
  --patience 0 --save-every "${SAVE_EVERY:-500}" --seed "${SEED:-42}"
