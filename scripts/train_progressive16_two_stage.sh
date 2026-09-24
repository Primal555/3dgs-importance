#!/usr/bin/env bash
# Historical local-response pretraining + current progressive render optimization.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="${1:-$PROJECT/output/truck_progressive16_two_stage_$(date +%Y%m%d_%H%M%S)}"
PLY="${PLY:-$PROJECT/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply}"
SCENE="${SCENE:-$PROJECT/data/tandt_db/tandt/truck}"
[[ -f "$PLY" ]] || { echo "Missing PLY: $PLY" >&2; exit 1; }
[[ -d "$SCENE/sparse" || -f "$SCENE/transforms_train.json" ]] || { echo "Missing scene: $SCENE" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Output exists: $OUT" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN.' >&2; exit 1; }
EXTRA=()
INITIALIZATION="${INITIALIZATION:-random}"
case "$INITIALIZATION" in
  random) [[ -z "${INIT:-}" ]] || echo 'Random initialization: ignoring inherited INIT.' ;;
  checkpoint)
    [[ -n "${INIT:-}" && -f "$INIT" ]] || { echo 'checkpoint mode requires an existing INIT file.' >&2; exit 1; }
    EXTRA+=(--init "$INIT")
    echo 'Requires progressive 16-bit delta_zlib checkpoint. Fresh Adam; NOT exact resume.' ;;
  *) echo 'INITIALIZATION must be random or checkpoint.' >&2; exit 1 ;;
esac
printf 'Initialization: %s\nGPU: %s\nOutput: %s\n' "$INITIALIZATION" "$CUDA_VISIBLE_DEVICES" "$OUT"
echo 'Stage A: historical local-response attribute loss; minibatches, no full-scene rasterization per update.'
echo 'Stage B: full-scene multiview MSE, replay. XYZ fixed throughout; no mask training.'
echo 'Weights retained across phases; Adam moments reset. Constant LR, no automatic decay.'
exec "$PYTHON_BIN" -u -m gaussian_jscc train-learned \
  --ply "$PLY" --source "$SCENE" --out "$OUT" "${EXTRA[@]}" \
  --device cuda --snr 10 --channel awgn --prefix-mode progressive \
  --position-delivery quantized --position-bits 16 --position-compression delta_zlib \
  --position-net-bits-per-use 2 \
  --bootstrap-steps "${BOOTSTRAP_STEPS:-5000}" --bootstrap-objective local-response --local-response-views 4 \
  --render-steps "${RENDER_STEPS:-5000}" --joint-steps 0 \
  --block-size 256 --decoder-window 32 --blocks-per-batch "${BLOCKS_PER_BATCH:-64}" \
  --rates 0 8 16 32 --lr "${LR:-0.0001}" --render-lr "${RENDER_LR:-0.0001}" --drop 0 --clip-mode none \
  --render-backward replay --training-data-device cpu --resolution 2 \
  --train-views 0 --views-per-step 2 \
  --validate-every "${VALIDATE_EVERY:-500}" --validation-views 4 --validation-trials 2 \
  --patience 0 --save-every "${SAVE_EVERY:-500}" --seed 42
