#!/usr/bin/env bash
# Dedicated pure-render baseline. Compatible with historical commit 9f2810e.
# Deliberately does NOT source train_codec_learned.sh or two-stage launchers.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="${1:-$PROJECT/output/truck_quantized12_render_only_$(date +%Y%m%d_%H%M%S)}"
PLY="${PLY:-$PROJECT/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply}"
SCENE="${SCENE:-$PROJECT/data/tandt_db/tandt/truck}"
[[ -f "$PLY" ]] || { echo "Missing PLY: $PLY" >&2; exit 1; }
[[ -d "$SCENE/sparse" || -f "$SCENE/transforms_train.json" ]] || { echo "Missing scene: $SCENE" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Output exists: $OUT" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN.' >&2; exit 1; }
EXTRA=()
INITIALIZATION="${INITIALIZATION:-random}"
case "$INITIALIZATION" in
  random)
    [[ -z "${INIT:-}" ]] || echo 'Random initialization: ignoring inherited INIT.'
    ;;
  checkpoint)
    [[ -n "${INIT:-}" && -f "$INIT" ]] || { echo 'checkpoint mode requires an existing INIT file.' >&2; exit 1; }
    EXTRA+=(--init "$INIT")
    echo 'Weight continuation only: Adam and step counter restart; NOT exact resume.'
    ;;
  *) echo 'INITIALIZATION must be random or checkpoint.' >&2; exit 1 ;;
esac
printf 'Pure render baseline (9f2810e lineage); XYZ=12 bit; LR=1e-4\nInitialization: %s\nSteps in THIS run: %s\nGPU: %s\nOutput: %s\n' \
  "$INITIALIZATION" "${RENDER_STEPS:-5000}" "$CUDA_VISIBLE_DEVICES" "$OUT"
echo 'Locked: bootstrap=0, joint=0, all training views, AWGN 10 dB, no gradient clipping.'
exec "$PYTHON_BIN" -u -m gaussian_jscc train-learned \
  --ply "$PLY" --source "$SCENE" --out "$OUT" "${EXTRA[@]}" \
  --device cuda --snr 10 --channel awgn \
  --position-delivery quantized --position-bits 12 --position-net-bits-per-use 2 \
  --bootstrap-steps 0 --render-steps "${RENDER_STEPS:-5000}" --joint-steps 0 \
  --block-size 256 --decoder-window 32 --blocks-per-batch "${BLOCKS_PER_BATCH:-64}" \
  --rates 0 8 16 32 --lr 0.0001 --render-lr 0.0001 --drop 0 --clip-mode none \
  --render-backward replay --training-data-device cpu --resolution 2 \
  --train-views 0 --views-per-step 2 \
  --validate-every "${VALIDATE_EVERY:-500}" --validation-views 4 --validation-trials 2 \
  --patience 0 --save-every "${SAVE_EVERY:-500}" --seed 42
