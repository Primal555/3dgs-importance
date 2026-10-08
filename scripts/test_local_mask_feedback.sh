#!/usr/bin/env bash
# One run, one output folder: frozen-codec mask learning followed by joint fine-tuning.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INIT="${1:?Usage: CUDA_VISIBLE_DEVICES=2 bash scripts/test_local_mask_feedback.sh CODEC_BEST_RENDER_PT [OUT]}"
OUT="${2:-$PROJECT/output/truck_local_mask_feedback_$(date +%Y%m%d_%H%M%S)}"
PLY="${PLY:-$PROJECT/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply}"
SCENE="${SCENE:-$PROJECT/data/tandt_db/tandt/truck}"
[[ -f "$INIT" ]] || { echo "Missing codec checkpoint: $INIT" >&2; exit 1; }
[[ -f "$PLY" ]] || { echo "Missing PLY: $PLY" >&2; exit 1; }
[[ -d "$SCENE/sparse" || -f "$SCENE/transforms_train.json" ]] || { echo "Missing scene: $SCENE" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Output exists: $OUT" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN.' >&2; exit 1; }
echo "GPU=$CUDA_VISIBLE_DEVICES; checkpoint=$INIT; output=$OUT"
echo 'One local-feedback allocation run. Frozen codec first, then optional joint updates.'
"$PYTHON_BIN" -u -m gaussian_jscc train-learned \
  --ply "$PLY" --source "$SCENE" --out "$OUT" --init "$INIT" \
  --device cuda --snr 10 --channel awgn --prefix-mode progressive \
  --position-delivery quantized --position-bits 16 --position-compression delta_zlib \
  --rates 0 8 16 32 --position-net-bits-per-use 2 \
  --bootstrap-steps 0 --render-steps 0 --joint-steps "${JOINT_STEPS:-1000}" \
  --mask-only-steps "${MASK_ONLY_STEPS:-${JOINT_STEPS:-1000}}" --mask-lr "${MASK_LR:-0.001}" \
  --keep-lr "${KEEP_LR:-0.01}" --mask-adam-eps "${MASK_ADAM_EPS:-1e-15}" \
  --existence-prior "${EXISTENCE_PRIOR:-auto}" --beta "${BETA:-0.01}" \
  --render-lr "${RENDER_LR:-0.0001}" --clip-mode none \
  --blocks-per-batch "${BLOCKS_PER_BATCH:-64}" --render-backward replay \
  --training-data-device cpu --resolution 2 --views-per-step 2 \
  --validate-every "${VALIDATE_EVERY:-100}" --validation-views 4 \
  --validation-trials 2 --patience 0 --save-every "${SAVE_EVERY:-500}" --seed 42
"$PYTHON_BIN" -u -m gaussian_jscc export-route2 \
  --ply "$PLY" --checkpoint "$OUT/codec_best_joint.pt" \
  --allocation "$OUT/route2_best_joint.pt" --out "$OUT/deployment_best" \
  --snr 10 --device cuda
echo "Check $OUT/mask_renderer_check.json, $OUT/allocation_history.jsonl, $OUT/validation.jsonl and $OUT/validation_images/"
