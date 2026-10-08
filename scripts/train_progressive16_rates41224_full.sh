#!/usr/bin/env bash
# Random 4/12/24 codec -> render -> frozen-codec allocation -> joint -> deployment/test.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="${1:-$PROJECT/output/truck_rates41224_full_$(date +%Y%m%d_%H%M%S)}"
PLY="${PLY:-$PROJECT/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply}"
SCENE="${SCENE:-$PROJECT/data/tandt_db/tandt/truck}"
BOOTSTRAP_STEPS="${BOOTSTRAP_STEPS:-5000}"
RENDER_STEPS="${RENDER_STEPS:-5000}"
ALLOCATION_STEPS="${ALLOCATION_STEPS:-2000}"
JOINT_FINETUNE_STEPS="${JOINT_FINETUNE_STEPS:-1000}"
for count in "$BOOTSTRAP_STEPS" "$RENDER_STEPS" "$ALLOCATION_STEPS" "$JOINT_FINETUNE_STEPS"; do
  [[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo 'All four stage lengths must be positive integers.' >&2; exit 1; }
done
TOTAL_JOINT_STEPS=$((ALLOCATION_STEPS + JOINT_FINETUNE_STEPS))
[[ -f "$PLY" ]] || { echo "Missing PLY: $PLY" >&2; exit 1; }
[[ -d "$SCENE/sparse" || -f "$SCENE/transforms_train.json" ]] || { echo "Missing scene: $SCENE" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Output exists: $OUT" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN.' >&2; exit 1; }
[[ -z "${INIT:-}${ALLOCATION_INIT:-}" ]] || echo 'Random initialization: ignoring inherited INIT/ALLOCATION_INIT; old rate tables cannot be reused.'
echo 'q0=drop; q1/q2/q3=4/12/24 COMPLEX symbols. Incremental layers: 4+8+12.'
echo 'XYZ: fixed 16-bit quantization, lossless delta_zlib level 6, for retained points only.'
printf 'Stages: attribute=%s, render=%s, allocation-only=%s, joint-finetune=%s updates.\n' \
  "$BOOTSTRAP_STEPS" "$RENDER_STEPS" "$ALLOCATION_STEPS" "$JOINT_FINETUNE_STEPS"
echo 'Rate penalty is normalized by actual all-q3 cost: 24 payload symbols plus measured XYZ/tier-map cost.'
echo 'Separate keep/tier hard Gumbel feedback; all sampled q0 rows get shadow feedback; ten-draw deployment.'
echo 'All stages use the same OUT. Final test uses held-out test cameras, not validation cameras.'
printf 'GPU: %s\nOutput: %s\n' "$CUDA_VISIBLE_DEVICES" "$OUT"
"$PYTHON_BIN" -u -m gaussian_jscc train-learned \
  --ply "$PLY" --source "$SCENE" --out "$OUT" \
  --device cuda --snr 10 --channel awgn --prefix-mode progressive \
  --position-delivery quantized --position-bits 16 --position-compression delta_zlib \
  --position-compression-level 6 --position-net-bits-per-use 2 --rates 0 4 12 24 \
  --bootstrap-steps "$BOOTSTRAP_STEPS" --bootstrap-objective local-response --local-response-views 4 \
  --render-steps "$RENDER_STEPS" --joint-steps "$TOTAL_JOINT_STEPS" \
  --mask-only-steps "$ALLOCATION_STEPS" \
  --existence-prior "${EXISTENCE_PRIOR:-auto}" \
  --mask-lr "${MASK_LR:-0.001}" --keep-lr "${KEEP_LR:-0.01}" --mask-adam-eps "${MASK_ADAM_EPS:-1e-15}" \
  --beta "${BETA:-0.01}" \
  --lr "${LR:-0.0001}" --render-lr "${RENDER_LR:-0.0001}" --drop 0.05 --clip-mode none \
  --block-size 256 --decoder-window 32 --blocks-per-batch "${BLOCKS_PER_BATCH:-64}" \
  --render-backward replay --training-data-device cpu --resolution 2 \
  --train-views 0 --views-per-step 2 --validate-every "${VALIDATE_EVERY:-500}" \
  --joint-validate-every "${ALLOCATION_VALIDATE_EVERY:-100}" \
  --validation-views 4 --validation-trials 2 --patience 0 --save-every "${SAVE_EVERY:-500}" --seed "${SEED:-42}"

# Save both deployment maps, always with their own matching codec checkpoints.
for suffix in end_allocation best_joint; do
  "$PYTHON_BIN" -u -m gaussian_jscc export-route2 \
    --ply "$PLY" --checkpoint "$OUT/codec_${suffix}.pt" --allocation "$OUT/route2_${suffix}.pt" \
    --out "$OUT/deployment_${suffix}" --snr 10 --allocation-seed "${SEED:-42}" --device cuda
done
if [[ "${FINAL_EVAL:-1}" == 1 ]]; then
  # Fixed positive budgets are evaluated with the SAME selected codec as the mask.
  "$PYTHON_BIN" -u -m gaussian_jscc evaluate \
    --ply "$PLY" --checkpoint "$OUT/codec_best_joint.pt" \
    --source "$SCENE" --out "$OUT/test_best_uniform" --device cuda --snrs 10 --channel awgn \
    --tiers 1 2 3 --trials "${TEST_TRIALS:-2}" --resolution 2 --save-images \
    --seed "${SEED:-42}" --metadata-code-rate 1 --metadata-modulation-bits 2
  "$PYTHON_BIN" -u -m gaussian_jscc evaluate \
    --ply "$PLY" --checkpoint "$OUT/codec_best_joint.pt" --allocation "$OUT/route2_best_joint.pt" \
    --source "$SCENE" --out "$OUT/test_best_mask" --device cuda --snrs 10 --channel awgn \
    --allocation-seed "${SEED:-42}" --trials "${TEST_TRIALS:-2}" --resolution 2 --save-images \
    --seed "${SEED:-42}" --metadata-code-rate 1 --metadata-modulation-bits 2
fi
echo "Completed. Logs/curves: $OUT/loss.jsonl, $OUT/validation.jsonl, $OUT/charts"
echo "Hard deployment counts: $OUT/deployment_best_joint/allocation.json"
echo "Pre-finetune deployment: $OUT/deployment_end_allocation/allocation.json"
echo 'Best joint checkpoint may be from the frozen allocation stage if fine-tuning did not help.'
