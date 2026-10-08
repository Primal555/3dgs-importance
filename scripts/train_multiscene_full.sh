#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="${1:-output/multiscene_$(date +%Y%m%d_%H%M%S)}"
MANIFEST="${MANIFEST:-configs/multiscene_tandt_db.json}"
BOOTSTRAP_STEPS="${BOOTSTRAP_STEPS:-5000}"
RENDER_STEPS="${RENDER_STEPS:-5000}"
ALLOCATION_STEPS="${ALLOCATION_STEPS:-1000}"
JOINT_STEPS="${JOINT_STEPS:-1000}"
VALIDATE_EVERY="${VALIDATE_EVERY:-500}"
BETA="${BETA:-0.01}"
echo "Output: $OUT; GPU: $CUDA_VISIBLE_DEVICES; steps are PER SCENE, not global."
"$PYTHON_BIN" -u -m gaussian_jscc.multiscene --manifest "$MANIFEST" --out "$OUT" \
  --bootstrap-steps "$BOOTSTRAP_STEPS" --render-steps "$RENDER_STEPS" \
  --allocation-steps "$ALLOCATION_STEPS" --joint-steps "$JOINT_STEPS" --beta "$BETA" \
  --validate-every "$VALIDATE_EVERY" --save-every "$VALIDATE_EVERY" \
  --blocks-per-batch "${BLOCKS_PER_BATCH:-64}" --lr "${LR:-1e-4}" --render-lr "${RENDER_LR:-1e-4}"
if [[ "${RUN_EVALUATION:-1}" == 1 ]]; then
  # Boundary checkpoints deliberately used: joint vs frozen is an explicit comparison.
  for STAGE in end_allocation final; do
    if [[ -f "$OUT/checkpoints/$STAGE/codec.pt" ]]; then
      "$PYTHON_BIN" -u -m gaussian_jscc.multiscene_experiments --manifest "$MANIFEST" \
        --checkpoint "$OUT/checkpoints/$STAGE/codec.pt" --out "$OUT/evaluation/$STAGE" \
        --trials "${TEST_TRIALS:-3}" --test-views "${TEST_VIEWS:-0}" --snrs 10
    fi
  done
fi
if [[ "${RUN_ADAPTATION:-1}" == 1 ]]; then
  "$PYTHON_BIN" -u -m gaussian_jscc.multiscene --adapt --manifest "$MANIFEST" \
    --checkpoint "$OUT/checkpoints/final/codec.pt" --out "$OUT/heldout_adaptation" \
    --bootstrap-steps 0 --render-steps 0 --joint-steps 0 --allocation-steps "$ALLOCATION_STEPS" \
    --validate-every "$VALIDATE_EVERY" --save-every "$VALIDATE_EVERY" --beta "$BETA" \
    --blocks-per-batch "${BLOCKS_PER_BATCH:-64}"
  "$PYTHON_BIN" -u -m gaussian_jscc.multiscene_experiments --manifest "$MANIFEST" --role heldout \
    --checkpoint "$OUT/heldout_adaptation/checkpoints/final/codec.pt" --out "$OUT/evaluation/heldout_adapted" \
    --trials "${TEST_TRIALS:-3}" --test-views "${TEST_VIEWS:-0}" --snrs 10
fi
