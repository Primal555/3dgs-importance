#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
INIT="${1:?Usage: bash scripts/train_multiscene_joint_aligned.sh END_ALLOCATION_DIR [OUT]}"
if [[ -d "$INIT" ]]; then INIT="$INIT/codec.pt"; fi
[[ -f "$INIT" ]] || { echo "Missing initializer: $INIT" >&2; exit 1; }
OUT="${2:-output/multiscene_joint_aligned_$(date +%Y%m%d_%H%M%S)}"
[[ ! -e "$OUT" ]] || { echo "Output already exists: $OUT" >&2; exit 1; }
MANIFEST="${MANIFEST:-configs/multiscene_tandt_db.json}"
PROFILE="${OUTPUT_PROFILE:-compact}"
echo "Initializer: $INIT (paired tables; fresh Adam, not exact resume)"
echo "Output: $OUT; GPU: $CUDA_VISIBLE_DEVICES; joint updates are PER SCENE."
# Metrics use ALL test views by default; only four representative panels/layout
# are saved. Complete both paired evaluations before publishing review.zip.
"$PYTHON_BIN" -u -m gaussian_jscc.multiscene --joint-only --manifest "$MANIFEST" \
  --checkpoint "$INIT" --out "$OUT" --bootstrap-steps 0 --render-steps 0 --allocation-steps 0 \
  --joint-steps "${JOINT_STEPS:-1000}" --allocation-sampling deployment \
  --prefix-anchor-every "${PREFIX_ANCHOR_EVERY:-4}" --rate-chunk-size "${RATE_CHUNK_SIZE:-1024}" \
  --beta "${BETA:-0.01}" --render-lr "${LR:-1e-4}" --lr "${LR:-1e-4}" \
  --validate-every "${VALIDATE_EVERY:-500}" --save-every "${VALIDATE_EVERY:-500}" \
  --blocks-per-batch "${BLOCKS_PER_BATCH:-64}" --output-profile "$PROFILE" --image-policy none
for STAGE in initial final; do
  CHECKPOINT="$INIT"
  if [[ "$STAGE" == final ]]; then CHECKPOINT="$OUT/checkpoints/final/codec.pt"; fi
  "$PYTHON_BIN" -u -m gaussian_jscc.multiscene_experiments --manifest "$MANIFEST" \
    --checkpoint "$CHECKPOINT" --out "$OUT/evaluation/$STAGE" --output-profile "$PROFILE" \
    --image-views 4 --trials "${TEST_TRIALS:-3}" --test-views "${TEST_VIEWS:-0}" --snrs 10
done
"$PYTHON_BIN" -u -m gaussian_jscc.compact_output "$OUT"
echo "Done. Download $OUT/review.zip for logs, metrics, charts and initial/final images."
