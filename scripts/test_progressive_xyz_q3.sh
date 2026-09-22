#!/usr/bin/env bash
# Matched context/refinement experiment. All outputs live under OUT.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export BLOCK_SIZE="${BLOCK_SIZE:-512}"
case "$BLOCK_SIZE" in
  256) DEFAULT_BATCH=32 ;;
  512) DEFAULT_BATCH=16 ;;
  *) echo 'This comparison supports BLOCK_SIZE=256 or 512.' >&2; exit 1 ;;
esac
export BLOCKS_PER_BATCH="${BLOCKS_PER_BATCH:-$DEFAULT_BATCH}"
export VALIDATION_REGION_SIZE=512
export ENCODER_NEIGHBORS=16
export DECODER_MEMORY=none
export DECODER_REFINEMENT="${DECODER_REFINEMENT:-progressive}"
OUT="${1:-$PROJECT/output/truck_${DECODER_REFINEMENT}_b${BLOCK_SIZE}_q3_$(date +%Y%m%d_%H%M%S)}"
echo "Block=$BLOCK_SIZE; batch=$BLOCKS_PER_BATCH; refinement=$DECODER_REFINEMENT; matched 512-point validation regions."
exec bash "$PROJECT/scripts/test_transformer_trunk_q3.sh" "$OUT"
