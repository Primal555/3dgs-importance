#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_channel_prefix_truck.sh COMPLETED_Y_RUN NEW_OUT [extra options]}"
OUT="${2:?Usage: test_channel_prefix_truck.sh COMPLETED_Y_RUN NEW_OUT [extra options]}"
shift 2
if [[ ! -f "$RUN/training.json" && -f "$RUN/progressive/training.json" ]]; then RUN="$RUN/progressive"; fi
test -f "$RUN/training.json" || { echo "Missing metadata: $RUN/training.json" >&2; exit 1; }
test -f "$RUN/checkpoints/final/codec.pt" || { echo "Missing channel checkpoint: $RUN/checkpoints/final/codec.pt" >&2; exit 1; }
command -v python >/dev/null || { echo 'Activate conda maskgs first' >&2; exit 1; }
echo "Source channel weights + Adam: $RUN/checkpoints/final/codec.pt"
echo "Two sequential 10dB arms: suffix10 / full_prefix10; default +2000 updates EACH"
echo "Budget, semantic modules, XYZ, normalization and loss unchanged"
echo "New earlier-prefix projections start at zero; old tensors and moments retained"
echo "Output: $OUT; GPU: $CUDA_VISIBLE_DEVICES; arm logs: suffix10.log / full_prefix10.log"
python -u -m gaussian_jscc.channel_inverse_controls \
  --run "$RUN" --out "$OUT" --suite prefix --steps 2000 \
  --validate-every 500 --save-every 500 \
  --diagnostic-points 32768 --diagnostic-trials 8 --diagnostic-batch 2048 "$@"
