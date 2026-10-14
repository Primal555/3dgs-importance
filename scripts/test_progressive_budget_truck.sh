#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_progressive_budget_truck.sh CLEAN_COLOR_RUN NEW_OUT [extra options]}"
OUT="${2:?Usage: test_progressive_budget_truck.sh CLEAN_COLOR_RUN NEW_OUT [extra options]}"
shift 2
INIT="$RUN/checkpoints/end_semantic_render/codec.pt"
test -f "$RUN/training.json" || { echo "Missing metadata: $RUN/training.json" >&2; exit 1; }
test -f "$INIT" || { echo "Missing semantic checkpoint: $INIT" >&2; exit 1; }
command -v python >/dev/null || { echo 'Activate conda maskgs first' >&2; exit 1; }
echo "Frozen semantic teacher: $INIT"
echo "Default increased budget: 8 / 24 / 48 cumulative complex symbols (increments 8 + 16 + 24)"
echo "4000 pure latent-MSE updates; semantic y stays 48-dimensional; no render-training updates"
echo "Fresh channel weights; reliable 16bit XYZ unchanged"
echo "Output: $OUT; GPU: $CUDA_VISIBLE_DEVICES"
python -u -m gaussian_jscc.channel_capacity_controls \
  --run "$RUN" --out "$OUT" --init-semantic "$INIT" --groups progressive \
  --rates 8 24 48 --channel-bootstrap-steps 4000 --channel-render-steps 0 \
  --channel-encoder-lr 1e-4 --channel-decoder-lr 1e-4 \
  --channel-hidden 192 --train-snrs 10 --validate-every 500 --save-every 500 "$@"
