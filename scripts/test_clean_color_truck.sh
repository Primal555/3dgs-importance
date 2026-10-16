#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_clean_color_truck.sh SOURCE_RUN NEW_OUT [extra options]}"
OUT="${2:?Usage: test_clean_color_truck.sh SOURCE_RUN NEW_OUT [extra options]}"
shift 2
test -f "$RUN/training.json" || { echo "Missing source metadata: $RUN/training.json" >&2; exit 1; }
command -v python >/dev/null || { echo 'Activate conda maskgs first' >&2; exit 1; }
echo "Source metadata only (random weights): $RUN"
echo "Output: $OUT; GPU: $CUDA_VISIBLE_DEVICES"
echo 'Both spatial contexts; CLEAN semantic bypass; 4000 bootstrap + 1000 render updates; no channel/allocator training.'
python -u -m gaussian_jscc.modular_experiment \
  --run "$RUN" --out "$OUT" --context-variant both --clean-only \
  --semantic-bootstrap-steps 4000 --semantic-render-steps 1000 \
  --channel-bootstrap-steps 0 --channel-render-steps 0 \
  --semantic-color-every 4 --semantic-color-directions 8 \
  --semantic-encoder-lr 1e-4 --semantic-decoder-lr 1e-4 \
  --color-metrics --save-semantic-end-images --validate-every 500 --save-every 500 --lpips "$@"
