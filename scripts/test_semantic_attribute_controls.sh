#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES}"
RUN="${1:?Usage: test_semantic_attribute_controls.sh SHARED_RUN NEW_OUT [options]}"
OUT="${2:?Usage: test_semantic_attribute_controls.sh SHARED_RUN NEW_OUT [options]}"
shift 2
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; scene metadata: $RUN; output: $OUT"
echo 'Random semantic-only controls: shared128, shared256, split128, split256; sequential by default.'
echo 'Each arm: TOTAL 3000 attribute updates + 1000 render updates, alternating Truck/Playroom.'
echo '32 rotating training views/update; 32 separate validation views; original test views (up to32).'
echo 'DrJohnson heldout; no channel/noise/allocation. Fixed bootstrap/final endpoints; no test selection.'
echo 'Shared source renders cached; one full-scene codec forward/replay per multi-view render update.'
echo 'Download OUT/review.tar.gz. Exact resume also needs each arm/checkpoints/latest.pt.'
exec "${PYTHON_BIN:-python}" -u -m gaussian_jscc.semantic_attribute_experiment \
  --run "$RUN" --out "$OUT" "${INIT[@]}" "$@"
