#!/usr/bin/env bash
# Same frozen semantic teacher; RANDOM channel weights, train step0 -> 8000.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_channel_response_scratch_truck.sh TEACHER_SOURCE_RUN_OR_STABILITY_SUITE NEW_OUT [options]}"
OUT="${2:?Usage: test_channel_response_scratch_truck.sh TEACHER_SOURCE_RUN_OR_STABILITY_SUITE NEW_OUT [options]}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate conda maskgs or set PYTHON_BIN' >&2; exit 1; }
echo "GPU: $CUDA_VISIBLE_DEVICES; teacher source: $RUN; output: $OUT"
echo '8000 updates FROM RANDOM CHANNEL WEIGHTS; source channel weights/Adam NOT copied.'
echo 'Both semantic modules frozen; 10dB; 8/24/48; block power; NEW Adam at1e-4.'
echo 'Recalibrate response weight at random initialization; no joint stage or allocator.'
exec "$PYTHON" -u -m gaussian_jscc.channel_response_training \
  --run "$RUN" --out "$OUT" --channel-init random --steps 8000 --lr 1e-4 \
  --response-directions 8 --calibration-batches 4 \
  --validate-every 500 --save-every 100 --test-views 32 --render-trials 4 "$@"
