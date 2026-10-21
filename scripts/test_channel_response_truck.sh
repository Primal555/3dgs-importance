#!/usr/bin/env bash
# Task-aware CHANNEL training only. No semantic/joint updates or allocator.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_channel_response_truck.sh BLOCK_CHANNEL_RUN_OR_STABILITY_SUITE NEW_OUT [options]}"
OUT="${2:?Usage: test_channel_response_truck.sh BLOCK_CHANNEL_RUN_OR_STABILITY_SUITE NEW_OUT [options]}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate conda maskgs or set PYTHON_BIN' >&2; exit 1; }
echo "GPU: $CUDA_VISIBLE_DEVICES; source: $RUN; output: $OUT"
echo '2000 additional CHANNEL updates; both semantic modules FROZEN; no joint stage.'
echo '10dB; 8/24/48; block power; NEW Adam at1e-4; latent MSE + calibrated local RGB response.'
echo 'Full-scene rendering is evaluation-only; only initial/final images are saved.'
exec "$PYTHON" -u -m gaussian_jscc.channel_response_training \
  --run "$RUN" --out "$OUT" --steps 2000 --lr 1e-4 \
  --response-directions 8 --calibration-batches 4 \
  --validate-every 500 --save-every 100 --test-views 32 --render-trials 4 "$@"
