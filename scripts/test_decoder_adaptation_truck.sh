#!/usr/bin/env bash
# Train ONLY the semantic decoder; originals and previous joint probes are read-only.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_decoder_adaptation_truck.sh CHANNEL_RUN_OR_STABILITY_SUITE NEW_OUT [options]}"
OUT="${2:?Usage: test_decoder_adaptation_truck.sh CHANNEL_RUN_OR_STABILITY_SUITE NEW_OUT [options]}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate conda maskgs or set PYTHON_BIN' >&2; exit 1; }
echo "GPU: $CUDA_VISIBLE_DEVICES; source: $RUN; output: $OUT"
echo '1000 decoder-only updates; LR=1e-5; fixed10dB; q1/q2/q3/mixed; frozen semantic encoder and BOTH channel modules.'
echo 'No allocator, no channel optimizer, no latent auxiliary loss, no gradient clipping.'
exec "$PYTHON" -u -m gaussian_jscc.decoder_adaptation \
  --run "$RUN" --out "$OUT" --steps 1000 --lr 1e-5 \
  --validate-every 250 --save-every 100 --probe-every 50 \
  --test-views 32 --render-trials 4 "$@"
