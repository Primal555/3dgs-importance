#!/usr/bin/env bash
# All six channel-only arms run sequentially on ONE explicitly selected GPU.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_channel_normalization_truck.sh TEACHER_RUN NEW_OUT [extra options]}"
OUT="${2:?Usage: test_channel_normalization_truck.sh TEACHER_RUN NEW_OUT [extra options]}"
shift 2
test -f "$RUN/training.json" || { echo "Missing teacher metadata: $RUN/training.json" >&2; exit 1; }
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate conda maskgs first, or set PYTHON_BIN' >&2; exit 1; }
echo "Frozen semantic teacher: $RUN; GPU: $CUDA_VISIBLE_DEVICES"
echo "3 power normalizers x noiseless/10dB; fresh matched channel weights and Adam; 4000 updates EACH."
echo "Rates: 8/24/48; LR: 1e-4; pure original-y MSE. No semantic or render optimization."
echo "Output: $OUT; worker progress: $OUT/{point,statistical,block}_{noiseless,awgn10}.log"
"$PYTHON" -u -m gaussian_jscc.channel_normalization_controls \
  --run "$RUN" --out "$OUT" --steps 4000 --lr 1e-4 \
  --validate-every 1000 --save-every 500 --diagnostic-blocks 32 \
  --response-blocks 8 --trials 8 --render-trials 2 --validation-views 16 --lpips "$@"
