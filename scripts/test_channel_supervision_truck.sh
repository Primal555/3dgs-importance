#!/usr/bin/env bash
# Single arm: frozen semantic teacher; RANDOM channels, 8000 total updates.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_channel_supervision_truck.sh TEACHER_SOURCE_RUN NEW_OUT color|separated [options]}"
OUT="${2:?Usage: test_channel_supervision_truck.sh TEACHER_SOURCE_RUN NEW_OUT color|separated [options]}"
VARIANT="${3:?Specify color or separated}"
shift 3
[[ "$VARIANT" == color || "$VARIANT" == separated ]] || { echo 'Variant must be color or separated' >&2; exit 2; }
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate conda maskgs or set PYTHON_BIN' >&2; exit 1; }
echo "GPU: $CUDA_VISIBLE_DEVICES; variant: $VARIANT; teacher: $RUN; output: $OUT"
echo 'RANDOM channel weights, EMPTY Adam; same frozen semantic teacher; 10dB, 8/24/48.'
echo '8000 TOTAL updates; render only for evaluation, no joint training or allocator.'
exec "$PYTHON" -u -m gaussian_jscc.channel_supervision_controls \
  --run "$RUN" --out "$OUT" --variant "$VARIANT" --steps 8000 --lr 1e-4 \
  --response-directions 8 --validate-every 500 --save-every 100 \
  --test-views 32 --render-trials 4 "$@"
