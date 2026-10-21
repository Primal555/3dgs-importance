#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_successive_channel_truck.sh TEACHER_RUN NEW_OUT [options]}"
OUT="${2:?Usage: test_successive_channel_truck.sh TEACHER_RUN NEW_OUT [options]}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
echo "GPU: $CUDA_VISIBLE_DEVICES; teacher: $RUN; output: $OUT"
echo 'Random channel; frozen semantics; 8/24/48; reliable binary16 block scales counted.'
echo '8000 main + 1000 neighbor + 1000 channel cooperation; images initial/final only.'
exec "$PYTHON" -u -m gaussian_jscc.successive_experiment \
  --run "$RUN" --out "$OUT" --main-steps 8000 --neighbor-steps 1000 --cooperative-steps 1000 \
  --lr 1e-4 --neighbor-lr 1e-4 --cooperative-lr 1e-5 \
  --validate-every 500 --save-every 100 --test-views 32 --render-trials 4 "$@"
