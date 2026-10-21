#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_joint_adam_reset_truck.sh STABILITY_SUITE_OR_CHANNEL_RUN NEW_OUT [options]}"
OUT="${2:?Usage: test_joint_adam_reset_truck.sh STABILITY_SUITE_OR_CHANNEL_RUN NEW_OUT [options]}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate conda maskgs or set PYTHON_BIN' >&2; exit 1; }
echo "GPU: $CUDA_VISIBLE_DEVICES; output: $OUT"
echo '30-step joint Adam RESET probe. Source weights, LR, losses and sampling are unchanged.'
echo 'Train BOTH channel modules and semantic decoder; semantic encoder stays frozen.'
exec "$PYTHON" -u -m gaussian_jscc.joint_adam_reset \
  --run "$RUN" --out "$OUT" --steps 30 --adam-mode reset --top-k 16 "$@"
