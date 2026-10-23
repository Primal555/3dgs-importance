#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_budgeted_allocation.sh COMPLETED_RUN SCENE NEW_OUT --budget-ratio-to-q3 R (or --budget-total-uses B) [options]}"
SCENE="${2:?Set an explicitly trained scene}"
OUT="${3:?Set a NEW output directory}"
shift 3
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; source: $RUN; scene: $SCENE; output: $OUT"
echo 'Frozen codec; 2000 actual allocation updates, 4 views/window; rates0/8/24/48, SNR10.'
echo 'Explicit total budget; single-draw training, ten-draw measured deployment; no joint finetuning.'
exec "$PYTHON" -u -m gaussian_jscc.budgeted_allocation \
  --run "$RUN" --scene "$SCENE" --out "$OUT" "${INIT[@]}" "$@"
