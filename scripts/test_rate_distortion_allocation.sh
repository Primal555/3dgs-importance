#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_rate_distortion_allocation.sh COMPLETED_RUN SCENE NEW_OUT --beta BETA [options]}"
SCENE="${2:?Set an explicitly trained scene}"
OUT="${3:?Set a NEW output directory}"
shift 3
# A budget-only invocation must not silently start the older experiment.
HAVE_BETA=0
for ARG in "$@"; do
  case "$ARG" in
    --beta|--beta=*) HAVE_BETA=1 ;;
    --budget-total-uses|--budget-total-uses=*|--budget-ratio-to-q3|--budget-ratio-to-q3=*)
      echo 'Fixed RD mode requires --beta and does not accept budget arguments' >&2; exit 2 ;;
  esac
done
if [[ "$HAVE_BETA" != 1 ]]; then echo 'Supply an explicit --beta BETA (finite, nonnegative)' >&2; exit 2; fi
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; source: $RUN; scene: $SCENE; output: $OUT"
echo 'Fixed beta RD; frozen codec; defaults: 2000 actual updates, 4 views/window; rates0/8/24/48, SNR10.'
echo 'No communication budget, adaptive price or joint finetuning; best_rd uses calibrated actual deployment cost.'
exec "$PYTHON" -u -m gaussian_jscc.budgeted_allocation \
  --run "$RUN" --scene "$SCENE" --out "$OUT" "${INIT[@]}" "$@"
