#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_structured_evidence_allocation.sh COMPLETED_CODEC_RUN SCENE NEW_OUT --beta BETA [options]}"
SCENE="${2:?Set a trained scene, e.g. truck}"
OUT="${3:?Set a NEW output directory}"
shift 3
if [[ ! -f "$RUN/training.json" && -f "$RUN/training/training.json" ]]; then RUN="$RUN/training"; fi
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
HAVE_BETA=0
for ARG in "$@"; do
  case "$ARG" in
    --beta|--beta=*) HAVE_BETA=1 ;;
    --budget-total-uses|--budget-total-uses=*|--budget-ratio-to-q3|--budget-ratio-to-q3=*)
      echo 'This experiment requires fixed --beta, not an explicit budget' >&2; exit 2 ;;
  esac
done
if [[ "$HAVE_BETA" != 1 ]]; then echo 'Supply --beta BETA' >&2; exit 2; fi
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; frozen codec: $RUN; scene: $SCENE; output: $OUT"
echo 'Defaults: 2000 updates; MSE + calibrated SSIM; structural reference share 20%.'
echo 'V2: independent action residuals; view-limited mean uncertainty; 32 finite checks/update; log-odds cap 0.25.'
echo 'Single-draw learning / ten-draw deployment; frozen codecs; no joint phase.'
exec "$PYTHON" -u -m gaussian_jscc.structured_evidence_allocation \
  --run "$RUN" --scene "$SCENE" --out "$OUT" "${INIT[@]}" "$@"
