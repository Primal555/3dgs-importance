#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_direct_rd_diagnostic.sh DIRECT_RUN NEW_OUT [options]}"
OUT="${2:?Set a NEW diagnostic output directory}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
DRY=0
for arg in "$@"; do if [[ "$arg" == '--dry-run' ]]; then DRY=1; fi; done
if [[ "$DRY" == 0 ]]; then
  PAUSE=()
  if [[ "${PAUSE_SOURCE:-0}" == 1 ]]; then PAUSE=(--pause); fi
  "$PYTHON" -u -m gaussian_jscc.direct_rd_process "$RUN" "${PAUSE[@]}"
fi
echo "GPU: $CUDA_VISIBLE_DEVICES; fixed source: $RUN; diagnostic output: $OUT"
echo 'Frozen map/codec/benefit; default10 fresh proposal windows; top/middle/bottom/ordinary x16 actions.'
echo 'Matched exact singles, view-only shift with decoded candidates reused, original calibration AWGN trials, joint and disjoint groups.'
echo 'Source is read-only. Download NEW_OUT/review.tar.gz; no model, replay or image dump.'
exec "$PYTHON" -u -m gaussian_jscc.direct_rd_diagnostic --run "$RUN" --out "$OUT" "$@"
