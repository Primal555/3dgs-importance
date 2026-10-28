#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
OUT="${1:?Usage: test_allocation_update_audit.sh NEW_OUT ALLOCATION_RUN [ALLOCATION_RUN...]}"
shift
if [[ "$#" -lt 1 ]]; then echo 'Supply at least one completed calibrated allocation run' >&2; exit 2; fi
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
EXTRA=()
if [[ "${DRY_RUN:-0}" == 1 ]]; then EXTRA=(--dry-run); fi
for RUN in "$@"; do
  NAME="$(basename -- "$RUN")"
  echo "GPU: $CUDA_VISIBLE_DEVICES; audit source: $RUN; output: $OUT/$NAME"
  "$PYTHON" -u -m gaussian_jscc.allocation_update_audit \
    --run "$RUN" --out "$OUT/$NAME" --windows "${WINDOWS:-20}" \
    --samples "${AUDIT_SAMPLES:-32}" --checkpoint "${CHECKPOINT:-best_rd}" "${EXTRA[@]}"
done
