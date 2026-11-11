#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
SOURCE="${1:?Usage: test_student_rd_step_audit.sh STUDENT_RUN NEW_OUT [options]}"
OUT="${2:?Set NEW output directory}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
echo "GPU: $CUDA_VISIBLE_DEVICES; student: $SOURCE; audit output: $OUT"
echo 'Default: latest RD checkpoint, 3 independent one-step probes; original weights and Adam restored per probe.'
echo 'Paired train/calibration renders: full, small step, keep-only, positive-only, evaluated/unseen positive maps.'
echo 'Compare decayed vs original teacher weight when distinct. Source features reused; source files read only.'
echo 'Progress prints each evaluated map. Download OUT/review.tar.gz.'
exec "$PYTHON" -u -m gaussian_jscc.student_rd_step_audit --source "$SOURCE" --out "$OUT" "$@"
