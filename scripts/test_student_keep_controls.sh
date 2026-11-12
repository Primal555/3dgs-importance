#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
SOURCE="${1:?Usage: test_student_keep_controls.sh STUDENT_RUN NEW_OUT [options]}"
OUT="${2:?Set NEW_OUT}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
[[ -f "$SOURCE/preparation/features.pt" ]] || { echo "Missing source feature cache: $SOURCE" >&2; exit 1; }
ARGS=()
[[ "${RESUME:-0}" == 1 ]] && ARGS+=(--resume)
echo "GPU: $CUDA_VISIBLE_DEVICES; source student: $SOURCE; output: $OUT"
echo 'Two arms, sequentially on this GPU: exact replacement / importance-weighted keep residual.'
echo 'Default: source best RD checkpoint and Adam; 200 updates/arm; 64 keep draws + original24 positive checks.'
echo 'Shared student learning, original training windows; original32 test views,4 trials, every50 updates.'
echo 'Fixed final update; test metrics are reporting only. Independent keep checks and paired map diagnostics every25 updates.'
echo 'Download OUT/review.tar.gz. RESUME=1 resumes both arms using the same SOURCE and OUT.'
exec "$PYTHON" -u -m gaussian_jscc.student_keep_controls --source "$SOURCE" --out "$OUT" "${ARGS[@]}" "$@"
