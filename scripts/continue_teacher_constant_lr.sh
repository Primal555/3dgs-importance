#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES}"
SOURCE="${1:?Usage: continue_teacher_constant_lr.sh COMPLETED_TEACHER NEW_OUT [options]}"
OUT="${2:?Usage: continue_teacher_constant_lr.sh COMPLETED_TEACHER NEW_OUT [options]}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
DIGITAL="${DIGITAL_PYTHON:-$HOME/miniforge3/envs/gs_digital/bin/python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
[[ -x "$DIGITAL" ]] || { echo "Missing gs_digital interpreter: $DIGITAL" >&2; exit 1; }
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; teacher source: $SOURCE; output: $OUT"
echo 'Continue latest table/history/view scheduler/RNG for 500 additional updates; same constant LRs and caps.'
echo 'Fixed codec, beta and digital RD references; calibration/test views inherited. Latest and best maps both reported.'
echo 'Final selected/latest/q2 actual side delivery: LDPC4096/6144 + 16QAM, no ARQ.'
echo 'Download OUT/review.tar.gz. Resume: same SOURCE/OUT/options with RESUME=1.'
exec "$PYTHON" -u -m gaussian_jscc.teacher_constant_continue --source "$SOURCE" --out "$OUT" \
  --digital-python "$DIGITAL" "${INIT[@]}" "$@"
