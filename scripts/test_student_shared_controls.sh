#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
SOURCE="${1:?Usage: test_student_shared_controls.sh STUDENT_RUN NEW_OUT [options]}"
OUT="${2:?Set NEW_OUT}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
[[ -f "$SOURCE/preparation/features.pt" ]] || { echo "Missing source feature cache: $SOURCE" >&2; exit 1; }
ARGS=()
[[ "${RESUME:-0}" == 1 ]] && ARGS+=(--resume)
echo "GPU: $CUDA_VISIBLE_DEVICES; source student: $SOURCE; output: $OUT"
echo 'One shared-residual arm, 200 updates; source best checkpoint and its Adam state.'
echo 'Default: 104 global keep draws + 24 retained four-state groups; 128 keep and 48 positive action draws.'
echo 'Shared backgrounds/noise; batched original-block decoding; q0 toggles re-encode semantic context.'
echo 'Both residual branches use sampling weights. Output mean KL budgets: keep/tier 2e-5; Adam advances once/update.'
echo 'Inherit original test views/trials (source: 32 views, 4 trials); report every 50 updates; fixed final endpoint.'
echo 'Download OUT/review.tar.gz. RESUME=1 continues the same SOURCE and OUT.'
exec "$PYTHON" -u -m gaussian_jscc.student_shared_controls --source "$SOURCE" --out "$OUT" "${ARGS[@]}" "$@"
