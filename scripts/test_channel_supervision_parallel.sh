#!/usr/bin/env bash
# nohup this parent: both independent arms keep running after SSH disconnect.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
RUN="${1:?Usage: test_channel_supervision_parallel.sh TEACHER_SOURCE_RUN NEW_SUITE_OUT [options]}"
OUT="${2:?Usage: test_channel_supervision_parallel.sh TEACHER_SOURCE_RUN NEW_SUITE_OUT [options]}"
shift 2
: "${GPU_COLOR:?Set GPU_COLOR to an available GPU}"
: "${GPU_SEPARATED:?Set GPU_SEPARATED to an available GPU (may equal GPU_COLOR)}"
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate conda maskgs or set PYTHON_BIN' >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Use a NEW suite output: $OUT" >&2; exit 1; }
[[ -f "$RUN/training.json" || -f "$RUN/block_continuation/block/training.json" ]] || {
  echo "Missing teacher source metadata: $RUN" >&2; exit 1;
}
mkdir -p -- "$OUT"
echo "Launching color on GPU $GPU_COLOR and separated on GPU $GPU_SEPARATED."
CUDA_VISIBLE_DEVICES="$GPU_COLOR" PYTHON_BIN="$PYTHON" \
  bash scripts/test_channel_supervision_truck.sh "$RUN" "$OUT/color" color "$@" >"$OUT/color.log" 2>&1 &
COLOR_PID=$!
printf '%s\n' "$COLOR_PID" >"$OUT/color.pid"
CUDA_VISIBLE_DEVICES="$GPU_SEPARATED" PYTHON_BIN="$PYTHON" \
  bash scripts/test_channel_supervision_truck.sh "$RUN" "$OUT/separated" separated "$@" >"$OUT/separated.log" 2>&1 &
SEPARATED_PID=$!
printf '%s\n' "$SEPARATED_PID" >"$OUT/separated.pid"
echo "color PID=$COLOR_PID log=$OUT/color.log"
echo "separated PID=$SEPARATED_PID log=$OUT/separated.log"
# Explicit parent termination stops only the two jobs launched here.
trap 'kill -TERM "$COLOR_PID" "$SEPARATED_PID" 2>/dev/null || true; exit 130' INT TERM
COLOR_STATUS=0; SEPARATED_STATUS=0
wait "$COLOR_PID" || COLOR_STATUS=$?
wait "$SEPARATED_PID" || SEPARATED_STATUS=$?
printf '{"color_exit_code":%s,"separated_exit_code":%s}\n' "$COLOR_STATUS" "$SEPARATED_STATUS" >"$OUT/exit_status.json"
if (( COLOR_STATUS != 0 || SEPARATED_STATUS != 0 )); then
  echo "One or both arms failed: color=$COLOR_STATUS separated=$SEPARATED_STATUS; inspect arm logs." >&2
  exit 1
fi
if [[ -f "$OUT/color/complete.json" && -f "$OUT/separated/complete.json" ]]; then
  printf '{"status":"complete","arms":["color","separated"]}\n' >"$OUT/complete.json"
  echo 'Both arms completed.'
else
  echo 'Both commands exited successfully without complete.json (e.g. dry-run); suite not marked complete.'
fi
