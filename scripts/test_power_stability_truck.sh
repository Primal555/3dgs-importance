#!/usr/bin/env bash
# One GPU, sequential independent workers; never resumes failed joint weights.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
POWER="${1:?Usage: test_power_stability_truck.sh POWER_SUITE PARTIAL_FOLLOWUP NEW_OUT [options]}"
FOLLOWUP="${2:?Usage: test_power_stability_truck.sh POWER_SUITE PARTIAL_FOLLOWUP NEW_OUT [options]}"
OUT="${3:?Usage: test_power_stability_truck.sh POWER_SUITE PARTIAL_FOLLOWUP NEW_OUT [options]}"
shift 3
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate conda maskgs or set PYTHON_BIN' >&2; exit 1; }
echo "GPU: $CUDA_VISIBLE_DEVICES; output: $OUT"
echo 'Block channel continuation +4000; then point/block plain and instrumented 16-step probes.'
echo 'Original endpoints are read-only. No clipping, no LR changes, no long joint training.'
echo "Logs: $OUT/block_continuation.log (then block_continuation/block.log), point_plain.log, point_instrumented.log, block_plain.log, block_instrumented.log"
"$PYTHON" -u -m gaussian_jscc.power_stability_controls \
  --power-run "$POWER" --followup "$FOLLOWUP" --out "$OUT" \
  --channel-steps 4000 --audit-steps 16 --top-k 16 "$@"
