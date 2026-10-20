#!/usr/bin/env bash
# Sequential point/block diagnostics, channel continuation and joint finetuning.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_power_followup_truck.sh POWER_SUITE NEW_OUT [extra options]}"
OUT="${2:?Usage: test_power_followup_truck.sh POWER_SUITE NEW_OUT [extra options]}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate conda maskgs or set PYTHON_BIN' >&2; exit 1; }
echo "Source power suite: $RUN; GPU: $CUDA_VISIBLE_DEVICES"
echo "Point/block each: clean-attribute interventions -> +4000 channel -> +1000 joint."
echo "Channel Adam retained; original y encoder frozen. Joint decoder LR=1e-5; channel LR=1e-4."
echo "32 actual test views, 4 paired image noise trials; original validation views logged separately."
echo "Worker logs: $OUT/point.log and $OUT/block.log"
"$PYTHON" -u -m gaussian_jscc.power_followup \
  --run "$RUN" --out "$OUT" --channel-steps 4000 --joint-steps 1000 \
  --channel-lr 1e-4 --semantic-decoder-lr 1e-5 \
  --validate-every 1000 --joint-validate-every 250 --save-every 250 \
  --test-views 32 --render-trials 4 --diagnostic-blocks 32 --response-blocks 8 --trials 8 --lpips "$@"
