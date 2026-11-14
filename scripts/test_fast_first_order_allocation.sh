#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_fast_first_order_allocation.sh CODEC_RUN SCENE NEW_OUT [options]}"
SCENE="${2:?Set scene, e.g. truck}"
OUT="${3:?Set a NEW output directory}"
shift 3
if [[ ! -f "$RUN/training.json" && -f "$RUN/training/training.json" ]]; then RUN="$RUN/training"; fi
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; frozen codec: $RUN; scene: $SCENE; output: $OUT"
echo 'Calibrated first-order allocation; beta defaults to 0.5, maximum 2000 updates, candidate batch128.'
echo 'Dense keep feedback, two candidate rounds, 32 independent corrections; keep/tier caps default0.25.'
echo 'Override positive-tier cap with --tier-max-log-odds; --max-log-odds controls the keep cap when overridden.'
echo 'Resolved learning rates, calibration interval and stopping mode are printed by the experiment.'
echo 'Download OUT/review.tar.gz; exact resume needs checkpoints plus the source codec and scene.'
exec "$PYTHON" -u -m gaussian_jscc.fast_first_order_allocation \
  --run "$RUN" --scene "$SCENE" --out "$OUT" --beta "${BETA:-0.5}" "${INIT[@]}" "$@"
