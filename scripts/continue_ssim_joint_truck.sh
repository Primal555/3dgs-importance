#!/usr/bin/env bash
# Extend completed joint training, retaining optimizer moments and RNG.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: continue_ssim_joint_truck.sh COMPLETED_RUN NEW_OUT [extra options]}"
OUT="${2:?Usage: continue_ssim_joint_truck.sh COMPLETED_RUN NEW_OUT [extra options]}"
shift 2
test -f "$RUN/training.json" || { echo "Missing source metadata: $RUN/training.json" >&2; exit 1; }
test -f "$RUN/checkpoints/final/codec.pt" || { echo "Missing completed checkpoint: $RUN/checkpoints/final/codec.pt" >&2; exit 1; }
MODE=()
if [[ "${RESUME:-0}" == 1 ]]; then
  test -f "$OUT/checkpoints/latest/codec.pt" || { echo "Missing extension checkpoint: $OUT/checkpoints/latest/codec.pt" >&2; exit 1; }
  MODE=(--resume)
else
  test ! -e "$OUT" || { echo "Use a NEW output directory or RESUME=1: $OUT" >&2; exit 1; }
fi
echo "Completed source: $RUN; continuation output: $OUT; GPU: $CUDA_VISIBLE_DEVICES"
echo "Add 1000 JOINT updates, validate/save every 250; keep Adam, RNG, architecture, objective and LRs"
"${PYTHON_BIN:-python}" -u -m gaussian_jscc.modular_joint_continue \
  --run "$RUN" --out "$OUT" "${MODE[@]}" \
  --additional-steps 1000 --validate-every 250 --save-every 250 "$@"
