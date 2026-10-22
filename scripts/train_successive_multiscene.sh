#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU (this experiment: 3)}"
RUN="${1:?Usage: train_successive_multiscene.sh COMPLETED_JOINT_B_RUN PRUNED_MANIFEST NEW_OUT [options]}"
MANIFEST="${2:?Usage: train_successive_multiscene.sh COMPLETED_JOINT_B_RUN PRUNED_MANIFEST NEW_OUT [options]}"
OUT="${3:?Usage: train_successive_multiscene.sh COMPLETED_JOINT_B_RUN PRUNED_MANIFEST NEW_OUT [options]}"
shift 3
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
test -f "$RUN/checkpoints/final/codec.pt" || { echo "Missing Joint B: $RUN" >&2; exit 1; }
test -f "$MANIFEST" || { echo "Missing pruned training manifest: $MANIFEST" >&2; exit 1; }
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; source Joint B: $RUN; output: $OUT"
echo 'Train/Truck/Playroom: 2000 actual updates EACH, total6000; shared four-module Adam LR1e-5.'
echo 'DrJohnson: ZERO-SHOT initial/final evaluation only; no fitting, calibration, or allocation learning.'
echo '10dB; 8/24/48; reliable16bit XYZ and binary16 block scales; source RGB MSE + latent MSE.'
exec "$PYTHON" -u -m gaussian_jscc.successive_multiscene \
  --run "$RUN" --manifest "$MANIFEST" --out "$OUT" "${INIT[@]}" \
  --heldout-manifest "${HELDOUT_MANIFEST:-configs/multiscene_tandt_db.json}" --heldout-scene drjohnson \
  --scenes train truck playroom --steps-per-scene 2000 --lr 1e-5 --latent-weight 1 \
  --validation-views 4 --validate-every 250 --save-every 100 \
  --test-views 32 --render-trials 4 --lpips "$@"
