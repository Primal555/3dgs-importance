#!/usr/bin/env bash
# Keep the trained SSIM semantic AND channel weights; only redo joint adaptation.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_ssim_joint_lr_truck.sh SSIM_RUN NEW_OUT [extra options]}"
OUT="${2:?Usage: test_ssim_joint_lr_truck.sh SSIM_RUN NEW_OUT [extra options]}"
shift 2
test -f "$RUN/training.json" || { echo "Missing metadata: $RUN/training.json" >&2; exit 1; }
if [[ "${RESUME:-0}" == 1 ]]; then
  CHECKPOINT="$OUT/checkpoints/latest/codec.pt"
  test -f "$CHECKPOINT" || { echo "Missing resume checkpoint: $CHECKPOINT" >&2; exit 1; }
  INIT=(--resume "$CHECKPOINT")
else
  CHECKPOINT="$RUN/checkpoints/end_channel_bootstrap/codec.pt"
  test -f "$CHECKPOINT" || { echo "Missing channel initializer: $CHECKPOINT" >&2; exit 1; }
  INIT=(--init-codec "$CHECKPOINT")
fi
echo "Source: $RUN; checkpoint: $CHECKPOINT"
echo "Retain all trained codec weights/config; fresh Adam for a NEW joint run"
echo "RESUME=1 instead restores the new run's optimizer/RNG/progress"
echo "Channel encoder/decoder LR=1e-4; semantic decoder LR=1e-5; semantic encoder frozen"
echo "2000 joint updates ONLY; 10 dB; MSE + (1/3)*(1-SSIM) + latent MSE"
echo "Output: $OUT; GPU: $CUDA_VISIBLE_DEVICES"
"${PYTHON_BIN:-python}" -u -m gaussian_jscc.modular_experiment \
  --run "$RUN" --out "$OUT" "${INIT[@]}" \
  --semantic-bootstrap-steps 0 --semantic-render-steps 0 \
  --channel-bootstrap-steps 0 --channel-render-steps 2000 \
  --channel-encoder-lr 1e-4 --channel-decoder-lr 1e-4 --semantic-decoder-lr 1e-5 \
  --image-objective mse_ssim --image-aux-weight 0.3333333333333333 \
  --channel-latent-weight 1 --train-snrs 10 --test-snrs 10 --seed 42 \
  --validate-every 500 --save-every 500 --color-metrics --lpips "$@"
