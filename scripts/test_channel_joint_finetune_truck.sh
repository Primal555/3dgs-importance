#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_channel_joint_finetune_truck.sh TRAINED_RUN NEW_OUT [extra options]}"
OUT="${2:?Usage: test_channel_joint_finetune_truck.sh TRAINED_RUN NEW_OUT [extra options]}"
shift 2
# Passing a suite root selects the explicit plain-neighbor arm. An arm path
# (including neighbor_gated10) is accepted unchanged, without altering it.
if [[ ! -f "$RUN/training.json" && -f "$RUN/neighbor_plain10/training.json" ]]; then
  RUN="$RUN/neighbor_plain10"
fi
test -f "$RUN/training.json" || { echo "Missing metadata: $RUN/training.json" >&2; exit 1; }
command -v python >/dev/null || { echo 'Activate conda maskgs first' >&2; exit 1; }
INIT=(--init-codec "$RUN/checkpoints/final/codec.pt")
if [[ "${RESUME:-0}" == 1 ]]; then
  test -f "$OUT/checkpoints/latest/codec.pt" || { echo 'Missing latest checkpoint for exact resume' >&2; exit 1; }
  INIT=(--resume "$OUT/checkpoints/latest/codec.pt")
else
  test -f "$RUN/checkpoints/final/codec.pt" || { echo "Missing trained checkpoint: $RUN/checkpoints/final/codec.pt" >&2; exit 1; }
fi
echo "Source metadata: $RUN; checkpoint: ${INIT[1]}"
echo "No architecture change; full saved config/rates retained; 10 dB AWGN"
echo "First 2000 updates: channel encoder + decoder; both semantic modules frozen; latent MSE"
echo "Then 1000 updates: channel encoder + decoder + semantic decoder; semantic encoder frozen"
echo "Joint objective: source PLY RGB MSE + 1 * latent MSE; replay backward"
echo "Channel LR=1e-4; semantic decoder joint LR=1e-5; no extra color/bootstrap updates"
echo "q1/q2/q3/mixed cycle; numeric validation/checkpoint every 500; initial/final images only"
echo "Output: $OUT; GPU: $CUDA_VISIBLE_DEVICES"
python -u -m gaussian_jscc.modular_experiment \
  --run "$RUN" --out "$OUT" "${INIT[@]}" \
  --semantic-bootstrap-steps 0 --semantic-render-steps 0 \
  --channel-bootstrap-steps 2000 --channel-render-steps 1000 \
  --channel-encoder-lr 1e-4 --channel-decoder-lr 1e-4 --semantic-decoder-lr 1e-5 \
  --train-snrs 10 --channel-latent-weight 1 \
  --validate-every 500 --save-every 500 --color-metrics --lpips "$@"
