#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_channel_error_gates_truck.sh TRAINED_SELF_ONLY_RUN NEW_OUT [extra options]}"
OUT="${2:?Usage: test_channel_error_gates_truck.sh TRAINED_SELF_ONLY_RUN NEW_OUT [extra options]}"
shift 2
if [[ ! -f "$RUN/training.json" ]]; then
  if [[ -f "$RUN/final_pointwise10/training.json" ]]; then RUN="$RUN/final_pointwise10";
  elif [[ -f "$RUN/pointwise10/training.json" ]]; then RUN="$RUN/pointwise10"; fi
fi
test -f "$RUN/training.json" || { echo "Missing metadata: $RUN/training.json" >&2; exit 1; }
test -f "$RUN/checkpoints/final/codec.pt" || { echo "Missing checkpoint: $RUN/checkpoints/final/codec.pt" >&2; exit 1; }
command -v python >/dev/null || { echo 'Activate conda maskgs first' >&2; exit 1; }
echo "Source: $RUN/checkpoints/final/codec.pt; ALL source/self/semantic/TX weights frozen"
echo "First: read-only joint error / fixed decoder color sensitivity; log: source_geometry.log"
echo "Then: neighbor_plain10 / neighbor_gated10 sequentially, 2000 new updates EACH"
echo "New final neighbor increment only; latent MSE; 10 dB AWGN; LR=1e-4"
echo "Same payload, rates and XYZ path; no rendering updates or double-noise accumulation"
echo "Diagnostic directions do NOT automatically reweight the loss or change the transmitter"
echo "Every 500 updates: numeric validation/checkpoint; only initial/final images"
echo "Output: $OUT; GPU: $CUDA_VISIBLE_DEVICES"
python -u -m gaussian_jscc.channel_inverse_controls \
  --run "$RUN" --out "$OUT" --suite gated --steps 2000 --lr 1e-4 \
  --denoiser-hidden 64 --denoiser-neighbors 16 \
  --sensitivity-blocks 4 --sensitivity-views 8 --sensitivity-step .01 \
  --validate-every 500 --save-every 500 \
  --diagnostic-points 32768 --diagnostic-trials 8 --diagnostic-batch 2048 "$@"
