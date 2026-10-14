#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_channel_inverse_truck.sh COMPLETED_Y_RUN NEW_OUT [extra options]}"
OUT="${2:?Usage: test_channel_inverse_truck.sh COMPLETED_Y_RUN NEW_OUT [extra options]}"
shift 2
if [[ ! -f "$RUN/training.json" && -f "$RUN/progressive/training.json" ]]; then
  RUN="$RUN/progressive"
fi
test -f "$RUN/training.json" || { echo "Missing metadata: $RUN/training.json" >&2; exit 1; }
test -f "$RUN/checkpoints/final/codec.pt" || {
  echo "Missing completed channel checkpoint: $RUN/checkpoints/final/codec.pt" >&2
  echo "Use the server's original completed run; do not initialize from the clean semantic teacher." >&2
  exit 1
}
command -v python >/dev/null || { echo 'Activate conda maskgs first' >&2; exit 1; }
echo "Source weights + Adam: $RUN/checkpoints/final/codec.pt"
echo "Two sequential arms on GPU $CUDA_VISIBLE_DEVICES: noiseless / 10dB AWGN"
echo "Default: +2000 updates EACH; same batches and tier masks; frozen semantic modules"
echo "Initial/final 32768-point, 8-trial latent diagnostics; render evaluation only"
echo "Output: $OUT; arm logs: $OUT/noiseless.log and $OUT/awgn10.log"
python -u -m gaussian_jscc.channel_inverse_controls \
  --run "$RUN" --out "$OUT" --steps 2000 \
  --validate-every 500 --save-every 500 \
  --diagnostic-points 32768 --diagnostic-trials 8 --diagnostic-batch 2048 "$@"
