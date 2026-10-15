#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_channel_spatial_truck.sh COMPLETED_FULL_PREFIX_RUN NEW_OUT [extra options]}"
OUT="${2:?Usage: test_channel_spatial_truck.sh COMPLETED_FULL_PREFIX_RUN NEW_OUT [extra options]}"
shift 2
if [[ ! -f "$RUN/training.json" && -f "$RUN/full_prefix10/training.json" ]]; then RUN="$RUN/full_prefix10"; fi
test -f "$RUN/training.json" || { echo "Missing metadata: $RUN/training.json" >&2; exit 1; }
test -f "$RUN/checkpoints/final/codec.pt" || { echo "Missing checkpoint: $RUN/checkpoints/final/codec.pt" >&2; exit 1; }
command -v python >/dev/null || { echo 'Activate conda maskgs first' >&2; exit 1; }
echo "Source: $RUN/checkpoints/final/codec.pt"
echo "Source full-prefix weights FROZEN; new zero-output receiver residuals ONLY"
echo "pointwise10 / spatial10: sequential 10dB arms, default 2000 new updates EACH"
echo "Same new parameter count and initialization; no added transmitted symbols"
echo "Only initial/final render images; metrics/checkpoints every 500 updates"
echo "Output: $OUT; GPU: $CUDA_VISIBLE_DEVICES; logs: pointwise10.log / spatial10.log"
python -u -m gaussian_jscc.channel_inverse_controls \
  --run "$RUN" --out "$OUT" --suite spatial --steps 2000 \
  --denoiser-hidden 64 --denoiser-neighbors 16 \
  --validate-every 500 --save-every 500 \
  --diagnostic-points 32768 --diagnostic-trials 8 --diagnostic-batch 2048 "$@"
