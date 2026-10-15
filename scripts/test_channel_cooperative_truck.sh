#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_channel_cooperative_truck.sh COMPLETED_FULL_PREFIX_RUN NEW_OUT [extra options]}"
OUT="${2:?Usage: test_channel_cooperative_truck.sh COMPLETED_FULL_PREFIX_RUN NEW_OUT [extra options]}"
shift 2
if [[ ! -f "$RUN/training.json" && -f "$RUN/full_prefix10/training.json" ]]; then RUN="$RUN/full_prefix10"; fi
test -f "$RUN/training.json" || { echo "Missing metadata: $RUN/training.json" >&2; exit 1; }
test -f "$RUN/checkpoints/final/codec.pt" || { echo "Missing checkpoint: $RUN/checkpoints/final/codec.pt" >&2; exit 1; }
command -v python >/dev/null || { echo 'Activate conda maskgs first' >&2; exit 1; }
echo "Source: $RUN/checkpoints/final/codec.pt (read-only, all source modules frozen)"
echo "Three sequential arms: stage_spatial10 / final_pointwise10 / final_spatial10"
echo "2000 additional updates EACH; 10dB AWGN; LR=1e-4; pure y MSE; original payload budgets"
echo "Final pair: one zero-output residual each, identical parameters and initialization"
echo "Stage reference: three residuals, NOT parameter matched to final arms"
echo "Every 500 updates: numeric validation/checkpoint; only initial/final images"
echo "Output: $OUT; GPU: $CUDA_VISIBLE_DEVICES; per-arm logs inside this directory"
python -u -m gaussian_jscc.channel_inverse_controls \
  --run "$RUN" --out "$OUT" --suite cooperative --steps 2000 --lr 1e-4 \
  --denoiser-hidden 64 --denoiser-neighbors 16 \
  --validate-every 500 --save-every 500 \
  --diagnostic-points 32768 --diagnostic-trials 8 --diagnostic-batch 2048 "$@"
