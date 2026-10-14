#!/usr/bin/env bash
set -euo pipefail
RUN="${1:?Usage: test_latent_decoder_sensitivity.sh POINTNET_SUITE NEW_OUT [extra options]}"
OUT="${2:?Usage: test_latent_decoder_sensitivity.sh POINTNET_SUITE NEW_OUT [extra options]}"
shift 2
DEVICE="${DEVICE:-cpu}"
THREADS="${THREADS:-4}"
if [[ "$DEVICE" == cuda* ]]; then
  : "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES when using CUDA}"
fi
test -f "$RUN/baseline/training.json" || { echo "Missing baseline metadata in $RUN" >&2; exit 1; }
test -f "$RUN/both/training.json" || { echo "Missing both-upgraded metadata in $RUN" >&2; exit 1; }
command -v python >/dev/null || { echo 'Activate conda maskgs first' >&2; exit 1; }
echo "Frozen decoder audit only; no training/channel/tier changes. Device: $DEVICE; CPU threads: $THREADS"
python -u -m gaussian_jscc.latent_sensitivity --run "$RUN" --out "$OUT" \
  --device "$DEVICE" --cpu-threads "$THREADS" "$@"
