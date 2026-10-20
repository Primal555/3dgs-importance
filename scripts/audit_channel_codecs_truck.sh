#!/usr/bin/env bash
# Diagnostic only: never starts an optimizer or modifies source runs.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
CURRENT="${1:?Usage: audit_channel_codecs_truck.sh CURRENT_RUN REFERENCE_RUN NEW_OUT [extra options]}"
REFERENCE="${2:?Usage: audit_channel_codecs_truck.sh CURRENT_RUN REFERENCE_RUN NEW_OUT [extra options]}"
OUT="${3:?Usage: audit_channel_codecs_truck.sh CURRENT_RUN REFERENCE_RUN NEW_OUT [extra options]}"
shift 3
for RUN in "$CURRENT" "$REFERENCE"; do
  test -f "$RUN/training.json" || { echo "Missing metadata: $RUN/training.json" >&2; exit 1; }
  test -f "$RUN/checkpoints/final/codec.pt" || { echo "Missing checkpoint: $RUN/checkpoints/final/codec.pt" >&2; exit 1; }
done
test ! -e "$OUT" || { echo "Use a NEW independent output directory: $OUT" >&2; exit 1; }
echo "Read-only channel audit; GPU: $CUDA_VISIBLE_DEVICES; output: $OUT"
echo "32 original blocks, 32 paired AWGN trials; physical response on 4 blocks. NO training or rasterization."
"${PYTHON_BIN:-python}" -u -m gaussian_jscc.channel_audit \
  --run "$CURRENT" --reference-run "$REFERENCE" --out "$OUT" \
  --blocks 32 --trials 32 --response-blocks 4 --directions 8 --probe-nmse .02 --snr 10 "$@"
