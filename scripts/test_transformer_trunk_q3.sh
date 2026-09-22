#!/usr/bin/env bash
# Decoder-only experiment, random weights, full q3, no channel noise.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DECODER_ATTENTION=transformer_trunk
export DECODER_DEPTH="${DECODER_DEPTH:-4}"
export XYZ_DECODER=additive
OUT="${1:-$PROJECT/output/truck_transformer_trunk_q3_$(date +%Y%m%d_%H%M%S)}"
echo "Receiver: ${DECODER_DEPTH}-layer block Transformer; refinement=${DECODER_REFINEMENT:-none} (none=multi-depth readout); no coordinate side stream."
exec bash "$PROJECT/scripts/test_q3_noiseless_bootstrap.sh" "$OUT"
