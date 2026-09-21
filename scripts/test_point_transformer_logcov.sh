#!/usr/bin/env bash
# Encoder-only update of multiscale_self. Random start; unchanged loss/receiver.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export ENCODER_ATTENTION=geometric_point
OUT="${1:-$PROJECT/output/truck_point_transformer_logcov_$(date +%Y%m%d_%H%M%S)}"
echo "Sender kNN grouped geometric attention; decoder attention=${DECODER_ATTENTION:-window}; fixed payload and spatial_logcov_v1."
exec bash "$PROJECT/scripts/test_multiscale_logcov_bootstrap.sh" "$OUT"
