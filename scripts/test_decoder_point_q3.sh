#!/usr/bin/env bash
# Experimental received-feature Point Transformer; no coordinate side stream.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DECODER_ATTENTION=feature_point
export XYZ_DECODER="${XYZ_DECODER:-additive}"
OUT="${1:-$PROJECT/output/truck_decoder_point_q3_$(date +%Y%m%d_%H%M%S)}"
exec bash "$PROJECT/scripts/test_q3_noiseless_bootstrap.sh" "$OUT"
