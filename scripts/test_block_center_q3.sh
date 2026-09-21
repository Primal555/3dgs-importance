#!/usr/bin/env bash
# Decoder-only upgrade of the frozen q3/noiseless experiment.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export XYZ_DECODER=block_center
OUT="${1:-$PROJECT/output/truck_block_center_q3_$(date +%Y%m%d_%H%M%S)}"
exec bash "$PROJECT/scripts/test_q3_noiseless_bootstrap.sh" "$OUT"
