#!/usr/bin/env bash
# Fixed full-payload diagnosis; changes only tier sampling and channel noise.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export BOOTSTRAP_TIER=3
export CHANNEL=none
export SNR=10
export BOOTSTRAP_STEPS="${BOOTSTRAP_STEPS:-10000}"
# No channel randomness or mixed tiers: repeated validation trials are redundant.
export VALIDATION_TRIALS=1
OUT="${1:-$PROJECT/output/truck_q3_noiseless_$(date +%Y%m%d_%H%M%S)}"
echo 'Fixed q3: all 32 complex symbols, no noise, no random tiers; SNR=10 is conditioning only.'
exec bash "$PROJECT/scripts/test_point_transformer_logcov.sh" "$OUT"
