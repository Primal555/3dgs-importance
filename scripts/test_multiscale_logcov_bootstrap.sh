#!/usr/bin/env bash
# Architecture-only experiment: same logcov bootstrap loss and payload lengths.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CONTEXT_MODE=multiscale_self
OUT="${1:-$PROJECT/output/truck_multiscale_logcov_$(date +%Y%m%d_%H%M%S)}"
echo 'Pointwise self paths + gated multiscale context; unchanged spatial_logcov_v1; random weights.'
exec bash "$PROJECT/scripts/test_logcov_codec_bootstrap.sh" "$OUT"
