#!/usr/bin/env bash
# Opt-in experimental upgrade; no checkpoint migration or reserved XYZ stream.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export SPLIT_REPRESENTATION=logcov
OUT="${1:-$PROJECT/output/truck_logcov_codec_$(date +%Y%m%d_%H%M%S)}"
exec bash "$PROJECT/scripts/test_split_codec_bootstrap.sh" "$OUT"
