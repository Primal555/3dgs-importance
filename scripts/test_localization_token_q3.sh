#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DECODER_LOCALIZATION=token_translation
OUT="${1:-$PROJECT/output/truck_localization_token_q3_$(date +%Y%m%d_%H%M%S)}"
echo 'Zero-start block translation from received multi-depth features; unchanged loss and payload.'
exec bash "$PROJECT/scripts/test_transformer_trunk_q3.sh" "$OUT"
