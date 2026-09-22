#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DECODER_MEMORY=received
OUT="${1:-$PROJECT/output/truck_received_memory_q3_$(date +%Y%m%d_%H%M%S)}"
echo 'Per-layer rereading of the SAME received payload; no localization token or extra channel symbols.'
exec bash "$PROJECT/scripts/test_transformer_trunk_q3.sh" "$OUT"
