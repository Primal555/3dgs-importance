#!/usr/bin/env bash
# Shared encoder/decoder; encode once, transmit 8 / 8+8 / 8+8+16 symbols.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PREFIX_MODE=progressive
OUT="${1:-$PROJECT/output/truck_progressive16_render_$(date +%Y%m%d_%H%M%S)}"
echo 'Progressive JSCC: same base symbols for all positive tiers; SNR conditioning retained.'
echo '8 + 8 + 16 complex symbols, independently normalized layers; shared decoder.'
echo 'Validation uses paired point/symbol noise; prefix_gains records actual enhancement benefit.'
exec bash "$PROJECT/scripts/train_quantized16_render_only.sh" "$OUT"
