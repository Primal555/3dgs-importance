#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="${1:-$ROOT/output/maskgaussian_three_$(date +%Y%m%d_%H%M%S)}"
if (( $# > 0 )); then shift; fi
exec "$PYTHON_BIN" -u -m utils.maskgaussian_batch \
    --out "$OUT" --steps "${PRUNE_STEPS:-5000}" \
    --lambda-mask "${LAMBDA_MASK:-0.1}" --resolution "${RESOLUTION:-2}" "$@"
