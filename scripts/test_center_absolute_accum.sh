#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CENTER_ACCUMULATION_STEPS="${CENTER_ACCUMULATION_STEPS:-2}"
OUT="${1:-$PROJECT/output/truck_center_absolute_accum${CENTER_ACCUMULATION_STEPS}_$(date +%Y%m%d_%H%M%S)}"
exec bash "$PROJECT/scripts/test_center_absolute_soft.sh" "$OUT"
