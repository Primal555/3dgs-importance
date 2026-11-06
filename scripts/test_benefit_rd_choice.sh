#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
SOURCE="${1:?Usage: test_benefit_rd_choice.sh COMPLETED_STAGE_ONE NEW_OUT [options]}"
OUT="${2:?Set NEW output}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cpu}"
echo "Fixed comparison: $SOURCE; RD-choice output: $OUT; device: $DEVICE"
echo 'Reuse initial weights and fixed records. One new arm; original control is verified, not retrained.'
echo 'Uniform shuffled coverage; centered RD + per-view components + joint soft choice.'
echo 'Use the original update budget (normally 2000). Zero renders and allocation updates.'
echo 'Download OUT/review.tar.gz; final.pt stays on server.'
exec "$PYTHON" -u -m gaussian_jscc.benefit_rd_choice \
  --run "$SOURCE" --out "$OUT" --device "$DEVICE" "$@"
