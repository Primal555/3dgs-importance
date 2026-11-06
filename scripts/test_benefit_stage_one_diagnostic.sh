#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
SOURCE="${1:?Usage: test_benefit_stage_one_diagnostic.sh OFFLINE_COMPARISON NEW_OUT [options]}"
OUT="${2:?Set NEW output}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cpu}"
echo "Completed comparison: $SOURCE; diagnostic: $OUT; device: $DEVICE"
echo 'Read both final checkpoints and the original fixed validation manifest.'
echo 'Zero optimizer updates and renders. Download OUT/review.tar.gz.'
exec "$PYTHON" -u -m gaussian_jscc.benefit_stage_one_diagnostic \
  --run "$SOURCE" --out "$OUT" --device "$DEVICE" "$@"
