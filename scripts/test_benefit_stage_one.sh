#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
SOURCE="${1:?Usage: test_benefit_stage_one.sh WIDE_HISTORY_RUN NEW_OUT [options]}"
OUT="${2:?Set NEW output}"; shift 2
PYTHON="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cpu}"
echo "Source snapshot: $SOURCE; offline output: $OUT; device: $DEVICE"
echo 'Paired evaluator-only experiment: original loss versus positive-pair repair.'
echo 'Defaults: 4096 fixed training records, 2000 optimizer updates PER ARM, fresh AdamW.'
echo 'Zero scene renders, zero mask updates. Source trainer may continue.'
echo 'Download OUT/review.tar.gz; final evaluator weights stay under original/ and repaired/.'
exec "$PYTHON" -u -m gaussian_jscc.benefit_stage_one \
  --run "$SOURCE" --out "$OUT" --device "$DEVICE" "$@"
