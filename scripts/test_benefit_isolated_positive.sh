#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
SOURCE="${1:?Usage: test_benefit_isolated_positive.sh FOOTPRINT_RUN NEW_OUT [options]}"
OUT="${2:?Set NEW output}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "Source footprint comparison: $SOURCE; output: $OUT"
echo 'Reuse immutable candidate data and both completed reference predictions; zero new renders.'
echo 'Frozen common network; independent positive point encoder and region branch; direct positive pairs.'
echo 'Default CPU, 2000 updates, evaluate every250. No probability updates. Download OUT/review.tar.gz.'
exec "$PYTHON" -u -m gaussian_jscc.benefit_isolated_positive_controls \
  --source "$SOURCE" --out "$OUT" --device "${TRAIN_DEVICE:-cpu}" "${INIT[@]}" "$@"
