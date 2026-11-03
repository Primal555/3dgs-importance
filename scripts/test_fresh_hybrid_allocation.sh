#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
SOURCE="${1:?Usage: test_fresh_hybrid_allocation.sh SOURCE_HYBRID_RUN NEW_OUT [options]}"
OUT="${2:?Set an independent output directory}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; evaluator source: $SOURCE; fresh output: $OUT"
echo 'Frozen evaluator and codec; fresh keep/tier probabilities, empty evidence and correction histories.'
echo 'Default: 2000 allocation steps from step1, positive cap1.0, original dense keep.'
echo 'Optional: --unbounded-tier-updates. Review: OUT/review.tar.gz.'
exec "$PYTHON" -u -m gaussian_jscc.fresh_hybrid_allocation \
  --source "$SOURCE" --out "$OUT" "${INIT[@]}" "$@"
