#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
SOURCE="${1:?Usage: test_hybrid_tier_step_controls.sh SOURCE_RUN_OR_SNAPSHOT NEW_OUT [options]}"
OUT="${2:?Set an independent output directory}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; source snapshot: $SOURCE; new output: $OUT"
echo 'Allocation fork only; no evaluator pretraining. Frozen codec/network, original dense keep and history.'
echo 'Default positive cap1.0; explicit --unbounded-tier-updates removes the positive step limit.'
echo 'Global steps continue to original target (normally2000). Review: OUT/review.tar.gz.'
exec "$PYTHON" -u -m gaussian_jscc.hybrid_tier_step_controls \
  --source "$SOURCE" --out "$OUT" "${INIT[@]}" "$@"
