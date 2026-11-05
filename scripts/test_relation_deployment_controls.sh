#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
SOURCE="${1:?Usage: test_relation_deployment_controls.sh RELATION_RUN NEW_OUT [options]}"
OUT="${2:?Set NEW controls output}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
INIT=(); if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; completed allocation: $SOURCE; controls output: $OUT"
echo 'Controls: restore all q0 to q1; preserve q0 mask and set retained points to q2.'
echo 'Frozen codec, zero optimizer updates; original preparation/test cameras, noise trials and RD references.'
echo 'Reuse matched original test baselines; combine preparation/test rendering for each new map.'
echo 'No relation bank, evaluator weights or replay required. Download OUT/review.tar.gz.'
exec "$PYTHON" -u -m gaussian_jscc.relation_deployment_controls \
  --source "$SOURCE" --out "$OUT" "${INIT[@]}" "$@"
