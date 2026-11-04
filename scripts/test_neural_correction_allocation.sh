#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
CODEC="${1:?Usage: test_neural_correction_allocation.sh CODEC_RUN SCENE NEW_OUT [options]}"
SCENE="${2:?Set scene}"; OUT="${3:?Set NEW output}"; shift 3
if [[ ! -f "$CODEC/training.json" && -f "$CODEC/training/training.json" ]]; then CODEC="$CODEC/training"; fi
PYTHON="${PYTHON_BIN:-python}"
INIT=(); if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; codec: $CODEC; output: $OUT"
echo 'Aligned 700-window pretraining, then 2000 online allocation steps; positive cap1.0, keep cap0.25.'
echo 'Exact corrections supervise action gaps; residuals are recomputed by the current network in matched contexts.'
echo 'Download OUT/review.tar.gz. Resume needs checkpoints, benefit_network.pt and allocation/neural_replay.'
exec "$PYTHON" -u -m gaussian_jscc.neural_correction_allocation \
  --run "$CODEC" --scene "$SCENE" --out "$OUT" "${INIT[@]}" "$@"
