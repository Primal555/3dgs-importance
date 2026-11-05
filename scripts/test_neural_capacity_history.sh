#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
CODEC="${1:?Usage: test_neural_capacity_history.sh CODEC_RUN SCENE NEW_OUT [options]}"
SCENE="${2:?Set scene}"; OUT="${3:?Set NEW output}"; shift 3
if [[ ! -f "$CODEC/training.json" && -f "$CODEC/training/training.json" ]]; then CODEC="$CODEC/training"; fi
PYTHON="${PYTHON_BIN:-python}"
INIT=(); if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; frozen codec: $CODEC; scene: $SCENE; output: $OUT"
echo 'Defaults: width256, 206082 parameters. All historical teachers retained on disk; bounded read cache.'
echo '700 fresh pretraining windows (first50 fit scales), then2000 online allocation steps; beta0.1.'
echo 'Dense keep feedback, positive cap1.0, keep cap0.25, original teachers/losses/corrections.'
echo 'Download OUT/review.tar.gz. Resume retains checkpoints, benefit_network.pt and both replay directories.'
exec "$PYTHON" -u -m gaussian_jscc.neural_capacity_history \
  --run "$CODEC" --scene "$SCENE" --out "$OUT" "${INIT[@]}" "$@"
