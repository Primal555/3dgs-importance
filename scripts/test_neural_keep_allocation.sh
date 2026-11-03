#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
CODEC="${1:?Usage: test_neural_keep_allocation.sh CODEC_RUN SCENE NEW_OUT [options]}"
SCENE="${2:?Set a trained scene}"
OUT="${3:?Set a new output directory}"
shift 3
if [[ ! -f "$CODEC/training.json" && -f "$CODEC/training/training.json" ]]; then CODEC="$CODEC/training"; fi
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; frozen codec: $CODEC; output: $OUT"
echo 'Fresh 700-window spatial pretraining, then 2000 allocation steps; positive network frozen by default.'
echo 'Default learned KEEP: all-point smoothed proposals, dense render/rate VJP and paired q0-q3 calibration.'
echo 'beta0.1; 0/8/24/48; 10dB; ten-draw deployment. Review: OUT/review.tar.gz.'
exec "$PYTHON" -u -m gaussian_jscc.neural_keep_allocation \
  --run "$CODEC" --scene "$SCENE" --out "$OUT" "${INIT[@]}" "$@"
