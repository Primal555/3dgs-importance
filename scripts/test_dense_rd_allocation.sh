#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
CODEC="${1:?Usage: test_dense_rd_allocation.sh CODEC_RUN SCENE NEW_OUT [options]}"
SCENE="${2:?Set a trained scene}"
OUT="${3:?Set a new output directory}"
shift 3
if [[ ! -f "$CODEC/training.json" && -f "$CODEC/training/training.json" ]]; then CODEC="$CODEC/training"; fi
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; frozen codec: $CODEC; scene: $SCENE; output: $OUT"
echo 'Dense keep + conditional tier Adam; single hard-Gumbel training, ten-draw deployment; 10dB, 0/8/24/48.'
echo 'Normalized MSE+SSIM and actual channel-use reference; default beta0.1, maximum2000 updates.'
echo 'Deployment RD plateau reduces LR; stable map/cost at the LR floor permits early ending.'
echo 'Review: OUT/review.tar.gz. Resume requires latest/state.pt and the original codec/scene inputs.'
exec "$PYTHON" -u -m gaussian_jscc.dense_rd_allocation \
  --run "$CODEC" --scene "$SCENE" --out "$OUT" "${INIT[@]}" "$@"
