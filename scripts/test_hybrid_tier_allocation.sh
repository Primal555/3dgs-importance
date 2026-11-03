#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
CODEC="${1:?Usage: test_hybrid_tier_allocation.sh CODEC_RUN SCENE NEW_OUT [options]}"
SCENE="${2:?Set a trained scene}"
OUT="${3:?Set a new output directory}"
shift 3
if [[ ! -f "$CODEC/training.json" && -f "$CODEC/training/training.json" ]]; then CODEC="$CODEC/training"; fi
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; frozen codec: $CODEC; output: $OUT"
echo 'Fresh spatial value network: 700 windows, then automatically 2000 allocation updates.'
echo 'Dense calibrated first-order KEEP; sparse neural positive tiers; mirror updates; default network frozen.'
echo '10dB; 0/8/24/48; normalized MSE+SSIM RD; beta0.1; batch128. Use --network-mode online explicitly.'
echo 'Download OUT/review.tar.gz. Frozen allocation resume needs benefit_network.pt and allocation checkpoints; pretraining replay is separate.'
exec "$PYTHON" -u -m gaussian_jscc.hybrid_tier_allocation \
  --run "$CODEC" --scene "$SCENE" --out "$OUT" "${INIT[@]}" "$@"
