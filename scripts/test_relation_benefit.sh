#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
CODEC="${1:?Usage: test_relation_benefit.sh CODEC_RUN SCENE NEW_OUT [options]}"
SCENE="${2:?Set scene}"
OUT="${3:?Set NEW output}"
shift 3
if [[ ! -f "$CODEC/training.json" && -f "$CODEC/training/training.json" ]]; then CODEC="$CODEC/training"; fi
PYTHON="${PYTHON_BIN:-python}"
INIT=(); if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; frozen codec: $CODEC; scene: $SCENE; output: $OUT"
echo 'Relation model: 128 channels; 16 spatial and 16 projection neighbors; q0/8/24/48.'
echo '700 teacher windows; first50 fit scales; 64 full groups/window; 64 network updates/window after fitting.'
echo 'Rotating q1/q2/q3/mixed backgrounds; original PLY rendering target; 2 paired-noise trials; 10dB.'
echo 'Then 4 conditional graph passes, equal signed-value mean, one RD tier selection, and held-out PSNR/SSIM/LPIPS.'
echo 'Frozen codec; 41600 evaluator optimizer updates; shared static/candidate cache; background-indexed replay.'
echo 'Digital side costs: LDPC4096/6144 +16QAM, with separate file headers/CRC/padding.'
echo 'CUDA consistency runs before training. Download OUT/review.tar.gz; resume also needs cache and checkpoints.'
exec "$PYTHON" -u -m gaussian_jscc.relation_benefit_experiment \
  --run "$CODEC" --scene "$SCENE" --out "$OUT" "${INIT[@]}" "$@"
