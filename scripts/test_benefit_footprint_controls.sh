#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to one available GPU}"
COMPARISON="${1:?Usage: test_benefit_footprint_controls.sh STAGE_ONE_RUN CODEC_RUN NEW_OUT [options]}"
CODEC="${2:?Set frozen codec run}"
OUT="${3:?Set NEW output}"
shift 3
if [[ ! -f "$CODEC/training.json" && -f "$CODEC/training/training.json" ]]; then CODEC="$CODEC/training"; fi
PYTHON="${PYTHON_BIN:-python}"
COLLECT_RESUME=()
TRAIN_RESUME=()
if [[ "${RESUME:-0}" == 1 ]]; then
  if [[ -d "$OUT/data" ]]; then COLLECT_RESUME=(--resume); fi
  if [[ -d "$OUT/training" ]]; then TRAIN_RESUME=(--resume); fi
fi
echo "GPU: $CUDA_VISIBLE_DEVICES; comparison: $COMPARISON; codec: $CODEC; output: $OUT"
echo 'Collect 4096 training / 512 validation records once; original whole-block holdout and RD references.'
echo 'Both models start with matched predictions, train 2000 updates each using the same original loss.'
echo 'New input: target/block candidate-region grids. Independent common and relative output heads.'
echo 'Collection uses CUDA; paired offline training defaults to CPU. Download OUT/review.tar.gz.'
"$PYTHON" -u -m gaussian_jscc.benefit_footprint_controls --phase collect \
  --comparison "$COMPARISON" --codec "$CODEC" --out "$OUT/data" --device cuda "${COLLECT_RESUME[@]}" "$@"
"$PYTHON" -u -m gaussian_jscc.benefit_footprint_controls --phase train \
  --data "$OUT/data" --out "$OUT/training" --device "${TRAIN_DEVICE:-cpu}" "${TRAIN_RESUME[@]}" "$@"
cp -- "$OUT/training/review.tar.gz" "$OUT/review.tar.gz"
