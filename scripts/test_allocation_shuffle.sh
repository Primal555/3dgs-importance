#!/usr/bin/env bash
# One shuffled map, same histogram, no optimizer or training.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
TRAINING="${1:?Usage: CUDA_VISIBLE_DEVICES=2 bash scripts/test_allocation_shuffle.sh TRAINING_DIR [OUT]}"
OUT="${2:-$TRAINING/allocation_shuffle}"
PYTHON_BIN="${PYTHON_BIN:-python}"
for file in training.json codec_best_joint.pt route2_best_joint.pt; do
  [[ -f "$TRAINING/$file" ]] || { echo "Missing: $TRAINING/$file" >&2; exit 1; }
done
command -v "$PYTHON_BIN" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN.' >&2; exit 1; }
"$PYTHON_BIN" -u -m gaussian_jscc.allocation_shuffle \
  --training "$TRAINING" --out "$OUT" --device cuda \
  --trials "${TRIALS:-2}" --shuffle-seed "${SHUFFLE_SEED:-2026}"
