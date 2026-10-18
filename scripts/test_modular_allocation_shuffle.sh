#!/usr/bin/env bash
# Evaluation only: same final codec, keep set and positive-tier histogram.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:-$PROJECT/output/truck_beta_parallel_gpu2_20261017_213438/beta0p005}"
OUT="${2:-$PROJECT/output/truck_beta005_shuffle_$(date +%Y%m%d_%H%M%S)}"
if [[ $# -ge 2 ]]; then shift 2; elif [[ $# -eq 1 ]]; then shift; fi
PYTHON_BIN="${PYTHON_BIN:-python}"
for file in training.json complete.json checkpoints/final/codec.pt deployment/final/tiers.npy deployment/final/allocation.json; do
  [[ -f "$RUN/$file" ]] || { echo "Missing: $RUN/$file" >&2; exit 1; }
done
command -v "$PYTHON_BIN" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN.' >&2; exit 1; }
echo "Source: $RUN; output: $OUT; GPU: $CUDA_VISIBLE_DEVICES; evaluation only (no training)."
"$PYTHON_BIN" -u -m gaussian_jscc.modular_allocation_shuffle \
  --run "$RUN" --out "$OUT" --device cuda --expected-beta 0.005 \
  --trials "${TRIALS:-2}" --test-views "${TEST_VIEWS:-0}" --image-views "${IMAGE_VIEWS:-2}" "$@"
