#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
CODEC="${1:?Usage: test_direct_rd_allocation.sh CODEC_RUN BENEFIT_RUN SCENE NEW_OUT [options]}"
BENEFIT="${2:?Set the existing calibrated spatial benefit run}"
SCENE="${3:?Set a trained scene}"
OUT="${4:?Set a new output directory}"
shift 4
if [[ ! -f "$CODEC/training.json" && -f "$CODEC/training/training.json" ]]; then CODEC="$CODEC/training"; fi
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; codec: $CODEC; benefit: $BENEFIT; output: $OUT"
echo 'Frozen codec AND benefit network; start uniform q2, direct integer tiers; no replay or network updates.'
echo '2000 candidate windows, batch128; paired calibration RD audit every10, up to2 backtracks; 10dB, 0/8/24/48.'
echo 'Final/latest source benefit checkpoint by default; actual source step is recorded. Use --benefit-checkpoint for an explicit saved state.'
echo 'Download OUT/review.tar.gz for analysis; checkpoints/latest/state.pt remains separate for exact resume.'
exec "$PYTHON" -u -m gaussian_jscc.direct_rd_allocation \
  --run "$CODEC" --benefit-run "$BENEFIT" --scene "$SCENE" --out "$OUT" "${INIT[@]}" "$@"
