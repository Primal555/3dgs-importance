#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
CODEC="${1:?Usage: test_matched_rd_allocation.sh CODEC_RUN BENEFIT_RUN SCENE NEW_OUT [options]}"
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
echo 'Frozen codec and benefit network; uniform q2 initialization; direct integer allocation.'
echo 'Common 4 optimization views / 2 paired noise conditions for rescoring, exact labels and acceptance.'
echo 'Every10 windows: at most512 rescored,64 exact groups; verified16-action chunks first, network-only groups separately; at most12 checks.'
echo 'Cached baseline decodes and projections; full-render verification every100 windows. Budgets are configurable.'
echo 'Download OUT/review.tar.gz; keep checkpoints/latest/state.pt separately for resume.'
exec "$PYTHON" -u -m gaussian_jscc.matched_rd_allocation \
  --run "$CODEC" --benefit-run "$BENEFIT" --scene "$SCENE" --out "$OUT" "${INIT[@]}" "$@"
