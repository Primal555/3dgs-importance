#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_response_value_allocation.sh COMPLETED_CODEC_RUN SCENE NEW_OUT --beta BETA [options]}"
SCENE="${2:?Set a trained scene}"
OUT="${3:?Set a new output directory}"
shift 3
if [[ ! -f "$RUN/training.json" && -f "$RUN/training/training.json" ]]; then RUN="$RUN/training"; fi
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; frozen codec: $RUN; scene: $SCENE; output: $OUT"
echo 'Unified q0-anchored values; candidate reconstruction and footprint patches; no first-order inputs/backward.'
echo '200 pretraining windows (50 normalization); 32 exact intervention renders/window; 10dB, 0/8/24/48.'
echo 'Allocation resumes explicitly with same OUT and --phase allocation; default 2000 updates.'
exec "$PYTHON" -u -m gaussian_jscc.learned_benefit_allocation \
  --run "$RUN" --scene "$SCENE" --out "$OUT" "${INIT[@]}" "$@" --benefit-variant response
