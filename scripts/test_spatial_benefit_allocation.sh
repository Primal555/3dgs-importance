#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_spatial_benefit_allocation.sh CODEC_RUN SCENE NEW_OUT --beta BETA [options]}"
SCENE="${2:?Set a trained scene}"
OUT="${3:?Set a new output directory}"
shift 3
if [[ ! -f "$RUN/training.json" && -f "$RUN/training/training.json" ]]; then RUN="$RUN/training"; fi
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate environment or set PYTHON_BIN' >&2; exit 1; }
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; frozen codec: $RUN; scene: $SCENE; output: $OUT"
echo 'Spatially paired full-block neighbor tokens, cached ordered footprint profiles, exact paired render labels.'
echo '200 windows (first 50 fit fixed scales); 64 label groups/window; then 64 updates x 8 groups/window.'
echo 'Physical RD-unit value/common/pair supervision; MSE+SSIM, 0/8/24/48 at 10dB.'
echo 'Logs include per-record optimizer coverage. Float32 spatial replay is larger; retain replay with checkpoints.'
echo 'Resume allocation explicitly: RESUME=1, same OUT and --phase allocation. Default 2000 allocation windows.'
exec "$PYTHON" -u -m gaussian_jscc.spatial_benefit_allocation \
  --run "$RUN" --scene "$SCENE" --out "$OUT" "${INIT[@]}" "$@"
