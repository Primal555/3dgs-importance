#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_calibrated_spatial_benefit.sh CODEC_RUN SCENE NEW_OUT --beta BETA [options]}"
SCENE="${2:?Set a trained scene}"
OUT="${3:?Set the output directory}"
shift 3
if [[ ! -f "$RUN/training.json" && -f "$RUN/training/training.json" ]]; then RUN="$RUN/training"; fi
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate environment or set PYTHON_BIN' >&2; exit 1; }
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; frozen codec: $RUN; scene: $SCENE; output: $OUT"
if [[ "${RESUME:-0}" == 1 ]]; then
  echo 'Exact resume: retain network, Adam, replay, validation and fixed scales. --pretrain-until sets an absolute target.'
else
  echo 'Fresh spatial benefit network; calibrated Bernoulli KL from true RD gaps; one experiment only.'
fi
echo 'Default: 200 windows, first 50 fit scales; 64 true label groups/window, then 64 updates x 8 groups/window.'
echo 'Unchanged spatial inputs, replay, MSE+SSIM, 0/8/24/48, 10dB; direction weight 0.25.'
echo 'Default phase is pretrain: allocation table remains frozen. Resume uses the SAME output directory.'
echo 'Allocation acceleration: --phase allocation --allocation-batch 64; runtime batch does not change the saved training plan.'
exec "$PYTHON" -u -m gaussian_jscc.calibrated_spatial_benefit_allocation \
  --run "$RUN" --scene "$SCENE" --out "$OUT" "${INIT[@]}" "$@"
