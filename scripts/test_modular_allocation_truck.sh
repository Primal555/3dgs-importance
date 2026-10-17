#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_modular_allocation_truck.sh TRAINED_RUN NEW_OUT [extra options]}"
OUT="${2:?Usage: test_modular_allocation_truck.sh TRAINED_RUN NEW_OUT [extra options]}"
shift 2
command -v python >/dev/null || { echo 'Activate conda maskgs first' >&2; exit 1; }
test -f "$RUN/training.json" || { echo "Missing source metadata: $RUN/training.json" >&2; exit 1; }
test -f "$RUN/checkpoints/final/codec.pt" || { echo 'Missing trained full codec' >&2; exit 1; }
RESUME_ARGS=()
if [[ "${RESUME:-0}" == 1 ]]; then
  test -f "$OUT/checkpoints/latest/state.pt" || { echo 'Missing allocation resume state' >&2; exit 1; }
  RESUME_ARGS=(--resume)
fi
echo "Source: $RUN; output: $OUT; physical GPU: $CUDA_VISIBLE_DEVICES"
echo 'Trained 8/24/48 full codec retained; fixed 10 dB; reliable 16bit XYZ'
echo '2000 frozen-codec allocator updates, then 1000 joint updates of table + all four codec modules'
echo 'Single-draw local feedback training; fixed ten-draw deployment; beta=0.01'
echo 'Joint codec LR=1e-5; independent q1/q2/q3 prefix anchor every 4 joint updates'
echo 'PSNR / SSIM / pretrained LPIPS; initial/final deployment images and tier contribution maps'
python -u -m gaussian_jscc.modular_allocation \
  --run "$RUN" --out "$OUT" "${RESUME_ARGS[@]}" \
  --allocation-steps 2000 --joint-steps 1000 --snr 10 --beta 0.01 \
  --codec-lr 1e-5 --keep-lr 0.01 --mask-lr 0.001 --mask-adam-eps 1e-15 \
  --prefix-anchor-every 4 --validate-every 250 --save-every 250 \
  --trials 2 --image-views 2 --lpips "$@"
