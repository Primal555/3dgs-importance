#!/usr/bin/env bash
# Only two new loss arms. Reuse the existing pure-MSE run; never launch it here.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:-$PROJECT/output/truck_beta_parallel_gpu2_20261017_213438/beta0p005}"
OUT="${2:-$PROJECT/output/truck_perceptual_finetune_$(date +%Y%m%d_%H%M%S)}"
if [[ $# -ge 2 ]]; then shift 2; elif [[ $# -eq 1 ]]; then shift; fi
case "${LOSS:-both}" in
  both) ARMS=(mse_ssim mse_lpips) ;;
  mse_ssim|mse_lpips) ARMS=("$LOSS") ;;
  *) echo 'LOSS must be both, mse_ssim or mse_lpips (pure MSE is not rerun)' >&2; exit 1 ;;
esac
for file in training.json complete.json checkpoints/final/codec.pt deployment/final/tiers.npy; do
  [[ -f "$RUN/$file" ]] || { echo "Missing source file: $RUN/$file" >&2; exit 1; }
done
PYTHON_BIN="${PYTHON_BIN:-python}"
# Sequential by default: LPIPS backprop needs extra VRAM, even with codec replay.
for ARM in "${ARMS[@]}"; do
  case "$ARM" in
    mse_ssim) WEIGHT="${SSIM_WEIGHT:-0.3333333333333333}" ;;
    mse_lpips) WEIGHT="${LPIPS_WEIGHT:-1}" ;;
  esac
  printf '\n=== %s: weight=%s; source=%s; out=%s ===\n' "$ARM" "$WEIGHT" "$RUN" "$OUT/$ARM"
  "$PYTHON_BIN" -u -m gaussian_jscc.modular_allocation \
    --run "$RUN" --out "$OUT/$ARM" --fixed-deployment --allocation-steps 0 \
    --joint-steps "${STEPS:-1000}" --beta 0.005 --codec-lr "${LR:-1e-5}" --snr 10 \
    --image-objective "$ARM" --image-aux-weight "$WEIGHT" \
    --prefix-anchor-every 4 --validate-every 250 --save-every 250 \
    --trials 2 --image-views 2 --seed 42 --device cuda "$@"
done
