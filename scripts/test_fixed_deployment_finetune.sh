#!/usr/bin/env bash
# Default: fixed beta=.005 deployment; MODE=joint gives a continuing-table control.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:-$PROJECT/output/truck_beta_parallel_gpu2_20261017_213438/beta0p005}"
OUT="${2:-$PROJECT/output/truck_fixed_finetune_$(date +%Y%m%d_%H%M%S)}"
if [[ $# -ge 2 ]]; then shift 2; elif [[ $# -eq 1 ]]; then shift; fi
case "${MODE:-fixed}" in
  fixed) MODE_FLAG=--fixed-deployment ;;
  joint) MODE_FLAG=--continue-allocation ;;
  *) echo 'MODE must be fixed or joint' >&2; exit 1 ;;
esac
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -u -m gaussian_jscc.modular_allocation \
  --run "$RUN" --out "$OUT" "$MODE_FLAG" --allocation-steps 0 \
  --joint-steps "${STEPS:-1000}" --beta 0.005 --codec-lr "${LR:-1e-5}" --snr 10 \
  --prefix-anchor-every 4 --validate-every 250 --save-every 250 \
  --trials 2 --image-views 2 --device cuda "$@"
