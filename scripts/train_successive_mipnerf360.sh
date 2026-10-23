#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: train_successive_mipnerf360.sh COMPLETED_SHARED_RUN NEW_OUT [options]}"
OUT="${2:?Usage: train_successive_mipnerf360.sh COMPLETED_SHARED_RUN NEW_OUT [options]}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; completed shared codec: $RUN; output: $OUT"
echo 'Prune all 9 Mip-NeRF 360 assets sequentially (5000 steps each).'
echo 'Train: Train/Truck/Playroom + bicycle/garden/stump/bonsai/counter/kitchen; 1250 updates EACH, 11250 total.'
echo 'Hold out flowers/treehill/room from shared codec training; initial/final evaluation only.'
echo 'Four modules LR1e-5; NEW Adam; fixed source attribute statistics; SNR10, rates8/24/48; no allocation training.'
exec "$PYTHON" -u -m utils.mipnerf360_pipeline --run "$RUN" --out "$OUT" "${INIT[@]}" "$@"
