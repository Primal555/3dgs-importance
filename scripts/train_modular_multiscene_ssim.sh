#!/usr/bin/env bash
# Independent random-weight multi-scene modular experiment. No Truck initializer.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
MANIFEST="${1:?Usage: train_modular_multiscene_ssim.sh PRUNED_MANIFEST NEW_OUT [extra options]}"
OUT="${2:?Usage: train_modular_multiscene_ssim.sh PRUNED_MANIFEST NEW_OUT [extra options]}"
shift 2
test -f "$MANIFEST" || { echo "Missing pruned manifest: $MANIFEST" >&2; exit 1; }
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then
  CHECKPOINT="$OUT/checkpoints/latest/codec.pt"
  test -f "$CHECKPOINT" || { echo "Missing exact-resume checkpoint: $CHECKPOINT" >&2; exit 1; }
  INIT=(--resume "$CHECKPOINT")
else
  test ! -e "$OUT" || { echo "Use a NEW output directory (or RESUME=1): $OUT" >&2; exit 1; }
fi
echo "Manifest: $MANIFEST; output: $OUT; GPU: $CUDA_VISIBLE_DEVICES"
echo "Shared four-module model; Train/Truck/Playroom; RANDOM weights unless RESUME=1"
echo "Budgets PER SCENE: 5000 attribute/color + 1000 clean-render + 2000 channel + 2000 joint"
echo "8/24/48 complex symbols; reliable 16bit XYZ; AWGN 10 dB; SSIM relative weight=1/3"
echo "Semantic LR=1e-4; channel LR=1e-4; ONLY joint semantic decoder LR=1e-5; replay rendering"
"${PYTHON_BIN:-python}" -u -m gaussian_jscc.modular_multiscene \
  --manifest "$MANIFEST" --out "$OUT" "${INIT[@]}" \
  --scenes train truck playroom \
  --semantic-bootstrap-steps 5000 --semantic-render-steps 1000 \
  --channel-bootstrap-steps 2000 --channel-render-steps 2000 \
  --semantic-lr 1e-4 --channel-lr 1e-4 --joint-semantic-decoder-lr 1e-5 \
  --semantic-color-every 4 --semantic-color-directions 8 \
  --snr 10 --image-aux-weight 0.3333333333333333 --channel-latent-weight 1 \
  --blocks-per-batch 64 --views-per-step 2 --resolution 2 \
  --validate-every 500 --save-every 500 --lpips "$@"
