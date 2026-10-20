#!/usr/bin/env bash
# Continue PRE-JOINT channel recovery; never load the final joint checkpoint.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: continue_ssim_channel_truck.sh SOURCE_SSIM_RUN NEW_OUT [extra options]}"
OUT="${2:?Usage: continue_ssim_channel_truck.sh SOURCE_SSIM_RUN NEW_OUT [extra options]}"
shift 2
test -f "$RUN/training.json" || { echo "Missing source metadata: $RUN/training.json" >&2; exit 1; }
MODE=()
if [[ "${RESUME:-0}" == 1 ]]; then
  test -f "$OUT/checkpoints/latest/codec.pt" || { echo "Missing extension checkpoint: $OUT/checkpoints/latest/codec.pt" >&2; exit 1; }
  MODE=(--resume)
else
  test -f "$RUN/checkpoints/end_channel_bootstrap/codec.pt" || { echo "Missing PRE-JOINT checkpoint: $RUN/checkpoints/end_channel_bootstrap/codec.pt" >&2; exit 1; }
  test ! -e "$OUT" || { echo "Use a NEW output directory or RESUME=1: $OUT" >&2; exit 1; }
fi
echo "Source: $RUN; output: $OUT; GPU: $CUDA_VISIBLE_DEVICES"
echo "Add 4000 CHANNEL recovery updates (2000 -> 6000); preserve channel Adam/RNG and saved LRs (source: 1e-4)."
echo "Freeze BOTH semantic modules; latent MSE only; NO joint training. Validate/save every 500."
"${PYTHON_BIN:-python}" -u -m gaussian_jscc.modular_channel_continue \
  --run "$RUN" --out "$OUT" "${MODE[@]}" \
  --additional-steps 4000 --validate-every 500 --save-every 500 "$@"
