#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_integrated_progressive_color_truck.sh SOURCE_METADATA_RUN NEW_OUT [extra options]}"
OUT="${2:?Usage: test_integrated_progressive_color_truck.sh SOURCE_METADATA_RUN NEW_OUT [extra options]}"
shift 2
test -f "$RUN/training.json" || { echo "Missing source metadata: $RUN/training.json" >&2; exit 1; }
command -v python >/dev/null || { echo 'Activate conda maskgs first' >&2; exit 1; }
CONTINUE=()
if [[ "${RESUME:-0}" == 1 ]]; then
  test -f "$OUT/checkpoints/latest/codec.pt" || { echo 'Missing latest checkpoint for resume' >&2; exit 1; }
  CONTINUE=(--resume "$OUT/checkpoints/latest/codec.pt")
fi
echo "INDEPENDENT one-pair experiment; source is metadata only: $RUN"
echo "RANDOM initialization; cumulative 12/24/48 complex symbols; reliable 16bit XYZ; fixed 10 dB"
echo "5000 bootstrap updates (3750 local response + 1250 color), then 1000 replay render updates"
echo "Independent objective-local q1/q2/q3/mixed cycles; LR=1e-4; no latent MSE or allocator"
echo "Numeric validation/checkpoint every 500; initial/final images only; GPU: $CUDA_VISIBLE_DEVICES"
echo "Output: $OUT"
python -u -m gaussian_jscc.integrated_experiment \
  --run "$RUN" --out "$OUT" \
  --bootstrap-steps 5000 --render-steps 1000 --color-every 4 --color-directions 8 \
  --rates 12 24 48 --lr 1e-4 --validate-every 500 --save-every 500 \
  "${CONTINUE[@]}" "$@"
