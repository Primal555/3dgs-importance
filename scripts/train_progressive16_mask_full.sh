#!/usr/bin/env bash
# Three positive prefixes + learned drop; source-specific categorical table.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="${1:-$PROJECT/output/truck_progressive16_mask_full_$(date +%Y%m%d_%H%M%S)}"
PLY="${PLY:-$PROJECT/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply}"
SCENE="${SCENE:-$PROJECT/data/tandt_db/tandt/truck}"
[[ -f "$PLY" ]] || { echo "Missing PLY: $PLY" >&2; exit 1; }
[[ -d "$SCENE/sparse" || -f "$SCENE/transforms_train.json" ]] || { echo "Missing scene: $SCENE" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Output exists: $OUT" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN.' >&2; exit 1; }
EXTRA=()
case "${INITIALIZATION:-random}" in
  random) [[ -z "${INIT:-}" ]] || echo 'Random initialization: ignoring inherited INIT.' ;;
  checkpoint)
    [[ -n "${INIT:-}" && -f "$INIT" ]] || { echo 'checkpoint mode requires an existing INIT file.' >&2; exit 1; }
    EXTRA+=(--init "$INIT") ;;
  *) echo 'INITIALIZATION must be random or checkpoint.' >&2; exit 1 ;;
esac
echo 'q0=drop, q1/q2/q3=8/16/32 complex symbols. Reliable 16-bit XYZ for retained points only.'
echo '5000 attribute + 5000 render + 3000 allocation/joint updates by default.'
echo 'First 500 joint updates freeze codec; learn per-point categorical table, NOT a new predictor network.'
echo 'BETA is an engineering tradeoff, NOT a bandwidth guarantee. Existence prior is initialization only.'
printf 'GPU: %s\nOutput: %s\n' "$CUDA_VISIBLE_DEVICES" "$OUT"
"$PYTHON_BIN" -u -m gaussian_jscc train-learned \
  --ply "$PLY" --source "$SCENE" --out "$OUT" "${EXTRA[@]}" \
  --device cuda --snr 10 --channel awgn --prefix-mode progressive \
  --position-delivery quantized --position-bits 16 --position-compression delta_zlib \
  --position-net-bits-per-use 2 --rates 0 8 16 32 \
  --bootstrap-steps "${BOOTSTRAP_STEPS:-5000}" --bootstrap-objective local-response --local-response-views 4 \
  --render-steps "${RENDER_STEPS:-5000}" --joint-steps "${JOINT_STEPS:-3000}" \
  --mask-only-steps "${MASK_ONLY_STEPS:-500}" --mask-samples "${MASK_SAMPLES:-2}" \
  --existence-prior "${EXISTENCE_PRIOR:-auto}" --mask-lr "${MASK_LR:-0.001}" --beta "${BETA:-0.01}" \
  --lr "${LR:-0.0001}" --render-lr "${RENDER_LR:-0.0001}" --drop 0.05 --clip-mode none \
  --block-size 256 --decoder-window 32 --blocks-per-batch "${BLOCKS_PER_BATCH:-64}" \
  --render-backward replay --training-data-device cpu --resolution 2 \
  --train-views 0 --views-per-step 2 --validate-every "${VALIDATE_EVERY:-500}" \
  --validation-views 4 --validation-trials 2 --patience 0 --save-every "${SAVE_EVERY:-500}" --seed 42

# Export the matched BEST joint codec/table, not last table with best codec.
"$PYTHON_BIN" -u -m gaussian_jscc export-route2 \
  --ply "$PLY" --checkpoint "$OUT/codec_best_joint.pt" --allocation "$OUT/route2_best_joint.pt" \
  --out "$OUT/deployment_best" --snr 10 --device cuda
if [[ "${FINAL_EVAL:-1}" == 1 ]]; then
  "$PYTHON_BIN" -u -m gaussian_jscc evaluate \
    --ply "$PLY" --checkpoint "$OUT/codec_best_joint.pt" --allocation "$OUT/route2_best_joint.pt" \
    --source "$SCENE" --out "$OUT/test_best_mask" --device cuda --snrs 10 --channel awgn \
    --trials "${TEST_TRIALS:-2}" --resolution 2 --save-images \
    --metadata-code-rate 1 --metadata-modulation-bits 2
fi
echo "Completed. Hard deployment counts: $OUT/deployment_best/allocation.json"
