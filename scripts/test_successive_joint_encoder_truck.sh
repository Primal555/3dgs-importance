#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_successive_joint_encoder_truck.sh COMPLETED_SUCCESSIVE_RUN NEW_OUT [options]}"
OUT="${2:?Usage: test_successive_joint_encoder_truck.sh COMPLETED_SUCCESSIVE_RUN NEW_OUT [options]}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
echo "GPU: $CUDA_VISIBLE_DEVICES; full trained source: $RUN; output: $OUT"
echo 'Joint B: train semantic encoder + BOTH channel modules + semantic decoder; XYZ/statistics fixed.'
echo '2000 updates, all four LR=1e-5, NEW Adam; source PLY RGB MSE + 1 * current-y MSE; replay.'
echo 'Current y is recomputed live; only target-side stop-gradient. Log y power, mean/std, NMSE and encoder updates.'
echo 'Keep 8/24/48, reliable16 XYZ, delivered block scales and trained neighbor branch; no allocator.'
exec "$PYTHON" -u -m gaussian_jscc.successive_joint \
  --run "$RUN" --out "$OUT" --train-semantic-encoder \
  --steps 2000 --lr 1e-5 --latent-weight 1 \
  --validate-every 250 --save-every 100 --test-views 32 --render-trials 4 "$@"
