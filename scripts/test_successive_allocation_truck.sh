#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_successive_allocation_truck.sh COMPLETED_JOINT_A_RUN NEW_OUT [options]}"
OUT="${2:?Usage: test_successive_allocation_truck.sh COMPLETED_JOINT_A_RUN NEW_OUT [options]}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
echo "GPU: $CUDA_VISIBLE_DEVICES; Joint A source: $RUN; output: $OUT"
echo '2000 allocator-only updates: freeze ALL codec modules; new hierarchical keep/tier table.'
echo '1000 joint updates: semantic encoder FROZEN; train table + channel E/D + semantic D.'
echo 'beta=0.005; 8/24/48; 10dB; reliable16 XYZ + delivered binary16 scale frames.'
echo 'Source-PLY RGB MSE; joint channel latent anchor=1; independent uniform-prefix anchor every 4 updates.'
exec "$PYTHON" -u -m gaussian_jscc.successive_allocation \
  --run "$RUN" --out "$OUT" --allocation-steps 2000 --joint-steps 1000 \
  --beta 0.005 --codec-lr 1e-5 --channel-latent-weight 1 \
  --keep-lr 0.01 --mask-lr 0.001 --mask-adam-eps 1e-15 --snr 10 \
  --prefix-anchor-every 4 --validate-every 250 --save-every 100 \
  --test-views 32 --trials 4 --image-views 2 --lpips "$@"
