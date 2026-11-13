#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES}"
RUN="${1:?Usage: train_successive_multisnr.sh COMPLETED_SHARED_RUN NEW_OUT [options]}"
OUT="${2:?Usage: train_successive_multisnr.sh COMPLETED_SHARED_RUN NEW_OUT [options]}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
DIGITAL="${DIGITAL_PYTHON:-$HOME/miniforge3/envs/gs_digital/bin/python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
[[ -x "$DIGITAL" ]] || { echo "Missing gs_digital interpreter: $DIGITAL" >&2; exit 1; }
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; source: $RUN; output: $OUT"
echo 'Shared codec, continuous 5..15 dB; fixed semantic encoder and attribute statistics.'
echo 'Defaults per training scene: 1000 channel updates + 250 joint decoder updates; NEW Adam at phase transition.'
echo 'Original source scenes/splits; room zero-shot evaluation; rates8/24/48 and fixed mixed layouts.'
echo 'Initial/final paired test matrix; channel-end calibration matrix; fixed final weights.'
echo 'Final actual side-file delivery: Truck/q2, five SNRs, four trials; LDPC4096/6144 + 16QAM, no ARQ.'
echo 'Review: OUT/review.tar.gz. Resume: same RUN/OUT/options with RESUME=1.'
exec "$PYTHON" -u -m gaussian_jscc.successive_multisnr --run "$RUN" --out "$OUT" \
  --digital-python "$DIGITAL" "${INIT[@]}" "$@"
