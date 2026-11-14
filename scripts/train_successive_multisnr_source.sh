#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES}"
RUN="${1:?Usage: train_successive_multisnr_source.sh TEN_DB_SHARED_RUN NEW_OUT [options]}"
OUT="${2:?Usage: train_successive_multisnr_source.sh TEN_DB_SHARED_RUN NEW_OUT [options]}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
DIGITAL="${DIGITAL_PYTHON:-$HOME/miniforge3/envs/gs_digital/bin/python}"
[[ -x "$DIGITAL" ]] || { echo "Missing gs_digital interpreter: $DIGITAL" >&2; exit 1; }
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; 10dB source: $RUN; output: $OUT"
echo 'Per scene: 1000 channel updates + 250 joint updates; LR1e-5, new Adam at transition.'
echo 'Source PLY supervision for all three delivered prefixes; channel-phase full rendering every4 visits.'
echo 'Frozen 10dB mapping anchor every4 visits; semantic encoder/statistics stay fixed.'
echo 'Side files below10dB: LDPC4096/8192 + QPSK; at/above10dB: LDPC4096/6144 + 16QAM. No ARQ.'
echo 'Fixed validation curves and initial/final test curves in figures/. Review: OUT/review.tar.gz.'
exec "$PYTHON" -u -m gaussian_jscc.multisnr_adaptation --run "$RUN" --out "$OUT" \
  --digital-python "$DIGITAL" "${INIT[@]}" "$@"
