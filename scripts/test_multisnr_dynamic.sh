#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES}"
RUN="${1:?Usage: test_multisnr_dynamic.sh COMPLETED_MULTISNR_RUN NEW_OUT [options]}"
OUT="${2:?Usage: test_multisnr_dynamic.sh COMPLETED_MULTISNR_RUN NEW_OUT [options]}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
DIGITAL="${DIGITAL_PYTHON:-$HOME/miniforge3/envs/gs_digital/bin/python}"
[[ -x "$DIGITAL" ]] || { echo "Missing gs_digital interpreter: $DIGITAL" >&2; exit 1; }
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; multi-SNR source: $RUN; dynamic output: $OUT"
echo 'Frozen initial10dB versus final multi-SNR; same CSI/MCS, paired AWGN, original test cameras.'
echo 'Default Truck q2; 12 changing-SNR deliveries per trial, inherited trial count. Zero optimizer updates.'
echo 'Actual LDPC/QAM for three side files; failures charged, missing quality shown as gaps; no ARQ.'
echo 'Figures: OUT/figures. Download OUT/review.tar.gz.'
exec "$PYTHON" -u -m gaussian_jscc.multisnr_dynamic --run "$RUN" --out "$OUT" \
  --digital-python "$DIGITAL" "${INIT[@]}" "$@"
