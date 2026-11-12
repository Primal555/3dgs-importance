#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES}"
SOURCE="${1:?Usage: test_digital_side_delivery.sh COMPLETED_ALLOCATION NEW_OUT [options]}"
OUT="${2:?Set NEW_OUT}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
DIGITAL_PYTHON="${DIGITAL_PYTHON:-$HOME/miniforge3/envs/gs_digital/bin/python}"
[[ -x "$DIGITAL_PYTHON" ]] || { echo "Missing digital interpreter: $DIGITAL_PYTHON" >&2; exit 1; }
echo "GPU: $CUDA_VISIBLE_DEVICES; allocation: $SOURCE; output: $OUT"
echo 'Actual XYZ/metadata/scales delivery: LDPC4096/6144 + 16QAM + AWGN; no ARQ.'
echo 'Default: selected and uniform_q2; inherit original SNR, 32 test cameras and 4 trials.'
echo 'Report success rate and quality conditional on successful delivery. Download OUT/review.tar.gz.'
exec "$PYTHON" -u -m gaussian_jscc.digital_side_delivery --allocation "$SOURCE" --out "$OUT" \
  --digital-python "$DIGITAL_PYTHON" "$@"
