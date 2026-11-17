#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the physical GPU}"
RUN="${1:?Usage: train_successive_multisnr_adapter.sh SHARED_10DB_RUN NEW_OUT [options]}"
OUT="${2:?Missing NEW_OUT}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
DIGITAL="${DIGITAL_PYTHON:-$HOME/miniforge3/envs/gs_digital/bin/python}"
command -v "$PYTHON" >/dev/null || { echo "Missing Python: $PYTHON" >&2; exit 1; }
[[ -f "$DIGITAL" ]] || { echo "Missing digital Python: $DIGITAL" >&2; exit 1; }
INIT=()
[[ "${RESUME:-0}" == 1 ]] && INIT+=(--resume)
echo "GPU: $CUDA_VISIBLE_DEVICES; fixed 10dB source: $RUN; output: $OUT"
echo "Zero-initialized anchored SNR adapters only; original codec, neighbor weights and attributes frozen."
echo "Truck/Playroom: 100 updates EACH, 200 total; THREE SNR conditions/update, 600 full-scene recoveries."
echo "32 rotating train views + 32 separate calibration views; Room initial/final evaluation only."
echo "Per-scene noiseless q1/q2/q3 rotation; LR1e-5; fixed final checkpoint; original test cameras."
echo "Low-SNR side files use QPSK LDPC4096/8192; >=10dB uses 16QAM LDPC4096/6144; no ARQ."
echo "Actual Truck q2 delivery at five SNRs x four trials. Download OUT/review.tar.gz."
echo "Resume: RESUME=1 with the same source, OUT and options."
exec "$PYTHON" -u -m gaussian_jscc.successive_multisnr_adapter \
  --run "$RUN" --out "$OUT" --digital-python "$DIGITAL" "${INIT[@]}" "$@"
