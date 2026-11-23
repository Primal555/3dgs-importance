#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES}"
SOURCE="${1:?Usage: test_q1_content_gate.sh LAYER_01_PT NEW_OUT [options]}"
OUT="${2:?Usage: test_q1_content_gate.sh LAYER_01_PT NEW_OUT [options]}"
shift 2
RESUME_ARGS=()
if [[ "${RESUME:-0}" == 1 ]]; then RESUME_ARGS=(--resume); fi
echo "GPU $CUDA_VISIBLE_DEVICES | q1 checkpoint $SOURCE | output $OUT"
echo 'Paired baseline and transmitter gate at 8 complex symbols per Gaussian. Truck and Playroom, 4/10/16 dB AWGN.'
echo 'Defaults per arm: 1000 attribute updates, 500 render updates, 32 rotating train views.'
echo 'Both phases retain grouped attribute supervision. Same final weights for content gate and mean-gate intervention.'
echo 'Final baseline/gate delivery verifies three digital side files. Download OUT/review.tar.gz. RESUME=1 reuses OUT.'
exec "${PYTHON_BIN:-python}" -u -m gaussian_jscc.q1_content_gate_experiment \
  --source "$SOURCE" --out "$OUT" \
  --digital-python "${DIGITAL_PYTHON:-$HOME/miniforge3/envs/gs_digital/bin/python}" \
  "${RESUME_ARGS[@]}" "$@"
