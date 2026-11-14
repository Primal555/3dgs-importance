#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES}"
RUN="${1:?Usage: test_successive_q3_refinement.sh TEN_DB_SHARED_RUN NEW_OUT [options]}"
OUT="${2:?Usage: test_successive_q3_refinement.sh TEN_DB_SHARED_RUN NEW_OUT [options]}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
DIGITAL="${DIGITAL_PYTHON:-$HOME/miniforge3/envs/gs_digital/bin/python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
[[ -x "$DIGITAL" ]] || { echo "Missing gs_digital interpreter: $DIGITAL" >&2; exit 1; }
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; 10dB source: $RUN; output: $OUT"
echo 'Train only q3 sender/receiver stages; keep semantic codec, q1/q2 and neighbor refinement fixed.'
echo 'Defaults: 250 updates per original training scene; 2250 total for the nine-scene source; LR1e-5.'
echo 'Source RGB MSE + latent MSE; cache immutable semantic features; fresh 10dB AWGN.'
echo 'Initial/final paired q2/q3 test evaluation; Room heldout; fixed final weights.'
echo 'Final Truck q2/q3 side-file delivery: LDPC4096/6144 + 16QAM; four trials, no ARQ.'
echo 'Download OUT/review.tar.gz. Resume: same RUN/OUT/options with RESUME=1.'
exec "$PYTHON" -u -m gaussian_jscc.successive_q3_refinement --run "$RUN" --out "$OUT" \
  --digital-python "$DIGITAL" "${INIT[@]}" "$@"
