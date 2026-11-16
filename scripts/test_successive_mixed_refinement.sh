#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the physical GPU}"
RUN="${1:?Usage: test_successive_mixed_refinement.sh COMPLETED_TIER_REFINE NEW_OUT [options]}"
OUT="${2:?Missing NEW_OUT}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
DIGITAL="${DIGITAL_PYTHON:-$HOME/miniforge3/envs/gs_digital/bin/python}"
command -v "$PYTHON" >/dev/null || { echo "Missing Python: $PYTHON" >&2; exit 1; }
[[ -f "$DIGITAL" ]] || { echo "Missing digital Python: $DIGITAL" >&2; exit 1; }
INIT=()
[[ "${RESUME:-0}" == 1 ]] && INIT+=(--resume)
echo "GPU: $CUDA_VISIBLE_DEVICES; sequential refinement: $RUN; output: $OUT"
echo "Mixed q1/q2/q3 only; all channel stages train jointly; semantic codec and neighbor repair frozen."
echo "Defaults: 150 updates/scene, 32 rotating train views; 16 validation views x 4 trials; 10dB, LR1e-5."
echo "Initial/final uniform q1/q2/q3 and mixed evaluation; actual side delivery. Download OUT/review.tar.gz."
exec "$PYTHON" -u -m gaussian_jscc.successive_mixed_refinement \
  --run "$RUN" --out "$OUT" --digital-python "$DIGITAL" "${INIT[@]}" "$@"
