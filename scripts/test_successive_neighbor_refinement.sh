#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the physical GPU}"
RUN="${1:?Usage: test_successive_neighbor_refinement.sh COMPLETED_TIER_REFINE NEW_OUT [options]}"
OUT="${2:?Missing NEW_OUT}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
DIGITAL="${DIGITAL_PYTHON:-$HOME/miniforge3/envs/gs_digital/bin/python}"
command -v "$PYTHON" >/dev/null || { echo "Missing Python: $PYTHON" >&2; exit 1; }
[[ -f "$DIGITAL" ]] || { echo "Missing digital Python: $DIGITAL" >&2; exit 1; }
INIT=()
[[ "${RESUME:-0}" == 1 ]] && INIT+=(--resume)
echo "GPU: $CUDA_VISIBLE_DEVICES; sequential refinement: $RUN; output: $OUT"
echo "Train inherited receiver neighbor module only; freeze semantic codec and all channel stages."
echo "Defaults: Truck + Playroom, 150 updates EACH, 300 total; mixed/q1/mixed/q2/mixed/q3 per-scene cycle."
echo "32 rotating train views/update, 16 validation views, four paired trials; 10dB, LR1e-5."
echo "Fixed final weights; uniform/mixed paired evaluation; frozen-symbol probes; actual Truck side delivery."
echo "Download OUT/review.tar.gz. Resume with RESUME=1 and the same RUN/OUT/options."
exec "$PYTHON" -u -m gaussian_jscc.successive_neighbor_refinement \
  --run "$RUN" --out "$OUT" --digital-python "$DIGITAL" "${INIT[@]}" "$@"
