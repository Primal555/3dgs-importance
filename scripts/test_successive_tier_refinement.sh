#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES}"
RUN="${1:?Usage: test_successive_tier_refinement.sh TEN_DB_SHARED_RUN NEW_OUT [options]}"
OUT="${2:?Usage: test_successive_tier_refinement.sh TEN_DB_SHARED_RUN NEW_OUT [options]}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
DIGITAL="${DIGITAL_PYTHON:-$HOME/miniforge3/envs/gs_digital/bin/python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
[[ -x "$DIGITAL" ]] || { echo "Missing gs_digital interpreter: $DIGITAL" >&2; exit 1; }
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; original 10dB source: $RUN; output: $OUT"
echo 'Sequential q1 -> q2 -> q3; only current channel sender/receiver stage trains; fresh Adam each stage.'
echo 'Defaults: 150 updates per scene per stage; nine scenes: 1350/stage, 4050 total; LR1e-5.'
echo 'Defaults: 32 distinct rotating training views/update; separate scene/stage cycles and exact resume.'
echo 'Defaults: 16 validation views x 4 paired noise trials every 50 updates/scene; fixed stage endpoints.'
echo 'Fixed semantic codec and neighbor refinement; source RGB MSE + latent MSE; immutable y cached once.'
echo 'All tiers and mixed map evaluated at initial/after_q1/after_q2/final; Room heldout.'
echo 'Final Truck q1/q2/q3/mixed actual side delivery: LDPC4096/6144 + 16QAM; four trials.'
echo 'Download OUT/review.tar.gz. Resume: same RUN/OUT/options with RESUME=1.'
exec "$PYTHON" -u -m gaussian_jscc.successive_tier_refinement --run "$RUN" --out "$OUT" \
  --digital-python "$DIGITAL" "${INIT[@]}" "$@"
