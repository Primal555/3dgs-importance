#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
RUN="${1:?Usage: test_teacher_views64_shuffle.sh CODEC_RUN NEW_OUT [options]}"
OUT="${2:?Set NEW output directory}"
shift 2
echo 'Teacher: 64 training views/update, beta0.5, positive cap2, keep cap0.25, up to2000 steps.'
echo 'Cost: actual side-file byte accounting, LDPC4096/6144 + 16QAM, including C_ref and selection.'
echo 'After training: freeze selected map and codec; one positive-tier permutation, q0 mask fixed.'
bash "$PROJECT/scripts/test_fast_first_order_allocation.sh" "$RUN" "${SCENE:-truck}" "$OUT" \
  --allocation-steps 2000 --candidate-batch 128 --beta 0.5 "$@" \
  --window-views 64 --tier-max-log-odds 2.0 --cost-protocol digital_side_file_v1
"${PYTHON_BIN:-python}" -u -m gaussian_jscc.first_order_positive_shuffle \
  --allocation "$OUT" --out "$OUT/positive_shuffle" --shuffle-seed "${SHUFFLE_SEED:-20261112}"
