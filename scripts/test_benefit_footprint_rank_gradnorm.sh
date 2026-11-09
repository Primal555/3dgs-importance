#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
SOURCE="${1:?Usage: test_benefit_footprint_rank_gradnorm.sh FOOTPRINT_RUN NEW_OUT [options]}"
OUT="${2:?Set NEW output}"
shift 2
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "Source footprint data: $SOURCE; output: $OUT"
echo 'Shared footprint CNN/point encoder and common/relative heads all trainable.'
echo 'Initialize from data/initial.pt with original construction order; original uniform sampling.'
echo 'Original numeric supervision + six-pair RD RankNet; two-task GradNorm.'
echo 'Default CPU, 2000 updates, no rendering. Download OUT/review.tar.gz.'
exec "${PYTHON_BIN:-python}" -u -m gaussian_jscc.benefit_footprint_rank_gradnorm_controls \
  --source "$SOURCE" --out "$OUT" --device "${TRAIN_DEVICE:-cpu}" "${INIT[@]}" "$@"
