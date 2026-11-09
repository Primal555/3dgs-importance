#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
SOURCE="${1:?Usage: test_benefit_rank_gradnorm.sh FOOTPRINT_RUN NEW_OUT [options]}"
OUT="${2:?Set NEW output}"
shift 2
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "Source candidate dataset: $SOURCE; output: $OUT"
echo 'One physical magnitude loss, continuous RD RankNet, GradNorm ordinary/tail task updates.'
echo 'Initial references use the complete training set; frozen common network and codecs.'
echo 'Default CPU, 2000 optimizer steps; existing reference predictions reused. Download OUT/review.tar.gz.'
exec "${PYTHON_BIN:-python}" -u -m gaussian_jscc.benefit_rank_gradnorm_controls \
  --source "$SOURCE" --out "$OUT" --device "${TRAIN_DEVICE:-cpu}" "${INIT[@]}" "$@"
