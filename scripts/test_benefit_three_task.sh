#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
SOURCE="${1:?Usage: test_benefit_three_task.sh FOOTPRINT_RUN NEW_OUT [options]}"
OUT="${2:?Set NEW output}"
shift 2
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "Source footprint data: $SOURCE; output: $OUT"
echo 'Common regression / relative regression / positive-only RankNet: three-task GradNorm.'
echo 'Shared footprint and two heads; initialize both arms from data/initial.pt.'
echo 'Default: CPU; each arm 16000 record visits; batch8=2000 updates, batch64=250 updates.'
echo 'Batch64 uses proportional strata and importance correction, microbatch8 gradient accumulation.'
echo 'Zero renders. Download OUT/review.tar.gz. RESUME=1 continues the same OUT.'
exec "${PYTHON_BIN:-python}" -u -m gaussian_jscc.benefit_three_task_controls \
  --source "$SOURCE" --out "$OUT" --device "${TRAIN_DEVICE:-cpu}" "${INIT[@]}" "$@"
