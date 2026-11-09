#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
SOURCE="${1:?Usage: test_benefit_view_gap.sh FOOTPRINT_RUN NEW_OUT [options]}"
OUT="${2:?Set NEW output}"
shift 2
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "Source footprint data: $SOURCE; output: $OUT"
echo 'View-wise RD gap weight 2; component contrast weight 1; shared footprint and split heads.'
echo 'RankNet KL gain=tau/scale. Per-update ranking gradient <= numerical gradient (all parameters).'
echo 'Same data/initial.pt initialization; default CPU, batch64/micro8, 64000 visits = 1000 updates.'
echo 'Zero renders. Download OUT/review.tar.gz. RESUME=1 continues the same OUT.'
exec "${PYTHON_BIN:-python}" -u -m gaussian_jscc.benefit_three_task_controls \
  --source "$SOURCE" --out "$OUT" --objective view_gap --arm batch64 \
  --samples 64000 --device "${TRAIN_DEVICE:-cpu}" "${INIT[@]}" "$@"
