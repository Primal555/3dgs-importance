#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
SOURCE="${1:?Usage: test_benefit_independent_response.sh RESPONSE_RUN NEW_OUT [options]}"
OUT="${2:?Set new output}"
shift 2
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "Response labels: $SOURCE/data; independent output: $OUT"
echo 'CPU only: existing exact A/B/S labels; same source backbone, fresh independent prediction heads.'
echo '1000 updates / 64000 record visits; batch64, micro8; zero renders and zero allocation updates.'
echo 'Main value weight1; auxiliary GradNorm sum1; ordinary gradient EMA0.95 limits high-gap numerical gradients.'
echo 'Logs report raw/applied group norms and alpha; rank budget follows group balancing.'
echo 'Download OUT/review.tar.gz. Resume needs source data/replay and OUT/training/latest.pt.'
exec "${PYTHON_BIN:-python}" -u -m gaussian_jscc.benefit_independent_controls \
  --source "$SOURCE" --out "$OUT" --phase train --device cpu "${INIT[@]}" "$@"
