#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
SOURCE="${1:?Usage: test_benefit_response_terms.sh FOOTPRINT_RUN NEW_OUT [options]}"
OUT="${2:?Set new output}"
shift 2
COLLECT_DEVICE="${COLLECT_DEVICE:-cuda}"
if [[ "$COLLECT_DEVICE" == cuda* ]]; then
  : "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to one available GPU for collection}"
fi
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "Source footprint: $SOURCE; output: $OUT; collection GPU: ${CUDA_VISIBLE_DEVICES:-CPU}"
echo 'Collect exact cross / energy / structure labels from the same frozen four-state renders.'
echo 'Default 4096 training + 512 whole-block-held-out validation records; no allocation training.'
echo 'Keep footprint inputs and common/relative value heads; add a nonnegative energy head.'
echo 'Fixed primary numerical weight1; GradNorm only balances the three auxiliary terms.'
echo 'CPU training: batch64/micro8, 64000 record visits = 1000 optimizer updates.'
echo 'Download OUT/review.tar.gz. Resume requires OUT/data and OUT/training/latest.pt.'
exec "${PYTHON_BIN:-python}" -u -m gaussian_jscc.benefit_response_controls \
  --source "$SOURCE" --out "$OUT" --collect-device "$COLLECT_DEVICE" \
  --device "${TRAIN_DEVICE:-cpu}" "${INIT[@]}" "$@"
