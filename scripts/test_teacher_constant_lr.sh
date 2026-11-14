#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:?Usage: test_teacher_constant_lr.sh CODEC_RUN REFERENCE_TEACHER NEW_OUT [options]}"
REFERENCE="${2:?Set completed digital-file teacher run}"
OUT="${3:?Set NEW output directory}"
shift 3
echo 'Constant-LR teacher: keep LR0.1, positive-tier LR0.4 for all 2000 allocation updates.'
echo '32 training views/update and 32 calibration views; select by calibration RD and also report the final step.'
exec bash "$PROJECT/scripts/test_teacher_calibration32.sh" "$RUN" "$REFERENCE" "$OUT" "$@" \
  --allocation-steps 2000 --keep-lr 0.1 --tier-lr 0.4 --no-early-stop
