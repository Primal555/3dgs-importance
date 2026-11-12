#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:?Usage: train_student_calibration32.sh CODEC_RUN VIEWS32_CAP2_TEACHER NEW_OUT [options]}"
TEACHER="${2:?Set completed views32 cap2 teacher}"
OUT="${3:?Set NEW output directory}"
shift 3
echo 'Student calibration: 32 fixed cameras excluded from RD updates and 64-view feature preparation.'
echo 'Calibration every100 updates; original test views and RD references retained.'
exec bash "$PROJECT/scripts/train_student_views32.sh" "$RUN" "$TEACHER" "$OUT" "$@" \
  --calibration-views 32
