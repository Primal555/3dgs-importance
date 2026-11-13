#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:?Usage: test_teacher_calibration32.sh CODEC_RUN REFERENCE_TEACHER NEW_OUT [options]}"
REFERENCE="${2:?Set completed digital-file teacher run}"
OUT="${3:?Set NEW output directory}"
shift 3
echo 'Teacher: fresh probability tables; 64 training views/update, 32 fixed calibration views.'
echo '2000 updates maximum; beta0.5, positive cap2, keep cap0.25; calibration every100.'
echo "Fixed objective scales from: $REFERENCE/training.json"
echo '32 original test views, 4 paired noise trials; actual side-file accounting. Download OUT/review.tar.gz.'
exec bash "$PROJECT/scripts/test_fast_first_order_allocation.sh" "$RUN" "${SCENE:-truck}" "$OUT" \
  --allocation-steps 2000 --candidate-batch 128 "$@" \
  --beta 0.5 --window-views 64 --calibration-views 32 --validate-every 100 \
  --tier-max-log-odds 2.0 --max-log-odds 0.25 --cost-protocol digital_side_file_v1 \
  --objective-reference-run "$REFERENCE"
