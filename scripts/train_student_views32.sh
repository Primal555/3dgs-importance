#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:?Usage: train_student_views32.sh CODEC_RUN VIEWS32_CAP2_TEACHER NEW_OUT [options]}"
TEACHER="${2:?Set completed views32 cap2 teacher}"
OUT="${3:?Set NEW output directory}"
shift 3
echo 'Teacher: 32 views/update, cap2; verify completed source and selected soft targets.'
echo 'Student: random initialization, 64 preparation views, 32 RD views/update.'
echo '1000 distillation passes + 1000 RD updates; beta0.5; constant teacher weight0.1.'
echo 'Cost: attribute symbols + separately framed XYZ/metadata/scales, LDPC4096/6144 + 16QAM.'
exec bash "$PROJECT/scripts/train_student_soft_policy.sh" "$RUN" "$TEACHER" "$OUT" \
  --distill-epochs 1000 --rd-steps 1000 --beta 0.5 --teacher-weight 0.1 "$@" \
  --required-teacher-window-views 32 --preparation-views 64 --window-views 32 \
  --cost-protocol digital_side_file_v1
