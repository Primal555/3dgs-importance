#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:?Usage: train_student_soft_policy.sh CODEC_RUN CAP1_TEACHER NEW_OUT [options]}"
TEACHER="${2:?Set completed cap1 first-order teacher}"
OUT="${3:?Set NEW output directory}"
shift 3
echo 'Soft deployment distillation: exact ten-draw teacher probabilities, required positive cap1.0.'
echo 'Random student initialization; 1000 distillation passes + 1000 RD updates by default.'
echo 'Teacher probabilities must reproduce the selected map in original PLY order.'
echo 'RD teacher constraint: constant weight (default0.1) throughout RD training.'
exec bash "$PROJECT/scripts/train_student_policy.sh" "$RUN" "$TEACHER" "$OUT" "$@" \
  --distill-target soft_deployment --required-teacher-tier-cap 1.0 --teacher-weight-schedule constant
