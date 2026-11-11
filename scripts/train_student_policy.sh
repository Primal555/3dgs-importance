#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: train_student_policy.sh CODEC_RUN COMPLETED_TEACHER NEW_OUT [options]}"
TEACHER="${2:?Set completed first-order teacher}"
OUT="${3:?Set NEW output directory}"
shift 3
if [[ ! -f "$RUN/training.json" && -f "$RUN/training/training.json" ]]; then RUN="$RUN/training"; fi
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; codec: $RUN; teacher: $TEACHER; output: $OUT"
echo 'Student: source attributes/latent/visibility, two-scale spatial context, independent keep/tier heads.'
echo 'Defaults: Truck, beta0.5, 1000 full distillation passes + 1000 RD updates; frozen codec.'
echo 'Distillation: calibration RD every100 updates; keep/action/positive-tier and q0-q3 accuracies.'
echo 'Preparation: 8 independent training cameras, original-row caches; hard policy in training and deployment.'
echo 'RD: dense keep, 2 uniform positive points/block, inverse-inclusion weighting, 32 paired corrections.'
echo 'Download OUT/review.tar.gz. Resume uses same OUT/settings and RESUME=1.'
exec "$PYTHON" -u -m gaussian_jscc.student_policy_experiment \
  --run "$RUN" --teacher-run "$TEACHER" --scene "${SCENE:-truck}" --out "$OUT" \
  --beta "${BETA:-0.5}" "${INIT[@]}" "$@"
