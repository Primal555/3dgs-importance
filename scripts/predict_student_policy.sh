#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
CHECKPOINT="${1:?Usage: predict_student_policy.sh STUDENT_PT CODEC_RUN SCENE NEW_OUT [options]}"
RUN="${2:?Set codec run}"
SCENE="${3:?Set scene name}"
OUT="${4:?Set NEW output}"
shift 4
if [[ ! -f "$RUN/training.json" && -f "$RUN/training/training.json" ]]; then RUN="$RUN/training"; fi
PYTHON="${PYTHON_BIN:-python}"
echo "GPU: $CUDA_VISIBLE_DEVICES; student: $CHECKPOINT; scene: $SCENE; output: $OUT"
echo 'Source preprocessing and one deterministic student forward; zero optimizer updates.'
exec "$PYTHON" -u -m gaussian_jscc.student_policy_experiment --phase predict \
  --student-checkpoint "$CHECKPOINT" --run "$RUN" --scene "$SCENE" --out "$OUT" \
  --beta "${BETA:-0.5}" "$@"
