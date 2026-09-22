#!/usr/bin/env bash
# Aggressive anisotropic position supervision. Architecture stays unchanged.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${INIT:-}" ]] || (( ${ADAPTER_STEPS:-0} != 0 || ${JOINT_STEPS:-0} != 0 )); then
  echo 'This experiment starts randomly and trains representation only. Unset INIT and disable adapter/joint steps.' >&2
  exit 1
fi
export POSITION_OBJECTIVE=teacher-axis
export ADAPTER_STEPS=0 JOINT_STEPS=0
export AXIS_FLOOR_PERCENTILE="${AXIS_FLOOR_PERCENTILE:-1}"
OUT="${1:-$PROJECT/output/truck_teacher_axis_$(date +%Y%m%d_%H%M%S)}"
echo 'Teacher-axis pseudo-Huber REPLACES coarse/fine XYZ terms. No gradient clipping; constant LR by default.'
exec bash "$PROJECT/scripts/test_representation_codec.sh" "$OUT"
