#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Select an available GPU}"
SOURCE_RUN="${1:?Usage: script CENTER_RUN [NEW_OUTPUT]}"
OUT="${2:-$PROJECT/output/truck_attributes_joint_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
[[ -f "$SOURCE_RUN/training_state.pt" ]] || { echo 'Missing source training_state.pt' >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo 'Output must be a NEW directory' >&2; exit 1; }
exec "$PYTHON_BIN" -u -m gaussian_jscc train-center-attributes \
  --after-center-run "$SOURCE_RUN" --out "$OUT" --device cuda \
  --attribute-steps "${ATTRIBUTE_STEPS:-5000}" --joint-steps "${JOINT_STEPS:-1000}" \
  --attribute-lr "${ATTRIBUTE_LR:-2e-4}" \
  --joint-center-lr "${JOINT_CENTER_LR:-1e-5}" \
  --joint-attribute-lr "${JOINT_ATTRIBUTE_LR:-1e-4}" \
  --later-phase-policy budget --render-backward replay \
  --render-blocks-per-batch "${RENDER_BLOCKS:-64}" \
  --validate-every 500 --render-every 500 --save-every 500 --profile-every 50
