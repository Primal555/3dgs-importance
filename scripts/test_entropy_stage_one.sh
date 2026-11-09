#!/usr/bin/env bash
set -euo pipefail
RUN=${1:?Usage: test_entropy_stage_one.sh CODEC_RUN SCENE NEW_OUT [options]}
SCENE=${2:?Usage: test_entropy_stage_one.sh CODEC_RUN SCENE NEW_OUT [options]}
OUT=${3:?Usage: test_entropy_stage_one.sh CODEC_RUN SCENE NEW_OUT [options]}
shift 3
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ ! -f "$RUN/training.json" && -f "$RUN/training/training.json" ]]; then RUN="$RUN/training"; fi
EXTRA=()
if [[ "${RESUME:-0}" == 1 ]]; then EXTRA+=(--resume); fi
printf 'Frozen codec: %s; scene: %s; output: %s\n' "$RUN" "$SCENE" "$OUT"
printf '%s\n' 'Grouped entropy: 16/16/16; fixed input keep set; actual 8/24/48 complex-symbol tiers.'
printf '%s\n' '20 full density epochs; 9 eta candidates calibrated by real mixed-map RD; final heldout q1/q2/q3 comparisons.'
printf '%s\n' 'Download OUT/review.tar.gz. Resume also needs cache, checkpoints and the frozen source codec.'
exec "${PYTHON_BIN:-python}" -u -m gaussian_jscc.entropy_stage_one \
    --run "$RUN" --scene "$SCENE" --out "$OUT" "${EXTRA[@]}" "$@"
