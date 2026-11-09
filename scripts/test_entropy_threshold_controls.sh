#!/usr/bin/env bash
set -euo pipefail
SOURCE=${1:?Usage: test_entropy_threshold_controls.sh ENTROPY_RUN NEW_OUT [options]}
OUT=${2:?Usage: test_entropy_threshold_controls.sh ENTROPY_RUN NEW_OUT [options]}
shift 2
cd "$(dirname "${BASH_SOURCE[0]}")/.."
EXTRA=()
if [[ "${RESUME:-0}" == 1 ]]; then EXTRA+=(--resume); fi
printf 'Entropy source: %s; output: %s\n' "$SOURCE" "$OUT"
printf '%s\n' 'Frozen codec and information scores; zero optimizer updates; independent low/high thresholds.'
printf '%s\n' 'Beta 0.1/0.5 share actual mixed-map measurements; calibrated q2-budget candidate also evaluated.'
printf '%s\n' 'Same source cameras, noise trials, RD reference and complete side-information cost.'
printf '%s\n' 'Source needs cache/information_scores.pt and evaluation reports. Download OUT/review.tar.gz.'
exec "${PYTHON_BIN:-python}" -u -m gaussian_jscc.entropy_threshold_controls \
    --source "$SOURCE" --out "$OUT" "${EXTRA[@]}" "$@"
