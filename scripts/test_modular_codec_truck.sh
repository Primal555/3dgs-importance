#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_modular_codec_truck.sh SOURCE_RUN NEW_OUT [extra options]}"
OUT="${2:?Usage: test_modular_codec_truck.sh SOURCE_RUN NEW_OUT [extra options]}"
shift 2
test -f "$RUN/training.json" || { echo "Missing source metadata: $RUN/training.json" >&2; exit 1; }
command -v python >/dev/null || { echo 'Activate conda maskgs first' >&2; exit 1; }
python -u -m gaussian_jscc.modular_experiment --run "$RUN" --out "$OUT" --lpips "$@"
