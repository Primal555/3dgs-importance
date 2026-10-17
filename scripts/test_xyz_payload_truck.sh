#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_xyz_payload_truck.sh SOURCE_RUN NEW_OUT [extra options]}"
OUT="${2:?Usage: test_xyz_payload_truck.sh SOURCE_RUN NEW_OUT [extra options]}"
shift 2
test -f "$RUN/training.json" || { printf 'Missing source training.json: %s\n' "$RUN" >&2; exit 1; }
command -v python >/dev/null || { printf 'Activate the maskgs environment first.\n' >&2; exit 1; }
printf 'Source metadata only: %s\nOutput: %s\nGPU: %s\n' "$RUN" "$OUT" "$CUDA_VISIBLE_DEVICES"
printf 'XYZ is inside the 8/24/48-symbol payload; NO reliable coordinate stream.\n'
python -u -m gaussian_jscc.xyz_payload_experiment --run "$RUN" --out "$OUT" --lpips "$@"
