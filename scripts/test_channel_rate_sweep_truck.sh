#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_channel_rate_sweep_truck.sh MODULAR_SOURCE_RUN NEW_OUT [options]}"
OUT="${2:?Usage: test_channel_rate_sweep_truck.sh MODULAR_SOURCE_RUN NEW_OUT [options]}"
shift 2
[[ -f "$RUN/training.json" ]] || { printf 'Missing modular metadata: %s/training.json\n' "$RUN" >&2; exit 1; }
command -v python >/dev/null || { printf 'Activate the maskgs environment first.\n' >&2; exit 1; }
printf 'Frozen semantic teacher: %s\nOutput: %s\nGPU: %s\n' "$RUN" "$OUT" "$CUDA_VISIBLE_DEVICES"
exec python -u -m gaussian_jscc.channel_rate_sweep --run "$RUN" --out "$OUT" "$@"
