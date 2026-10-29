#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_pixel_benefit_allocation.sh COMPLETED_CODEC_RUN SCENE NEW_OUT --beta BETA [options]}"
SCENE="${2:?Set a trained scene}"
OUT="${3:?Set a new output directory}"
shift 3
if [[ ! -f "$RUN/training.json" && -f "$RUN/training/training.json" ]]; then RUN="$RUN/training"; fi
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate environment or set PYTHON_BIN' >&2; exit 1; }
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; frozen codec: $RUN; scene: $SCENE; output: $OUT"
echo 'CUDA consistency must pass on the server before training; CUDA has not been validated locally.'
echo '200 pretraining windows, first 50 for scale fitting; 64 groups and 16 network updates per window.'
echo 'Replay retains up to 8192 indexed training groups; unreferenced temporary training shards are reclaimed.'
echo 'Keep the entire replay directory with checkpoints for exact resume.'
echo 'Resume allocation with RESUME=1 and the same OUT, adding --phase allocation; default 2000 steps.'
exec "$PYTHON" -u -m gaussian_jscc.pixel_benefit_allocation \
  --run "$RUN" --scene "$SCENE" --out "$OUT" "${INIT[@]}" "$@"
