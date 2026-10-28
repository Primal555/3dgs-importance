#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
RUN="${1:?Usage: test_learned_benefit_allocation.sh COMPLETED_CODEC_RUN SCENE NEW_OUT --beta BETA [options]}"
SCENE="${2:?Set a trained scene, e.g. truck}"
OUT="${3:?Set an output directory}"
shift 3
if [[ ! -f "$RUN/training.json" && -f "$RUN/training/training.json" ]]; then RUN="$RUN/training"; fi
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
HAVE_BETA=0
for ARG in "$@"; do
  case "$ARG" in
    --beta|--beta=*) HAVE_BETA=1 ;;
    --budget-total-uses|--budget-total-uses=*|--budget-ratio-to-q3|--budget-ratio-to-q3=*)
      echo 'Supply fixed --beta for learned benefit allocation' >&2; exit 2 ;;
  esac
done
if [[ "$HAVE_BETA" != 1 ]]; then echo 'Supply --beta BETA' >&2; exit 2; fi
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; frozen codec: $RUN; scene: $SCENE; output: $OUT"
echo 'Default phase: 200 pretraining windows, fixed probabilities; first 50 fit normalization.'
echo '32 paired interventions/window; full-benefit MLP; auxiliary masking; 10 dB, 0/8/24/48.'
echo 'Start allocation explicitly after review: same OUT, --phase allocation --resume (2000 updates).'
exec "$PYTHON" -u -m gaussian_jscc.learned_benefit_allocation \
  --run "$RUN" --scene "$SCENE" --out "$OUT" "${INIT[@]}" "$@"
