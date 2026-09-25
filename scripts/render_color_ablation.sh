#!/usr/bin/env bash
# Evaluation only; never trains or overwrites the original run/checkpoints.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
TRAIN_DIR="${1:?Usage: render_color_ablation.sh TRAIN_DIR [NEW_OUT]}"
OUT="${2:-$TRAIN_DIR/color_ablation_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
[[ -f "$TRAIN_DIR/training.json" && -f "$TRAIN_DIR/codec.pt" ]] || { echo 'Missing training.json or codec.pt' >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Output exists: $OUT" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
EXTRA=()
[[ -z "${PLY:-}" ]] || EXTRA+=(--ply "$PLY")
[[ -z "${SCENE:-}" ]] || EXTRA+=(--source "$SCENE")
exec "$PYTHON_BIN" -u -m gaussian_jscc.color_ablation \
  --training "$TRAIN_DIR" --out "$OUT" --tiers 3 --trials 2 --device cuda "${EXTRA[@]}"
