#!/usr/bin/env bash
# Small observation budget, FULL Gaussian scene, deterministic held-out views.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$PROJECT/output/truck_render_first_short_$(date +%Y%m%d_%H%M%S)}"
export TRAIN_VIEWS="${TRAIN_VIEWS:-12}"
export VIEWS_PER_STEP="${VIEWS_PER_STEP:-2}"
export RENDER_STEPS="${RENDER_STEPS:-300}"
export JOINT_STEPS="${JOINT_STEPS:-0}"
export VALIDATE_EVERY="${VALIDATE_EVERY:-25}"
export VALIDATION_VIEWS="${VALIDATION_VIEWS:-4}"
export VALIDATION_TRIALS="${VALIDATION_TRIALS:-2}"
export PATIENCE="${PATIENCE:-0}"
export SAVE_EVERY="${SAVE_EVERY:-100}"
printf 'Short render-first run: %s render steps, %s training views, full Gaussian scene.\n' "$RENDER_STEPS" "$TRAIN_VIEWS"
exec bash "$PROJECT/scripts/train_codec_learned.sh" "$OUT"
