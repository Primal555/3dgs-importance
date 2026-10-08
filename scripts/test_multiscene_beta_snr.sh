#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
cd "$(dirname "$0")/.."
CHECKPOINT="${1:?Supply shared end_render/codec.pt}"
OUT="${2:?Supply a new experiment output directory}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MANIFEST="${MANIFEST:-configs/multiscene_tandt_db.json}"
read -r -a BETAS <<< "${BETAS:-0 0.001 0.01 0.03}"
for BETA in "${BETAS[@]}"; do
  RUN="$OUT/beta_$BETA"
  "$PYTHON_BIN" -u -m gaussian_jscc.multiscene --allocation-only --manifest "$MANIFEST" \
    --checkpoint "$CHECKPOINT" --out "$RUN" --bootstrap-steps 0 --render-steps 0 --joint-steps 0 \
    --allocation-steps "${ALLOCATION_STEPS:-1000}" --validate-every 100 --beta "$BETA"
  "$PYTHON_BIN" -u -m gaussian_jscc.multiscene_experiments --manifest "$MANIFEST" --role train \
    --checkpoint "$RUN/checkpoints/final/codec.pt" --out "$RUN/test" --snrs 0 5 10 15 20 \
    --trials "${TEST_TRIALS:-3}" --test-views "${TEST_VIEWS:-0}"
done
# Pass explicit directories: no cross-run averaging of beta-dependent total losses.
EVALUATIONS=()
for BETA in "${BETAS[@]}"; do EVALUATIONS+=("$OUT/beta_$BETA/test"); done
"$PYTHON_BIN" -m gaussian_jscc.multiscene_plots --runs "${EVALUATIONS[@]}" --out "$OUT/comparison"
