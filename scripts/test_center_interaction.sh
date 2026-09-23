#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Select an available GPU}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${1:?Usage: bash scripts/test_center_interaction.sh CHECKPOINT [OUT]}"
OUT="${2:-$PROJECT/output/truck_center_interaction_$(date +%Y%m%d_%H%M%S)}"
PLY="${PLY:-$PROJECT/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply}"
SCENE="${SCENE:-$PROJECT/data/tandt_db/tandt/truck}"
[[ -f "$CHECKPOINT" && -f "$PLY" && -d "$SCENE" ]] || { echo 'Missing checkpoint, PLY or scene' >&2; exit 1; }
# Validate root without ever overwriting an existing experiment.
"$PYTHON_BIN" -c 'import pathlib,sys; p=pathlib.Path(sys.argv[1]); assert not p.exists() or all(x.name in ("console.log","run.pid") for x in p.iterdir()), "Choose a new output"; p.mkdir(parents=True,exist_ok=True)' "$OUT"
"$PYTHON_BIN" -u -m gaussian_jscc.center_interaction_diagnostics \
  --checkpoint "$CHECKPOINT" --ply "$PLY" --out "$OUT/diagnostics" --device cuda \
  --probe-steps "${PROBE_STEPS:-1000}" --seed "${SEED:-42}"
# Both cases start randomly; CHECKPOINT is used only by diagnostics above.
# Default affine matches the latest experiment. Both cases always use the same norm.
exec "$PYTHON_BIN" -u scripts/compare_center_decoders.py \
  --comparison interaction --readout-norm "${READOUT_NORM:-affine}" \
  --ply "$PLY" --source "$SCENE" --out "$OUT/training" --device cuda \
  --steps "${STEPS:-2000}" --blocks-per-batch "${BLOCKS_PER_BATCH:-32}" \
  --lr "${LR:-0.0002}" --seed "${SEED:-42}" --resolution "${RESOLUTION:-2}" \
  --validate-every 500 --render-every 500
