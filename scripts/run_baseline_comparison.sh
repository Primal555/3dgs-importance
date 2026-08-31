#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 SCENE_DIR VANILLA_OUTPUT MASK_OUTPUT [GPU_ID]" >&2
  exit 2
fi

scene_dir=$1
vanilla_output=$2
mask_output=$3
gpu_id=${4:-0}

mkdir -p "$vanilla_output"

CUDA_VISIBLE_DEVICES="$gpu_id" python -u train_3dgs.py \
  -s "$scene_dir" \
  -m "$vanilla_output" \
  --eval \
  --data_device cpu \
  --test_iterations 7000 30000 \
  --checkpoint_iterations 7000 15000 30000 \
  2>&1 | tee "$vanilla_output/train.log"

CUDA_VISIBLE_DEVICES="$gpu_id" python render.py -m "$vanilla_output" --skip_train
CUDA_VISIBLE_DEVICES="$gpu_id" python render.py -m "$mask_output" --skip_train
CUDA_VISIBLE_DEVICES="$gpu_id" python metrics.py -m "$vanilla_output" "$mask_output"

python compare_models.py \
  --baseline "$vanilla_output" \
  --mask "$mask_output" \
  --make_visuals
