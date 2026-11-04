#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
git submodule update --init --depth 1 -- comparison_baselines/MesonGS comparison_baselines/MaskGaussian
git -C comparison_baselines/MaskGaussian \
  -c url.https://github.com/.insteadOf=git@github.com: \
  submodule update --init --recursive --depth 1 -- \
  submodules/diff-gaussian-rasterization \
  submodules/mask-diff-gaussian-rasterization submodules/simple-knn
for method in MesonGS MaskGaussian; do
  test -f "$PROJECT/comparison_baselines/$method/README.md"
  printf '%s: ' "$method"
  git -C "$PROJECT/comparison_baselines/$method" rev-parse HEAD
done
