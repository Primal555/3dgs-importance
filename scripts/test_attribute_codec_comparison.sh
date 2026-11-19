#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
RUN="${1:?Usage: test_attribute_codec_comparison.sh SEMANTIC_RUN NEW_OUT [options]}"
OUT="${2:?Usage: test_attribute_codec_comparison.sh SEMANTIC_RUN NEW_OUT [options]}"
shift 2
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES}"
CONDA_BASE="$(conda info --base)"
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; semantic run: $RUN; output: $OUT"
echo 'Fixed 16-bit XYZ and Gaussian rows. Semantic final.pt, FCGS and C3DGS file round trips.'
echo 'Report full attributes, geometry/opacity only and color only using the original test cameras.'
echo 'Download OUT/review.tar.gz. RESUME=1 skips completed scene/method rows.'
exec "${PYTHON_BIN:-python}" -u -m comparison_baselines.attribute_codec_comparison \
  --run "$RUN" --out "$OUT" \
  --fcgs-python "${FCGS_ENV:-$CONDA_BASE/envs/gs_fcgs}/bin/python" \
  --c3dgs-python "${C3DGS_ENV:-$CONDA_BASE/envs/gs_c3dgs}/bin/python" "${INIT[@]}" "$@"
