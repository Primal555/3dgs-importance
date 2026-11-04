#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-all}"
case "$MODE" in all|sources|digital|mesongs) ;; *) echo 'Usage: setup_comparison_baselines.sh [all|sources|digital|mesongs]' >&2; exit 2;; esac
if [[ "$MODE" == all || "$MODE" == sources ]]; then
  bash "$PROJECT/scripts/prepare_comparison_sources.sh"
fi
if [[ "$MODE" == sources ]]; then exit 0; fi
command -v conda >/dev/null || { echo 'Activate conda before running setup.' >&2; exit 1; }
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
DIGITAL_ENV="${DIGITAL_ENV:-$CONDA_BASE/envs/gs_digital}"
MESONGS_ENV="${MESONGS_ENV:-$CONDA_BASE/envs/gs_mesongs}"
if [[ "$MODE" == all || "$MODE" == digital ]]; then
  if [[ ! -x "$DIGITAL_ENV/bin/python" ]]; then conda create -y -p "$DIGITAL_ENV" python=3.11 pip; fi
  "$DIGITAL_ENV/bin/python" -m pip install 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cu128
  "$DIGITAL_ENV/bin/python" -m pip install -r "$PROJECT/comparison_baselines/requirements-digital.txt"
  # The PHY path does not require the optional ray-tracing distribution.
  "$DIGITAL_ENV/bin/python" -m pip install --no-deps 'sionna==2.2.0'
  (cd "$PROJECT" && "$DIGITAL_ENV/bin/python" -m comparison_baselines.digital_link self-test --device cpu)
fi
if [[ "$MODE" == all || "$MODE" == mesongs ]]; then
  test -f "$PROJECT/comparison_baselines/MesonGS/mesongs.py" || { echo 'Run the sources setup first.' >&2; exit 1; }
  if [[ ! -x "$MESONGS_ENV/bin/python" ]]; then conda create -y -p "$MESONGS_ENV" python=3.10 pip; fi
  "$MESONGS_ENV/bin/python" -m pip install 'numpy==1.26.4' 'setuptools<70' wheel ninja
  "$MESONGS_ENV/bin/python" -m pip install 'torch==2.1.0+cu121' 'torchvision==0.16.0+cu121' \
    'torchaudio==2.1.0+cu121' --index-url https://download.pytorch.org/whl/cu121
  "$MESONGS_ENV/bin/python" -m pip install -r "$PROJECT/comparison_baselines/requirements-mesongs.txt"
  "$MESONGS_ENV/bin/python" -m pip install --only-binary=torch-scatter 'torch-scatter==2.1.2' \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
  if [[ -n "${MESONGS_CUDA_HOME:-}" ]]; then
    export CUDA_HOME="$MESONGS_CUDA_HOME"
  else
    conda install -y -p "$MESONGS_ENV" -c nvidia/label/cuda-12.1.1 cuda-toolkit
    export CUDA_HOME="$MESONGS_ENV"
  fi
  test -x "$CUDA_HOME/bin/nvcc" || { echo "nvcc missing: $CUDA_HOME/bin/nvcc" >&2; exit 1; }
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
  for module in diff-gaussian-rasterization simple-knn weighted_distance; do
    "$MESONGS_ENV/bin/python" -m pip install --no-build-isolation \
      "$PROJECT/comparison_baselines/MesonGS/submodules/$module"
  done
fi
printf '\nDigital Python: %s\nMesonGS Python: %s\n' "$DIGITAL_ENV/bin/python" "$MESONGS_ENV/bin/python"
