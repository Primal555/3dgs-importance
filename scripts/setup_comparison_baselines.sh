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
# Conda compiler activation hooks may read unset CONDA_BACKUP_* variables.
# Relax nounset only inside Conda operations and preserve their exit status.
conda_command() {
  local conda_status=0
  set +u
  conda "$@" || conda_status=$?
  set -u
  return "$conda_status"
}
CONDA_BASE="$(conda_command info --base)"
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
set -u
DIGITAL_ENV="${DIGITAL_ENV:-$CONDA_BASE/envs/gs_digital}"
MESONGS_ENV="${MESONGS_ENV:-$CONDA_BASE/envs/gs_mesongs}"
if [[ "$MODE" == all || "$MODE" == digital ]]; then
  if [[ ! -x "$DIGITAL_ENV/bin/python" ]]; then conda_command create -y -p "$DIGITAL_ENV" python=3.11 pip; fi
  "$DIGITAL_ENV/bin/python" -m pip install 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cu128
  "$DIGITAL_ENV/bin/python" -m pip install -r "$PROJECT/comparison_baselines/requirements-digital.txt"
  # The PHY path does not require the optional ray-tracing distribution.
  "$DIGITAL_ENV/bin/python" -m pip install --no-deps 'sionna==2.2.0'
  (cd "$PROJECT" && "$DIGITAL_ENV/bin/python" -m comparison_baselines.digital_link self-test --device cpu)
fi
if [[ "$MODE" == all || "$MODE" == mesongs ]]; then
  test -f "$PROJECT/comparison_baselines/MesonGS/mesongs.py" || { echo 'Run the sources setup first.' >&2; exit 1; }
  if [[ ! -x "$MESONGS_ENV/bin/python" ]]; then conda_command create -y -p "$MESONGS_ENV" python=3.10 pip; fi
  "$MESONGS_ENV/bin/python" -m pip install 'numpy==1.26.4' 'setuptools<70' wheel ninja
  "$MESONGS_ENV/bin/python" -m pip install 'torch==2.1.0+cu121' 'torchvision==0.16.0+cu121' \
    'torchaudio==2.1.0+cu121' --index-url https://download.pytorch.org/whl/cu121
  "$MESONGS_ENV/bin/python" -m pip install -r "$PROJECT/comparison_baselines/requirements-mesongs.txt"
  "$MESONGS_ENV/bin/python" -m pip install --only-binary=torch-scatter 'torch-scatter==2.1.2' \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
  if [[ -n "${MESONGS_CUDA_HOME:-}" ]]; then
    export CUDA_HOME="$MESONGS_CUDA_HOME"
  else
    conda_command install -y -p "$MESONGS_ENV" -c nvidia/label/cuda-12.1.1 cuda-toolkit
    export CUDA_HOME="$MESONGS_ENV"
  fi
  test -x "$CUDA_HOME/bin/nvcc" || { echo "nvcc missing: $CUDA_HOME/bin/nvcc" >&2; exit 1; }
  # CUDA 12.1 and the PyTorch 2.1 bundled pybind11 headers need a compatible
  # host toolchain (https://github.com/pybind/pybind11/issues/4606).
  conda_command install -y -p "$MESONGS_ENV" -c conda-forge 'gcc_linux-64=11' 'gxx_linux-64=11'
  conda_command activate "$MESONGS_ENV"
  # Absolute Python paths alone do not isolate the compiler environment. In
  # particular, a parent maskgs shell can inject its includes and nvcc -ccbin.
  unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH
  unset NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS CUDAHOSTCC GCC_EXEC_PREFIX COMPILER_PATH
  export CC="$MESONGS_ENV/bin/x86_64-conda-linux-gnu-cc"
  export CXX="$MESONGS_ENV/bin/x86_64-conda-linux-gnu-c++"
  export CUDAHOSTCXX="$CXX"
  test -x "$CC" && test -x "$CXX" || { echo 'MesonGS GCC 11 toolchain missing' >&2; exit 1; }
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
  export MAX_JOBS="${MAX_JOBS:-4}"
  printf 'MesonGS build: CC=%s CXX=%s CUDA_HOME=%s MAX_JOBS=%s\n' "$CC" "$CXX" "$CUDA_HOME" "$MAX_JOBS"
  "$CXX" --version
  for module in diff-gaussian-rasterization simple-knn weighted_distance; do
    "$MESONGS_ENV/bin/python" -m pip install --no-build-isolation \
      "$PROJECT/comparison_baselines/MesonGS/submodules/$module"
  done
fi
printf '\nDigital Python: %s\nMesonGS Python: %s\n' "$DIGITAL_ENV/bin/python" "$MESONGS_ENV/bin/python"
