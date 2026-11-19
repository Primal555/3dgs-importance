#!/usr/bin/env bash
set -eo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
MODE="${1:-all}"
case "$MODE" in all|sources|fcgs|c3dgs) ;; *) echo "Usage: $0 [all|sources|fcgs|c3dgs]" >&2; exit 2;; esac
BASE="$PROJECT/external/attribute_codecs"
fetch_source() {
  local dir="$1" url="$2" commit="$3"
  if [[ ! -d "$dir/.git" ]]; then
    git init -q "$dir"
    git -C "$dir" remote add origin "$url"
  fi
  if ! git -C "$dir" rev-parse --verify HEAD >/dev/null 2>&1; then
    git -C "$dir" fetch --depth 1 "$url" "$commit"
    git -C "$dir" checkout --detach "$commit"
  fi
  test "$(git -C "$dir" rev-parse HEAD)" = "$commit" || {
    echo "Unexpected upstream revision in $dir. Expected $commit." >&2
    exit 1
  }
}
if [[ "$MODE" != c3dgs ]]; then
  fetch_source "$BASE/FCGS" https://github.com/YihangChen-ee/FCGS.git 31e59a46f7e51505b46a7bbe6e53268ee3155fbf
fi
if [[ "$MODE" != fcgs ]]; then
  fetch_source "$BASE/C3DGS" https://github.com/KeKsBoTer/c3dgs.git 2a234af55fbe8b90c8829c1436ce80088c4b622b
  git -C "$BASE/C3DGS" -c url.https://github.com/.insteadOf=git@github.com: \
    submodule update --init --recursive --depth 1
fi
if [[ "$MODE" == sources ]]; then exit 0; fi

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
FCGS_ENV="${FCGS_ENV:-$CONDA_BASE/envs/gs_fcgs}"
C3DGS_ENV="${C3DGS_ENV:-$CONDA_BASE/envs/gs_c3dgs}"
setup_env() (
  local prefix="$1" toolkit="$2" tag="$3" arch="$4"
  shift 4
  if [[ ! -x "$prefix/bin/python" ]]; then
    conda create -y -p "$prefix" python=3.10 pip
  fi
  conda install -y -p "$prefix" -c "nvidia/label/cuda-$toolkit" -c conda-forge \
    cuda-toolkit 'gcc_linux-64=11' 'gxx_linux-64=11'
  conda activate "$prefix"
  unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH
  unset NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS CUDAHOSTCC GCC_EXEC_PREFIX COMPILER_PATH
  export CUDA_HOME="$prefix" CC="$prefix/bin/x86_64-conda-linux-gnu-cc" CXX="$prefix/bin/x86_64-conda-linux-gnu-c++"
  export CUDAHOSTCXX="$CXX"
  export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib64:${LD_LIBRARY_PATH:-}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$arch}" MAX_JOBS="${MAX_JOBS:-4}"
  local py="$prefix/bin/python"
  "$py" -m pip install 'numpy==1.26.4' 'plyfile==0.8.1' 'setuptools<70' wheel ninja tqdm
  "$py" -m pip install "$@" --index-url "https://download.pytorch.org/whl/$tag"
  if [[ "$tag" == cu117 ]]; then
    for ext in freqencoder gridencoder gridcreater arithmetic; do
      "$py" -m pip install --no-build-isolation "$BASE/FCGS/submodules/$ext"
    done
    PYTHONPATH="$BASE/FCGS" "$py" -c 'from model.FCGS_model import FCGS'
  else
    "$py" -m pip install --only-binary=torch-scatter 'torch-scatter==2.1.2' \
      -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
    for ext in diff-gaussian-rasterization weighted_distance; do
      "$py" -m pip install --no-build-isolation "$BASE/C3DGS/submodules/$ext"
    done
    (cd "$BASE/C3DGS" && "$py" -c 'from compress import calc_importance; from compression.vq import compress_gaussians')
  fi
)
if [[ "$MODE" != c3dgs ]]; then
  setup_env "$FCGS_ENV" 11.7.1 cu117 '8.6+PTX' 'torch==1.13.1+cu117'
fi
if [[ "$MODE" != fcgs ]]; then
  setup_env "$C3DGS_ENV" 12.1.1 cu121 '8.9' 'torch==2.1.0+cu121' 'torchvision==0.16.0+cu121'
fi
