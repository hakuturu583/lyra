#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
CC_BIN="${CC_BIN:-/usr/bin/gcc-13}"
CXX_BIN="${CXX_BIN:-/usr/bin/g++-13}"
EIGEN_INCLUDE="${EIGEN_INCLUDE:-/usr/include/eigen3}"
MAX_JOBS="${MAX_JOBS:-8}"

cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found in PATH." >&2
  exit 1
fi

if [ ! -x "$CC_BIN" ] || [ ! -x "$CXX_BIN" ]; then
  echo "gcc-13/g++-13 not found. Installing via apt." >&2
  if ! command -v sudo >/dev/null 2>&1; then
    echo "sudo is required to install gcc-13 and g++-13." >&2
    exit 1
  fi
  sudo apt-get update
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
  sudo apt-get update
  sudo apt-get install -y gcc-13 g++-13
fi

if [ ! -d "$CUDA_HOME" ]; then
  echo "CUDA_HOME does not exist: $CUDA_HOME" >&2
  echo "Set CUDA_HOME to your CUDA 12.8 toolkit path and retry." >&2
  exit 1
fi

if [ ! -x "$CC_BIN" ]; then
  echo "C compiler not found: $CC_BIN" >&2
  exit 1
fi

if [ ! -x "$CXX_BIN" ]; then
  echo "C++ compiler not found: $CXX_BIN" >&2
  exit 1
fi

echo "[1/6] Creating uv-managed Python environment"
uv python install "$PYTHON_VERSION"
uv venv --clear --python "$PYTHON_VERSION"

# shellcheck disable=SC1091
source "$ROOT_DIR/.venv/bin/activate"

export CUDA_HOME
export CC="$CC_BIN"
export CXX="$CXX_BIN"

echo "[2/6] Syncing Python dependencies"
uv sync

SITE="$(python -c "import site; print(site.getsitepackages()[0])")"
export CPATH="$CUDA_HOME/include:$EIGEN_INCLUDE:$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include${CPATH:+:$CPATH}"
export LD_LIBRARY_PATH="$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[3/6] Installing transformer_engine"
uv pip install --no-build-isolation "transformer_engine[pytorch]"
ln -sfn "$SITE/nvidia/cuda_runtime" "$SITE/nvidia/cudart"

echo "[4/6] Installing flash-attn"
MAX_JOBS="$MAX_JOBS" uv pip install --no-build-isolation --no-binary flash-attn flash-attn==2.6.3

echo "[5/6] Building vendored CUDA extensions"
USE_SYSTEM_EIGEN=1 uv pip install --no-build-isolation -e "lyra_2/_src/inference/vipe"
uv pip install --no-build-isolation -e "lyra_2/_src/inference/depth_anything_3[gs]"

echo "[6/6] Verifying imports"
PYTHONPATH=. python -c "
import torch, flash_attn, transformer_engine.pytorch, vipe_ext, depth_anything_3.api, moge.model.v1
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
print('all imports OK')
"

echo
echo "Environment setup completed."
echo "Activate it with: source .venv/bin/activate"
