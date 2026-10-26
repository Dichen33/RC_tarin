#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
if [[ "$REQUIRE_CUDA" != "0" && "$REQUIRE_CUDA" != "1" ]]; then
  echo "ERROR: REQUIRE_CUDA must be 0 or 1 (got '$REQUIRE_CUDA')." >&2
  exit 1
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.cache/uv}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$ROOT_DIR/.cache/warp}"
# Mainland-China PyPI mirror for ordinary Python packages.  The explicit
# PyTorch CUDA, NVIDIA Warp, and MuJoCo indexes in pyproject.toml are kept
# unchanged because the mirror may not provide matching binary wheels.
export UV_INDEX_URL="${UV_INDEX_URL:-${PYPI_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}}"
mkdir -p "$UV_CACHE_DIR" "$WARP_CACHE_PATH"

if ! command -v uv >/dev/null 2>&1; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: uv is missing and curl is not installed." >&2
    exit 1
  fi
  echo "uv was not found; installing it into the user account..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi was not found. CUDA training needs an NVIDIA driver." >&2
  if [[ "$REQUIRE_CUDA" == "1" ]]; then
    echo "Set REQUIRE_CUDA=0 only for CPU installation/debugging." >&2
    exit 1
  fi
else
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
fi

echo "Installing Python $PYTHON_VERSION and native MJLab dependencies..."
uv python install "$PYTHON_VERSION"
# The lock file is committed with the portable package.  Use it strictly so
# setup does not perform another slow dependency resolution after a mirror
# refresh; refresh the lock explicitly only when intentionally changing it.
uv sync --locked --python "$PYTHON_VERSION" --extra native --extra cu128

echo ""
uv run python scripts/native_check.py
echo ""
if [[ "$REQUIRE_CUDA" == "1" ]]; then
  uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA is required but unavailable'; print('torch=', torch.__version__); print('cuda_available=True'); print('cuda_device=', torch.cuda.get_device_name(0))"
else
  uv run python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda_device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
fi

echo ""
echo "Python package index: $UV_INDEX_URL"
echo "Environment setup complete. Quick Go2W validation: bash quick_go2w_test.sh"
echo "Full unified training: uv run python scripts/train_go2w_unified.py --num-envs 2048 --iterations 132000 --run-name go2w_unified --logger tensorboard"
