#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.cache/uv}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$ROOT_DIR/.cache/warp}"
# Optional per-invocation package mirror. For example:
# PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/ bash check_linux.sh
# This does not modify pyproject.toml or uv.lock.
if [[ -n "${PYPI_INDEX_URL:-}" ]]; then
  export UV_INDEX_URL="$PYPI_INDEX_URL"
fi

REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
if [[ "$REQUIRE_CUDA" != "0" && "$REQUIRE_CUDA" != "1" ]]; then
  echo "ERROR: REQUIRE_CUDA must be 0 or 1 (got '$REQUIRE_CUDA')." >&2
  exit 1
fi

uv run python scripts/native_check.py
if [[ "$REQUIRE_CUDA" == "1" ]]; then
  uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA is required but unavailable'; print('torch=', torch.__version__); print('cuda_available=True'); print('cuda_device=', torch.cuda.get_device_name(0))"
else
  uv run python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda_device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
fi

SMOKE_TASK="${SMOKE_TASK:-Robot-Go2W-Mixed-v0}"
SMOKE_DEVICE="${SMOKE_DEVICE:-}"
if [[ "$REQUIRE_CUDA" == "0" && -z "$SMOKE_DEVICE" ]]; then
  SMOKE_DEVICE="cpu"
fi
SMOKE_ARGS=(--task "$SMOKE_TASK" --num-envs "${SMOKE_NUM_ENVS:-64}" --steps "${SMOKE_STEPS:-4}")
if [[ -n "$SMOKE_DEVICE" ]]; then
  SMOKE_ARGS+=(--device "$SMOKE_DEVICE")
fi
uv run python scripts/smoke_test.py "${SMOKE_ARGS[@]}"
