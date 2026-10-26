#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.cache/uv}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$ROOT_DIR/.cache/warp}"

TASK="${1:-Robot-Go2W-Mixed-v0}"
if [[ "$#" -gt 0 ]]; then
  shift
fi

exec uv run --locked --extra native --extra cu128 python scripts/train_native.py "$TASK" "$@"
