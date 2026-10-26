#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.cache/uv}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$ROOT_DIR/.cache/warp}"

exec uv run python scripts/train_native.py Robot-PlumBlossom-v0 "$@"
