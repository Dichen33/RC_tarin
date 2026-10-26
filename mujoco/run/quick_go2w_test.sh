#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.cache/uv}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$ROOT_DIR/.cache/warp}"

# This is a short PPO validation run, not the full 132000-update training.
# The 48 GiB server GPU can use the normal 2048-env rollout while keeping the
# run short enough to validate physics, observations, AME/MoE, logging, and
# checkpoint creation before committing to the long run.
NUM_ENVS="${GO2W_QUICK_NUM_ENVS:-2048}"
ITERATIONS="${GO2W_QUICK_ITERATIONS:-200}"
RUN_NAME="${GO2W_QUICK_RUN_NAME:-go2w_quick}"
LOGGER="${GO2W_QUICK_LOGGER:-tensorboard}"

exec uv run --locked --extra native --extra cu128 python scripts/train_go2w_unified.py \
  --num-envs "$NUM_ENVS" \
  --iterations "$ITERATIONS" \
  --run-name "$RUN_NAME" \
  --logger "$LOGGER" \
  "$@"
