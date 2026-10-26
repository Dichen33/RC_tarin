#!/usr/bin/env bash
# =============================================================================
# 梅花桩预设 A：2048 环境（当前基线，采集约占 83%）
#
# 用法（在服务器 /root/run 下）：
#   DRY=1 bash scripts/preset_plum_2048.sh            # 只打印命令，不启动（先看这个）
#   bash scripts/preset_plum_2048.sh                  # 500 轮，软地形（BeamDojo 阶段 1）
#   ITER=1000 bash scripts/preset_plum_2048.sh        # 改轮数
#   NAME=plum_try2 bash scripts/preset_plum_2048.sh   # 改 run 名
#   SOFT=0 bash scripts/preset_plum_2048.sh           # 关软地形（阶段 2 / 真赛道）
#
#   阶段 2（带 checkpoint 续训）——注意 --stage-checkpoint 必须放在 EXTRA_ARGS 里，
#   它不是 train_go2w_unified.py 的参数，而是透传给 train_native.py 的：
#     SOFT=0 EXTRA_ARGS="--stage-checkpoint /root/run/model/go2w_unified/<dir>/model_500.pt --stage-transfer full_resume" \
#       bash scripts/preset_plum_2048.sh
#
# 超参：全部使用 go2w_native/config/rl_cfg.py 的默认值（已用 --help 核实过真实参数名）
#   num_envs            = 2048
#   num_steps_per_env   = 24      -> batch 49,152
#   num_mini_batches    = 4       -> mini-batch 12,288
#   num_learning_epochs = 3
#   learning_rate       = 2e-4  (adaptive, desired_kl 0.02)
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."          # -> /root/run
export MINIDOG_OBSTACLES=plum
export GO2W_PLUM_PILOT=1
export GO2W_PLUM_SOFT="${SOFT:-1}"   # 1 = 软地形（掉不下去，必须读雷达）

ITER="${ITER:-500}"
NAME="${NAME:-plum_soft1}"
NUM_ENVS=2048
LOGDIR="logs"

# --- 输入校验：避免 ``set -e`` 下出现难懂的错误 --------------------------------
if ! [[ "$ITER" =~ ^[0-9]+$ ]] || (( ITER < 1 )); then
  echo "ERROR: ITER must be a positive integer, got '$ITER'" >&2; exit 2
fi
if [[ -z "$NAME" || "$NAME" =~ [[:space:]] ]]; then
  echo "ERROR: NAME must be non-empty and contain no spaces, got '$NAME'" >&2; exit 2
fi
case "$GO2W_PLUM_SOFT" in
  0|1) ;;
  *) echo "ERROR: SOFT must be 0 or 1, got '$GO2W_PLUM_SOFT'" >&2; exit 2 ;;
esac
mkdir -p "$LOGDIR"

CMD=(uv run --frozen --extra native --extra cu128 python scripts/train_go2w_unified.py
     --num-envs "$NUM_ENVS"
     --iterations "$ITER"
     --run-name "$NAME")

# 透传参数（故意不加引号以支持多个参数；不要在这里放含空格的值）
EXTRA_ARGS="${EXTRA_ARGS:-}"
if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  CMD+=(--extra $EXTRA_ARGS)
fi

echo "------------------------------------------------------------------"
echo " preset    : 2048 envs / 24 steps / 4 mini-batches (batch 49152)"
echo " obstacles : MINIDOG_OBSTACLES=$MINIDOG_OBSTACLES"
echo " pilot     : GO2W_PLUM_PILOT=$GO2W_PLUM_PILOT"
echo " soft fill : GO2W_PLUM_SOFT=$GO2W_PLUM_SOFT   (1=软地形 0=真赛道)"
echo " iterations: $ITER"
echo " run name  : $NAME"
echo " log       : $LOGDIR/$NAME.log"
echo " extra     : ${EXTRA_ARGS:-(none)}"
echo "------------------------------------------------------------------"

if [[ "${DRY:-0}" == "1" ]]; then
  printf 'DRY RUN: '; printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

pkill -f train_native 2>/dev/null || true
sleep 2
PYTHONUNBUFFERED=1 nohup "${CMD[@]}" > "$LOGDIR/$NAME.log" 2>&1 &
echo "started pid $!  ->  tail -f $LOGDIR/$NAME.log"
