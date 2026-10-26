#!/usr/bin/env bash
# =============================================================================
# 梅花桩预设 B：4096 环境（只为压缩墙钟时间；学习效果本身不变）
#
# 用法（在服务器 /root/run 下）：
#   DRY=1 bash scripts/preset_plum_4096.sh                     # 只打印命令
#   ITER=20 NAME=probe4096 bash scripts/preset_plum_4096.sh    # 20 轮吞吐探针
#   bash scripts/preset_plum_4096.sh                           # 500 轮
#   MINI=8 bash scripts/preset_plum_4096.sh                    # 保 mini-batch 不变
#
# 4096 的超参怎么定（别乱调）：
#
#   我们的默认值（go2w_native/config/rl_cfg.py，已核实）：
#     num_steps_per_env = 24 , num_mini_batches = 4 , num_learning_epochs = 3
#     learning_rate = 2e-4 (adaptive, desired_kl = 0.02) , entropy_coef = 0.0
#
#   legged_gym / rsl_rl 的常见默认几何（建议以仓库源码为准）：
#     num_envs 4096 , num_steps_per_env 24 , num_mini_batches 4 ,
#     num_learning_epochs 5 , learning_rate 1e-3 , desired_kl 0.01 ,
#     entropy_coef 0.01 , clip 0.2 , gamma 0.99 , lam 0.95
#
#   ==> 4096 时我们的默认值（24 steps / 4 mini-batches）恰好等于这套几何：
#       batch = 4096 x 24 = 98,304，mini-batch = 24,576。
#       所以默认不加任何 --extra 覆盖，风险最低。
#
#   两种可选：
#     A) MINI=4（默认）  每个 mini-batch 24,576（= 参考实现）。adaptive lr 自动
#                        收小步长，最稳；前 50 轮可能略慢。
#     B) MINI=8          每个 mini-batch 12,288（= 与 2048 基线一致）。每样本梯度
#                        步数翻倍，早期更快，代价是时间/显存。
#
#   不要做：把 learning_rate 按 envs 线性放大 —— 我们的 schedule 是 adaptive
#   （desired_kl 控制），手动放大 lr 会和 KL 控制器打架。
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."          # -> /root/run
export MINIDOG_OBSTACLES=plum
export GO2W_PLUM_PILOT=1
export GO2W_PLUM_SOFT="${SOFT:-1}"

ITER="${ITER:-500}"
NAME="${NAME:-plum_soft1_4096}"
NUM_ENVS=4096
MINI="${MINI:-4}"
LOGDIR="logs"

# --- 输入校验 ------------------------------------------------------------------
if ! [[ "$ITER" =~ ^[0-9]+$ ]] || (( ITER < 1 )); then
  echo "ERROR: ITER must be a positive integer, got '$ITER'" >&2; exit 2
fi
if [[ -z "$NAME" || "$NAME" =~ [[:space:]] ]]; then
  echo "ERROR: NAME must be non-empty and contain no spaces, got '$NAME'" >&2; exit 2
fi
if ! [[ "$MINI" =~ ^[0-9]+$ ]] || (( MINI < 1 )); then
  echo "ERROR: MINI must be a positive integer, got '$MINI'" >&2; exit 2
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

EXTRA_NOTE="(默认：24 steps / 4 mini-batches = legged_gym 标准几何)"
if (( MINI != 4 )); then
  # 参数名已用 ``train_native.py --help`` 核实：--agent.algorithm.num-mini-batches
  CMD+=(--extra --agent.algorithm.num-mini-batches "$MINI")
  EXTRA_NOTE="(--extra 覆盖 num_mini_batches=$MINI)"
fi

# 额外透传（阶段 2 的 --stage-checkpoint 就放这里）
EXTRA_ARGS="${EXTRA_ARGS:-}"
if [[ -n "$EXTRA_ARGS" ]]; then
  if (( MINI != 4 )); then
    echo "ERROR: 不能同时用 MINI=$MINI 和 EXTRA_ARGS（--extra 只能出现一次）" >&2
    exit 2
  fi
  # shellcheck disable=SC2206
  CMD+=(--extra $EXTRA_ARGS)
  EXTRA_NOTE="$EXTRA_NOTE + EXTRA_ARGS"
fi

echo "------------------------------------------------------------------"
echo " preset    : 4096 envs   $EXTRA_NOTE"
echo " batch     : $NUM_ENVS x 24 = $((NUM_ENVS * 24))  -> mini-batch $((NUM_ENVS * 24 / MINI))"
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
