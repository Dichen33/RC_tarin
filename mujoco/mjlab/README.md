# HighTorque MiniDog MJLab 强化学习工作区

这个目录是 HTDW-4438 MiniDog 的 MuJoCo 强化学习工作区。
当前第一阶段使用的是：

```text
command(vx, vy, yaw) -> IK gait q_nominal -> policy residual dq -> joint PD torque
```

这是一种比较稳妥的路线。机器人先尽量贴近现有 IK 步态，再让策略学习平衡、横向修正、yaw 抑制和地形/接触补偿。

如果你现在要看正式的训练、回放、模型路径和奖励说明，直接看
[TRAINING_TASKS.md](./TRAINING_TASKS.md)。

## 目录结构

```text
mujoco/mjlab/
  legged_gym/                         IsaacGym / legged_gym 风格任务层
    envs/base/                        通用配置基类
    envs/htdw_4438/                   HTDW4438 任务配置和环境适配
    scripts/train.py                  IsaacGym 风格训练入口
    scripts/play.py                   IsaacGym 风格回放入口
    utils/task_registry.py            任务注册接口
  rsl_rl/                             RSL-RL 风格占位目录
    rsl_rl/runners/on_policy_runner.py 原生 runner 预留位置
  resources/robots/htdw_4438/         机器人资源入口
  runs/                               legged_gym 风格输出目录
  model/                              可直接回放的模型输出目录
  minidog_rl/                         MuJoCo / SB3 实现
    envs/minidog_env.py               Gymnasium residual RL 环境
    configs/                          环境和 PPO 配置说明
    scripts/train_sb3.py              当前可用训练入口
  mjlab_tasks/                        原生 MJLab 迁移笔记
  scripts/                            Windows 辅助脚本
```

## 安装

先进入工程目录：

```powershell
cd D:\mujoco_test\HighTorque-MiniDog\mujoco\mjlab
```

如果你已经有可用的 Python 环境，也可以先直接安装：

```powershell
pip install -e .
pip install -r requirements.txt
```

原生 MJLab 这条路线更推荐直接用 uv：

```powershell
scripts\uv_setup_native.bat
```

这个脚本会自动把 uv 缓存放到 `D:\uv`。

## 快速检查

先跑一个最小检查：

```powershell
uv run python scripts\native_check.py
```

如果一切正常，通常会看到：

```text
mjlab=1.6.0
minidog_native=ok
registered_tasks=Robot-Flat-v0, Robot-Rough-v0, Robot-Crawl-v0
```

## 烟雾测试

用零残差动作跑一段最小回合：

```powershell
python -m minidog_rl.scripts.rollout --steps 3000 --vx 0.08 --vy 0.00 --yaw 0.00
python -m minidog_rl.scripts.rollout --steps 3000 --vx 0.00 --vy 0.09 --yaw 0.00
```

输出会包含类似这样的轨迹记录：

```text
step,x,y,z,yaw,body_vx,body_vy,body_wz,reward,fallen
```

## 训练

当前推荐的训练命令仍然是 legged_gym 风格：

```powershell
python legged_gym\scripts\train.py --task=htdw_4438 --headless --num_envs 4 --max_iterations 1000 --num_steps_per_env 24 --run_name residual_v1
```

这个训练方式仍然是 SB3 PPO，但命令形状和目录结构已经尽量对齐 IsaacGym / legged_gym 的习惯。

## 模型保存位置

推荐模型目录：

```text
mujoco\mjlab\model\<experiment_name>\<run_name>\
```

可直接回放的模型文件：

```text
mujoco\mjlab\model\<experiment_name>\<run_name>\final_policy.pt
```

`final_model.zip` 也会保留，用来续训或重新导出。

## 回放

回放命令：

```powershell
python legged_gym\scripts\play.py --task=htdw_4438 --experiment_name htdw_4438_standard --load_run residual_v1 --checkpoint final_policy --vx 0.0 --vy 0.09 --yaw 0.0
```

`play.py` 默认会打开 MuJoCo viewer。
如果只想看 CSV 日志，不要窗口，可以加 `--headless`。

## 原生 MJLab 迁移

`mjlab_tasks/minidog_residual` 里放的是原生 MJLab 迁移笔记。
后续会把同样的观测、奖励、命令采样和 residual 动作契约，迁移到原生 MJLab 的 manager-based 任务里。

