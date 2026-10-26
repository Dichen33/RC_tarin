# MiniDog 原生 MJLab 训练说明

本文档说明 `D:\mujoco_test\HighTorque-MiniDog\mujoco\mjlab` 下的原生 MJLab + RSL-RL 训练方式。当前推荐路线是：

```text
MJLab ManagerBasedRlEnvCfg + RSL-RL runner
```

注意：MJLab 虽然风格接近 IsaacLab，但后端是 MuJoCo/Warp，不是 IsaacGym。奖励项、接触传感器、地形和训练入口不要直接照搬 IsaacGym 的实现细节。

## 1. 环境准备

进入目录：

```bat
cd /d D:\mujoco_test\HighTorque-MiniDog\mujoco\mjlab
```

安装原生 MJLab + CUDA 依赖：

```bat
scripts\uv_setup_native.bat
```

这个脚本会把 `UV_CACHE_DIR` 指到 `D:\uv`，把 `WARP_CACHE_PATH` 指到 `D:\uv\warp_cache`。前者控制 uv 缓存和下载包，后者控制 MuJoCo/Warp 编译缓存，避免默认写到 C 盘时出现权限问题。

快速检查：

```bat
uv run python scripts\native_check.py
```

正常时应看到类似：

```text
mjlab=1.6.0
minidog_native=ok
registered_tasks=Robot-Flat-v0, Robot-Rough-v0, Robot-Crawl-v0
```

## 2. 任务列表

| 任务名 | 用途 |
|---|---|
| `Robot-Flat-v0` | 平地速度跟踪，先学稳定站立和基本行走 |
| `Robot-Rough-v0` | 崎岖地形，包含平地、台阶、方块、随机粗糙地形、坡面 |
| `Robot-Crawl-v0` | 低姿态爬行/低矮障碍任务 |

## 3. 推荐训练命令

当前 rough 奖惩已经参考 MJLab 官方 Go1 风格进一步调整：关闭 `feet_air_time`，加强摆腿高度/防滑，加入速度分档姿态奖励，并增加分组碰撞惩罚。建议新开一个训练名：

```bat
scripts\train_native.bat Robot-Rough-v0 --env.scene.num-envs 2048 --agent.max-iterations 2000 --agent.run-name rough_v5 --agent.logger tensorboard --agent.upload-model False
```

平地训练：

```bat
scripts\train_native.bat Robot-Flat-v0 --env.scene.num-envs 2048 --agent.max-iterations 10000 --agent.run-name flat_v1 --agent.logger tensorboard --agent.upload-model False
```

低姿态 crawl 训练：

```bat
scripts\train_native.bat Robot-Crawl-v0 --env.scene.num-envs 1024 --agent.max-iterations 15000 --agent.run-name crawl_v1 --agent.logger tensorboard --agent.upload-model False
```

常用参数含义：

| 参数 | 作用 |
|---|---|
| `Robot-Rough-v0` | 任务名，决定使用哪套 env cfg、地形、奖励和默认高度 |
| `--env.scene.num-envs` | 并行环境数量，越大采样越快，但越吃 GPU/显存 |
| `--agent.max-iterations` | PPO 更新轮数，不是单纯仿真步数 |
| `--agent.num-steps-per-env` | 每轮每个环境采样步数，默认 24 |
| `--agent.run-name` | 本次训练名字，也是 `model/` 下的目录名 |
| `--agent.logger tensorboard` | 记录 TensorBoard 曲线，Zh-db 也可以读取这些历史曲线 |
| `--agent.upload-model False` | 不上传 wandb，只保存本地 |

## 4. 模型保存位置

训练模型默认保存到：

```text
D:\mujoco_test\HighTorque-MiniDog\mujoco\mjlab\model\<run_name>\<timestamp>_<run_name>\
```

例如：

```text
D:\mujoco_test\HighTorque-MiniDog\mujoco\mjlab\model\rough_v5\2026-08-26_xx-xx-xx_rough_v5\
```

常见文件：

| 文件 | 作用 |
|---|---|
| `model_*.pt` | RSL-RL checkpoint，可用于 play |
| `params\env.yaml` | 本次训练环境配置快照 |
| `params\agent.yaml` | 本次训练 agent/PPO 配置快照 |
| `events.out.tfevents.*` | TensorBoard 曲线数据 |
| `git\*.diff` | 本次训练对应代码差异 |

`--checkpoint latest` 会选择该 run 目录里编号最大的 `model_*.pt`。

## 5. Play 命令

播放最新 checkpoint：

```bat
scripts\play_native.bat Robot-Rough-v0 --run-name rough_v5 --checkpoint latest
```

播放指定 checkpoint：

```bat
scripts\play_native.bat Robot-Rough-v0 --run-name rough_v5 --checkpoint model_1999.pt
```

零动作检查环境和默认站姿：

```bat
scripts\play_native.bat Robot-Rough-v0 --run-name rough_v5 --checkpoint latest --agent zero
```

随机动作检查环境是否正常：

```bat
scripts\play_native.bat Robot-Rough-v0 --run-name rough_v5 --checkpoint latest --agent random
```

## 6. 当前观测维度

Actor 观测：

| 名称 | 作用 |
|---|---|
| `base_ang_vel` | 机身角速度，让策略知道 roll/pitch/yaw 方向的转动状态 |
| `projected_gravity` | 重力在机身坐标系下的投影，用来判断机身姿态是否倾斜 |
| `command` | 目标速度 `[vx, vy, yaw_rate]` |
| `gait_phase` | 步态相位 `[sin(phi), cos(phi)]`，给策略学习 trot 节奏，不写死 action |
| `joint_pos` | 12 个关节相对默认站姿的位置 |
| `joint_vel` | 12 个关节速度 |
| `actions` | 上一步 action，帮助策略减少突变和抖动 |

Critic 额外观测：

| 名称 | 作用 |
|---|---|
| `base_lin_vel` | 真实机身线速度，只给 critic，看训练评价更准 |
| `foot_contact` | 足端接触比例 |
| `height_scan` | 机身前方高度扫描，用于 rough 地形判断 |

## 7. 动作定义

当前动作仍然是 12 维关节位置 residual：

```text
policy action -> default_joint_pos + residual * ACTION_SCALE -> position actuator
```

也就是说，策略不是直接输出力矩，也不是 IK 轨迹。trot 相位只作为观测和奖励软约束，不把迈腿时序硬写进 action。

## 8. Rough 奖励/惩罚

基础项：

| 名称 | 权重 | 作用 |
|---|---:|---|
| `track_lin_vel` | `+3.0` | 跟踪 x/y 平面速度 |
| `track_yaw_vel` | `+1.0` | 跟踪 yaw 角速度 |
| `upright` | `+1.0` | 保持机身朝上 |
| `base_height` | `-1.0` | 惩罚机身高度偏离 `STAND_HEIGHT` |
| `flat_orientation` | `-1.1` | 惩罚 roll/pitch 倾斜 |
| `lin_vel_z` | `-0.35` | 惩罚机身上下乱跳 |
| `pose` | `+0.80` | 官方 Go1 风格速度分档姿态奖励：站立严格、行走放松、跑动更放松 |
| `joint_torques` | `-2.0e-4` | 惩罚力矩过大 |
| `joint_acc` | `-2.5e-7` | 惩罚关节加速度过大 |
| `action_rate` | `-0.01` | 惩罚动作变化太快 |
| `joint_pos_limits` | `-1.0` | 惩罚顶到关节限位 |
| `stand_still` | `-0.55` | 零速度命令时抑制漂移、转动、掉高和倾斜 |
| `feet_contact` | `+0.10` | 鼓励保持合理接触，不让足端长期全飞 |
| `is_terminated` | `-50.0` | 摔倒/非法接触等终止惩罚 |
| `terrain_level` | `+0.15` | 课程地形等级奖励 |

新增/调整的 trot 与姿态约束：

| 名称 | 权重 | 作用 |
|---|---:|---|
| `feet_air_time` | `0.0` | 关闭单纯足端浮空时间奖励，避免策略为了浮空奖励学出异常摆腿 |
| `feet_gait_consistency` | `+0.45` | trot 接触节奏软约束：FL+RR 同相，FR+RL 同相，两组反相 |
| `foot_clearance` | `-2.2` | 摆腿时足端高度不足会被惩罚，减少拖地小碎步 |
| `foot_swing_height` | `-0.35` | 脚落地时检查本次摆腿最高点是否合理，抑制拖地式摆腿 |
| `foot_slip` | `-0.35` | 触地时足端水平滑动会被惩罚，提高支撑腿稳定性 |
| `stand_pose` | `-0.45` | 零速度时靠近默认站姿，并惩罚关节速度，避免 enable 后无命令仍做训练动作 |
| `stand_contact` | `-0.80` | 零速度时要求四足都接地，避免站立时只靠两只脚 |
| `stand_foot_geometry` | `-1.20` | 零速度时约束足端站宽和前后距离，避免站姿收腿、歪腿 |
| `leg_mirror` | `-0.25` | 只在有速度命令时生效，约束左右腿镜像，抑制一侧摆动更大 |
| `diagonal_leg_symmetry` | `-0.30` | 行走时约束 trot 对角腿关节 residual 和关节速度相近 |
| `diagonal_foot_symmetry` | `-0.18` | 行走时约束对角腿足端高度和水平速度幅值相近 |
| `stance_width` | `-4.0` | 惩罚左右足端距离过窄，抑制四条腿向内收 |
| `self_collisions` | `-0.10` | 惩罚机器人自身碰撞 |
| `thigh_collision` | `-0.20` | 惩罚大腿触地 |
| `shank_collision` | `-0.12` | 惩罚小腿触地 |
| `trunk_collision` | `-0.30` | 惩罚机身触地 |

`pose` 的速度分档参数：

| 档位 | 触发条件 | hip std | thigh std | calf std | 含义 |
|---|---|---:|---:|---:|---|
| standing | 速度命令 `< 0.05` | `0.05` | `0.05` | `0.08` | 几乎贴近默认站姿 |
| walking | `0.05 ~ 1.0` | `0.22` | `0.32` | `0.60` | 允许正常走路摆腿 |
| running | `>= 1.0` | `0.28` | `0.42` | `0.75` | 给高速运动更大动作空间 |

分组碰撞传感器：

| 传感器 | 检测对象 |
|---|---|
| `self_collision` | 机器人自身碰撞 |
| `thigh_ground_touch` | 四条大腿简化碰撞体触地 |
| `shank_ground_touch` | 四条小腿简化碰撞体触地 |
| `trunk_ground_touch` | 机身碰撞体触地 |

为支持大腿/小腿触地检测，MJCF 中新增了四条大腿和四条小腿的简化 capsule collision geom。这些不是视觉模型，只用于碰撞和接触惩罚。

## 9. Trot 相位划分

相位 `phase` 范围为 `0~1`，观测输出为 `[sin(2*pi*phase), cos(2*pi*phase)]`。

当前 soft trot 期望：

| phase | 角度 | 期望支撑腿 | 期望摆动腿 |
|---|---|---|---|
| `0~0.5` | `0~pi` | `FL + RR` | `FR + RL` |
| `0.5~1.0` | `pi~2pi` | `FR + RL` | `FL + RR` |

`transition_width=0.08` 表示相位切换边缘有软过渡，不会在边界强制瞬间换腿。

这不是开环控制器，因为：

- action 仍由网络输出 12 维 residual；
- phase 只是观测，让网络知道当前节奏；
- `feet_gait_consistency` 只是 reward，鼓励接触节奏像 trot；
- 网络仍然需要自己学关节如何摆动、如何落脚、如何保持身体稳定。

## 10. 为什么这样改

之前只靠 `feet_air_time`，策略容易学到“有两只脚浮空就能拿奖励”，但没有人告诉它必须四条腿按 trot 对角节奏轮换，所以可能出现：

- 只有两条腿伸直摆动；
- 另外两条腿长期支撑不动；
- 小碎步前进，摆腿高度不够；
- 四条腿向内收；
- 零速度 enable 后仍保持训练出的移动姿态。

现在的修改把这些漏洞分别补上：

- 关闭 `feet_air_time`，避免单纯浮空奖励造成异常摆腿；
- `feet_gait_consistency` 约束“哪两条腿该同相/反相”；
- `foot_clearance` 和 `foot_swing_height` 约束摆腿必须抬起来；
- `foot_slip` 约束支撑腿不能乱滑；
- `pose` 采用速度分档：站立严、行走松；
- `stand_pose`、`stand_contact`、`stand_foot_geometry` 约束零速度时回到普通四足站立；
- `leg_mirror`、`diagonal_leg_symmetry`、`diagonal_foot_symmetry` 约束行走时左右腿和对角腿不要摆得不对称；
- `stance_width` 约束脚不要向身体中心收得太窄。
- `self_collisions`、`thigh_collision`、`shank_collision`、`trunk_collision` 把碰撞惩罚分组，方便在 Zh-db/TensorBoard 中判断是哪一类碰撞问题。

## 11. 调参建议

如果仍然小碎步：

- 适当提高 `foot_clearance` 惩罚，例如 `-1.5 -> -2.0`
- 稍微降低 `action_rate` 或 `joint_acc` 惩罚，给腿部更大动作空间
- 提高速度命令范围时同步提高 `gait_phase` 的频率上限

如果仍然只有两条腿主要运动：

- 提高 `feet_gait_consistency`，例如 `0.35 -> 0.45`
- 提高 `leg_mirror`，例如 `-0.18 -> -0.25`
- 保持 `feet_air_time=0.0`，先不要重新打开单纯浮空奖励

如果四条腿继续向内收：

- 提高 `stance_width` 惩罚，例如 `-3.0 -> -4.0`
- 检查 URDF/MJCF 足端 body 坐标和左右 hip 方向是否一致

如果零速度仍然动：

- 提高 `stand_pose` 或 `stand_still`
- 检查 play 时命令是否真的固定为 `[0, 0, 0]`
- 用 `--agent zero` 先确认模型和环境默认站姿是否正常
