# 观测维度与步态学习分析

## 结论

如果策略里没有显式的 trot 相位、足端轨迹、步态时钟、参考 IK 步态，策略就只能从奖励里自己“碰”出周期性迈腿。
一旦奖励更偏向站稳、少动、少错，策略很容易先学成“站住 + 微抖 + 少走”。

## 1. HTDW4438_Isaacgym 的观测

参考：
- [htdw_4438_config.py](<D:\mujoco_test\HTDW4438_Isaacgym\legged_gym\envs\htdw_4438\htdw_4438_config.py>)
- [legged_robot.py](<D:\mujoco_test\HTDW4438_Isaacgym\legged_gym\envs\base\legged_robot.py>)

这个项目的 actor 观测总维度是 `45`。

### 逐项说明

- `base_ang_vel` 3维  
  机身角速度，反映机器人当前在怎么转、怎么晃。  
  主要用于平衡和 yaw 控制。

- `projected_gravity` 3维  
  重力在机身坐标系下的投影。  
  它最直接反映 roll/pitch 倾角，是“站稳没站稳”的核心信号。

- `command` 3维  
  当前速度目标，通常是 `x速度、y速度、yaw速度`。  
  策略靠它知道现在要往哪走。

- `joint_pos` 12维  
  12 个关节当前相对位置。  
  告诉策略腿现在弯成什么样。

- `joint_vel` 12维  
  12 个关节速度。  
  告诉策略腿当前是收还是放、动得快不快。

- `actions` 12维  
  上一时刻策略动作。  
  这相当于给策略一点短期记忆，方便动作更平滑。

### 这个项目缺少什么

- 没有显式步态相位
- 没有足端轨迹
- 没有 IK 参考步态
- 没有步态时钟
- 没有地形高度扫描作为 actor 输入

所以它更像“纯状态 + 命令”的 PPO，周期性迈腿要策略自己从 reward 里学出来。

## 2. RC_Legged 的观测

参考：
- [env_cfgs.py](<D:\git\RC_Legged\05_software\train\src\robot\config\env_cfgs.py>)
- [him_mjlab_vec_env_wrapper.py](<D:\git\RC_Legged\05_software\train\src\robot\rsl_rl\wrappers\him_mjlab_vec_env_wrapper.py>)
- [him_actor_critic.py](<D:\git\RC_Legged\05_software\train\src\robot\rsl_rl\modules\him_actor_critic.py>)

### 单步 policy 观测

单步输入大致是：

- `base_ang_vel` 3维
- `projected_gravity` 3维
- `command` 4维
- `joint_pos` 12维
- `joint_vel` 12维
- `wheel_vel` 4维
- `actions` 12维

单步合计约 `50` 维。

### 历史堆叠后的 policy 输入

wrapper 里 `history_length = 5`，所以 policy buffer 会堆成：

- `50 * (5 + 1) = 300` 维

也就是说，策略不是只看一帧，而是看 6 帧历史。

### actor 真正用到的输入

`HIMActorCritic` 还会把历史估计成：

- velocity 3维
- latent 16维

所以 actor 最终有效输入是：

- 当前单步观测 50
- + 速度估计 3
- + latent 16
- = `69` 维

### critic 观测

critic 还会看 privileged obs：

- `base_lin_vel`
- `foot_contact`
- `height_scan`

所以 critic 比 actor 更“看得见环境”，尤其是地形和接触信息。

## 3. 这些观测各自干什么

- `base_ang_vel`：看机身转动趋势，帮助稳定和转向
- `projected_gravity`：看姿态倾斜，帮助保持直立
- `command`：告诉策略当前目标速度
- `joint_pos`：告诉策略腿现在的姿态
- `joint_vel`：告诉策略腿现在的运动速度
- `actions`：告诉策略上一时刻刚做了什么
- `wheel_vel`：告诉策略轮子当前怎么转
- `base_lin_vel`：告诉 critic 当前真实运动速度
- `foot_contact`：告诉 critic 哪些足端/轮子在接地
- `height_scan`：告诉 critic 前方地形是否有坡、台阶、坑

## 4. 为什么会学成站住不走

常见原因：

1. `stand_still`、姿态、生存类奖励太容易拿
2. 速度跟踪奖励不够强
3. 动作惩罚太重，策略倾向于少动
4. 没有 phase / clock / foot trajectory 先验
5. 只有 residual action，没有显式步态参考

## 5. 对你现在项目的启发

如果你想让策略更容易学出 trot，可以考虑：

- 加显式 gait phase
- 加足端目标轨迹
- 加参考 IK 步态
- 把 command 和 phase 绑定
- 把奖励从“少动”转成“稳定地周期性迈腿”

## 6. 你现在这个 native 机器人

参考：
- [minidog_native/config/env_cfgs.py](<D:\mujoco_test\HighTorque-MiniDog\mujoco\mjlab\minidog_native\config\env_cfgs.py>)

### Actor 观测

当前 actor 观测项是：

- `base_ang_vel` 3
- `projected_gravity` 3
- `command` 3
- `joint_pos` 12
- `joint_vel` 12
- `actions` 12

单步一共 `45` 维。

但你这里还设置了：

- `observations["actor"].history_length = 6`

所以 actor 最终输入会按 6 帧历史展开，约等于：

- `45 * 6 = 270` 维

如果你把 `last_action` 也按单独一帧理解，总体上可以记成“约 270 维级别的历史输入”。

### Critic 观测

critic 额外包含：

- `base_lin_vel` 3
- `foot_contact` 1
- `height_scan` 一组地形射线值

所以 critic 的单步维度大约是：

- `45 + 3 + 1 + height_scan`

其中 `height_scan` 取决于当前 `GridPatternCfg(size=(1.4, 0.9), resolution=0.08)` 的射线点数，实际通常是上百维。

所以你现在的 critic 观测可以理解成：

- **基础状态 + 命令 + 关节状态 + 接触信息 + 地形扫描**

## 7. 这些维度各自有什么用

### Actor 侧

- `base_ang_vel`：让策略知道机身是否在旋转、偏摆、失稳
- `projected_gravity`：让策略知道身体有没有歪，帮助维持站立和平衡
- `command`：告诉策略当前要追踪的速度目标
- `joint_pos`：告诉策略腿当前的姿态，决定下一步怎么摆腿
- `joint_vel`：告诉策略腿当前的运动趋势，帮助它做动态修正
- `actions`：告诉策略上一时刻刚输出了什么，帮助动作连续和平滑

### Critic 侧

- `base_lin_vel`：给价值函数看真实速度，帮助它判断当前动作到底有没有跑对
- `foot_contact`：告诉 critic 足端是不是在接地，帮助估计支撑相和稳定性
- `height_scan`：告诉 critic 前方地形是不是有坡、台阶、坑

### 这些维度为什么重要

- `base_ang_vel` 和 `projected_gravity` 主要负责“稳不稳”
- `command` 负责“往哪走”
- `joint_pos` / `joint_vel` 负责“腿现在是什么状态”
- `actions` 负责“动作有没有连续性”
- `base_lin_vel` / `foot_contact` / `height_scan` 负责“环境和真实运动是不是匹配”

如果没有这些状态，策略就更容易瞎猜；  
如果只有这些状态但没有步态相位、足端轨迹、IK 参考，策略又容易学成“少动最安全”。

## 8. Play 命令

### 你现在这个 MJLab native

```bat
cd D:\mujoco_test\HighTorque-MiniDog\mujoco\mjlab
scripts\play_native.bat Robot-Flat-v0 --run-name flat_v1 --checkpoint latest
scripts\play_native.bat Robot-Rough-v0 --run-name rough_v1 --checkpoint latest
scripts\play_native.bat Robot-Crawl-v0 --run-name crawl_v1 --checkpoint latest
```

### HTDW4438_Isaacgym

```bat
cd D:\mujoco_test\HTDW4438_Isaacgym
python legged_gym\scripts\play.py --task=htdw_4438 --experiment_name htdw_4438_standard --load_run layout_model_smoke --checkpoint final_policy
```

### RC_Legged

```bat
cd D:\git\RC_Legged\05_software\train
python sim2sim\main.py
```

如果你要看某个具体模型，通常要按这个项目自己的 play / sim2sim 入口，再指定对应模型文件。

## 9. 一句话总结

HTDW4438_Isaacgym 是更典型的“稀观测 PPO”，  
RC_Legged 是“历史堆叠 + 估计器 + 更强地形/命令约束”，  
但两者都还不是“显式步态时钟 + IK 参考轨迹”的结构。
D:\mujoco_test\HighTorque-MiniDog>cd D:\mujoco_test\HighTorque-MiniDog\Zh-db

D:\mujoco_test\HighTorque-MiniDog\Zh-db>python server.py --host 127.0.0.1 --port 8765


cd /d D:\mujoco_test\HighTorque-MiniDog\mujoco\mjlab
uv run train Robot-Rough-v0 --env.scene.num-envs 2048 --agent.max-iterations 2000 --agent.run-name rough_v2