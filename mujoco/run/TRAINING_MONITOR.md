# 训练监控手册（精简版 · 2026-10-23 重写）

> ⚠️ **本文档是重写版**。原 400 行版本在一次 PowerShell 文本往返中被双重编码损坏
> （详见 `PROJECT_STATUS.md` §13.6），原文已另存为 `TRAINING_MONITOR.md.corrupt`。
> 若你手上有备份，可以覆盖本文件；否则本版覆盖了全部**判读所需**的指标。
>
> 适用任务：`Robot-Go2W-{Mixed,PlumBlossom,Gap,HighWall,Step}-v0`，2048 envs × 24 steps/轮，50 Hz。

---

## 0. 30 秒速查

| # | 指标 | 干什么用 | 健康 | 该停 |
|---|---|---|---|---|
| 1 | `Mean episode length` | 回合存活多久（奖励的乘数、成功率的天花板） | 逐步爬向 1500 | 卡在 <300 超过 200 轮 |
| 2 | `Mean action std` | 探索幅度（策略自己调的旋钮） | 0.30–0.45 平稳 | 单调塌缩到 <0.15 或死贴 1.0 |
| 3 | `Mean value loss` | critic 是否跟得上回报 | <1e3 | 连续 5 轮 >1e4 |
| 4 | `Curriculum/terrain_levels/course_stage` | 课程开到第几段 | 单调不减（棘轮） | 回落（说明棘轮失效） |
| 5 | `Curriculum/terrain_levels/rough_plum_blossom` | 梅花桩平均档位（0–6） | 持续上升 | **贴 0（只在 L0 连续板）** |
| 6 | **`plum_reached_frac`** | **前进到出口的比例（真指标）** | 上升 | 恒 0 超过 500 轮 |
| 7 | `plum_success_frac` | 完整过桩比例 | 上升 | 恒 0 |
| 8 | **`Metrics/air_time_mean`** | **刷分哨兵** | 0.03–0.15 | >0.25 且上升 |
| 9 | `rough_plum_blossom_blocks_completed` | 摸过的桩数（**集合指标，会骗人**） | 增长 | 只作参考 |
| 10 | `Episode_Termination/fell_off_course` | 掉落终止是否生效 | 全程非零（硬终止） | 恒 0 |

> **当前这一跑（`plum_pilot1`，`GO2W_PLUM_PILOT=1`）先看 §6**：第 1 轮要确认 `plum_pilot=1`、
> `command_max_x=0.3500`，20 轮内 `plum_progress` 必须离开 0.11（=卡在入口平台边）。

---

## 1. 指标分组

### 1.1 数值健康

| 指标 | 含义 | 判读 |
|---|---|---|
| `Mean value loss` | critic 回归回报的损失（Huber，Δ=20） | <1e3 正常；>1e4 连续 5 轮 → 停（健康守卫会自动终止） |
| `Mean surrogate loss` | PPO 策略损失 | 在 0 附近震荡即正常 |
| `Mean entropy loss` | 策略熵 | **骤降**说明探索塌缩 |
| `Metrics/mean_action_acc` | 动作加速度 | >1.0 说明动作在抖 |
| `Mean action std` | 探索宽度 | 0.30–0.45；**持续上升 = 熵奖励过大**（bootcamp1 在 `entropy_coef=0.005` 下升到 0.71 并摔崩）。当前 `entropy_coef=0.0`，正常应缓慢下降 |

### 1.2 基础运动（放梅桩前的门槛）

| 指标 | 含义 | 判读 |
|---|---|---|
| `Mean episode length` | 回合长度 | 应逐步接近 1500 |
| `Episode_Termination/bad_orientation` | 摔倒终止 | <1.0 健康 |
| `Episode_Reward/stand_still` | 站立不动惩罚 | −1 ~ −5 正常；< −20 说明机身塌陷 |
| `Curriculum/terrain_levels/base_ready` | 基础运动放行 | 变 1 后（棘轮）永不回 0 |
| `Curriculum/terrain_levels/base_success_rate_ema` | 基础成功率 | ≥0.75 才放行（会抖，靠尖峰 + 棘轮） |

> **基础闸门是第一道门，不是梅花桩。** `plum_bootcamp1` 的教训：`base_ready` 未开时
> `plum_*` 全为 0，先查 `eplen`/`bad_orientation`/`action std`，不要调梅花桩奖励。
> 判据：iter 50 `eplen>500`、`bad_ori<1.5`、`std` 不再上升；iter 150 `base_ready=1`。
>
> **例外：`GO2W_PLUM_PILOT=1` 时这道门被强制打开**（`base_ready=True`，≈98% 回合在桩上），
> 所以上面这些指标不再是本轮的阻塞项；代价是基础能力暂时退化，是**故意接受**的，
> 过桩后按 `PROJECT_STATUS.md` §15.5 P1/P2 接回来。见 §6。

### 1.3 课程与难度

| 指标 | 含义 | 判读 |
|---|---|---|
| `course_stage` | 放开到第几段（0=仅基础） | 单调不减 |
| `focus_mode` | 单障碍聚焦采样 | 单族任务应为 1 |
| `rough_plum_blossom` | 梅花桩平均档位（0–6，L0=连续板，L6=赛题） | 应逐轮上升；贴 0 说明卡在 bootcamp |
| `rough_plum_blossom_level_cap` | 该族难度上限 | 升档后应 +1 |
| `lp_*` | LP-ACRL 采样统计 | 仅诊断 |

### 1.4 梅花桩专用

| 指标 | 含义 | 健康方向 |
|---|---|---|
| **`plum_reached_frac`** | 位移 > 2.7 m 的回合比例 | ↑ |
| `plum_success_frac` | 到出口 + 姿态/高度/支撑合格 + 保持 3 步 | ↑ |
| `plum_progress` | 平均位移 / 2.7（中位数最有用） | ↑（0.167 ≈ 0.45 m） |
| `plum_tilt_frac` | 结束时倾角 <0.70 的比例 | ↑（>0.8 好） |
| `plum_height_frac` | 结束时机身高度 >0.14 的比例 | ≈1.0 |
| `plum_collision_frac` | 回合内有碰撞的比例 | ↓ |
| **`radar_valid_fraction`** | 雷达有效单元比例（1 − dropout − clip/miss） | ≈0.96；<0.90 说明雷达地图/高度越界 |
| **`plum_success_frac_l*`** | **每一档**的成功率（`l0`/`l1`/…） | 低档 >0.5，逐档下降是正常的 |
| **`plum_blocks_l*`** | **每一档**的平均过桩数 | 低档应接近 3；高档从 0 开始 |
| **`plum_gap_seen_frac`** | 采到 L3+（真缝）的回合比例 | >0.1，否则说明课程没升上去 |
| `plum_wheels_on_stone` | 每步踩在桩上的轮数加权（稠密收入） | ↑ **且必须非零**（权重 0 时恒 0，等于失去观测） |
| `plum_foothold` | 每块**新**桩一次性 +12 | 出现尖峰 |
| `plum_overhang` | 悬空采样惩罚（负） | 出生应 ≈0（不是 −0.6） |
| `plum_landing` | 落地缓冲（轻放高） | ↑ |
| `obstacle_foot_clearance` | 抬腿超过桩面且在动 | ↑ **非零 = 在尝试抬腿** |
| `plum_airborne` | 接触 <2 轮（负） | ≈0 |
| `plum_edge_hit` | 撞桩侧面（负） | ↓ |
| `plum_next_foothold` | 摆动轮靠近下一块桩（potential） | 小正值，不悬停 |

### 1.5 终止分布

| 指标 | 含义 | 判读 |
|---|---|---|
| `fell_off_course` | 掉出赛道 | 硬终止，全程必须非零 |
| `bad_orientation` | 翻倒 | <1.0 |
| `illegal_thigh_contact` | 大腿碰地 | 越少越好 |
| `time_out` | 超时 | 高说明"活着但没进展" |

---

## 2. 三个必看组合

### 2.1 "骗奖励"识别

| `plum_reached_frac` | `blocks_completed` | 结论 |
|---|---|---|
| 高 | 低 | ❌ 在无桩档位开车（`lvl` 会 ≈0） |
| 中 | 高 | ✅ 真在过桩 |
| 0 | 1–2 | ⚠️ 卡在第一块（`plum_fixed1` 的状态，P0 前的旧阶梯） |
| 0 | **从 1 开始** | ❌ 平台被误判为第 1 块桩（`overhang` 会恒为负） |

### 2.2 "悬停刷分"识别

```
air_time_mean 从 0.05 涨到 >0.25 且持续上升  →  某条腿被举着刷分
同时 tilt_frac 下降、bad_orientation 上升    →  三足支撑不稳
```
历史值：step5 涨到 2.15（`plum_next_foothold` 的 level 形式），已改为 potential 形式。

### 2.3 "掉落不判死"识别

```
fell_off_course 恒 0  且  eplen 长期贴 1500   →  终止没生效（或机器人从没靠近过边缘）
```
P0 起硬终止从第 1 步开始：`fell` 应非零，`eplen` 回落到 600–1000。
如果 `fell` 恒 0，先查 `course_start_difficulty` 和 `terrain_levels`，不要再加软窗口。

---

## 3. 判据速查（阈值）

| 阶段 | 轮数 | 该看的 | 门槛 |
|---|---|---|---|
| **课程** | **0–150** | **`level_cap`、`plum_success_frac_l0`、`plum_gap_seen_frac`** | **cap 50 轮 ≥2、150 轮 ≥4；L0 success >0.5** |
| 基础运动 | 0–300 | `base_success_rate_ema`、`bad_orientation`、`eplen` | ≥0.75 且 `bad_ori`<1.5 |
| 开闸 | 300–700 | `course_stage`、`focus_mode` | 变 1 后不回退 |
| 试着上桩 | 150–300 | `blocks`、`wheels_on_stone`、`clearance` | 从 0 开始增长 |
| 硬约束 | 全程 | `fell_off_course`、`eplen` | `fell` 非零、`eplen` 600–1000 |
| 学前进 | 300–1500 | **`reached`**、`progress`、逐档 `blocks_l*` | `reached` ≥0.4 |
| 学收尾 | 1500+ | `succ`、`tilt_frac` | `succ` ≥0.2 |

---

## 4. 参数一览（当前值）

| 项 | 值 |
|---|---|
| 回合 | 30 s / 1500 步 |
| 指令速度 | 基础 **0.30–0.60 m/s**；其他障碍 0.15–0.30；plum 释放后 **0.20–0.35**（实机低速死区，日志看 `command_max_x`：0.60→0.35 表示已切到梅花桩） |
| 软窗口 | **已移除**（硬终止从第 1 步开始）；`recovery_steps` 只给一次爬回窗口 |
| 掉落恢复宽限 | **60 步（1.2 s）**（旧 25 步 = 0.5 s，而 0.20 m 空档在 0.20 m/s 下要 1.0 s，任何入洞都必然终止） |
| 轮线桥板 | 入口（行 B y=+0.2）/出口（行 A y=−0.2）各一块与桩面齐平的桥，长 `(0.5−size/2)×(1−d)`：**0.20/0.19/0.18/0.15/0.11/0.06/0.00 m**（L0–L6），剩余入口空档 0.00/0.04/0.09/0.15/0.22/0.31/0.40 m。奖励侧已把桥面从 `course_span` 挖掉（§17），桥上滚行不会被算成踩桩、也不会被算成悬空 |
| 站桩环境 | 桩任务 `rel_standing_envs=0.0`（零指令回合会让 `stand_still` 罚无界 `joint_deviation_l2`，bootcamp2 实测 −6275/回合） |
| 地形阶梯 | 7 行：`0.600/0.533/0.467/0.400/0.333/0.267/0.200`；L0 连续板，L3 起 gap > 轮径 0.172 |
| 认证 | `gap > 0.02` 或 `gutter > 0.02` 的桩才计入 `blocks`；L0 连续板 `blocks=0`，成功只看 `stable_exit` |
| 升档 | `blocks ≥ 2.5` 或成功率 ≥0.45（plum，`PLUM_UPGRADE_SUCCESS`）；障碍 cap 只升不降；**pilot 模式下 plum 的证据窗口 256→1024 回合**（≈98% 采样下 256 回合不足 2 个迭代，会冲太快） |
| 终止惩罚 | plum 任务 `is_terminated = −8`（全局仍 −15） |
| 任务收入 | `course_completion +40`、`plum_foothold +12`；障碍行 `upright/yaw ×0.25`、`track_lin_vel ×0.05` |
| 聚焦采样 | 非聚焦地形占 0.2；**pilot 模式下 0.02（采样器下限）= ≈98% 在桩上** |
| Huber Δ | 20.0 |
| 健康守卫 | 连续 5 次越界 |
| `entropy_coef` | 0.0（0.005 会导致 std 升到 0.7 并摔崩） |
| 地形抖动 | 位置 ±1 cm、高度 ±4 mm |

---

## 5. 日志一行示例

```
Curriculum/terrain_levels/plum_reached_frac: 0.0000
Episode_Reward/plum_wheels_on_stone: 0.0356
Metrics/air_time_mean: 0.0578
```

**读法**：`reached` 是结论，`wheels_on_stone` 是过程，`air_time_mean` 是哨兵。

---

## 6. `plum_pilot1` 这一跑怎么读（当前，2026-10-24）

开跑条件：`MINIDOG_OBSTACLES=plum GO2W_PLUM_PILOT=1`，200 轮，2048 envs。改动与理由见 `PROJECT_STATUS.md` §16/§17。

**第 1 轮就要确认的 5 个值**（不对就别往下等）：

| 指标 | 期望 | 不对说明 |
|---|---|---|
| `Curriculum/terrain_levels/plum_pilot` | **1.0** | 环境变量没传进容器 / 没重启 |
| `course_stage`、`focus_mode` | 1、1 | profile 不是单族（`MINIDOG_OBSTACLES=plum` 没生效），开机日志会打 `[Go2W] GO2W_PLUM_PILOT=1 expects a single-family run` |
| `command_max_x` | 0.3500 | 还在基础档，指令没切到桩档 |
| `rough_plum_blossom_level_cap` | 起步 1.0 | — |
| `Episode_Reward/stand_still` | 量级 −1~−5 | 若又是几千量级，说明 `rel_standing_envs=0` 没生效 |

**按轮次的判据（顺序看，不要只看 `Mean reward`）**：
- ~20 轮：`plum_progress` 必须**离开 0.11**（=0.31 m，即卡在平台边）；`plum_foothold>0`；`Episode_Reward/course_completion>0`。
- ~60 轮：`plum_progress_l0>0.7`、`plum_success_frac_l0>0.2`、`plum_lateral_hold>0`（说明真的越过了 x=0.45）。
- ~120 轮：`plum_reached_frac>0.1`、`blocks_completed>1.0`、`level_cap≥2`。
- 200 轮：`plum_blossom_success_rate_ema>0.1`，`Mean episode length` 在桩档上升（不再是 ~267）。

**失败分支（提前定好，不要临场改奖励）**：
1. 40 轮后 `plum_progress` 仍 ≈0.11 且 `plum_foothold≈0` → 入口仍不通：查 `Episode_Termination/fell_off_course` 与 play 回放的轮线实际落点，**不要动奖励权重**。
2. `plum_progress` 上去了但 `air_time_mean>0.25` 且 `blocks` 不涨 → 悬跳刷分：查 `plum_overhang`/`foot_clearance` 的门控，而不是加新项。
3. `level_cap` 冲到 ≥4 但 `plum_success_frac_l0` 掉回 0 → 升档太快：把 `PLUM_PILOT_EPISODES_PER_UPGRADE` 提到 2048，这是本轮唯一预留的旋钮。
4. `Mean value loss` 连续 5 轮 >1e4 → 停，回报尺度问题（先确认 `stand_still` 已按 §4 收敛）。
