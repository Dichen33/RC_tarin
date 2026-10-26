# `model_199.pt` 的 sim2sim 结论

## 一句话

**这个 checkpoint 只能在平地上稳定前进；梅花桩、高墙、台阶、rough、沟壑都不能通过。**
最小的 10 cm × 4 cm 沟壑可以通过，但这只是轮距跨过小坑，不代表学会了越障。

---

## 1. checkpoint 是什么

```
D:\mujoco_test\model_199.pt
iter = 199
architecture = AME + wheel-first residual MoE
action = 16 (12 legs + 4 wheels)
proprio = 6 × 59 = 354
scan = 403 (31 × 13, 2.4 m × 1.0 m)
```

checkpoint 的 `infos` 里有 `go2w_curriculum`、`_course_stage = 0`、`_lp_stage = 28`，且
`fix_smoke.log` 记录的命令是：

```
train_native.py Robot-Go2W-Mixed-v0 --env.scene.num-envs 2048 \
  --agent.max-iterations 200 --agent.run-name fix_smoke
```

所以它是 **Mixed-v0 的 199 iteration 冒烟模型**，不是训练充分的 Flat-v0，更不是障碍策略。
它的“平地基础运动”是 wheel locomotion：四轮滚动 + 腿基本保持站立姿态。

---

## 2. 评估结果（nominal，seed=42）

完整矩阵：`results/full_model199_v4/report.md`（24 episodes，20 s/个）。
最小难度矩阵：`results/easiest_model199_v4/report.md`（每个障碍 d=0，15 s/个）。
平地随机化：`results/flat_stress_model199_v4/report.md`（5 个随机硬件实例）。

| 障碍 | 最小难度（d=0） | 最高难度（d=1） | 判定 |
|---|---|---|---|
| flat | **PASS** 4.05 m / 4.0 m，9.78 s | — | 平地可走 |
| flat + 域随机化 | **5/5 PASS**（摩擦、质量、延迟、噪声等） | — | 平地较稳 |
| rough | FAIL，0.75 m 后停住 | FAIL，0.35 m 后停住 | 不能过 |
| plum（梅花桩） | FAIL，fallen，0.53 m，只碰到 1 块 | FAIL，fallen，0.47 m | 不能过 |
| high_wall（高墙） | FAIL，2.06 m（墙高仅 0.06 m） | FAIL，1.61 m（墙高 0.42 m） | 不能过 |
| gap（沟壑） | **PASS** 3.39 m（宽 0.10 m，深 0.04 m） | FAIL，fallen（宽 0.65 m，深 0.45 m） | 只能过小于轮距的小坑 |
| stairs（台阶） | FAIL，1.35 m（第一级仅 0.04 m） | FAIL，1.02 m | 不能过 |

完整矩阵总计：**1 PASS / 23 FAIL / 0 ERROR，通过率 4.2%**。
最小难度矩阵：**2 PASS / 4 FAIL**（flat + 10 cm gap）。

关键细节：

- `high_wall` 在 d=0.25（墙高约 0.137 m）时，机器人在第一道墙前约 0.2 m 就停下，
  不是“撞倒后滑过去”，而是策略自己拒绝继续；
- `stairs` 在 d=0（踏步高 0.04 m）时，机器人在第一级前停住；
- `plum` 在 d=0（块高 0.10 m、平台 0.14 m）时，从入口平台走出去约 0.5 m 就失去平衡；
- `gap` 在 d=0 能过，是因为 0.10 m 的缝小于轮子直径 0.172 m，轮子直接滚过去；
  到 d=0.2（宽 0.21 m）就停在边缘，d≥0.6 掉进沟里。

---

## 3. 为什么旧 sim2sim 会给出错误/不可用的结论

旧 `sim2sim` 不只是“参数没调好”，有几处会直接让当前 checkpoint 无法正确推理：

1. **radar encoder 通道数**：训练 checkpoint 的
   `terrain_encoder.convolution.0.weight` 是 `[32, 4, 5, 5]`（height + validity +
   2 个坐标通道），旧版按 3 通道构造，`load_state_dict` 直接报 shape mismatch。
2. **高度扫描符号**：旧版 `origin[2] - distance - base[2]` 等于 `-distance`，平地扫描约
   -0.45 m；训练是 `frame_z - hit_z`，约 +0.45 m。策略看到的“地形”完全反了。
3. **MoE 融合**：训练是 wheel policy + **argmax 单个 residual expert**，按
   `selected_weight / (wheel_weight + selected_weight)` 归一化，`residual_scale=0.40`，
   最后对 `last_action` 和输出 clamp ±4；旧版把 3 个 expert 全部加权、`residual_scale=0.35`，
   没有 clamp。动作分布不一致。
4. **扫描保持频率**：训练 `safe_height_scan(scan_update_hz=10)`，50 Hz 控制下每 5 步刷新；
   旧版 `scan_update_interval=2`，每 2 步刷新。
5. **物理选项**：训练 `implicitfast + elliptic + impratio=100`；旧版组合出的模型是
   `Euler + pyramidal + impratio=1`。
6. **判定口径**：训练成功条件允许 calf/shank 贴边，旧版把所有非 foot 接触都算 illegal；
   梅花桩训练允许“完成一排 3 块”，旧版要求 6 块；旧版还没有 stairs/rough 判定。
7. **障碍几何是手写的**：旧版 lane 的 0.18 m 墙、0.15 m 石、0.42 m 沟与训练
   `terrain.py` 的难度映射不是同一套，结论不可比。

修复 1–5 后，同一 checkpoint 从“radar 形状不匹配、无法加载 / 扫描符号反了”变为
4.05 m / 9.78 s 的平地 PASS；说明修复是有效的。修复后仍然过不了障碍，说明**能力缺口在策略**，
不在仿真。

---

## 4. 策略层面的判断

从轨迹看：

- 平地前向速度约 0.4 m/s（命令 0.25 m/s），base 高度从 0.45 m 逐步蹲到 0.28–0.33 m；
- 四轮持续支撑，腿主要在维持姿态/下蹲（动作绝对值均值约 0.2–1.2），没有形成抬腿跨障；
- 遇到 4–6 cm 的垂直障碍就停住或后退，没有抬腿/踩踏动作；
- 梅花桩上会走到边缘后失去平衡，而不是主动把轮子放到下一块石头上。

这与 199 iteration、curriculum stage 0 的 training log 一致：策略只学会了 wheel policy，
obstacle expert 的 residual 还没有被训练出来。

---

## 5. 建议的下一步

1. **不要用这个 checkpoint 继续做 sim2sim 障碍验收**；先完成训练课程。
2. 按 `Robot-Go2W-Flat-v0 → Rough-v0 → Step-v0 → Gap-v0 → HighWall-v0 →
   PlumBlossom-v0 → Mixed-v0` 逐段训练，每段用本框架的 suite 做 gate。
3. 先解决“腿不动”：检查 leg action scale、`feet_air_time` / `foot_clearance` /
   `pose` 奖励，以及 wheel 与 leg 动作的耦合；当前策略在 4 cm 台阶前就停下，说明
   抬腿/支撑行为没有形成。
4. 训练过程中每 50–100 iteration 跑一次 `configs/suites/quick.yaml`，记录
   `PASS/FAIL`、`max_tilt`、`min_z_rel`、`illegal`、`stones_visited`，不要只看 reward。
5. 用 `configs/suites/flat_stress.yaml` 或 `--randomize` 做域随机化；平地已经 5/5 PASS，
   可以继续加宽摩擦/质量/延迟范围。
6. 只有当 `quick.yaml` 至少 5/6 PASS、且 `full.yaml` 在目标难度行稳定通过时，再考虑
   上实机或 ODIN1。

---

## 6. 复现命令

```powershell
$py = "D:\mujoco_test\HighTorque-MiniDog\mujoco\mjlab\.venv\Scripts\python.exe"
cd D:\mujoco_test\sim2sim_next

& $py sim2sim.py run --headless --obstacle flat --duration 15
& $py sim2sim.py eval --suite configs/suites/easiest.yaml --out results\easiest_recheck
& $py sim2sim.py eval --suite configs/suites/full.yaml --out results\full_recheck
& $py sim2sim.py eval --suite configs/suites/flat_stress.yaml --out results\flat_stress_recheck
```
