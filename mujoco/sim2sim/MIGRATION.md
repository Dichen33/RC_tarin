# 从旧 sim2sim 迁移

## 0. 当前状态（2026-10-19 已合并）

```
主运行目录：D:\mujoco_test\HighTorque-MiniDog\mujoco\sim2sim
源副本：    D:\mujoco_test\sim2sim_next
备份：      D:\mujoco_test\sim2sim_backup_20261019_233908
旧入口：    D:\mujoco_test\HighTorque-MiniDog\mujoco\sim2sim\legacy
```

原目录已经通过管理员脚本完成合并、取得所有权并授予当前用户完全控制。
`selfcheck` 和 `quick` suite 都已在原目录直接跑通。

## 1. 背景：原目录权限问题

原目录：

```
D:\mujoco_test\HighTorque-MiniDog\mujoco\sim2sim
```

合并前，Windows 权限把 `sim2sim` 及其 `generated` 子目录设置成旧 sandbox 账户所有，
当前登录用户没有 `WRITE_DAC`，因此连新建文件都会被拒绝：

```
PermissionError: [Errno 13] Permission denied:
  ...\sim2sim\generated\minidog_obstacle_course.xml
```

这不是代码问题，也不是 DSH 的 workspace 策略（`D:\mujoco_test` 根目录可写）。合并前的
完整重构放在：

```
D:\mujoco_test\sim2sim_next
```

现在它作为源副本保留，方便重新同步；原目录是主运行位置。

---

## 2. 推荐用法：直接运行原目录

```powershell
$py = "D:\mujoco_test\HighTorque-MiniDog\mujoco\mjlab\.venv\Scripts\python.exe"
cd D:\mujoco_test\HighTorque-MiniDog\mujoco\sim2sim
& $py sim2sim.py selfcheck
& $py sim2sim.py eval --suite configs/suites/quick.yaml --out results\quick
```

`configs/go2w.yaml` 的路径解析对“原目录”和“sim2sim_next 源副本”都做了回退：
机器人 XML 通过 profile hints 找到 `../mjlab/...`，checkpoint 通过文件名在
`D:\mujoco_test` 及祖先目录中找到 `model_199.pt`，`ROBOCON2027.xml` 也按祖先目录回退。
如果移动整个项目，仍建议用 `--checkpoint` 和 `robot.xml` 覆盖。

---

## 3. 重新同步/安装到原目录

推荐用已经准备好的安装脚本。它会先把旧目录备份到
`D:\mujoco_test\sim2sim_backup_<时间戳>`，再取得所有权、授权当前用户、把旧入口文件
移到 `legacy\`，最后把新项目复制进去。脚本会自己请求 UAC，不需要手动开管理员窗口。

在 **cmd.exe** 里执行这一行（不要按回车换行，也不要用 PowerShell 的反引号续行）：

```bat
D:\mujoco_test\sim2sim_next\tools\install_into_original.cmd -RunSelfcheck
```

看到 UAC 弹窗时选“是”。等价的一行 PowerShell 命令是：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\mujoco_test\sim2sim_next\tools\install_into_original.ps1" -RunSelfcheck
```

安装完成后，原目录 `D:\mujoco_test\HighTorque-MiniDog\mujoco\sim2sim` 就是新的积木式项目；
旧文件在 `legacy\`，旧 `config.yaml` / `config_minidog.yaml` 仍然保留，新 CLI 可以读取。

如果你不想运行脚本，也可以手动操作。先备份旧目录：

```powershell
$old = "D:\mujoco_test\HighTorque-MiniDog\mujoco\sim2sim"
Copy-Item $old "$old.backup_20261019" -Recurse
```

然后解决 `sim2sim` 的 ACL。两种方式：

### 方式 A：管理员 PowerShell 取得所有权（推荐，一次解决）

```powershell
# 以管理员身份打开 PowerShell
takeown /F "D:\mujoco_test\HighTorque-MiniDog\mujoco\sim2sim" /R /D Y
icacls "D:\mujoco_test\HighTorque-MiniDog\mujoco\sim2sim" /grant "$env:USERNAME:(OI)(CI)F" /T
```

完成后当前用户对 `sim2sim`、`generated` 有完全控制，新代码可以写 `generated`。

### 方式 B：不解决 ACL，只把新项目放在旁边

保持 `D:\mujoco_test\sim2sim_next` 独立；评估结果和场景都写到新目录，完全不碰旧目录。
这也是本次交付采用的方式。

### 合并文件

```powershell
$src = "D:\mujoco_test\sim2sim_next"
$dst = "D:\mujoco_test\HighTorque-MiniDog\mujoco\sim2sim"

Copy-Item "$src\dogsim"            $dst -Recurse -Force
Copy-Item "$src\configs"           $dst -Recurse -Force
Copy-Item "$src\tests"             $dst -Recurse -Force
Copy-Item "$src\sim2sim.py"        $dst -Force
Copy-Item "$src\README.md"         $dst -Force
Copy-Item "$src\FINDINGS.md"       $dst -Force
Copy-Item "$src\MIGRATION.md"      $dst -Force
```

旧文件 `policy.py`、`scene.py`、`sim_config.py`、`ui.py`、`navigation.py` 可以保留；
新的 `sim2sim.py` 只 import `dogsim`，不会使用它们。旧 `config.yaml` / `config_minidog.yaml`
仍然可以被新 CLI 的 legacy 适配器读取。

### 合并后的路径调整

- `configs/go2w.yaml` 的 `policy.checkpoint` 是 `../../model_199.pt`；合并后从
  `sim2sim/configs` 解析为 `D:\mujoco_test\HighTorque-MiniDog\mujoco\model_199.pt`，
  不存在。直接用 `--checkpoint D:\mujoco_test\model_199.pt`，或把该行改成绝对路径。
- `configs/course.yaml` 的 `params.xml` 同理，可直接传
  `--course D:\mujoco_test\ROBOCON2027.xml`。
- 旧目录的 `generated` 如果仍不可写，用 `--scene D:\mujoco_test\...\scene.xml`
  指向一个可写路径。

---

## 4. 旧版必须修的问题（新版本已修）

| # | 旧版 | 训练契约 / 新版 |
|---|---|---|
| 1 | radar conv 3 通道 | checkpoint 是 4 通道（height + validity + 2 coords） |
| 2 | `origin[2] - distance - base[2]` = `-distance` | `frame_z - hit_z` = `distance` |
| 3 | 3 个 expert 全部加权，residual_scale 0.35 | wheel + argmax residual，按选中权重归一化，residual_scale 0.40，clamp ±4 |
| 4 | scan 每 2 步刷新 | 10 Hz，50 Hz 控制下每 5 步刷新 |
| 5 | Euler/pyramidal/impratio=1 | implicitfast/elliptic/impratio=100 |
| 6 | 所有非 foot 接触算 illegal | base/thigh/trunk 才算；calf/shank 允许 |
| 7 | 梅花桩必须 6 块 | 默认一排 3 块（可配 `plum_require_all_stones: true`） |
| 8 | 没有 stairs/rough 判定 | `stairs`/`rough` brick + evaluator |
| 9 | 手写 lane 尺寸 | 与训练 `terrain.py` 的难度映射对齐 |
| 10 | 无批量报告 | `eval` 输出 JSON/CSV/Markdown |

---

## 5. 兼容性

- 旧命令行参数：`--config --checkpoint --headless --spawn --command --duration --scene
  --pass-distance --build-only --no-panel --course` 都接受；
- 旧 `config.yaml` 字段（`robot_xml`, `control_decimation`, `scan_*`, `randomization`,
  `spawn`, `command`）会被自动转换成新配置；
- MiniDog 12-DoF 的 MLP checkpoint 路径可用（`configs/minidog.yaml`），但当前手头的
  `plum_v2` checkpoint 在旧版和新版中都不前进，不能作为正向对照；需要先核对它的训练契约。
- `--no-panel` 只是兼容参数；新运行时没有 pygame 面板，查看器用键盘 W/S/A/D/Q/E/X/R。
