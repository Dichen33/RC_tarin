# 原生 MJLab / RSL-RL 入口说明

这份笔记已经被 `TRAINING_TASKS.md` 取代，后者是当前主文档。

现在你只需要记住三件事：

```bat
cd /d D:\mujoco_test\HighTorque-MiniDog\mujoco\mjlab
scripts\uv_setup_native.bat
scripts\train_native.bat Robot-Rough-v0 --env.scene.num-envs 2048 --agent.max-iterations 2000 --agent.run-name rough_v2
scripts\play_native.bat Robot-Rough-v0 --run-name rough_v2 --checkpoint latest
```

默认模型目录：

```text
D:\mujoco_test\HighTorque-MiniDog\mujoco\mjlab\model\<run_name>\<timestamp>_<run_name>
```

如果你想看奖励、观测、动作和 `uv` 管理方式，直接看 [TRAINING_TASKS.md](./TRAINING_TASKS.md)。

