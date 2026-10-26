# MiniDog Linux Training Package

This directory is a self-contained training package for the native MJLab
HTDW-4438 MiniDog and 16-DoF Unitree Go2W tasks. It is intended to be copied to a Linux server with
an NVIDIA GPU. It contains the task code, locked Python dependencies, the
robot XML and all STL meshes required by the task.

The native training task generates its own curriculum terrain. The ROBOCON
2027 XML is used by the separate `sim2sim` validation package and is not
needed for this training package.

Go2W tasks are registered as `Robot-Go2W-Flat-v0`, `Robot-Go2W-Rough-v0`,
`Robot-Go2W-Step-v0`, `Robot-Go2W-Gap-v0`, `Robot-Go2W-HighWall-v0`,
`Robot-Go2W-PlumBlossom-v0`, and `Robot-Go2W-Mixed-v0`. Their first 12 actions
are leg position targets and their final 4 actions are wheel velocity targets.
The Go2W policy uses a 57-value frame, six-frame proprioception history (342
values), a 403-value height scan, AME radar encoding, and a wheel-first
four-expert MoE. Start Go2W training from `Robot-Go2W-Flat-v0` or run the
single-policy Mixed curriculum. MiniDog checkpoints cannot be reused because
the robot, action semantics, and observation dimensions differ.

The terrain profile is centralized in
`minidog_native/config/course_profiles.py`. The curriculum follows the
reference Go2 task structure: flat/shallow rough walking, standard stairs and
blocks, gap, plum blossom, high wall, then mixed terrain. Each release is
gated by measured success and the command range is increased with the stage;
the initial stage is capped at 0.22 m/s forward speed. The plum-blossom
reward pays once per new cube plus a stable-exit condition (`require_all_stones=False`).
Go2W model parameters are randomized at episode reset, including wheel
traction, joint friction/armature/damping, leg PD gains, effort limits,
inertial scale, and encoder bias. The actor receives the biased joint-position
observation, while the radar stream retains update-rate, noise, and dropout
simulation for sim2sim robustness.

## Server prerequisites

- Linux x86_64
- NVIDIA driver with a CUDA-capable GPU
- `curl` for automatic uv installation when uv is not already installed
- Network access to PyPI, the PyTorch CUDA index, and the MJLab package index

The default setup uses Python 3.10. Override it with `PYTHON_VERSION=3.11`
or another version supported by the locked dependencies.

The scripts require an NVIDIA GPU by default. If the server is only being used
for CPU installation or debugging, set `REQUIRE_CUDA=0`; this is not suitable
for normal training.

## Install

From this directory:

```bash
bash setup_linux.sh
```

The script creates `.venv`, installs native MJLab with CUDA-enabled PyTorch,
registers all native MiniDog and Go2W tasks, and checks CUDA availability. It stores uv
and Warp caches inside this directory by default, so the directory remains
portable. Set `UV_CACHE_DIR` or `WARP_CACHE_PATH` to use a shared server cache.
On mainland-China networks, the setup script uses the Aliyun PyPI mirror for
ordinary packages by default. The explicit CUDA/Warp/MuJoCo indexes remain
unchanged. Override it with `PYPI_INDEX_URL=https://pypi.org/simple/` when
the mirror is unavailable. Setup installs from the committed `uv.lock` without
re-resolving dependencies; use `uv lock --refresh` explicitly when changing
the package index or dependency versions.

## Train

Smoke test:

```bash
bash train_plum.sh \
  --env.scene.num-envs 64 \
  --agent.max-iterations 20 \
  --agent.run-name plum_smoke \
  --agent.logger tensorboard \
  --agent.upload-model False
```

Go2W flat-wheel smoke run:

```bash
bash train.sh Robot-Go2W-Flat-v0 \
  --env.scene.num-envs 64 \
  --agent.max-iterations 20 \
  --agent.run-name go2w_flat_smoke \
  --agent.logger tensorboard \
  --agent.upload-model False
```

Server training example:

```bash
bash train_plum.sh \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 10000 \
  --agent.run-name plum_v1 \
  --agent.logger tensorboard \
  --agent.upload-model False
```

Recommended one-model Go2W training (one PPO run with adaptive mixed terrain):

```bash
uv run --locked --extra native --extra cu128 python scripts/train_go2w_unified.py \
  --num-envs 2048 \
  --iterations 132000 \
  --run-name go2w_unified \
  --logger tensorboard
```

`132000` is the sum of the seven stage budgets in
`go2w_native/config/stage_profiles.py` (15k+15k+12k+18k+20k+22k+30k). It is a
long-horizon ceiling that matches the old staged total, not a requirement.
For a focused plum-blossom model, use `Robot-Go2W-PlumBlossom-v0` with 300
iterations first; `go2w_mixed_env_cfg()` is currently identical to the plum
configuration except for `residual_scale` (0.40 vs 0.35).

This uses one actor, one critic, and one optimizer. The terrain curriculum
releases flat/rough walking, stairs, gaps, high walls, and plum-blossom terrain
by measured success and learning progress. The seven-stage launcher remains
available for controlled ablations:

```bash
uv run python scripts/train_go2w_curriculum.py --dry-run
```

Short server validation run (48 GiB GPU):

```bash
bash quick_go2w_test.sh
```

The default is `2048` parallel environments and `200` PPO updates (about
`9,830,400` environment steps with the current 24-step rollout). Override
without editing the package, for example:

```bash
GO2W_QUICK_NUM_ENVS=1024 GO2W_QUICK_ITERATIONS=50 bash quick_go2w_test.sh
```

This short run checks the full Mixed-task training path. It is separate from
`check_linux.sh`, whose default smoke test uses 64 environments and 4 physics
steps to validate startup and observation/action contracts.

The generic launcher can select another registered task:

```bash
bash train.sh Robot-Rough-v0 \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 10000 \
  --agent.run-name rough_v1 \
  --agent.logger tensorboard \
  --agent.upload-model False
```

Training outputs are written below `model/<run-name>/`. To resume, use the
native MJLab checkpoint/resume options supported by the installed MJLab
version, while keeping the same task and observation contract. Each training
run also writes `metrics.csv` beside the TensorBoard event file. It contains
one row per PPO iteration, including mean reward, episode length, losses, and
terrain-specific success metrics.

To rank saved checkpoints by obstacle success, run:

```bash
uv run python scripts/rank_checkpoints.py \
  model/plum_v1/<timestamp> \
  --top 15
```

For an older run that only has TensorBoard events, first convert the events:

```bash
uv run python scripts/tensorboard_to_csv.py model/plum_v1/<timestamp>
```

The ranking is a first-pass filter. The selected checkpoints should still be
verified in sim2sim on a fixed map, because the best training reward is not
necessarily the best obstacle completion policy.

## Verify

```bash
bash check_linux.sh
```

Expected task registration includes:

```text
Robot-Flat-v0
Robot-Rough-v0
Robot-Crawl-v0
Robot-PlumBlossom-v0
Robot-Go2W-Flat-v0
Robot-Go2W-Rough-v0
Robot-Go2W-Step-v0
Robot-Go2W-Gap-v0
Robot-Go2W-HighWall-v0
Robot-Go2W-PlumBlossom-v0
Robot-Go2W-Mixed-v0
```

The package does not include `.venv`, `.cache`, `model`, `logs`, or TensorBoard
outputs. Those are created locally on the server and should not be copied from
Windows.

`check_linux.sh` performs both registration checks and a real 4-environment
reset/step test for the Go2W Mixed task by default. It verifies the actor and
critic observations, action dimensions, terrain registration, sensor
execution, domain-randomization initialization, and finite rewards. Set
`SMOKE_TASK=Robot-PlumBlossom-v0` to validate the MiniDog task instead. Use
`REQUIRE_CUDA=0 bash check_linux.sh` for a CPU-only debug check; use the
default `bash check_linux.sh` on the training server.

Use the Tsinghua PyPI mirror for one check invocation only:

```bash
PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/ bash check_linux.sh
```

This only affects that shell process and does not modify the lock file.

For a direct smoke test without the wrapper:

```bash
uv run python scripts/smoke_test.py --num-envs 4 --steps 2
```

## Curriculum gates

Robot-PlumBlossom-v0 now starts with only flat and shallow rough terrain.
The following stages are released only after enough completed episodes meet
their success thresholds:

    flat/rough -> stairs/blocks/slopes -> gap -> plum blossom -> high wall -> mixed

The gates use per-terrain exponential success statistics, not a fixed training
iteration. A failed episode lowers that terrain's row; a successful episode
raises it up to the current stage cap. The base locomotion gate requires at
least 2048 base-terrain episodes and a 75% aggregate success rate before any
obstacle terrain can be sampled.

Plum-blossom completion requires reaching the exit distance with a stable
attitude (`require_all_stones=False`: visiting every cube is no longer required,
because the sparse course reward already pays once per new cube). The
high-wall lane uses full-width transverse walls so walking around an edge is
not counted as passing the obstacle.

These changes require a new training run. Existing plum_gap_v1 checkpoints
were trained with the older fixed-step curriculum and distance-only success
metric and should not be resumed as the final obstacle policy.
