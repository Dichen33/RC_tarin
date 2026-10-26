# Go2W 16-DoF Training Plan

This is the current training contract for the four-wheel-leg Go2W task.

## Policy and action contract

- Actor input: six-frame proprioception history (`6 x 59 = 354`) plus a `31 x 13 = 403` local height scan.
- Actor: AME state encoder and radar encoder, followed by a wheel-first Soft-MoE.
- Experts: `wheel`, `step`, `gap`, `recovery`.
- Action order: 12 leg position actions (`FR, FL, RR, RL`, each `hip/thigh/calf`) followed by four wheel velocity actions (`FR, FL, RR, RL`).
- The wheel expert is the base action. Obstacle experts add a bounded residual, and action smoothing/rate limiting is applied after blending.
- The critic receives the clean instantaneous scan. The actor scan has ODIN1-like 10 Hz update, noise, and dropout simulation.

## Curriculum order

Train and validate each stage before moving to the next one:

1. `Robot-Go2W-Flat-v0`: wheel locomotion and stable standing.
2. `Robot-Go2W-Rough-v0`: shallow rough ground while keeping `feet_air_time` disabled.
3. `Robot-Go2W-Step-v0`: low stairs and blocks; gated leg swing is released.
4. `Robot-Go2W-Gap-v0`: focused ditch curriculum, starting below the XML medium gap.
5. `Robot-Go2W-HighWall-v0`: focused wall curriculum, starting below the XML medium wall.
6. `Robot-Go2W-PlumBlossom-v0`: stepping-stone course.
7. `Robot-Go2W-Mixed-v0`: adaptive mixed course after all focused tasks are stable.

The terrain rows begin below the `ROBOCON2027.xml` medium reference and extend beyond it. A terrain type is promoted only after a measured success window; it is downgraded when the window success rate falls. This prevents a single lucky rollout from releasing a high wall or a deep gap.

## Single-run alternative

The seven-stage process is not the only valid design. `Robot-Go2W-Mixed-v0`
already contains all terrain families and the ETH-style per-terrain release
gates. It now also maintains a bounded LP-ACRL task table over terrain family
and difficulty level. The sampler uses normalized progress and completion,
while release gates still protect the wheel/rough baseline. It can therefore
train one AME+Soft-MoE actor/critic with one optimizer in a single run:

```powershell
.venv\Scripts\python scripts\train_go2w_unified.py --iterations 132000 --num-envs 2048
```

This is one continuous PPO run, not seven checkpoints. The curriculum and LP
sampler state are stored in each checkpoint, so an interrupted run resumes
without resetting terrain probabilities. Do not disable the terrain release
gates or the history revisit quota. The policy should be accepted only after
evaluating each terrain family separately; a high mixed reward alone is not
sufficient.

## Example commands

Run from `D:\mujoco_test\HighTorque-MiniDog\mujoco\mjlab`:

```powershell
.venv\Scripts\python scripts\native_check.py
.venv\Scripts\python scripts\train_native.py Robot-Go2W-Flat-v0 --env.scene.num-envs 2048 --agent.max-iterations 10000 --agent.logger tensorboard --agent.upload-model False
.venv\Scripts\python scripts\train_native.py Robot-Go2W-Rough-v0 --env.scene.num-envs 2048 --agent.max-iterations 8000 --agent.logger tensorboard --agent.upload-model False
.venv\Scripts\python scripts\train_native.py Robot-Go2W-Step-v0 --env.scene.num-envs 2048 --agent.max-iterations 10000 --agent.logger tensorboard --agent.upload-model False
.venv\Scripts\python scripts\train_native.py Robot-Go2W-Gap-v0 --env.scene.num-envs 2048 --agent.max-iterations 12000 --agent.logger tensorboard --agent.upload-model False
.venv\Scripts\python scripts\train_native.py Robot-Go2W-HighWall-v0 --env.scene.num-envs 2048 --agent.max-iterations 15000 --agent.logger tensorboard --agent.upload-model False
.venv\Scripts\python scripts\train_native.py Robot-Go2W-Mixed-v0 --env.scene.num-envs 2048 --agent.max-iterations 20000 --agent.logger tensorboard --agent.upload-model False

For the complete ordered run, use the stage orchestrator. It creates an
independent run directory and manifest per stage, passes the previous actor
checkpoint explicitly, and resets Adam/iteration when the stage profile says
to do so:

```powershell
.venv\Scripts\python scripts\train_go2w_curriculum.py --dry-run
.venv\Scripts\python scripts\train_go2w_curriculum.py --output-root D:\mujoco_test\training_logs\go2w_curriculum --require-evaluation
```

The external sim/sim2sim evaluator should write one
`stage_evaluation.json` per stage with at least:

```json
{"success_rate": 0.72, "fall_rate": 0.08, "illegal_collision_rate": 0.03}
```

Use `scripts/evaluate_go2w_suite.py --enforce-gates` to audit those files.
```

When the default `model` directory is not writable, add `--log-root D:\mujoco_test\training_logs`.

## Acceptance metrics

Do not select a checkpoint by mean reward alone. Track, per terrain family:

- course success rate and fall/illegal-contact rate;
- final displacement and average commanded speed;
- wheel velocity tracking and leg activity while the course is flat;
- `Policy/wheel_expert_weight`, `Policy/step_expert_weight`, `Policy/gap_expert_weight`, `Policy/recovery_expert_weight`;
- `Policy/residual_norm` and `Policy/action_delta`.
- `Policy/radar_valid_fraction` and LP metrics (`lp_stage`,
  `lp_effective_tasks`, `lp_max_probability`, `lp_mean_abs_progress`,
  `lp_history_revisit_probability`).

The diagnostics are written to `metrics.csv` and TensorBoard by `CsvMetricsRunner`. A focused task should be stable before its terrain is released in `Mixed`.

## Obstacle success events

An episode is not successful merely because the base passes an x-coordinate:

- All obstacles: the exit must have positive forward command and velocity, valid body height and tilt, bounded vertical velocity, at least one foot/wheel supporting the terrain, and no base/trunk/thigh/shank terrain contact. Illegal body/leg contact is remembered for the whole episode, so hitting an obstacle and recovering cannot be scored as success.
- Gap: the robot must pass the gap exit with the common stable-exit test and hold that condition for five policy steps.
- High wall: the same stable-exit test is applied after the last wall; falling, stopping, airborne drift, or collision-assisted progress is not success.
- Plum blossom: all six stepping stones must each receive a valid top-surface foot contact for at least two consecutive policy steps, then the robot must pass the exit stable test for five policy steps. The foothold term already supplies this five-step exit hold, so it is not counted twice.

These events are used by both the one-time completion reward and the adaptive terrain curriculum. The curriculum therefore promotes a terrain only after a real obstacle completion, not after a collision-assisted displacement.

## Sim2sim and ODIN1 follow-up

Before real deployment, randomize wheel friction, joint friction, motor response, action delay, control frequency, mass/COM, scan delay, scan noise, and dropout. Keep the same scan normalization and frame convention for simulation and the future ODIN1 adapter. Real IMU orientation must be validated against the high-frequency odometry/fused estimate rather than assuming the raw driver quaternion is a physical attitude signal.

The unified task now implements the LP-ACRL terrain/level sampler, critic-only
terrain context, wheel-friction/inertia randomization, and a radar return-validity
channel. The current actor still consumes the simulated height scan rather than
an RGB camera tensor; camera fusion and a full concurrent teacher policy remain
separate follow-up work because they require a fixed ODIN1 image/point-cloud
interface and deployment-side preprocessing contract.
