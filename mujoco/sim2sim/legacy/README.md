# Go2W / MiniDog sim2sim

The MuJoCo runtime is configuration driven, following the reference
project's deployment layout. `config.yaml` owns the control period, action
scale, default pose, joint/actuator mapping, observation term sizes, scan
geometry, command and spawn defaults. Use `--config another.yaml` to validate
another robot-compatible checkpoint without editing Python files. When
`scan_size` is `null`, the grid is selected from the checkpoint: 216 values
uses the legacy 1.4 x 0.9 m grid and 403 values uses the 2.4 x 1.0 m grid.

The default configuration runs the 16-DoF Unitree Go2W actor in a standalone MuJoCo scene. `config_minidog.yaml` remains available for the legacy 12-DoF MiniDog actor. Both scenes combine:

- `go2w/go2w_description.xml` for the Go2W body, joints and mixed actuators;
- `D:\mujoco_test\ROBOCON2027.xml` for the ROBOCON 2027 obstacle-course map.
- training lanes for `rough + plum blossom`, independent flat high walls, and low bars.

The Go2W action order is 12 leg position targets followed by 4 wheel velocity targets, matching the Unitree reference deployment. The loader supports the legacy MiniDog actor and the terrain-aware AME actor. Go2W uses six history frames of a 59-value proprioceptive group (354 values total); MiniDog uses 47 values (282 total). Both use the current 403-value height map with the AME spatial encoder. The loader detects the scan geometry (`2.4 x 1.0 m`) and keeps compatibility with old 216-point (`1.4 x 0.9 m`) checkpoints.

The Go2W validation scene uses the training actuator gains (`leg kp=25`, `leg kd=0.5`, `wheel kd=0.15`, effort limit `23.5`) and reads the Go2W joint limits from its MJCF when composing position actuators. Current Go2W checkpoints with `wheel_policy/router/residual_experts` are loaded as the same four-expert residual MoE used in training; action smoothing and rate limiting are reproduced before actuator targets are sent to MuJoCo.

`config.yaml` enables reset-time sim2sim randomization for friction, damping,
actuator gain, mass/COM, motor response, action delay, control-period jitter,
IMU/encoder noise, and scan noise/dropout. Set `randomization.enabled: false`
for a deterministic nominal run. The evaluator requires stable upright motion
for five control steps, support contact, no illegal body collision, and
obstacle-specific completion (all six plum blocks or all three high walls),
rather than accepting distance alone.

## Run

From this directory, using the already prepared MJLab environment. In `cmd.exe`, run without the PowerShell `&` operator:

```bat
..\mjlab\.venv\Scripts\python.exe sim2sim.py --checkpoint ..\mjlab\model\go2w_plum_blossom\<timestamp>\model_1999.pt --command 0.25 0 0 --duration 20
```

In PowerShell, the equivalent command is:

```powershell
& ..\mjlab\.venv\Scripts\python.exe sim2sim.py --checkpoint ..\mjlab\model\go2w_plum_blossom\<timestamp>\model_1999.pt --command 0.25 0 0 --duration 20
```

For a non-graphical validation report:

```powershell
& ..\mjlab\.venv\Scripts\python.exe sim2sim.py --headless --checkpoint ..\mjlab\model\go2w_plum_blossom\<timestamp>\model_1999.pt --command 0.25 0 0 --duration 10
```

The default start is the ROBOCON start zone `(x,y,z,yaw)=(0,0,0.262,0)`. The supplied map follows the PDF's eight obstacle classes:

- `L_pile`: L-shaped slalom piles, 100 mm diameter x 100 mm high, 900 mm centre spacing;
- `limit_frame`: 800 x 800 mm frame with about 290 mm clearance;
- `high_wall_00`: 800 x 80 x 300 mm wall;
- `stairs_07`, `platform_08/09`, `ramp_10`: 20 degree ramp, 320 mm steps and 800 x 800 mm platforms;
- `pit_wall_*` and `gravel_*`: L-shaped 砖木坑 representation;
- `tire_ring_05/06`: 2 x 4 tire array;
- `stepping_stones_01` with `platform_02/03`: ROBOCON plum-blossom blocks and two side platforms;
- `platform_04`: 3 m x 400 mm single wooden bridge.

The PDF specifies an obstacle-course limit of 180 seconds and requires every plum-blossom block to receive at least one foot contact. The XML's unused `terrain.png` hfield declaration is removed during composition; the actual competition obstacles are the explicit geoms above, so the missing image does not change the obstacle geometry. Use `--spawn X Y Z YAW` or the control-panel scenario buttons to test a specific obstacle. The added rough + plum-blossom lane is on the north side at `y=3.35`; the independent high-wall lane is at `y=0.85` and the low-bar lane at `y=-0.85`.

The normal viewer launch also opens a pygame control panel modeled after the reference `RC_Legged` sim2sim. It provides buttons for obstacle-lane selection, `vx/vy/yaw` increments, start/pause/stop/reset, and simulation rate `0.5x/1x/2x`. The right-side map supports one-click automatic navigation. Enable `鼠标画线`, drag a path on the map, and release the mouse to convert it into waypoints; the controller follows each waypoint and stops at the final point. `取消导航` returns to manual velocity control. Use `--no-panel` to run only the MuJoCo viewer.

Set `--duration 0` for unlimited runtime. In viewer mode, close the MuJoCo window to exit. In headless mode, press `Ctrl+C`; a detected fall still stops the run so the failure can be inspected.

To test the added rough + plum-blossom lane after starting at its entrance, use:

```powershell
& ..\mjlab\.venv\Scripts\python.exe sim2sim.py --headless --spawn -2.9 3.35 0.262 0 --command 0.20 0 0 --duration 15
```

For the beam/wall/stepping-stone lane, a useful initial test is:

```powershell
& ..\mjlab\.venv\Scripts\python.exe sim2sim.py --headless --spawn 1.0 2.0 0.262 0 --command 0.20 0 0 --duration 15
```

Use `--build-only` to validate XML composition without running a policy. In the viewer, `W/S` changes forward speed, `A/D` changes lateral speed, `Q/E` changes yaw speed, `X` stops, and `R` resets.

## Important compatibility note

The policy must have the same action and observation contract as the MJLab run that produced it. Go2W checkpoints use 16 actions, `actor_proprio=(6 x 59)`, and `actor_radar=(31 x 13)`. MiniDog checkpoints use 12 actions and `actor_proprio=(6 x 47)`. The program stops with an explicit error when the checkpoint does not match the configured contract instead of silently feeding a misordered observation.
