# Native MJLab Task Port Notes

Use this folder to port the Gymnasium residual task into native MJLab after the
local MJLab installation is ready.

Suggested task id:

```text
Mjlab-Locomotion-Flat-HTDWMiniDog-Residual-v0
```

Task contract:

```text
observation:
  projected gravity in body frame: 3
  base rpy: 3
  base linear velocity in body frame: 3
  base angular velocity in body frame: 3
  command vx/vy/yaw_rate: 3
  joint position: 12
  joint velocity: 12
  previous action: 12
action:
  normalized 12 joint residuals over IK nominal targets
reward:
  linear velocity tracking
  yaw-rate tracking
  height tracking
  roll/pitch stability
  action-rate and torque penalties
termination:
  base z too low
  roll/pitch too large
```

Keep the first MJLab version residual-based.  Once it is stable, add a pure
joint-target task by replacing `q_nominal + residual` with direct policy joint
targets and curriculum the command range.
