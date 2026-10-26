$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

python -m py_compile `
  .\minidog_rl\envs\minidog_env.py `
  .\minidog_rl\scripts\rollout.py `
  .\minidog_rl\scripts\train_sb3.py `
  .\minidog_rl\scripts\eval_sb3.py `
  .\legged_gym\scripts\train.py `
  .\legged_gym\scripts\play.py

python -m minidog_rl.scripts.rollout --steps 300 --vx 0.08 --vy 0.00 --yaw 0.00 --log-every 50
python -m minidog_rl.scripts.rollout --steps 300 --vx 0.00 --vy 0.09 --yaw 0.00 --log-every 50
python legged_gym\scripts\train.py --task=htdw_4438 --headless --num_envs 1 --max_iterations 1 --num_steps_per_env 32 --batch_size 32 --num_learning_epochs 1 --run_name smoke_test --no-zhdb
