param(
    [int]$NumEnvs = 8,
    [int]$Iterations = 2000,
    [string]$RunName = "rough_v1"
)

$ErrorActionPreference = "Stop"

$env:UV_CACHE_DIR = "D:\uv"

New-Item -ItemType Directory -Force -Path $env:UV_CACHE_DIR | Out-Null

uv run python legged_gym\scripts\train.py `
    --task=Robot-Rough-v0 `
    --headless `
    --device cuda `
    --num_envs $NumEnvs `
    --max_iterations $Iterations `
    --num_steps_per_env 24 `
    --run_name $RunName

