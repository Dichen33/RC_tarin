$ErrorActionPreference = "Stop"

$env:UV_CACHE_DIR = "D:\uv"

New-Item -ItemType Directory -Force -Path $env:UV_CACHE_DIR | Out-Null

Write-Host "UV_CACHE_DIR=$env:UV_CACHE_DIR"
Write-Host "Creating/updating .venv with CUDA PyTorch source..."

uv sync --extra cu128

Write-Host ""
uv run python -c "import sys, torch; print('python', sys.executable); print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_device_count', torch.cuda.device_count()); print('device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

