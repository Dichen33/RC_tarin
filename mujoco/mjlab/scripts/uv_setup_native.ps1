$ErrorActionPreference = "Stop"

if (-not $env:UV_CACHE_DIR) {
    $env:UV_CACHE_DIR = "D:\uv"
}
if (-not $env:WARP_CACHE_PATH) {
    $env:WARP_CACHE_PATH = "D:\uv\warp_cache"
}
New-Item -ItemType Directory -Force -Path $env:UV_CACHE_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $env:WARP_CACHE_PATH | Out-Null

Write-Host "UV_CACHE_DIR=$env:UV_CACHE_DIR"
Write-Host "WARP_CACHE_PATH=$env:WARP_CACHE_PATH"
Write-Host "Installing native MJLab + CUDA dependencies..."

uv sync --extra native --extra cu128

Write-Host ""
uv run python scripts\native_check.py

Write-Host ""
uv run python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
