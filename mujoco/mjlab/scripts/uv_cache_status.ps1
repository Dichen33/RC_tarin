$ErrorActionPreference = "Stop"

$env:UV_CACHE_DIR = "D:\uv"

New-Item -ItemType Directory -Force -Path $env:UV_CACHE_DIR | Out-Null

Write-Host "UV_CACHE_DIR=$env:UV_CACHE_DIR"
uv cache dir
uv cache size

