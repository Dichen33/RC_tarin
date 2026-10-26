@echo off
if "%UV_CACHE_DIR%"=="" set "UV_CACHE_DIR=D:\uv"
if "%WARP_CACHE_PATH%"=="" set "WARP_CACHE_PATH=D:\uv\warp_cache"
if not exist "%UV_CACHE_DIR%" mkdir "%UV_CACHE_DIR%"
if not exist "%WARP_CACHE_PATH%" mkdir "%WARP_CACHE_PATH%"
uv run python "%~dp0play_native.py" %*
