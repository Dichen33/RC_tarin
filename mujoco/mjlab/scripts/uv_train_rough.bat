@echo off
setlocal
cd /d "%~dp0.."

set NUM_ENVS=%~1
set ITERATIONS=%~2
set RUN_NAME=%~3

if "%NUM_ENVS%"=="" set NUM_ENVS=8
if "%ITERATIONS%"=="" set ITERATIONS=2000
if "%RUN_NAME%"=="" set RUN_NAME=rough_v1

powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\uv_train_rough.ps1" -NumEnvs %NUM_ENVS% -Iterations %ITERATIONS% -RunName "%RUN_NAME%"
endlocal
