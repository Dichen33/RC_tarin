@echo off
setlocal
rem Works both in the original sim2sim folder and in sim2sim_next.
set "PY="
if defined MJLAB_PY set "PY=%MJLAB_PY%"
if not defined PY if exist "%~dp0..\mjlab\.venv\Scripts\python.exe" set "PY=%~dp0..\mjlab\.venv\Scripts\python.exe"
if not defined PY if exist "%~dp0..\HighTorque-MiniDog\mujoco\mjlab\.venv\Scripts\python.exe" set "PY=%~dp0..\HighTorque-MiniDog\mujoco\mjlab\.venv\Scripts\python.exe"
if not defined PY if exist "%~dp0..\..\mjlab\.venv\Scripts\python.exe" set "PY=%~dp0..\..\mjlab\.venv\Scripts\python.exe"
if not defined PY (
    echo Python not found. Set MJLAB_PY to the MJLab venv python.exe.
    exit /b 1
)
"%PY%" "%~dp0sim2sim.py" %*
exit /b %ERRORLEVEL%
