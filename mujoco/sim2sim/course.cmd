@echo off
setlocal
call "%~dp0sim2sim.cmd" run --config "%~dp0configs\course.yaml" %*
exit /b %ERRORLEVEL%
