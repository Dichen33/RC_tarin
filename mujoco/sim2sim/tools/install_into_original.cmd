@echo off
setlocal
set "SCRIPT=%~dp0install_into_original.ps1"
if not exist "%SCRIPT%" (
    echo Missing installer: "%SCRIPT%"
    pause
    exit /b 1
)
echo Starting modular sim2sim installer...
echo A UAC prompt will appear; choose "Yes" to allow the permission repair.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" %*
set "CODE=%ERRORLEVEL%"
if not "%CODE%"=="0" (
    echo.
    echo Installer failed with exit code %CODE%.
    pause
)
endlocal
