#requires -Version 5.1
<#
.SYNOPSIS
    Install sim2sim_next into the original HighTorque-MiniDog sim2sim folder.

.DESCRIPTION
    The original folder is owned by an old sandbox account and the current
    user has no WRITE_DAC, so a normal process cannot create files there.
    Run this script from an elevated PowerShell (Run as Administrator).

    It backs up the original folder to D:\mujoco_test, takes ownership,
    grants the current user full control, preserves the old Python entry
    points under legacy\, and copies the modular project in.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\tools\install_into_original.ps1
#>
param(
    [string]$Source = (Split-Path $PSScriptRoot -Parent),
    [string]$Target = "D:\mujoco_test\HighTorque-MiniDog\mujoco\sim2sim",
    [switch]$RunSelfcheck
)

$ErrorActionPreference = "Stop"

function Test-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Admin)) {
    Write-Host "Requesting administrator privileges (UAC) ..."
    $argList = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$PSCommandPath`""
    )
    foreach ($key in $PSBoundParameters.Keys) {
        $value = $PSBoundParameters[$key]
        if ($value -is [switch]) {
            if ($value.IsPresent) { $argList += "-$key" }
        }
        else {
            $argList += "-$key"
            $argList += "`"$value`""
        }
    }
    try {
        Start-Process -FilePath "powershell.exe" -Verb RunAs -ArgumentList $argList -Wait | Out-Null
    }
    catch {
        Write-Host "Elevation was cancelled or failed: $_"
        exit 1
    }
    exit 0
}

$principalName = [Security.Principal.WindowsIdentity]::GetCurrent().Name
Write-Host "Source: $Source"
Write-Host "Target: $Target"
Write-Host "User:   $principalName"

if (-not (Test-Path -LiteralPath $Source)) { throw "Source not found: $Source" }
if (-not (Test-Path -LiteralPath $Target)) { throw "Target not found: $Target" }
if (-not (Test-Path -LiteralPath (Join-Path $Source "dogsim\cli.py"))) {
    throw "Source does not look like sim2sim_next (missing dogsim\cli.py)"
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backup = Join-Path "D:\mujoco_test" "sim2sim_backup_$stamp"
Write-Host "Backing up to $backup ..."
Copy-Item -LiteralPath $Target -Destination $backup -Recurse -Force

Write-Host "Taking ownership ..."
& takeown.exe /F $Target /R /D Y | Out-Null
if ($LASTEXITCODE -ne 0) { throw "takeown failed with exit code $LASTEXITCODE" }

Write-Host "Granting full control to $principalName ..."
& icacls.exe $Target /grant "$($principalName):(OI)(CI)F" /T | Out-Null
if ($LASTEXITCODE -ne 0) { throw "icacls failed with exit code $LASTEXITCODE" }

# Keep the old entry points for reference; do not delete user files.
$legacy = Join-Path $Target "legacy"
New-Item -ItemType Directory -Path $legacy -Force | Out-Null
foreach ($name in @("policy.py", "scene.py", "sim_config.py", "ui.py", "navigation.py", "1.md", "README.md")) {
    $path = Join-Path $Target $name
    if (Test-Path -LiteralPath $path) {
        Move-Item -LiteralPath $path -Destination (Join-Path $legacy $name) -Force
        Write-Host "  legacy: $name"
    }
}

# Copy the modular project.  config.yaml / config_minidog.yaml are kept in
# place because the new CLI can read the legacy schema.
Copy-Item -LiteralPath (Join-Path $Source "dogsim") -Destination $Target -Recurse -Force
Copy-Item -LiteralPath (Join-Path $Source "configs") -Destination $Target -Recurse -Force
Copy-Item -LiteralPath (Join-Path $Source "tests") -Destination $Target -Recurse -Force
if (Test-Path -LiteralPath (Join-Path $Source "assets")) {
    Copy-Item -LiteralPath (Join-Path $Source "assets") -Destination $Target -Recurse -Force
}
if (Test-Path -LiteralPath (Join-Path $Source "model")) {
    Copy-Item -LiteralPath (Join-Path $Source "model") -Destination $Target -Recurse -Force
}
foreach ($name in @("sim2sim.py", "sim2sim.cmd", "course.cmd", "README.md", "FINDINGS.md", "MIGRATION.md", "pyproject.toml", ".gitignore")) {
    $path = Join-Path $Source $name
    if (Test-Path -LiteralPath $path) {
        Copy-Item -LiteralPath $path -Destination $Target -Force
    }
}
if (Test-Path -LiteralPath (Join-Path $Source "results")) {
    Copy-Item -LiteralPath (Join-Path $Source "results") -Destination $Target -Recurse -Force
}

Write-Host ""
Write-Host "Installed into $Target"
Write-Host "Backup: $backup"
Write-Host "Old entry points: $legacy"
Write-Host ""

$python = "D:\mujoco_test\HighTorque-MiniDog\mujoco\mjlab\.venv\Scripts\python.exe"
Write-Host "Next:"
Write-Host "  cd `"$Target`""
Write-Host "  & `"$python`" sim2sim.py selfcheck"
Write-Host "  & `"$python`" sim2sim.py eval --suite configs/suites/quick.yaml --out results\quick"

if ($RunSelfcheck -and (Test-Path -LiteralPath $python)) {
    Push-Location $Target
    try {
        & $python sim2sim.py selfcheck
    }
    finally {
        Pop-Location
    }
}
