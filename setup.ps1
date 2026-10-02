#!/usr/bin/env pwsh
# One-shot project setup for Windows (PowerShell 5.1+ or PowerShell 7+).
# Creates the venv, installs Python deps, and fetches the large CAD meshes
# from the project's GitHub Release.
#
# Usage:
#   .\setup.ps1
#
# Safe to re-run: skips venv creation if .venv already exists, and
# fetch_cad_assets.py skips any file that's already present with a
# matching checksum.

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

function Get-PythonCmd {
    foreach ($cand in @("python", "py")) {
        if (Get-Command $cand -ErrorAction SilentlyContinue) { return $cand }
    }
    throw "No Python interpreter found on PATH (tried 'python', 'py'). Install Python 3.10+ first."
}

$python = Get-PythonCmd
Write-Host "Using Python: $(& $python --version)"

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment (.venv)..."
    & $python -m venv .venv
} else {
    Write-Host ".venv already exists, skipping creation."
}

$venvPython = ".venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

if (Get-Command gh -ErrorAction SilentlyContinue) {
    Write-Host "Fetching large CAD meshes from the GitHub Release..."
    & $venvPython scripts\assets\fetch_cad_assets.py
} else {
    Write-Warning "GitHub CLI ('gh') not found - skipping CAD asset download."
    Write-Warning "Install it (https://cli.github.com/), run 'gh auth login', then:"
    Write-Warning "  .venv\Scripts\python.exe scripts\assets\fetch_cad_assets.py"
}

Write-Host ""
Write-Host "Setup complete. Activate the venv with:"
Write-Host "  .venv\Scripts\Activate.ps1"
Write-Host "Then render a sample dataset with:"
Write-Host "  blenderproc run scripts\blenderproc\generate_dataset.py --season 2026_biobuzz --camera limelight3a --num_images 10"
