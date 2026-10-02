#!/usr/bin/env bash
# One-shot project setup for macOS/Linux.
# Creates the venv, installs Python deps, and fetches the large CAD meshes
# from the project's GitHub Release.
#
# Usage:
#   ./setup.sh
#
# Safe to re-run: skips venv creation if .venv already exists, and
# fetch_cad_assets.py skips any file that's already present with a
# matching checksum.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="$(command -v python3 || command -v python || true)"
if [ -z "$PYTHON_BIN" ]; then
    echo "No Python interpreter found on PATH (tried 'python3', 'python'). Install Python 3.10+ first." >&2
    exit 1
fi
echo "Using Python: $("$PYTHON_BIN" --version)"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv)..."
    "$PYTHON_BIN" -m venv .venv
else
    echo ".venv already exists, skipping creation."
fi

VENV_PYTHON=".venv/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r requirements.txt

if command -v gh >/dev/null 2>&1; then
    echo "Fetching large CAD meshes from the GitHub Release..."
    "$VENV_PYTHON" scripts/assets/fetch_cad_assets.py
else
    echo "WARNING: GitHub CLI ('gh') not found - skipping CAD asset download." >&2
    echo "Install it (https://cli.github.com/), run 'gh auth login', then:" >&2
    echo "  .venv/bin/python scripts/assets/fetch_cad_assets.py" >&2
fi

echo ""
echo "Setup complete. Activate the venv with:"
echo "  source .venv/bin/activate"
echo "Then render a sample dataset with:"
echo "  blenderproc run scripts/blenderproc/generate_dataset.py --season 2026_biobuzz --camera limelight3a --num_images 10"
