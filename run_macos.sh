#!/usr/bin/env bash
#
# macOS launcher for the Multimodal Threat & Distress Detection System.
#
# Usage:
#   ./run_macos.sh            # launch the app
#   ./run_macos.sh --setup    # create .venv + install dependencies, then launch
#
# The dashboard will be available at http://localhost:8000
#
set -e

cd "$(dirname "$0")"

if [[ "${1:-}" == "--setup" ]]; then
  echo "[Launcher] Setting up Python 3.14 virtual environment..."
  if ! command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1; then
    echo "[Launcher] Homebrew Python 3.14 not found. Install it with:" >&2
    echo "  brew install python@3.14" >&2
    exit 1
  fi
  /opt/homebrew/bin/python3.14 -m venv .venv
  echo "[Launcher] Installing dependencies (this can take a few minutes)..."
  .venv/bin/python -m pip install --upgrade pip setuptools wheel
  .venv/bin/python -m pip install -r requirements.txt
fi

if [[ ! -x ".venv/bin/python" ]]; then
  echo "[Launcher] Virtual environment not found. Run:  ./run_macos.sh --setup" >&2
  exit 1
fi

echo "=================================================="
echo "   MULTIMODAL THREAT & DISTRESS DETECTION SYSTEM   "
echo "   Dashboard:  http://localhost:8000              "
echo "   Open it in your browser. Press Ctrl+C to stop. "
echo "=================================================="
exec .venv/bin/python run.py