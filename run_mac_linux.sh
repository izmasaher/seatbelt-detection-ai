#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Creating virtual environment (first run only, takes a few minutes)..."
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  if [ "$(uname)" = "Linux" ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  fi
  pip install -r requirements.txt
else
  source .venv/bin/activate
fi
python app.py "$@"
