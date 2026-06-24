#!/usr/bin/env bash
set -euo pipefail

# cd to the directory containing this script
cd "$(dirname "${BASH_SOURCE[0]}")"

# Create the virtualenv if it doesn't exist yet.
if [ ! -d ".venv" ]; then
  echo "Creating virtualenv at .venv ..."
  python3 -m venv .venv
fi

# Activate the venv and install dependencies.
# shellcheck disable=SC1091
source .venv/bin/activate
echo "Installing dependencies (pip install -q -r requirements.txt) ..."
pip install -q -r requirements.txt

# Ensure a .env exists; never fail if the key is missing -- the server must start.
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo ""
  echo "============================================================"
  echo "NOTICE: created .env from .env.example."
  echo "You MUST add your DASHSCOPE_API_KEY to .env."
  echo "Until you do, model-calling endpoints will return HTTP 503."
  echo "============================================================"
  echo ""
else
  # .env exists -- warn if the key is empty/unset.
  KEY="$(grep -E '^DASHSCOPE_API_KEY=' .env | head -n1 | cut -d= -f2- || true)"
  KEY="${KEY%\"}"; KEY="${KEY#\"}"; KEY="${KEY%\'}"; KEY="${KEY#\'}"
  if [ -z "${KEY// }" ]; then
    echo ""
    echo "============================================================"
    echo "WARNING: DASHSCOPE_API_KEY is empty in .env."
    echo "Model-calling endpoints will return HTTP 503 until a key is set."
    echo "============================================================"
    echo ""
  fi
fi

echo "Starting Eidetic-Plus -> http://localhost:8000"
exec .venv/bin/uvicorn eidetic.api:app --host 0.0.0.0 --port 8000 --reload
