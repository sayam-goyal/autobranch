#!/usr/bin/env bash
# Serve the real app locally against the live Modal registry.
#
#   ./serve_local.sh            # http://localhost:8080
#
# Same code as the deployed container: the frontend and the API come from one
# origin, and run/branch state is read from the same modal.Dict the sandboxes
# write to, so a run started on Modal shows up here live. The venv lives in
# the repo rather than a temp dir so a restarted shell can still serve.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "creating .venv (first run only)..."
  python3 -m venv --system-site-packages .venv
  .venv/bin/pip install -q "fastapi==0.115.12" "starlette<0.42" "uvicorn==0.34.3"
fi

PORT="${PORT:-8080}"
echo "autobranch -> http://localhost:${PORT}"
exec .venv/bin/python run_local.py
