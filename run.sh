#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

export JEPA_CHECKPOINT="${JEPA_CHECKPOINT:-checkpoints/run-1}"
exec "${PYTHON:-.venv/bin/python}" -m uvicorn api:app \
  --host 127.0.0.1 --port 8001 "$@"
