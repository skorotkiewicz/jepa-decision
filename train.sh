#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

exec "${PYTHON:-.venv/bin/python}" -u train.py \
  --train data/train.jsonl \
  --validation data/validation.jsonl \
  --calibration data/calibration.jsonl \
  --test data/test.jsonl \
  --output checkpoints/run-1 \
  "$@"
