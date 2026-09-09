#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export ADASPOT_FRAME_DIR="${ADASPOT_FRAME_DIR:-/tmp/adaspot-finegym-frames}"
mkdir -p artifacts/logs
bash scripts/setup_validation.sh 2>&1 | tee artifacts/logs/setup.log
.venv/bin/python scripts/validate_adaspot.py checkpoint 2>&1 | tee artifacts/logs/checkpoint.log
.venv/bin/python scripts/download_subset.py 2>&1 | tee artifacts/logs/download.log
.venv/bin/python scripts/validate_adaspot.py smoke 2>&1 | tee artifacts/logs/smoke.log
.venv/bin/python scripts/validate_adaspot.py evaluate 2>&1 | tee artifacts/logs/evaluate.log
.venv/bin/python scripts/validate_adaspot.py train 2>&1 | tee artifacts/logs/train.log
.venv/bin/python scripts/audit_validation.py 2>&1 | tee artifacts/logs/audit.log
