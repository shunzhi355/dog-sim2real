#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[vision,tof,dev]'
.venv/bin/python -m dog3588 doctor
.venv/bin/python -m pytest -q
.venv/bin/python scripts/smoke_ipc.py
.venv/bin/python -m dog3588 benchmark --count 1000
# Deliberately foreground and finite. No service activation or motor writes.
.venv/bin/python -m dog3588 run --seconds 5

