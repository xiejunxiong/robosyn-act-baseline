#!/usr/bin/env bash
set -euo pipefail
exec "${BASELINE_PYTHON:-python3}" -m robosyn_act train "$@"
