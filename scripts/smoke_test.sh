#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
"${BASELINE_PYTHON:-python3}" -m unittest discover -s tests -v
exec "${BASELINE_PYTHON:-python3}" -m robosyn_act smoke "$@"
