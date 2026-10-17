#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${1:-$repo_dir/.venv-tools}"
if [[ -e "$venv_dir" && ! -f "$venv_dir/pyvenv.cfg" ]]; then
  echo "Refusing to modify a directory that is not a venv: $venv_dir" >&2
  exit 2
fi
"${BASELINE_PYTHON:-python3}" -m venv "$venv_dir"
"$venv_dir/bin/python" -m pip install -e "$repo_dir"
echo "Installed launcher/config/NumPy smoke dependencies only. ACT and DexSim are not installed."
echo "Use: $venv_dir/bin/python -m robosyn_act --help"
