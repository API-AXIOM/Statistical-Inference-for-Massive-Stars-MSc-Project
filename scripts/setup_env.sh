#!/bin/bash
# Build the virtualenv used by the whole pipeline. Run once, on the login node.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the repository folder
source "$ROOT/env.sh"          # loads EESSI Python; venv does not exist yet

# SLURM writes each job's log into logs/ and needs the folder to exist
# before the job starts.
mkdir -p "$ROOT/logs"

VENV="${MCMC_GPU_VENV:-$ROOT/venv}"
if [ ! -d "$VENV" ]; then
    echo "Creating venv with $(command -v python3) ($(python3 -V 2>&1))"
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
python3 -V

python3 -m pip install --upgrade pip wheel setuptools
python3 -m pip install -r "$ROOT/requirements.txt"

python3 - <<'PY'
import torch
print("torch", torch.__version__, "| cuda build", torch.version.cuda)
print("cuda available (False is expected on the login node):", torch.cuda.is_available())
PY
