#!/usr/bin/env bash
# Shared environment for every job. EDIT THE THREE MARKED LINES for your cluster,
# then everything else works unchanged.
#
# Discover the right values with:
#   sinfo -o "%P %G %m %c %l"          partitions, GPUs, memory, cores, time limit
#   module avail 2>&1 | grep -i cuda   available CUDA / Python modules
#   sacctmgr -n show assoc user=$USER format=account,partition

# ---- EDIT ME ----------------------------------------------------------------
CLUSTER_MODULES="cuda/12.6 python/3.12"     # module names on your cluster
REPO="${REPO:-$HOME/fyp}"                    # where you cloned the repo
VENVDIR="${VENVDIR:-$REPO/.venv}"
# -----------------------------------------------------------------------------

set -euo pipefail
if command -v module >/dev/null 2>&1; then
  module purge  || true
  # shellcheck disable=SC2086
  module load $CLUSTER_MODULES || echo "WARN: module load failed; check names" >&2
fi
cd "$REPO"
# shellcheck disable=SC1091
source "$VENVDIR/bin/activate"
export PYTHON="$VENVDIR/bin/python"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p hrt/logs hrt/artifacts/runs
echo "host=$(hostname) repo=$REPO python=$(which python)"
python -c "import torch;print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} '
           f'dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"-\"}')"
