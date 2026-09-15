#!/bin/bash
#SBATCH --partition=gpu-long
#SBATCH --job-name=hrt-sweep
#SBATCH --output=hrt/logs/slurm-%x-%j.out
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=12:00:00
# All 60 jobs inside ONE allocation, sharing a single GPU (35 unrun; the 25
# committed runs are skipped by the resume logic).
#
# Sizing: each worker needs ~2.1 GB host RAM (the 2e5-transition replay buffer),
# ~0.3 GB of GPU memory and 2 CPU threads. So 21 concurrent workers want ~45 GB
# RAM, ~6 GB of GPU and 42 threads -- comfortably one node, one GPU. GPU memory
# is never the binding constraint here; host RAM is.
#
# Keep PARALLEL <= min(cpus-per-task / 2, mem_GB / 2.5).
#
# Resumable: relaunch after a timeout and finished runs are skipped.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../..}/hrt/slurm/00_env.sh"

export PARALLEL="${PARALLEL:-24}"
export STEPS="${STEPS:-500000}"

# train.py silently falls back to CPU when CUDA is missing (--device defaults to
# cpu), which would burn the whole sweep at a fraction of the speed with nothing
# in the logs to say so. Fail fast instead.
python - <<'PY' || { echo "FATAL: no CUDA on $(hostname); refusing to run the sweep on CPU" >&2; exit 1; }
import sys, torch
sys.exit(0 if torch.cuda.is_available() else 1)
PY

# NOTE: do not wrap this in `srun`. The batch step's CPU mask does not match what
# srun tries to bind, and the step dies with "Unable to satisfy cpu bind request".
# run_sweep.sh forks its own workers with xargs -P; it needs no step launcher.
./hrt/run_sweep.sh
python hrt/report.py
echo "SWEEP OK"
