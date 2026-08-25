#!/bin/bash
#SBATCH --job-name=hrt-sweep
#SBATCH --output=hrt/logs/slurm-%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=12:00:00
# All 21 runs inside ONE allocation, sharing a single H200.
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

export PARALLEL="${PARALLEL:-21}"
export STEPS="${STEPS:-500000}"
srun --unbuffered ./hrt/run_sweep.sh
python hrt/report.py
echo "SWEEP OK"
