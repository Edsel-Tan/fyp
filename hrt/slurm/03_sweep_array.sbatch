#!/bin/bash
#SBATCH --job-name=hrt-arr
#SBATCH --output=hrt/logs/slurm-%x-%A_%a.out
#SBATCH --array=1-21%8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
# Alternative to 03_sweep.sbatch: one run per array task.
#
# Use this when the cluster caps wall time below what the whole sweep needs, or
# when short jobs schedule far sooner than one long one. It costs 21 separate
# GPU allocations for work that fits on a single GPU, so it is the wrong default
# on a busy queue -- but it is the right answer when %-limits or QoS make one
# 12-hour reservation hard to get. The %8 throttles to 8 concurrent tasks.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../..}/hrt/slurm/00_env.sh"

JOB=$(sed -n "${SLURM_ARRAY_TASK_ID}p" hrt/jobs.txt)
echo "task ${SLURM_ARRAY_TASK_ID}: $JOB"
printf '%s\n' "$JOB" > "hrt/logs/job_${SLURM_ARRAY_TASK_ID}.txt"
JOBS="hrt/logs/job_${SLURM_ARRAY_TASK_ID}.txt" PARALLEL=1 ./hrt/run_sweep.sh
