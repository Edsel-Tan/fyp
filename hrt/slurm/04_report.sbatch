#!/bin/bash
#SBATCH --job-name=hrt-rep
#SBATCH --output=hrt/logs/slurm-%x-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
# Aggregate whatever runs exist and regenerate the HTML report. Safe to run
# while a sweep is still going -- it simply reports on the runs finished so far.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../..}/hrt/slurm/00_env.sh"
python hrt/report.py
python hrt/make_report.py
echo "report at hrt/artifacts/hrt_reproduction.html"
