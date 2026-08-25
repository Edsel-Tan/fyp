#!/bin/bash
#SBATCH --job-name=hrt-fc
#SBATCH --output=hrt/logs/slurm-%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
# Trains the two forward-return models. ~90 s each; the GPU request is mostly
# formality. Also runs the signal-value check and the HLC learning diagnostic.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../..}/hrt/slurm/00_env.sh"

python hrt/forecast.py --label causal
python hrt/forecast.py --label paper
python hrt/signal_strategy.py
python hrt/diag_hlc.py --signal paper --timesteps 30720   # alignment must climb
echo "FORECAST OK"
