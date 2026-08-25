#!/bin/bash
#SBATCH --job-name=hrt-prep
#SBATCH --output=hrt/logs/slurm-%x-%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
# No GPU: this stage is pure NumPy/pandas. Rebuilds the feature panel from the
# committed 15 MB bundle and computes every non-RL benchmark. ~15 min total.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../..}/hrt/slurm/00_env.sh"

python hrt/data.py                 # panel.npz from prices_bundle.npz  (~7 min)
python hrt/baselines.py            # min-variance, equal-weight, S&P 500
python hrt/passive.py              # buy-once-and-hold floor
python hrt/deadjust.py             # price-only panel
python hrt/test_alpha158.py        # features vs an independent implementation
python hrt/test_env.py             # environment invariants
echo "PREPARE OK"
