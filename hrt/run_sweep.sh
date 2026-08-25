#!/usr/bin/env bash
# 2x2 over signal timing (causal | paper) x alpha decay unit (step | episode),
# plus the standalone PPO/DDPG baselines. Seed-major, so a complete cross-arm
# table exists after the first wave rather than only at the end.
#
#   PARALLEL=12 ./hrt/run_sweep.sh          # concurrency (default 12)
#   PYTHON=.venv/bin/python ./hrt/run_sweep.sh
#
# Each worker holds a ~2 GB replay buffer, so keep PARALLEL under
# (free RAM in GB / 2.2). GPU memory is not the binding constraint: the networks
# are two 256-unit MLPs and use well under 1 GB per process.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PYTHON="${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}"
PARALLEL="${PARALLEL:-12}"
STEPS="${STEPS:-500000}"
JOBS="${JOBS:-hrt/jobs.txt}"
mkdir -p hrt/logs hrt/artifacts/runs

echo "repo=$REPO python=$PYTHON parallel=$PARALLEL steps=$STEPS"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

xargs -a "${JOBS:-hrt/jobs.txt}" -P "$PARALLEL" -I{} bash -c '
  set -- {}
  n=$(echo "$*" | tr -s " " | tr " " "_" | tr -d "\-")
  "'"$PYTHON"'" hrt/train.py $* --timesteps '"$STEPS"' --eval_every 25000 \
    > "hrt/logs/${n}.log" 2>&1
  tail -1 "hrt/logs/${n}.log"
'
echo "SWEEP COMPLETE"
