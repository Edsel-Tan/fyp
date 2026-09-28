#!/usr/bin/env bash
# Full pipeline on whatever frames are on disk. Logs to ../results/run_all.log.
set -euo pipefail
cd "$(dirname "$0")"
P=~/fyp/.venv/bin/python
{
  for step in "label_games.py raw" "label_games.py" "pregame.py" "eval_pregame.py" "ingame_data.py" "eval_ingame.py" \
              "pm_map.py" "market_compare.py 0 30 60 120 300" "backtest.py 0 5 15 30 60 120" "follow_wallets.py 0 5 15 30 60"; do
    echo "== $step"; $P $step || { echo "FAILED: $step"; exit 1; }
  done
  for s in 1 2 3 4 5; do
    NULL_SEED=$s $P eval_ingame.py > /dev/null
    echo "== null seed $s"; PREDS_SFX=_null$s $P backtest.py 0 30 | grep -E "^ +(0|30) +B|^ +(0|30) +A"
  done
} 2>&1 | grep -v -E "Warning|warnings.warn|bs = |return s.sum" | tee ../results/run_all.log
