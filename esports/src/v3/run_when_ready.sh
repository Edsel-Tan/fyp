#!/bin/bash
# Waits until >= 1200 train-period games have crops, then parses everything and runs the split-mode v3 evaluations.
cd /home/aimer/fyp/esports/src
source ~/fyp/.venv/bin/activate
while true; do
  n=$(python - <<'PY'
import pandas as pd, glob, json, os
t = pd.read_parquet('../data/game_table.parquet', columns=['game_id', 't_start']).set_index('game_id').t_start
S = pd.Timestamp('2025-07-01', tz='UTC')
n = 0
for f in glob.glob('../data/v3/crops/*/done.json'):
    g = os.path.basename(os.path.dirname(f))
    if g in t.index and t[g] < S and json.load(open(f)).get('status') == 'ok':
        n += 1
print(n)
PY
)
  echo "$(date -u +%H:%M) train games with crops: $n"
  [ "$n" -ge 1200 ] && break
  sleep 900
done
ls ../data/v3/crops/ > ../data/v3/ids_have.txt
python v3/parse2.py ../data/v3/ids_have.txt 24
python v3/eval_v3b.py > ../results/v3_eval_split.log 2>&1
python v3/eval_taker.py > ../results/v3_taker_all.log 2>&1
echo DONE
