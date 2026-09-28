#!/usr/bin/env bash
# Fine-tuning arms (RESULTS.md sec.3.6). Three test months so the answer is not
# one month's weather, and three arms per month so "text - price" is
# architecture-matched and "null" bounds what the architecture invents.
set -u
cd "$(dirname "$0")/.."
LOG=data/nlp/finetune.log
touch "$LOG"
for M in 2026-01 2026-02 2026-03; do
  for ARM in price text null; do
    if [ -f "data/nlp/finetune_${ARM}_text_${M}.json" ]; then
      echo "--- skip (done): arm=$ARM month=$M" >> "$LOG"; continue
    fi
    echo "### arm=$ARM month=$M" | tee -a "$LOG"
    mkdir -p data/nlp/arms
    .venv/bin/python pm/pmv1_text_finetune.py --arm $ARM --test-month $M --epochs 3 \
      > "data/nlp/arms/ft_${ARM}_${M}.out" 2>&1 || \
      echo "  ARM FAILED -- see data/nlp/arms/ft_${ARM}_${M}.out" | tee -a "$LOG"
    grep -E "^arm=|epoch|^test IC|book:" "data/nlp/arms/ft_${ARM}_${M}.out" | tee -a "$LOG"
    echo | tee -a "$LOG"
  done
done
echo "FINETUNE DONE" | tee -a "$LOG"
