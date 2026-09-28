#!/usr/bin/env bash
# Checkpoint-rule audit (RESULTS.md sec.10 items 3-4): the four leak-free arms on the real
# panel and on two null panels, every eval point scored on valid + both test years and its
# weights kept (train.py --probe). Seed-major; resumable -- a run whose JSON exists is
# skipped. Sized for the lum.id sandbox's 8 GB / 2-CPU cgroup (SANDBOX.md): PARALLEL=2.
#
#   SEEDS="0 1 2 3 4" PARALLEL=2 ./hrt/run_probe.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PYTHON="${PYTHON:-/opt/venv/bin/python}"; PARALLEL="${PARALLEL:-2}"
SEEDS="${SEEDS:-0 1 2 3 4}"; OUT="${OUT:-hrt/artifacts/runs_probe}"
A=hrt/artifacts
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT" hrt/logs/probe
jobs=$(mktemp)
for s in $SEEDS; do
  for pn in "real $A/panel.npz $A/fr_causal.npz" \
            "null0 $A/panel_synth.npz $A/fr_synth_causal.npz" \
            "null1 $A/panel_synth_s1.npz $A/fr_synth_causal_s1.npz"; do
    set -- $pn
    for arm in "hrt causal step _" "hrt causal episode _ep" "ppo causal step _" "ddpg causal step _"; do
      set -- $pn $arm
      tag=$([ "$7" = "_" ] && echo "" || echo "$7")
      echo "$1 $2 $3 $4 $5 $6 $tag $s" >> "$jobs"
    done
  done
done
echo "$(wc -l < "$jobs") jobs -> $OUT, parallel=$PARALLEL"
xargs -a "$jobs" -P "$PARALLEL" -L 1 bash -c '
  kind=$0 panel=$1 fr=$2 agent=$3 signal=$4 unit=$5 tag=$6 seed=$7
  [ -z "$seed" ] && { seed=$tag; tag=""; }
  d='"$OUT"'/$kind; out="$d/${agent}_${signal}${tag}_s${seed}.json"
  [ -s "$out" ] && { echo "skip $out"; exit 0; }
  mkdir -p "$d"
  '"$PYTHON"' hrt/train.py --agent $agent --signal $signal --alpha_unit $unit ${tag:+--tag $tag} \
     --seed $seed --panel $panel --fr $fr --runs_dir $d --probe \
     --timesteps 500000 --eval_every 25000 > hrt/logs/probe/${kind}_${agent}${tag}_s${seed}.log 2>&1 \
     && echo "done $out $(date +%H:%M)" || echo "FAIL $out"
'
rm -f "$jobs"; echo PROBE SWEEP COMPLETE
