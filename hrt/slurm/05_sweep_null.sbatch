#!/bin/bash
#SBATCH --job-name=hrt-null
#SBATCH --output=hrt/logs/slurm-%x-%j.out
#SBATCH --cpus-per-task=52
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --gpus=h100-96              
# RESULTS.md §10.1 / §9 -- the one experiment that has never been run.
#
# §4 falsifies the *forecaster* on synthetic null panels: under the paper's label
# timing the pipeline reports test IC +0.538 where there is provably nothing to
# find. The agents have never been put through that test. This script does it:
# the same 25-run sweep behind §2's tables, retrained end to end on panels whose
# returns are a martingale difference, and scored against the passive floor
# measured inside the same environment on the same synthetic panel.
#
# The question it answers: do the RL arms still beat "deploy the cash and sit
# still" when there is nothing to learn? If they do, §2's ordering is a property
# of the workflow rather than of the market.
#
# Arms (mirrors hrt/artifacts/runs/, 25 runs):
#   hrt  x {causal,paper} x {step,episode} x seeds 0-3   16
#   ppo  causal            x seeds 0-3                    4
#   ddpg causal            x seeds 0-4                    5
# Seed k trains on synthetic panel k mod 4, so the four independent null panels
# of §4 carry the seed dispersion instead of the seed alone.
#
# Sizing (CLUSTER.md §6, measured per worker): ~2.1 GB host RAM for the 2e5
# replay buffer, ~0.3 GB of GPU memory, 2 CPU threads. 25 concurrent workers
# want ~53 GB RAM, ~8 GB of GPU and 50 threads -- so the whole sweep fits in one
# wave on one GPU, and GPU memory is nowhere near binding on either an H200
# (141 GB) or an H100 (96 GB). Host RAM is the constraint:
#     PARALLEL <= min(cpus-per-task / 2, mem_GB / 2.5)
#
# Wall time is the duration of the slowest single run, not the sum. Calibrate
# with the srun recipe in CLUSTER.md §6 before trusting --time; the sweep is
# resumable, so an underestimate costs one resubmission and nothing else.
#
# Usage:
#   sbatch hrt/slurm/05_sweep_null.sbatch          # prep (if needed) + sweep + summary
#   STAGE=prep  sbatch ...                         # only build the null panels/forecasts
#   STAGE=sweep sbatch ...                         # only train (prep already done)
#   PARALLEL=16 STEPS=500000 sbatch ...            # knobs
#   RESUME=0 sbatch ...                            # force a full redo
#
# Depends on 01_prepare (panel.npz) having run. Chain it:
#   a=$(sbatch --parsable hrt/slurm/01_prepare.sbatch)
#   sbatch --dependency=afterok:$a hrt/slurm/05_sweep_null.sbatch
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../..}/hrt/slurm/00_env.sh"

ART=hrt/artifacts
RUNS="$ART/runs_null"
JOBS="$ART/jobs_null.txt"
NPANELS=4
export PARALLEL="${PARALLEL:-25}" STEPS="${STEPS:-500000}" EVAL_EVERY="${EVAL_EVERY:-25000}"
export RESUME="${RESUME:-1}" STAGE="${STAGE:-all}" RUNS PYTHON
mkdir -p "$RUNS" hrt/logs

# panel k: seed 0 keeps synth_panel.py's unsuffixed default name, as §4 wrote it
panel_of() { [ "$1" = 0 ] && echo "$ART/panel_synth.npz" || echo "$ART/panel_synth_s$1.npz"; }
fr_of()    { [ "$2" = 0 ] && echo "$ART/fr_synth_$1.npz" || echo "$ART/fr_synth_$1_s$2.npz"; }
floor_of() { echo "$ART/passive_synth_s$1.json"; }
export -f panel_of fr_of floor_of
export ART

# ---------------------------------------------------------------- stage: prep
# Four null panels, two forecasters each, one passive floor each. Skipped when
# the artifacts are already there, so this is free on a resubmission. Nothing
# here touches the network. ~10 min per panel (Alpha158 is the slow part) plus
# ~90 s per forecaster, so ~50 min cold.
if [ "$STAGE" = all ] || [ "$STAGE" = prep ]; then
  if [ ! -s "$ART/panel.npz" ]; then
    echo "ERROR: $ART/panel.npz missing -- run hrt/slurm/01_prepare.sbatch first" >&2
    exit 1
  fi
  for k in $(seq 0 $((NPANELS - 1))); do
    p=$(panel_of "$k")
    [ -s "$p" ] || python analysis/synth_panel.py --real "$ART/panel.npz" --seed "$k" --out "$p"
    for mode in causal paper; do
      f=$(fr_of "$mode" "$k")
      [ -s "$f" ] || python hrt/forecast.py --panel "$p" --label "$mode" --out "$f"
    done
    fl=$(floor_of "$k")
    [ -s "$fl" ] || python hrt/passive.py --panel "$p" --fr "$(fr_of causal "$k")" --out "$fl"
  done
  echo "PREP OK"
fi
if [ "$STAGE" = prep ]; then echo "stage=prep only, stopping"; exit 0; fi

# --------------------------------------------------------------- stage: sweep
# Job line: agent signal alpha_unit tag seed   ("-" = no tag)
: > "$JOBS"
for s in 0 1 2 3; do                       # seed-major: a full cross-arm table early
  printf 'hrt causal step - %s\n'      "$s" >> "$JOBS"
  printf 'hrt causal episode _ep %s\n' "$s" >> "$JOBS"
  printf 'hrt paper step - %s\n'       "$s" >> "$JOBS"
  printf 'hrt paper episode _ep %s\n'  "$s" >> "$JOBS"
  printf 'ppo causal step - %s\n'      "$s" >> "$JOBS"
  printf 'ddpg causal step - %s\n'     "$s" >> "$JOBS"
done
printf 'ddpg causal step - 4\n' >> "$JOBS"
echo "jobs: $(wc -l < "$JOBS")  parallel=$PARALLEL  steps=$STEPS  runs_dir=$RUNS"

# Resumable at run granularity: a run whose JSON exists is skipped, so a job
# killed at the time limit just needs resubmitting. A failed run logs and does
# not take the sweep down with it; the tally after the loop reports what is
# missing.
srun --unbuffered xargs -a "$JOBS" -P "$PARALLEL" -I{} bash -c '
  set -- {}
  agent=$1; signal=$2; unit=$3; tag=$4; seed=$5
  [ "$tag" = "-" ] && tag=""
  k=$(( seed % '"$NPANELS"' ))
  out="$RUNS/${agent}_${signal}${tag}_s${seed}.json"
  log="hrt/logs/null_${agent}_${signal}${tag}_s${seed}.log"
  if [ "$RESUME" = 1 ] && [ -s "$out" ]; then echo "skip (done): $out"; exit 0; fi
  if "$PYTHON" hrt/train.py --agent "$agent" --signal "$signal" --alpha_unit "$unit" \
       ${tag:+--tag "$tag"} --seed "$seed" \
       --panel "$(panel_of $k)" --fr "$(fr_of $signal $k)" --runs_dir "$RUNS" \
       --timesteps "$STEPS" --eval_every "$EVAL_EVERY" > "$log" 2>&1; then
    tail -1 "$log"
  else
    echo "FAIL (null panel $k): $out -- see $log"; tail -5 "$log"
  fi
  exit 0
'

want=$(wc -l < "$JOBS"); got=$(find "$RUNS" -name '*.json' -size +0 | wc -l)
echo "runs complete: $got / $want"

# ------------------------------------------------------------- stage: summary
# Each arm against the passive floor of the panel it was trained on. report.py
# is deliberately not reused: it scores against baselines.json, which is the
# REAL panel's benchmarks, and comparing a null-panel agent to those is
# meaningless.
python - <<'PY'
import glob, json, os, re
import numpy as np
ART = "hrt/artifacts"
floors = {int(re.search(r"_s(\d+)\.json$", f).group(1)): json.load(open(f))
          for f in sorted(glob.glob(f"{ART}/passive_synth_s*.json"))}

def load(d):
    runs = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        stem = os.path.basename(f)[:-5]
        m = re.search(r"_s(\d+)$", stem)
        if not m or "_smoke" in stem:
            continue
        runs.setdefault(stem[:m.start()], []).append((int(m.group(1)), json.load(open(f))))
    return runs

null, real = load(f"{ART}/runs_null"), load(f"{ART}/runs")
for period in ("test2021", "test2022"):
    fl = [floors[k][period]["cum_return"] for k in sorted(floors)]
    print(f"\n{'='*104}\n{period} on the synthetic null panels   "
          f"passive floor: mean {np.mean(fl):+.4f} over {len(fl)} panels\n{'='*104}")
    print(f"{'arm':<18}{'n':>3}{'null cum':>18}{'null sharpe':>16}"
          f"{'beats floor':>13}{'real cum':>18}{'real sharpe':>14}")
    print("-" * 104)
    for arm in sorted(null):
        rs = null[arm]
        cum = np.array([r[period]["cum_return"] for _, r in rs])
        shp = np.array([r[period]["sharpe"] for _, r in rs])
        beat = sum(r[period]["cum_return"] > floors[s % len(floors)][period]["cum_return"]
                   for s, r in rs)
        rc = rs_ = float("nan")
        if arm in real:
            rc = np.mean([r[period]["cum_return"] for _, r in real[arm]])
            rs_ = np.mean([r[period]["sharpe"] for _, r in real[arm]])
        sd = f"±{cum.std(ddof=1):.3f}" if len(cum) > 1 else ""
        print(f"{arm:<18}{len(rs):>3}{f'{cum.mean():+.4f}{sd}':>18}{shp.mean():>+16.4f}"
              f"{f'{beat}/{len(rs)}':>13}{rc:>+18.4f}{rs_:>+14.4f}")
print("\nA null arm that beats its own panel's floor as often as the real arm beats the"
      "\nreal floor is evidence the ordering in RESULTS.md §2 is a workflow artefact.")
PY
echo "NULL SWEEP OK -- runs in hrt/artifacts/runs_null/"
