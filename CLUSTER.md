# HRT reproduction — handoff to the H200 cluster

Everything needed to continue this work on a SLURM cluster. Written for a machine
where **you cannot touch the GPU directly** — all compute goes through `sbatch`.

Repo: `github.com/Edsel-Tan/fyp`, branch `hrt-reproduction`.
Report: <https://claude.ai/code/artifact/df6b17b8-9ba7-4821-9e5f-88384832a555>
Detail on the science: [`hrt/README.md`](hrt/README.md).

---

## 1. Do these two things first

**Rotate the findata token.** A live PAT was committed in `data.md` and pushed
before I got here. I redacted the working copy, but it remains in the pushed
history, where force-pushing would not reliably remove it. Issue a new token at
lum.id, put it in a git-ignored `.env`, and treat the old one as burned.

```bash
printf 'LUMID_TOKEN=<new-token>\n' > .env && chmod 600 .env
# load when needed:  set -a; source .env; set +a
```

**You almost certainly won't need it.** See §3 — the pipeline no longer touches
the network.

---

## 2. State of play

A from-scratch reproduction of Zhao & Welsch, *Hierarchical Reinforced Trader*
([arXiv:2410.14927v1](https://arxiv.org/abs/2410.14927)). 25 runs at 5×10⁵ steps
are committed under `hrt/artifacts/runs/`.

| | |
|---|---|
| **Reproduces** | The S&P 500 benchmark to four decimals. The paper's motivating pathology: standalone DDPG collapses to buy-and-hold (its whole annual turnover is the day-one deployment) while the hierarchy trades throughout. |
| **Does not** | HRT-FR's headline returns, in either test year, under any leak-free configuration. The claimed HRT-FR > DDPG > PPO ordering reverses in 2021. Every RL agent finishes below buy-and-hold in the bear year. |
| **Open question** | The published numbers sit *between* a causal implementation (+23.1% / −7.6%) and one wired to the paper's stated timing (+73% / +45%), matching neither. |

Three findings that stand independently of whether HRT reproduces:

1. **The label leaks.** The forward return is defined over a window containing
   the very close its features come from. Trained as written: test IC **+0.82**;
   causally, **+0.012**. Qlib's own Alpha158 default label skips exactly that
   window.
2. **Survivorship is worth 12.3 points.** 108 of the 133 unpriceable 2015
   constituents are names the index removed. Equal-weight survivors beat the
   index by 12.3 points in 2022 on a price-only basis, with no model.
3. **PPO breaks silently at N=370.** Summing entropy and the importance ratio
   over 370 action dimensions pins the policy at uniform random — and a random
   selection policy still *looks* fine, because it deploys capital and tracks the
   market. This cost a full sweep. `hrt/diag_hlc.py` is the regression test.

**The most promising direction** is the cost result: the hierarchy earns ~4.4
points of *gross* edge over passive in 2022, then pays 6.33 points in commission
at ~70× annual turnover. That makes the transaction-cost model the load-bearing
component, which is the argument for idea 6 in `fyp_ideas.md`.

---

## 3. The data is not a problem

`data/` (~2 GB) and `hrt/artifacts/panel.npz` (440 MB) are **not** in the repo and
do not need to be. `hrt/artifacts/prices_bundle.npz` (15 MB, committed) carries
the exact slice everything downstream reads, and `data.py` rebuilds the feature
panel from it **bit-for-bit** — verified, max abs diff 0.0 across all 158
features. `data.py`, `baselines.py` and `deadjust.py` use the SQLite mirror when
present and fall back to the bundle when absent.

This matters on a cluster: **compute nodes usually have no internet, and none is
needed.** Only `pip install` requires network, and that happens on the login node.

Re-pull from findata only if you need symbols or fields outside the bundle — then
run `python hrt/export_bundle.py` to refresh it.

---

## 4. One-time setup (login node)

Compute nodes are typically offline, so install here.

```bash
cd $SCRATCH                      # see §7 on scratch vs home
git clone -b hrt-reproduction https://github.com/Edsel-Tan/fyp
cd fyp

module avail 2>&1 | grep -iE 'cuda|python'      # find your module names
module load cuda/12.6 python/3.12               # ADJUST

python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126   # Hopper = sm_90
pip install -r hrt/requirements.txt

mkdir -p hrt/logs hrt/artifacts/runs            # SLURM will not create these
```

Then edit **`hrt/slurm/00_env.sh`** — three lines at the top: your module names,
your repo path, your venv path. Every job script sources it.

> **H200 note.** The repo was developed against `torch 2.6.0+cu124` on a GTX 1080
> Ti. Hopper (sm_90) wants cu126 or newer; a cu124 wheel may run but will warn or
> fall back. `requirements.txt` deliberately leaves torch unpinned for this
> reason — install it first, matched to the cluster's CUDA.

Discover your cluster's shape:

```bash
sinfo -o "%P %G %m %c %l"                        # partitions, GPUs, mem, cores, limits
sacctmgr -n show assoc user=$USER format=account,partition
```

Add `#SBATCH --partition=...` / `--account=...` to the job scripts if your site
requires them.

---

## 5. Running it

Four stages in `hrt/slurm/`. Submit from the repo root — SLURM's working
directory is the submission directory, and the `--output` paths are relative to
it.

```bash
sbatch hrt/slurm/01_prepare.sbatch        # CPU only, ~15 min
sbatch hrt/slurm/02_forecast.sbatch       # 1 GPU, ~5 min
sbatch hrt/slurm/03_sweep.sbatch          # 1 GPU, the long one
sbatch hrt/slurm/04_report.sbatch         # CPU, ~1 min
```

| Script | Does | Resources |
|---|---|---|
| `01_prepare` | rebuild panel from the bundle; all non-RL benchmarks; both test suites | 8 CPU, 24 GB, no GPU |
| `02_forecast` | both forward-return models; signal-value check; **HLC diagnostic** | 1 GPU, 8 CPU, 24 GB |
| `03_sweep` | all 21 RL runs on one GPU | 1 GPU, 48 CPU, 96 GB |
| `03_sweep_array` | *alternative*: one run per array task, `%8` concurrent | 1 GPU each, 4 CPU, 8 GB |
| `04_report` | aggregate + regenerate the HTML report | 4 CPU, 16 GB |

Chain them so each waits for the last:

```bash
a=$(sbatch --parsable hrt/slurm/01_prepare.sbatch)
b=$(sbatch --parsable --dependency=afterok:$a hrt/slurm/02_forecast.sbatch)
c=$(sbatch --parsable --dependency=afterok:$b hrt/slurm/03_sweep.sbatch)
sbatch --dependency=afterok:$c hrt/slurm/04_report.sbatch
```

### The sweep is resumable

Every run writes its own JSON on completion, and `run_sweep.sh` skips any run
whose JSON already exists. If the job dies at the time limit, **just resubmit** —
it picks up where it stopped. Nothing is lost below run granularity.

```bash
sbatch hrt/slurm/03_sweep.sbatch          # resumes automatically
RESUME=0 ...                              # only to force a full redo
```

### Which sweep script

`03_sweep.sbatch` is the default and the right one on a busy queue: 21 workers
share **one** GPU inside **one** allocation. Use `03_sweep_array.sbatch` instead
when your site caps wall time below what the sweep needs, or when short jobs
schedule far sooner than one long reservation — it costs 21 GPU allocations for
work that fits on a single GPU, so it is a scheduling workaround, not an
efficiency gain.

---

## 6. Sizing the request

Measured per worker process:

| Resource | Per worker | × 21 workers |
|---|---|---|
| Host RAM | **~2.1 GB** (the 2×10⁵-transition replay buffer) | ~45 GB |
| GPU memory | ~0.3 GB | ~6 GB |
| CPU threads | 2 | 42 |

**Host RAM is the binding constraint, not GPU memory.** An H200's 141 GB is never
close to full — the networks are two 256-unit MLPs. Keep:

```
PARALLEL ≤ min(cpus-per-task / 2, mem_GB / 2.5)
```

So all 21 runs fit in a single wave on one GPU, and the sweep's wall time is
roughly the duration of one HRT run rather than the sum of all of them.

### Calibrate before setting `--time`

The workload is **kernel-launch-bound**, not FLOP-bound — most steps are tiny
MLP forward passes, so an H200 helps less than its spec sheet suggests. Measure
rather than extrapolate:

```bash
srun --gres=gpu:1 --cpus-per-task=4 --mem=8G --time=00:20:00 --pty bash
source hrt/slurm/00_env.sh
time python hrt/train.py --agent hrt --seed 99 --timesteps 12288 \
     --eval_every 6144 --warmup 4096 --tag _cal
# steps/s = 12288 / elapsed;  full run = 500000 / (steps/s)
rm -f hrt/artifacts/runs/*_cal_*.json
```

Reference points from the 1080 Ti, at 12-way concurrency:

| Arm | Wall time per run |
|---|---|
| DDPG | ~60 min |
| PPO | ~92 min |
| HRT | ~127 min |

A single HRT run alone was ~32 min. Set `--time` to about 1.5× your measured
worst case; the resume behaviour makes an underestimate cheap.

---

## 7. Cluster gotchas

- **Never train on the login node.** `01_prepare` is CPU-only but still ~15
  minutes — submit it rather than running it interactively.
- **Scratch vs home.** `panel.npz` is 440 MB and home quotas are often small.
  Cloning into `$SCRATCH` is simplest, but scratch is usually **purged** after
  N days — copy `hrt/artifacts/runs/` and the report back to home when a sweep
  finishes. To keep the repo in home and only the big file on scratch:
  `ln -s $SCRATCH/panel.npz hrt/artifacts/panel.npz` before running `data.py`.
- **`$0` is not the repo.** SLURM copies the batch script to a node-local spool
  directory, so `dirname $0` does not resolve to your checkout. The scripts use
  `$SLURM_SUBMIT_DIR`; keep that if you write new ones.
- **`hrt/logs/` must exist before submission**, or SLURM fails to open
  `--output` and the job dies with no log to explain why.
- **Threads.** `00_env.sh` sets `OMP_NUM_THREADS=2`. Leave it — unset, each of 21
  workers grabs every core and they thrash.
- **`--gres` syntax varies.** Some sites want `--gres=gpu:h200:1` or
  `--gpus-per-node=1`. Check `sinfo -o "%P %G"`.

---

## 8. Checks that must pass

Run these before trusting any number. `01_prepare` and `02_forecast` already
include them.

```bash
python hrt/test_alpha158.py                     # features vs an independent impl
python hrt/test_env.py                          # 6 environment invariants
python hrt/diag_hlc.py --signal paper --timesteps 30720
```

**`diag_hlc.py` is the one that matters.** It trains the high-level controller on
the alignment reward alone under the leaked signal, where `action = sign(forecast)`
is nearly free to learn. Mean alignment must climb from ~0.00 toward ~0.6 and
entropy must fall from ln 3 = 1.0986. **If alignment stays flat near zero, the
policy is uniform random and every downstream number is meaningless** — that is
exactly the failure that cost a full sweep here, and it does not announce itself
in the returns.

---

## 9. Where to take it next

**Do not** start with the sentiment arm. `/news/{symbol}` caps at 200 items,
ignores every date parameter and paging, and serves only the trailing fortnight
(`since`/`until` returns HTTP 500). The 2015–2019 window is unreachable, so full
HRT cannot be built from findata **at any scale of hardware** — the H200 changes
nothing here. The route that works is FNSPID via the FinRL-DeepSeek benchmark on
HuggingFace, which ships precomputed LLM sentiment and risk scores, but it moves
the universe to an 89-name Nasdaq set. That is a different experiment, worth
choosing deliberately rather than drifting into.

Better uses of the hardware, in order:

1. **Transaction-cost modelling (idea 6).** The finding is already sitting there:
   4.4 points of gross edge, 6.33 points of commission. Add a turnover penalty
   and a non-constant cost model and measure whether the edge survives. Report
   net *and* gross — that split is where the story lives.
2. **More seeds.** Dispersion is wide (HRT-FR 2022 spans ~20 points across
   seeds); the committed runs are 4 seeds, the paper used 10. Cheap on an H200
   and it firms up every claim.
3. **Fix survivorship.** Needs CRSP, Sharadar or Norgate for delisted prices.
   Until then the 370-name universe is the acquired-and-survived cohort and the
   12.3-point gap is a lower bound.

---

## 10. Repo map

```
hrt/
  README.md            the science: ambiguities, the PPO trap, data gaps
  slurm/               00_env.sh + four sbatch scripts
  universe.py          point-in-time S&P 500 from the Wikipedia change log
  data.py              panel + 158 Alpha158 features (reads bundle or SQLite)
  forecast.py          Transformer forward-return model  --label causal|paper
  env.py agents.py     trading environment; PPO (factored) and DDPG
  train.py             phased alternating training  --signal, --alpha_unit
  run_sweep.sh         resumable driver; PARALLEL / STEPS / JOBS / RESUME
  jobs.txt             the 21-run 2×2 + baselines
  report.py            aggregate -> summary.json + console tables
  make_report.py       -> hrt_reproduction.html
  test_*.py diag_hlc.py the checks from §8
  portable.py          bundle loaders  |  export_bundle.py  rebuilds the bundle
  artifacts/
    prices_bundle.npz  15 MB, replaces the 2 GB mirror
    runs/              the 25 runs behind the tables
    runs_invalid/      10 runs from the defective PPO — evidence only,
                       DO NOT aggregate (report.py reads runs/ only)
```

Not committed, regenerated by `01_prepare` / `02_forecast`: `panel.npz` (440 MB),
`fr_causal.npz`, `fr_paper.npz`.
