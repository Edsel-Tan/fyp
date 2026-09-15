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

### Already done on the NUS cluster (login node `xcnf12`)

The repo is cloned at `/home/w/weekean/fyp/fyp` with a working `.venv`
(`torch 2.13.0+cu126`, correct for Hopper). No module system here — `00_env.sh`
has `CLUSTER_MODULES=""` and the module block is guarded. Partitions are set in
the job scripts:

| Stage | Partition | Why |
|---|---|---|
| `01_prepare`, `04_report` | `normal` | CPU only; 3 h cap is ample |
| `02_forecast` | `gpu` | 3 h cap, 30 min job |
| `03_sweep` | `gpu-long` | the only GPU partition above 3 h (3-day cap) |

Two site limits that bite:

- **`gpu` caps wall time at 3:00:00.** The 12 h sweep must go to `gpu-long`.
- **`gpu-long` will not allocate more than 48 CPUs.** A 72-CPU request fails
  with *Requested node configuration is not available* even though 96-core nodes
  are listed. So the sweep runs 48 CPU / 128 GB / `PARALLEL=24`.
- **`gpu-long` has no H200.** H200 exists only in `gpu`, whose 3 h cap the sweep
  cannot use in one allocation. The sweep lands on H100/A100 — irrelevant for a
  kernel-launch-bound workload.

**Three bugs fixed in the job scripts**, each of which broke the chain here:

1. **Stage ordering.** `01_prepare` ran `passive.py`, which loads `fr_causal.npz`
   — a file `02_forecast` produces. Stage 01 could never succeed on a clean
   checkout; it died with `FileNotFoundError` and took the whole dependency chain
   with it (`DependencyNeverSatisfied`). `passive.py` now runs in `02_forecast`,
   after `forecast.py`.
2. **`srun` inside the sweep.** `srun --unbuffered ./hrt/run_sweep.sh` dies
   instantly with *CPU binding outside of job step allocation … Unable to satisfy
   cpu bind request* — the batch step's CPU mask does not match what `srun` tries
   to bind. `run_sweep.sh` forks its own workers with `xargs -P` and needs no step
   launcher, so the `srun` is simply gone.
3. **Silent CPU fallback.** `train.py --device` defaults to `cpu` whenever
   `torch.cuda.is_available()` is false, so a GPU-less or broken-GPU node runs the
   entire sweep on CPU with nothing in the logs to say so. The sweep now asserts
   CUDA before launching and refuses to start otherwise. This is the same class of
   silent failure as the PPO trap in §2 — worth the four lines.

**GPU selection.** `gpu:1` on `gpu-long` schedules soonest but lands on `xgpe2`
(`gpu:nv`), where the batch step reported *CUDA unknown error* and `cuda=False`.
`gpu:nv` is not inherently broken — stage 02 ran fine on `xgpd0`, a TITAN V — but
the sweep now asks for `gpu:h100-96:1` explicitly. Approximate queue waits when
tested: `nv` immediate, `h100-96` ~25 min, `a100-80` ~1 h, `a100-40` ~2 h.

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
| `01_prepare` | rebuild panel from the bundle; benchmarks; both test suites | 8 CPU, 24 GB, no GPU |
| `02_forecast` | both forward-return models; passive floor; signal-value check; **HLC diagnostic** | 1 GPU, 8 CPU, 24 GB |
| `03_sweep` | all 60 RL jobs on one GPU (finished runs skipped) | 1 GPU, 48 CPU, 128 GB |
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

`03_sweep.sbatch` is the default and the right one on a busy queue: 24 workers
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

**The sentiment arm is still the wrong place to start, but for a different
reason than stated here before.** Re-probed 2026-08-25 from the login node (data
routes answer unauthenticated, so no token is needed to check any of this):

- `/news/{symbol}` and `/news/search` — dead as described. Date parameters are
  silently ignored; both serve only the trailing fortnight. `/news/AAPL` with
  `since`/`until` now returns `[]` rather than a 500, which is the same
  no-history outcome.
- **`/kols/tweets/search` does serve history.** `since`/`until` are honoured
  — `?q=$AAPL&cashtag=AAPL&since=2016-01-01&until=2016-03-01` returns 83 tweets
  strictly inside that window. `from`/`to` and `start`/`end` are ignored, which
  is likely what the earlier "ignores every date parameter" finding actually hit.
  Results cap at 200 per window, so page by narrowing the window, not by offset.

So the 2015–2019 window is **reachable** after all. What kills it is density,
not access. Coverage of a 20-symbol sample of the 2015 S&P constituents,
counting symbol-days with at least one tweet in a June:

| Window | Tweets | Symbols with zero | Symbol-days covered |
|---|---|---|---|
| 2015-06 | 87 | 8/20 | **13.1%** |
| 2019-06 | 173 | 8/20 | **24.8%** |
| 2021-06 | 873 | 5/20 | 45.5% |
| 2022-06 | 781 | 3/20 | 46.7% |

A per-name daily sentiment feature over the training window would be ~85% empty,
and the empties are not random — they concentrate in exactly the non-mega-cap
names. These are also curated KOL tweets carrying no sentiment score, a different
modality from the paper's news sentiment, and scoring them needs an LLM pass the
compute nodes cannot make (no internet) .

Three shapes that do fit the data, in order of how much they change the experiment:

1. **A market-level daily sentiment scalar** instead of a per-name feature.
   Unfiltered daily volume is ~17–31 tweets/day in 2017–2019 and hits the
   200 cap by 2022 — dense on every trading day. This is the cheapest honest
   sentiment arm and keeps the 370-name universe intact.
2. **Restrict the universe to the high-coverage names.** Dense-ish, but selects
   on attention, which is its own survivorship-flavoured bias.
3. **FNSPID** (`Zihan1004/FNSPID` on HuggingFace, reachable from the login node).
   `Stock_news/All_external.csv` is 5.3 GB and `nasdaq_exteral_data.csv` 21.6 GB;
   raw, with no sentiment scores. The precomputed scores are in
   `benstaf/nasdaq_news_sentiment` (0.5 GB, DeepSeek and Llama variants) but only
   for the 89-name Nasdaq set. Raw FNSPID is far broader than that benchmark
   subset, so scoring it yourself is the one route that could keep the S&P
   universe — at the cost of an LLM pass over millions of records.

Whichever way, it is a different experiment, worth choosing deliberately rather
than drifting into.

Better uses of the hardware, in order:

1. **Transaction-cost modelling (idea 6).** The finding is already sitting there:
   4.4 points of gross edge, 6.33 points of commission. Add a turnover penalty
   and a non-constant cost model and measure whether the edge survives. Report
   net *and* gross — that split is where the story lives.
2. ~~**More seeds.**~~ **Done / running.** `jobs.txt` now spans seeds 0–9 for
   every arm — 60 jobs, of which the 25 committed runs are skipped and 35 are
   new. Dispersion was wide (HRT-FR 2022 spans ~20 points across seeds) on 4
   seeds; this brings it to the paper's 10.
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
  jobs.txt             the 2×2 + baselines, seeds 0–9 (60 jobs)
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
