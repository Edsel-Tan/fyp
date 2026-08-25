# Reproducing the Hierarchical Reinforced Trader

A from-scratch reproduction of Zhao & Welsch, *"Hierarchical Reinforced Trader (HRT):
A Bi-Level Approach for Optimizing Stock Selection and Execution"*
([arXiv:2410.14927v1](https://arxiv.org/abs/2410.14927)), built against the local
findata SQLite mirror. No reference implementation was ever published, so every
underdetermined choice in the paper is pinned explicitly here and, where the text
admits two readings, both are run.

## What is reproduced

**HRT-FR** — the paper's forward-return-only ablation. Full HRT additionally
scores each stock daily with FinGPT sentiment over sampled news; that channel is
not reproduced (see *Data gaps*).

Test years are 2021 (bullish) and 2022 (bearish), each entered fresh with $1M.

## Setting up on a fresh machine

The findata SQLite mirror (~2 GB) and the feature panel (~440 MB) are **not** in
this repo. They do not need to be: `artifacts/prices_bundle.npz` (14 MB) carries
the exact slice everything downstream reads — the 370-name OHLCV panel, the
closes running past it for the dividend unwind, the dividend record, and the
S&P 500 index. `data.py`, `baselines.py` and `deadjust.py` use the mirror when it
is present and fall back to the bundle when it is not, so a clean checkout runs
with no re-pull.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126   # match your CUDA
pip install -r hrt/requirements.txt
python hrt/data.py            # rebuilds the feature panel from the bundle, ~7 min
```

Only re-pull from findata if you need symbols or fields outside the bundle:

```bash
export LUMID_TOKEN=...        # not in this repo; see "Secrets" below
python scripts/findata_pull.py
python hrt/export_bundle.py   # refresh the bundle afterwards
```

### Secrets

`LUMID_TOKEN` is a findata personal access token. It is **not** committed. Keep
it in a git-ignored `.env` at the repo root and load it with
`set -a; source .env; set +a`. Rotate it if it has ever been committed.

## Running it

```bash
python hrt/universe.py       # point-in-time S&P 500 from the Wikipedia change log
python hrt/data.py           # panel + 158 Qlib Alpha158 features   -> panel.npz
python hrt/forecast.py --label causal    # HLC signal               -> fr_causal.npz
python hrt/forecast.py --label paper     # the paper's literal timing
python hrt/baselines.py      # min-variance, equal-weight, S&P 500
python hrt/passive.py        # do-nothing floor inside the same env
python hrt/deadjust.py       # price-only panel, to size the dividend tailwind
python hrt/signal_strategy.py            # what the signal is worth without RL
./hrt/run_sweep.sh          # the RL sweep (2x2 x seeds + baselines)
python hrt/report.py         # aggregate -> summary.json + console tables
python hrt/make_report.py    # -> hrt_reproduction.html
```

Checks, all of which should pass before any result is trusted:

```bash
python hrt/test_alpha158.py  # features vs an independent pandas implementation
python hrt/test_env.py       # value conservation, long-only, cost monotonicity
python hrt/diag_hlc.py --signal paper    # can the HLC learn its own signal?
```

## Two ambiguities in the paper, both material

**Signal timing.** The forward return is defined as the change in opening price
from day *T* to day *T*+1, predicted from features observed through day *T*. Day
*T*'s close lies inside that window, and Alpha158 carries `KMID = (close-open)/open`
and twelve more features built on the same bar. Trained as written the model
scores test IC **+0.82**; trained causally, **+0.012**. Qlib's own Alpha158 handler
labels with `Ref($close,-2)/Ref($close,-1)-1`, which skips exactly this window.
`--label {causal,paper}` selects the reading.

**Alpha decay.** `alpha_t = alpha_0 * exp(-lambda*t)` with `lambda=1e-3`, but *t* is
never defined. Per env step, alpha is spent by step 3,000 of 500,000. Per episode,
it decays across the whole run. `--alpha_unit {step,episode}` selects the reading.

## One implementation trap

The high-level controller is a factorised categorical over 3^370. A stock PPO sums
the entropy bonus and importance ratio across action dimensions; at N=370 the
entropy term is `0.01 * 370 * ln 3 ~ 4.06` against a policy-gradient term of order
0.01, so the policy is pinned at uniform random and never learns. It is easy to
miss because a random selection policy still deploys capital and tracks the market,
so the returns look reasonable. `agents.PPO(factored=True)` takes the objective and
entropy per dimension; `adv_dim` supplies per-stock credit, which is what the
paper's own `alpha*sum_i r_i^align + (1-alpha)*r_l` decomposition implies.
`diag_hlc.py` is the check: under the leaked signal a working HLC must learn
`action = sign(forecast)`.

## Data gaps

| Gap | State | Needed |
|---|---|---|
| News text + FinGPT sentiment | **blocking — not solvable from findata** | see below |
| Prices for delisted securities | **blocking** | CRSP / Sharadar / Norgate. 108 of 133 unpriceable 2015 constituents are index removals |
| Price-only OHLCV | closed | reconstructed in `deadjust.py`, validated to ~0.2% on non-spin-off names |
| Authors' universe + label code | unavailable | never released |

### The news gap is a dead end in findata

Probed 2026-08-25 with a valid token: `/news/{symbol}` returns at most 200 items,
**ignores every date parameter** (`from`/`to`, `start`/`end`, `published_after`)
and ignores paging, serving only the trailing ~2 weeks. `since`/`until` returns
HTTP 500. There is no way to reach the 2015–2019 training window, so the full
HRT sentiment channel cannot be built from this source at any scale of hardware.

The workable path is the dataset HRT's own v2 revision uses: **FNSPID**
(Nasdaq news, 1999–2023) via the FinRL-DeepSeek benchmark on HuggingFace, which
also ships precomputed LLM sentiment and risk scores. That swaps the universe
from S&P 500 to an 89-name Nasdaq set, which is a change of experiment rather
than a port — worth deciding deliberately before starting.

## What is in `artifacts/`

| Path | |
|---|---|
| `prices_bundle.npz` | the portable data slice; everything else is derived from it |
| `universe.json`, `wiki/` | point-in-time membership and the cached Wikipedia source |
| `runs/` | the 25 runs behind the reported tables (4 arms x 4 seeds, plus baselines) |
| `runs_invalid/` | 10 runs from the defective PPO described above, kept because the report quotes them as evidence. **Do not aggregate these** — `report.py` reads `runs/` only |
| `summary.json`, `baselines.json`, `passive.json`, `signal_strategy.json`, `price_only.json` | aggregated results and non-RL benchmarks |
| `hrt_reproduction.html` | the written-up report |

`panel.npz` and `fr_*.npz` are deliberately not committed: the first is 440 MB and
the second takes ~90 s to regenerate. Both come back from `data.py` / `forecast.py`.

## Known limitations

- 370 names rather than 500; the shortfall is survivorship, not sampling.
- Panel gaps are forward-filled, with a back-fill at the very start of the
  warm-up window (2013) that never touches train, validation or test.
- Long-horizon seed dispersion is wide; means over 4 seeds indicate direction
  and magnitude, not precise levels.
