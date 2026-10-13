# RAW — every run, its methodology, and its numbers

FYP **H398280**. Working file, not a deliverable — git-ignored.

**Purpose.** `RESULTS.md` argues; this file does not. Part I states the methodology of
every experiment that has been run, in enough detail to re-derive it. Part II is the
numbers, tabulated, with no interpretation. Where `RESULTS.md` quotes a figure, this is
where it came from.

**Conventions.**
* Every number here was produced by a script in this repo and read back from its artifact
  (`*.json`, `*.npz`, `*.csv`), not transcribed from `RESULTS.md`.
* Sharpe ratios are **annualised** — √252 for daily equity bars, √(24·365) for hourly
  crypto bars — and are **gross of cost unless a cost column says otherwise**.
* `pp` = probability points (1 pp = 1 cent on a $1 binary contract). `bp` = basis points.
* "±" is the standard deviation across seeds or replicates, never a standard error, unless
  labelled `s.e.`.
* Run dates: the equity RL sweep 2026-08 (seeds 0–3) and **2026-09-15 on the NUS cluster**
  (seeds 4–9, plus the 25-run null sweep); everything else 2026-09-07, except §3.4 and §3.5
  of `RESULTS.md`, run 2026-09-14, and §3.6 run 2026-09-14.

---
---

# PART I — METHODOLOGY

## 0. Run inventory

| # | script | produces | runtime | hardware | status |
|---|---|---|---|---|---|
| A1 | `hrt/data.py`, `hrt/deadjust.py`, `hrt/universe.py` | `panel.npz` (2,414 d × 370 names × 158 feat), `prices_bundle.npz` | — | CPU | done |
| A2 | `hrt/forecast.py` | `fr_causal.npz`, `fr_paper.npz` | ~1 GPU-h each | GTX 1080 Ti | done |
| A3 | `hrt/train.py` × 60 | `artifacts/runs/*.json` (6 arms × 10 seeds) | ~3 GPU-h per run, 189 GPU-h total | GTX 1080 Ti (s0–3) + H100 (s4–9) | **extended 2026-09-15** |
| A4 | `hrt/train.py` × 10 (defective PPO) | `artifacts/runs_invalid/*.json` | — | — | done, retained for K |
| A5 | `hrt/passive.py`, `hrt/baselines.py`, `hrt/signal_strategy.py` | `passive.json`, `baselines.json`, `signal_strategy.json` | minutes | CPU | done |
| B1 | `analysis/dsr.py` | `dsr_test2021.json`, `dsr_test2022.json` | seconds | CPU | done |
| B2 | `analysis/cost_rerank.py` | `cost_rerank.json` + stdout | seconds | CPU | **rerun on 60 runs 2026-09-15** |
| C1 | `analysis/synth_panel.py` --seed 0..3 | `panel_synth{,_s1,_s2,_s3}.npz` | ~2 min each | CPU | done |
| C2 | `hrt/forecast.py` on each null × 2 timings | `fr_synth_*.npz` (8 files) | ~1 GPU-h each | GTX 1080 Ti | done |
| C3 | `analysis/falsify.py` | `falsify.json` + stdout | seconds | CPU | done |
| **C4** | `hrt/slurm/05_sweep_null.sbatch` → `hrt/train.py` × 25 on null panels | `artifacts/runs_null/*.json` | ~4.5 GPU-h per run, **111 GPU-h total** | H100-96 (NUS) | **done 2026-09-15** |
| **C5** | `hrt/passive.py --panel panel_synth_s{0..3}` | `passive_synth_s{0..3}.json` | seconds each | CPU | **new 2026-09-15** |
| **C6** | `analysis/null_sweep.py` | `null_sweep.json` + stdout | 2 s | CPU | **new 2026-09-15** |
| D1 | `crypto/pull.py 1d` / `1hour` | `data/crypto.sqlite` | ~20 min | CPU + network | done |
| D2 | `analysis/coverage.py` | stdout coverage tables | seconds | CPU | done |
| D3 | `crypto/panel.py --interval 1hour` | `panel_1hour.npz` (30,079 × 17 × 158, 45 blocks) | ~3 min | CPU | done |
| D4 | `crypto/forecast.py --label causal` / `paper` | `fr_causal.npz`, `fr_paper.npz` | ~20 min each | GTX 1080 Ti | done |
| D5 | `crypto/synth_panel.py --seed 0..2` + forecast | `fr_synth_causal_s{0,1,2}.npz` | ~20 min each | GTX 1080 Ti | done |
| D6 | `crypto/strategy.py` | `strategy_costs.json`, `strategy_paths.npz` | ~1 min | CPU | done |
| D7 | `crypto/dsr.py` | `dsr_strategy.json` | seconds | CPU | done |
| E1 | `analysis/censoring.py --reps 20` | `analysis/censoring.json` | ~10 min | CPU | done |
| F1 | `pm/pull.py discover` / `trades 900`, `pm/outcomes.py` | `data/pm.sqlite`, `pm_resolved.csv` | ~40 min | CPU + network | done, **superseded** |
| F2 | `pm/skill.py`, `pm/imbalance.py` | `pm_skill.csv`, `pm_bars.csv` | minutes | CPU | done, **superseded** |
| G1 | `pm/pmv1_pull.py` | `data/pmv1/` (2,105 files, 16.8 GB) | 13 min | network | done |
| G2 | `pm/pmv1_prep.py` | `pmv1_trades.parquet` (730 M rows, 7.9 GB) | 58 s | CPU (48 core) | done |
| G3 | `pm/pmv1_prep.py --only bars --freq {1h,4h}` | `pmv1_bars_{1h,4h}.parquet` | ~2 min | CPU | done |
| G4 | `pm/pmv1_skill.py --sims 10000 --sample 50000` | `pmv1_skill.{csv,json}`, `pmv1_accounts.parquet` | ~70 min (MC-bound) | CPU | done |
| G5 | `pm/pmv1_persistence.py --cut {2024-07-01,2025-01-01,2025-07-01}` | `pmv1_persistence_<cut>.{csv,json}` | ~6 min each | CPU | done |
| G6 | `pm/pmv1_imbalance.py --freq {1h,4h}` | `pmv1_imbalance_{1h,4h}.json` | ~5 min each | CPU | done |
| **G7** | `pm/pmv1_pooling.py` | `pmv1_pooling.json`, `pmv1_markets.parquet` | **33 s** | CPU | **new 2026-09-14** |
| **G8** | `pm/pmv1_spread.py --freq 1h` | `pmv1_spread_1h.json`, `pmv1_spreadbars_1h.parquet` | **100 s** | CPU | **new 2026-09-14** |

**Nothing on this list is now blocked on hardware.** C4 was the last item and it ran on
2026-09-15. Seeds 0–3 of A3 are byte-identical before and after the cluster run, so the
extension to ten seeds is an extension and not a re-run. **Still not run:** no CPCV or PBO
for any workstream, and no resampling of the *real* equity panel — so the null side of C4
has a dispersion across four panels while the real side has one draw of market history.
`CLUSTER.md` records the three job-script bugs fixed to make the chain run and the site
limits that bind.

## 1. Data

| dataset | source | span | size | key fields / shape |
|---|---|---|---|---|
| equity mirror | findata API → `data/findata.sqlite` | 2013-06 → 2022-12 (panel) | 2.23 GB | 2,414 trading days × 370 S&P 500 names |
| crypto | findata `/ohlc/{symbol}`, undocumented | 2020-01 → 2026-09 | 158 MB | 42 live pairs; 27,566 daily + 1,150,609 hourly bars |
| PM vendor tape | findata `/prediction-markets` | 2026-05-21 → 2026-09-04 | 191 MB | 382,119 trades, 887 markets, 109,053 takers |
| **Polymarket-v1** | HF `TimeSeventeen/Polymarket-v1`, CC-BY-4.0, ungated | 2022-11-21 → 2026-04-28 | 16.8 GB raw + 8.9 GB derived | 746,110,412 rows across 2 cleaned layers |

Polymarket-v1 layers pulled: `daily_aligned/` (standard binary) and `daily_aligned_multi/`
(negative-risk multi-outcome). `OrderFilled/` (1.20 B nominal rows) deliberately **not**
pulled — it retains platform relayer/router records that the cleaned layers remove.

Row-count chain, which must reconcile in the write-up: **1.20 B** raw `OrderFilled` →
**746,110,412** in the two cleaned layers → **730,492,126** after keeping resolved markets
with a winning label and `0.0005 < price < 0.9995` → **727,446,812** fills belonging to the
1,819,876 accounts with ≥10 fills.

## 2. Methodology by workstream

### A. HRT reproduction (equity RL)
Universe: point-in-time S&P 500 membership; 370 of 500 names survive the requirement that
prices exist for the whole window (108 of 133 unpriceable 2015 constituents are index
removals). Prices de-adjusted in `deadjust.py`, validated to ~0.2 % on non-spin-off names.
Features: **Alpha158** (Qlib), validated against an independent implementation in
`hrt/test_alpha158.py`. Split: train ≤2018, validate 2019, test 2021 and 2022 separately.

Two **label timings** are run throughout, and the distinction is load-bearing:
* `paper` — open-to-open return *T* → *T*+1, predicted from features observed through day
  *T*. Day *T*'s close lies inside the label window.
* `causal` — the same return with the same-bar window excluded.

Arms: `hrt_{causal,paper}` and `hrt_{causal,paper}_ep` (hierarchy, step- vs
episode-unit alpha), `ppo_causal`, `ddpg_causal` (flat baselines). **10 seeds each**
(0–3 local, 4–9 on the cluster; 0–3 are byte-identical across the two runs). Every run is
501,760 steps with a validation evaluation every 25,000. Environment charges a flat **10 bp**
per unit turnover. Reported Sharpe in the DSR table is the arm's **best seed**, since the
quantity being deflated is a maximum over a search.

**Passive floor.** `hrt/passive.py` pushes a do-nothing policy — deploy the cash evenly on
day 1 subject to the same `hmax` share cap, then hold — through the *same* `TradingEnv` the
agents use: integer share lots, cash-sequenced fills, the same 10 bp charge. This is
distinct from `baselines.py`'s equal-weight benchmark, which is a weight-space calculation
and is not achievable inside the environment. Excess-over-floor is the unit of comparison
throughout §4.5.

### B. Deflated Sharpe Ratio [8]
Bailey & López de Prado applied to every run. **K = 70** — the 60 runs in `runs/` *and* the
10 in `runs_invalid/`, on the grounds that the surviving configuration was chosen after
seeing both. (It was K = 35 before the sweep went to ten seeds; `dsr_test20**.json` now
carries `K` and `sr_star_ann` so that nothing downstream can quote a stale threshold.) σ_SR is the cross-trial standard deviation of daily Sharpe; SR\*₀ is the
expected maximum from K trials. PSR(0) is against a zero benchmark; DSR is against SR\*₀.
Skew and kurtosis enter through the PSR estimator.

For crypto (`crypto/dsr.py`), the search is 11 strategies × 4 cost models, so K is reported
at **11** (fix the cost model, search strategies) and **44** (choose the cost model
afterwards too). Under a cost model, two strategies lose nearly the whole book (Sharpe −57)
so σ_SR explodes; a **robust IQR-based scale** is reported alongside. **Flagged as a
methodological choice that needs checking against [8] — see `PREPARATION.md` Q2.**

### C. Cost re-ranking [2]
**Equities** (`analysis/cost_rerank.py`): policies were not checkpointed and each run cost
~2 h, so agents cannot be re-simulated. Run records carry the net value path, mean
per-step turnover and total cost paid at 10 bp, which permits a **first-order recharge** to
another flat rate: `r_net(ρ) = r_net(10bp) + (0.001 − ρ)·turnover`. This is an
approximation in level and is used **only for ordering**, which is MACE's actual claim.
Rates: 0, 10, 30, 60 bp.

**Crypto** (`crypto/strategy.py`): 11 strategies over the clean 2026 out-of-sample window
(2,665 hourly bars, 17 coins, 4 blocks) under 4 cost models — no cost, flat 10 bp, flat
30 bp, and a state-dependent square-root model (half-spread + `Y·σ_t·√(q/V_t)`, charging
each trade against its own bar's realised volatility and traded volume). Two variants
(`calm bars only`, `stressed bars only`) are built specifically to hold turnover near-equal
while inverting volatility exposure.

### D. Falsification audit [10]
`analysis/synth_panel.py` generates panels matching the real one in **shape, calendar,
per-name volatility, market-factor loading and overnight/intraday variance split**, in
which returns are a **martingale difference**. Conditional *variance* is left predictable
(stochastic volatility, volume clustering) because that is realistic and because Alpha158
is largely a volatility/volume feature set — the constraint is that no feature can carry
information about the **sign** of any future return. Realised lag-1 return autocorrelation
−0.027 on the null against −0.044 on the real panel.

Four independent equity panels (seeds 0–3), three crypto panels (seeds 0–2). The
**entire** pipeline — Alpha158 → Transformer → top-30 long book — is re-run on each, under
both label timings. Leakage mechanism is measured separately by regressing the label on
`KMID = (close − open)/open` alone.

**D2. The audit at agent level** (`hrt/slurm/05_sweep_null.sbatch`, `analysis/null_sweep.py`).
The same 25-run agent sweep behind table 1, retrained on those four null panels at the
identical 501,760-step budget. Seed *k* trains on panel *k* mod 4, so the four independent
panels carry the dispersion rather than the seed alone. `hrt/report.py` is deliberately not
reused for scoring: it compares against `baselines.json`, which is the *real* panel's
benchmarks. Instead each run is scored against `passive_synth_s{k}.json` — the do-nothing
floor measured inside the same environment on the same panel — because a null panel has no
drift by construction and its *realised* drift is an unconstrained noise draw that an agent
deploying capital simply inherits. Every quantity reported in table 4.5 is an excess over
that floor. Two `train.py` flags exist only for this sweep, both defaulting to prior
behaviour: `--fr` (the forecast file was hardcoded to `fr_causal`/`fr_paper`) and
`--runs_dir` (keeps null runs out of `runs/`, which `report.py` globs).

### E. Censoring cost
The crypto tape's censoring cannot be measured against itself — the uncensored series does
not exist. So the mask is applied **in reverse**, to a panel that *is* complete: the
2,414-day × 370-name S&P 500 panel, 2013-06 → 2022-12. Three views are compared:
* **truth** — the complete panel;
* **naive** — retained rows concatenated, gaps ignored (what pivoting a vendor response and
  calling `.diff()` produces);
* **block** — retained rows cut into contiguous runs, every window and every return confined
  to one run.

Two masks matching the two observed shapes of censoring: **`gap`** (the daily tape's
empirical 9-on/21-off run sequence, 36.8 % retention) and **`seasonal`** (the hourly tape's
month-of-year retention: Feb 0 %, Aug 7 %, Q2/Q4 ~97 %, 65.7 % retention). 20 phase
replicates each. Strategy: long top-30 on trailing momentum, at 5 and 20 days.

### F. Crypto panel construction
Coverage is not random (Part II §5), so `crypto/panel.py` cuts the tape into **contiguous
blocks** and confines every feature window, every label and every portfolio return to a
single block. Universe = the 17 coins with ≥98 % coverage in **every** block; requiring it
only in recent blocks would admit more coins at the cost of a look-ahead-shaped
survivorship filter. Result: 30,079 bars × 17 coins × 158 features over 45 blocks,
2020-11 → 2026-06. Train ≤2024-12 (21,961 bars), validate 2025 (5,453), test 2026 (2,665).
Model, feature set and timing ablation identical to the equity run.

### G. Prediction markets — payoff definition
A taker who takes direction `D ∈ {+1,−1}` on `shares = usdc_amount / price` of a token
settling at `win_event`, when the event-normalised price was `p_event`, realises

```
s = D · shares · (win_event − p_event)
```

`p_event` and `win_event` both live on the `outcome_seq = 1` axis, so no hand-rolled leg
alignment is needed. `s` is the quantity every sign-randomisation null flips; because
`D = ±1`, the null variance of a unit is exactly `Σ s²`, so sd(z) is a closed-form function
of sums and needs no simulation.

`event` = `neg_risk_market_id` where one exists (ground truth from the negative-risk
contract), `condition_id` otherwise.

### G1. Skill test [13]
Four nulls compared. The first three differ only in the unit receiving an independent coin
flip: **trade** (every fill, as published), **bet** (every trader × market position),
**event** (every trader × negative-risk event). The fourth, `event (recentred)`, shifts the
null by the **population median z** — in the statistic's own units — so the question becomes
whether an account beats the average taker rather than zero.

Two passes: sd(z) and mean z are **exact over all 1,819,876 accounts**; flagged fractions
need the exact Rademacher null rather than a normal approximation, so they are **simulated
(10,000 draws) on 50,000 randomly sampled accounts**, sorted by `taker_id` before sampling
because DuckDB's parallel aggregation does not fix row order.

### G2. Persistence (the design that uses no null)
Split each account's history at a cut date; compute event-level z in each half; report the
second half conditional on the first. **Confound closed first:** a market whose trading
straddles the cut puts the *same* resolution outcome on both sides, so an account holding it
looks skilled in both halves for one reason. **Events are therefore assigned whole** — an
event counts only if all its fills fall on one side — and the halves then share no outcome.
Accounts require ≥10 fills in *each* half. Three cuts: 2024-07-01, 2025-01-01, 2025-07-01.

### G3. Imbalance [14]
Hourly (and 4-hourly) bars; bar price = last `p_event` print; imbalance = signed notional
over gross notional, computed **separately for large trades** (top decile by notional
*within the market*) and the rest; markets require ≥$100k volume and ≥20 bars; consecutive
bars only. Returns in **log-odds**. Market fixed effects by **within-transformation** (a
dummy matrix at 25,317 markets does not fit). Standard errors **clustered two-way on market
and on hour** (Cameron–Gelbach–Miller).

### G4. Pooling ladder — `pm/pmv1_pooling.py` (new)
Extends G1's unit ladder past the negative-risk event, to test `RESULTS.md` §3.1's stated
conjecture that the residual over-dispersion is same-day markets sharing an underlying.
Market dimension built from the raw layers: 838,118 resolved markets → `close_day`
(`close_at`, falling back to the last print; `resolved_at` is null on 63 % of markets) and
`category_refined`. Rungs: trade → bet → event → **trader × close-day × category** →
**trader × close-day** → **trader** (degenerate, one flip). The coarse rungs deliberately
**over-pool**, which is the right direction for an upper bound: they cannot leave a pooling
artefact behind.

**Diagnostic corrected.** The null fixes the *second* moment — E[z] = 0 and Var(z) = 1 give
**E[z²] = 1 exactly** — so `rms(z)` is the quantity that must equal 1.000, decomposing as
`rms² = mean² + sd²`. §3.1 quotes centred `sd(z)`, which sees only the dispersion part. Both
are reported. The degenerate rung forces z = ±1 and therefore **must** return rms = 1.000;
it is a check on the arithmetic, not a result.

**Skill-variance decomposition.** Writing `z = δ_i·√n_i + N(0,1)` gives `Var(z) = 1 + V`
with `V = Var(δ√n)`. A split puts ~n/2 units each side, so each half carries V/2 and the
halves share only δ:

```
corr(z_in, z_out) = (V/2)/(1 + V/2)    ⟹    V = 2ρ/(1 − ρ)
```

ρ is **Pearson** (the decomposition is about variance), computed on z **winsorised at 0.5 %
a side** because a few accounts hold one enormous resolved position. Because the paired
accounts are a selected subpopulation, the event-level ladder is **recomputed on exactly
those accounts** for comparison.

### G5. Spread and tradeability — `pm/pmv1_spread.py` (new)
`fee_usdc` is **identically zero** across the whole archive (Polymarket v1 levied no
explicit taker fee) and `taker_base_fee` is set on only 3.6 % of fills, so the cost is the
spread and must be estimated. Because `D` is ground truth, the two taker sides separate
inside every bar:

```
spread_p = vwap(p_event | D = +1) − vwap(p_event | D = −1)
```

**The backtest does not use that estimate.** §3.3 regresses on the bar's *last print*, which
carries bid-ask bounce running against the signal; entering at the last print and then
subtracting a spread charges the crossing twice. So the round trip is priced at the VWAPs
themselves:

* signal known at close of bar *h*; position opened in *h+1*, closed in *h+1+k*;
* **long**: buy at `vwap_buy(h+1)`, sell at `vwap_sell(h+1+k)`;
* **short**: sell at `vwap_sell(h+1)`, buy back at `vwap_buy(h+1+k)`;
* **gross benchmark**: the same trip priced mid-to-mid, `mid = (vwap_buy + vwap_sell)/2`;
* cost = gross − net = the half-spread paid entering + the half-spread paid leaving;
* capital at risk = entry price when long, `1 − entry price` when short.

ROI is **value-weighted** (Σ PnL / Σ capital), not a mean of ratios, because capital can be
0.01 and a mean of `pnl/capital` is then decided by a handful of penny contracts. Holds
tested: 1, 2, 4, 8, 24 bars. Signal-strength conditioning: `|oi| = 1`, `> 0.5`, `≤ 0.5`.
Spread is observable only where both sides trade inside one bar — **51.4 % of bars**.

---
---

# PART II — RAW RESULTS

## 1. HRT reproduction — per-arm, per-year (mean ± sd across seeds)

Published target: **2021 +39.8 % / 2022 +2.3 %**.

**2021**

| arm | seeds | cum return | Sharpe (ann.) | vol | max DD | turnover/step |
|---|---:|---:|---:|---:|---:|---:|
| hrt_causal | 10 | +21.9 % ± 4.8 % | +1.70 ± 0.39 | 0.130 | −6.8 % | 0.195 |
| hrt_causal_ep | 10 | +21.1 % ± 4.1 % | +1.65 ± 0.35 | 0.129 | −6.4 % | 0.226 |
| hrt_paper *(leaky)* | 10 | +78.3 % ± 11.4 % | +6.29 ± 1.04 | 0.126 | −4.6 % | 0.330 |
| hrt_paper_ep *(leaky)* | 10 | +100.1 % ± 11.7 % | +8.16 ± 0.83 | 0.123 | −4.3 % | 0.375 |
| ppo_causal | 10 | +29.1 % ± 2.3 % | +2.42 ± 0.21 | 0.121 | −5.7 % | 0.004 |
| ddpg_causal | 10 | +27.3 % ± 2.0 % | +2.22 ± 0.14 | 0.124 | −5.9 % | 0.004 |

**2022**

| arm | seeds | cum return | Sharpe (ann.) | vol | max DD | turnover/step |
|---|---:|---:|---:|---:|---:|---:|
| hrt_causal | 10 | −11.3 % ± 6.1 % | −0.51 ± 0.26 | 0.222 | −23.2 % | 0.212 |
| hrt_causal_ep | 10 | −7.0 % ± 6.9 % | −0.32 ± 0.31 | 0.220 | −19.8 % | 0.245 |
| hrt_paper *(leaky)* | 10 | +54.3 % ± 17.7 % | +2.88 ± 1.12 | 0.193 | −10.6 % | 0.350 |
| hrt_paper_ep *(leaky)* | 10 | +67.8 % ± 7.7 % | +3.45 ± 0.42 | 0.199 | −10.0 % | 0.411 |
| ppo_causal | 10 | −11.4 % ± 2.6 % | −0.51 ± 0.10 | 0.224 | −21.5 % | 0.004 |
| ddpg_causal | 10 | −9.3 % ± 4.5 % | −0.42 ± 0.20 | 0.222 | −20.5 % | 0.004 |

**Passive floors.** 2021: passive-in-env **+26.5 %** (Sharpe +2.04), equal-weight
buy-and-hold +32.8 %, S&P 500 +28.8 %. 2022: passive-in-env **−5.6 %** (Sharpe −0.28),
equal-weight −5.4 %, S&P 500 −19.9 %.

**Excess over the in-environment passive floor**, per run, pooled by label timing (this is
the quantity table 4.5 compares against the null):

| year | arms | mean excess | t | runs above floor |
|---|---|---:|---:|---:|
| 2021 | leak-free (causal) | **−0.017** | −2.16 | 17/40 |
| 2021 | leaky (paper) | +0.627 | +17.70 | 20/20 |
| 2022 | leak-free (causal) | **−0.041** | −4.79 | 8/40 |
| 2022 | leaky (paper) | +0.667 | +19.89 | 20/20 |

At 4–5 seeds the 2021 leak-free shortfall was inside the noise; at 10 it is not. The *t* is
across seeds on a single panel and therefore carries no information about how the excess
would vary across market histories.

**Change from 4–5 seeds to 10.** Seeds 0–3 are unchanged. The leak-free arms moved
+22.4 → +21.9 % and −13.4 → −11.3 % (`hrt_causal`), i.e. within the seed sd; the leaky arms
moved more (`hrt_paper` 2022 +45.0 → +54.3 %, sd 3.4 → 17.7 pp), which is the arm with the
widest seed dispersion and the reason the 2022 best-seed Sharpe jumped 2.44 → 5.99.

## 2. Deflated Sharpe, equities (K = 70; best seed per arm)

`[invalid]` rows are the `runs_invalid/` arms, reported because they count towards K.

**2021** — σ_SR = 0.1004 daily, **SR\*₀ = 3.829 annualised**

| arm | seeds | Sharpe | skew | kurt | PSR(0) | **DSR** | at K = 35, 4–5 seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| hrt_paper_ep *(leaky)* | 10 | +9.35 | +0.17 | 5.55 | 1.000 | **0.988** | 9.35 → 0.999 |
| hrt_paper *(leaky)* | 10 | +8.79 | +0.27 | 5.82 | 1.000 | **0.983** | 6.02 → 0.936 |
| ppo_causal | 10 | +2.81 | −0.12 | 4.01 | 0.992 | 0.093 | 2.51 → 0.268 |
| ppo_causal *[invalid]* | 4 | +2.60 | −0.26 | 3.96 | 0.988 | 0.072 | 2.60 → 0.285 |
| hrt_causal *[invalid]* | 3 | +2.57 | −0.01 | 3.96 | 0.988 | 0.065 | 2.57 → 0.273 |
| ddpg_causal | 10 | +2.47 | −0.16 | 4.00 | 0.985 | 0.058 | 2.35 → 0.229 |
| hrt_paper *[invalid]* | 3 | +2.25 | +0.11 | 4.61 | 0.979 | 0.038 | 2.25 → 0.196 |
| hrt_causal | 10 | +2.11 | −0.29 | 4.20 | 0.971 | 0.033 | 2.11 → 0.175 |
| hrt_causal_ep | 10 | +2.04 | −0.09 | 3.47 | 0.969 | 0.028 | 2.04 → 0.159 |

**2022** — σ_SR = 0.0860 daily, **SR\*₀ = 3.281 annualised**

| arm | seeds | Sharpe | skew | kurt | PSR(0) | **DSR** | at K = 35, 4–5 seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| hrt_paper *(leaky)* | 10 | +5.99 | +0.15 | 4.63 | 1.000 | **0.817** | 2.44 → 0.328 |
| hrt_paper_ep *(leaky)* | 10 | +4.21 | +0.13 | 4.10 | 0.999 | **0.461** | 3.67 → 0.626 |
| ddpg_causal | 10 | +0.09 | −0.23 | 3.75 | 0.578 | 0.001 | −0.39 → 0.002 |
| hrt_causal_ep | 10 | +0.04 | +0.26 | 4.30 | 0.558 | 0.001 | −0.10 → 0.006 |
| hrt_causal | 10 | −0.13 | −0.08 | 3.60 | 0.496 | 0.001 | −0.18 → 0.005 |
| hrt_causal *[invalid]* | 3 | −0.15 | +0.06 | 3.69 | 0.486 | 0.000 | −0.15 → 0.005 |
| hrt_paper *[invalid]* | 3 | −0.24 | +0.02 | 3.79 | 0.445 | 0.000 | −0.24 → 0.004 |
| ppo_causal *[invalid]* | 4 | −0.30 | −0.08 | 3.74 | 0.422 | 0.000 | −0.30 → 0.003 |
| ppo_causal | 10 | −0.31 | −0.25 | 3.85 | 0.415 | 0.000 | −0.37 → 0.003 |

The last column is the previous version of this table. Doubling the sweep raised the best
observed Sharpe **and** the threshold, and the threshold faster: every leak-free DSR falls
by roughly two-thirds in 2021 and by an order of magnitude in 2022.

## 3. Cost re-ranking, equities — 2021 Sharpe by flat rate (first-order recharge)

| arm | turnover | 0 bp | 10 bp | 30 bp | 60 bp | rank 0 / 10 / 30 / 60 |
|---|---:|---:|---:|---:|---:|---|
| hrt_paper_ep | 0.375 | +6.47 | +5.70 | +4.17 | +1.86 | 1 / 1 / 1 / **3** |
| hrt_paper | 0.330 | +5.35 | +4.69 | +3.36 | +1.36 | 2 / 2 / 2 / **4** |
| ppo_causal | 0.004 | +2.19 | +2.18 | +2.16 | +2.14 | 3 / 3 / 3 / **1** |
| ddpg_causal | 0.004 | +2.03 | +2.02 | +2.01 | +1.98 | 4 / 4 / 4 / **2** |
| hrt_causal_ep | 0.226 | +2.00 | +1.55 | +0.67 | −0.66 | 5 / 6 / 6 / 6 |
| hrt_causal | 0.195 | +1.97 | +1.59 | +0.84 | −0.30 | 6 / 5 / 5 / 5 |

Rank changes vs the 10 bp default: **2 / 6** at 0 bp, **0 / 6** at 30 bp, **4 / 6** at 60 bp.
At 4–5 seeds the same computation gave **4 / 6**, **2 / 6**, **6 / 6**. The difference is
that at 4–5 seeds `ppo_causal` and `ddpg_causal` started *below* the causal hierarchy arms
at 0 bp and crossed them as the rate rose; at 10 seeds they start above and stay above until
60 bp, so there are fewer crossings left to count. 2022 is 3 / 6 at every rate.

2022: `hrt_causal_ep` Sharpe **+0.05 gross, −0.23 net**.

## 4. Falsification audit — full per-panel output

Real panel = S&P 500, 2,414 d × 370 names. Null panels = martingale-difference returns,
matched shape/calendar/vol/factor loading.

| panel | timing | IC train | IC valid | **IC test** | 2021 cum | 2021 Sh | 2022 cum | 2022 Sh |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| REAL S&P 500 | causal | +0.0688 | +0.0156 | **+0.0118** | +0.047 | +0.24 | −0.344 | −1.23 |
| REAL S&P 500 | paper | +0.8238 | +0.8275 | **+0.8155** | −0.111 | −0.49 | −0.374 | −1.44 |
| null #0 | causal | +0.3411 | +0.0341 | −0.0026 | −0.182 | −0.78 | −0.148 | −1.04 |
| null #1 | causal | +0.0332 | +0.0050 | +0.0077 | −0.153 | −0.92 | +0.480 | +1.76 |
| null #2 | causal | +0.0650 | +0.0165 | +0.0011 | −0.136 | −0.82 | +0.059 | +0.41 |
| null #3 | causal | +0.0233 | +0.0029 | +0.0032 | −0.176 | −0.86 | −0.409 | −1.37 |
| null #0 | paper | +0.5713 | +0.5535 | +0.5396 | −0.325 | −1.56 | −0.173 | −1.24 |
| null #1 | paper | +0.5539 | +0.5290 | +0.5317 | −0.243 | −1.42 | +0.200 | +0.64 |
| null #2 | paper | +0.5714 | +0.5410 | +0.5516 | −0.003 | −0.02 | −0.042 | −0.27 |
| null #3 | paper | +0.5512 | +0.5294 | +0.5303 | −0.168 | −0.81 | −0.320 | −1.14 |

| timing | null test IC mean | sd | range | real | **z vs null** |
|---|---:|---:|---|---:|---:|
| causal | +0.0023 | 0.0043 | [−0.0026, +0.0077] | +0.0118 | **+2.20** |
| paper | +0.5383 | 0.0097 | [+0.5303, +0.5516] | +0.8155 | **+28.46** |

Null 2021 top-30 return: causal mean −16.18 % (range −18.23 % … −13.62 %); paper mean
−18.47 % (range −32.53 % … −0.32 %).

**Leakage mechanism.** R²(`KMID` → label) = **0.653** equities, **0.580** crypto hourly.
Mechanical IC ceiling √0.653 = 0.81 against the observed 0.82.

## 4.5 Falsification audit at agent level — the 25-run null sweep

`hrt/slurm/05_sweep_null.sbatch` (H100-96, 111 GPU-h) + `analysis/null_sweep.py`.
25 runs on the four null panels of §4, seed *k* on panel *k* mod 4, at the identical
501,760-step budget as the 60 real runs. **Every figure is an excess over the
in-environment passive floor of the panel that run was trained on** (`passive_synth_s{k}.json`
for null runs, `passive.json` for real runs) — see methodology D2.

**2021** — null-panel floors −0.011, −0.015, +0.092, +0.162; real floor +0.265

| arm | *n* | null excess | *t* | beats floor | *n* | real excess | *t* | beats floor | Welch *p* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hrt_causal | 4 | +0.0855 | +0.92 | 3/4 | 10 | −0.0458 | −3.02 | 2/10 | 0.254 |
| hrt_causal_ep | 4 | −0.1107 | −2.86 | 0/4 | 10 | −0.0542 | −4.14 | 0/10 | 0.245 |
| hrt_paper *(leaky)* | 4 | +0.1023 | +1.33 | 3/4 | 10 | +0.5183 | +14.35 | 10/10 | 0.006 |
| hrt_paper_ep *(leaky)* | 4 | +0.2034 | +2.89 | 4/4 | 10 | +0.7360 | +19.91 | 10/10 | 0.001 |
| ppo_causal | 4 | +0.0017 | +0.03 | 2/4 | 10 | +0.0258 | +3.48 | 10/10 | 0.730 |
| ddpg_causal | 5 | +0.0345 | +1.31 | 3/5 | 10 | +0.0082 | +1.32 | 5/10 | 0.381 |

**2022** — null-panel floors +0.019, +0.554, +0.210, −0.158; real floor −0.056

| arm | *n* | null excess | *t* | beats floor | *n* | real excess | *t* | beats floor | Welch *p* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hrt_causal | 4 | +0.0080 | +0.07 | 2/4 | 10 | −0.0568 | −2.92 | 3/10 | 0.591 |
| hrt_causal_ep | 4 | −0.1378 | −1.01 | 1/4 | 10 | −0.0134 | −0.62 | 4/10 | 0.432 |
| hrt_paper *(leaky)* | 4 | +0.2861 | +3.22 | 4/4 | 10 | +0.5992 | +10.72 | 10/10 | 0.027 |
| hrt_paper_ep *(leaky)* | 4 | +0.2271 | +2.02 | 3/4 | 10 | +0.7349 | +30.00 | 10/10 | 0.018 |
| ppo_causal | 4 | +0.2483 | +2.79 | 3/4 | 10 | −0.0573 | −7.07 | 0/10 | 0.041 |
| ddpg_causal | 5 | +0.0565 | +1.16 | 4/5 | 10 | −0.0364 | −2.54 | 1/10 | 0.129 |

Note the floor spread on the null side: panel 1's 2022 floor is +55.4 % against panel 3's
−15.8 %, because a null panel has no drift by construction and its realised drift is an
unconstrained noise draw. Comparing raw returns across panels would be meaningless; this is
why the excess is the unit.

**Pooled by label timing** (this is where the *n* is):

| year | arms | null excess | *t* | beats floor | real excess | *t* | beats floor | null / real |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2021 | leaky (paper) | **+0.1529** | +2.94 | 7/8 | +0.6271 | +17.70 | 20/20 | **0.24** |
| 2021 | leak-free (causal) | +0.0046 | +0.15 | 8/17 | **−0.0165** | −2.16 | 17/40 | — |
| 2022 | leaky (paper) | **+0.2566** | +3.81 | 7/8 | +0.6671 | +19.89 | 20/20 | **0.38** |
| 2022 | leak-free (causal) | +0.0445 | +0.82 | 10/17 | **−0.0410** | −4.79 | 8/40 | — |

**Validation Sharpe** — the statistic `train.py` checkpoints on, one value per run:

| arms | null | real | Welch *t* (real vs null) | real runs below null median | mean percentile of a real run |
|---|---|---|---:|---:|---:|
| leak-free | **+0.933** [−0.403, +2.349], *n* = 17 | +0.431 [+0.116, +0.885], *n* = 40 | **−2.38** | **35/40** (median +0.607) | **35.7** |
| leaky | +2.384 [+1.280, +4.309], *n* = 8 | +2.840 [+2.160, +3.594], *n* = 20 | +1.05 | 0/20 (median +2.110) | 63.1 |

Per-arm null means: `hrt_causal` +1.245, `hrt_causal_ep` +1.188, `ddpg_causal` +0.790,
`ppo_causal` +0.545, `hrt_paper` +2.239, `hrt_paper_ep` +2.528. Real: +0.519, +0.349,
+0.423, +0.432, +2.447, +3.234.

**Arm ordering, null vs real** (by mean excess over floor):

| year | Spearman ρ | *p* | null order | real order |
|---|---:|---:|---|---|
| 2021 | **+0.771** | 0.072 | paper_ep > paper > causal > ddpg > ppo > causal_ep | paper_ep > paper > ppo > ddpg > causal > causal_ep |
| 2022 | +0.143 | 0.787 | paper > ppo > paper_ep > ddpg > causal > causal_ep | paper_ep > paper > causal_ep > ddpg > causal > ppo |

**The null sweep as a direct Monte Carlo of SR\*₀** (daily Sharpe, annualised by √252):

| year | subset | K | σ_SR (daily) | observed max | analytic E[max] | ratio | subset mean SR |
|---|---|---:|---:|---:|---:|---:|---:|
| 2021 | leak-free null runs | 17 | 0.0569 | **+2.139** | +1.650 | 1.30× | +0.330 |
| 2021 | leaky null runs | 8 | 0.0695 | +2.308 | +1.610 | 1.43× | +1.141 |
| 2021 | all null runs | 25 | 0.0645 | +2.308 | +2.044 | 1.13× | +0.590 |
| 2022 | leak-free null runs | 17 | 0.0880 | **+2.539** | +2.552 | 0.99× | +0.700 |
| 2022 | leaky null runs | 8 | 0.0954 | +3.708 | +2.209 | 1.68× | +1.693 |
| 2022 | all null runs | 25 | 0.0933 | +3.708 | +2.957 | 1.25× | +1.018 |

Only the leak-free rows are a valid zero-skill draw; the leaky rows carry the leak and are
shown for contrast. Against the real sweep's own threshold (K = 70, SR\*₀ = +3.829 in 2021
and +3.281 in 2022):

| year | best leak-free REAL run | best leak-free NULL run | null runs ≥ best real |
|---|---:|---:|---:|
| 2021 | +2.478 | +2.139 | 0/17 |
| 2022 | **+0.199** | **+2.539** | **10/17** |

**Non-parametric p-values.** Reference distribution = the 17 leak-free null runs' excess
Sharpe over their own panel's floor; statistic = the real arm's excess Sharpe over the real
floor. No Gaussian assumption, no zero centre.

| year | arm | best-seed excess SR | *p* | seed-mean excess SR | *p* |
|---|---|---:|---:|---:|---:|
| 2021 | ppo_causal | +0.767 | 0.059 | +0.373 | 0.176 |
| 2021 | ddpg_causal | +0.430 | 0.118 | +0.175 | 0.294 |
| 2021 | hrt_causal | +0.068 | 0.353 | −0.349 | 0.588 |
| 2021 | hrt_causal_ep | +0.000 | 0.353 | −0.398 | 0.588 |
| 2021 | hrt_paper *(leaky)* | +6.749 | 0.000 | +4.241 | 0.000 |
| 2021 | hrt_paper_ep *(leaky)* | +7.305 | 0.000 | +6.115 | 0.000 |
| 2022 | ddpg_causal | +0.373 | 0.176 | −0.134 | 0.529 |
| 2022 | hrt_causal_ep | +0.324 | 0.353 | −0.033 | 0.471 |
| 2022 | hrt_causal | +0.150 | 0.353 | −0.224 | 0.647 |
| 2022 | ppo_causal | −0.026 | 0.471 | −0.224 | 0.647 |
| 2022 | hrt_paper *(leaky)* | +6.273 | 0.000 | +3.161 | 0.000 |
| 2022 | hrt_paper_ep *(leaky)* | +4.490 | 0.000 | +3.729 | 0.000 |

**Runtime.** Null sweep 111.2 GPU-h over 25 runs (mean 4.45 h, range 2.05–5.07); real sweep
188.6 GPU-h over 60 runs (mean 3.14 h, range 0.72–5.70). Both at 501,760 steps with 20
validation evaluations — asserted rather than assumed by `null_sweep.py`.

## 5. Crypto tape coverage

| interval | Apr/May/Jun/Oct/Nov/Dec | Jan/Mar/Jul/Sep | Feb | Aug |
|---|---|---|---|---|
| 1 hour | **96.8 %** every year | ~50 % | **0.1 %** | absent except 2026 |
| 1 day | 30–42 % (100 % only in one 2025-08→2026-05 run) | 28–40 % | 40 % | 28 % |

Of 81 months in the hourly span: **43 ≥90 % complete, 15 under 50 %, 13 entirely absent**.
Daily tape: only **6 of 81** months clear 90 %. Outside one contiguous 299-day run
(2025-08-01 → 2026-05-26) the daily tape serves runs of median **9 consecutive days**
separated by median **21-day holes** (74 on-runs, 73 off-runs, **36.8 %** overall retention).

Discovery: 42 live pairs of 50 probed. Absent: `MATICUSD`, `SHIBUSD`, `PEPEUSD`, `ALGOUSD`,
`VETUSD`, `GRTUSD`, `SANDUSD`, `SNXUSD`.

## 6. Crypto forecaster IC (from `fr_*.npz`)

| panel | timing | IC train | IC valid | **IC test** |
|---|---|---:|---:|---:|
| crypto hourly, real | causal | +0.0643 | +0.0301 | **+0.0273** |
| crypto hourly, real | paper | +0.8615 | +0.7964 | **+0.8433** |
| null seed 0 | causal | +0.0179 | +0.0057 | +0.0017 |
| null seed 1 | causal | +0.0302 | +0.0069 | +0.0092 |
| null seed 2 | causal | +0.0070 | +0.0016 | +0.0128 |

Null mean **+0.0079**, sd **0.0057** (3 draws) → real **z = +3.4**.

## 7. Crypto strategy sweep — 11 strategies × 4 cost models

Test window 2026-03-08 → 2026-06-30, 2,665 hourly bars, 17 coins, 4 blocks.
Cells are **cumulative return / annualised Sharpe**.

| strategy | turnover/bar | no cost | flat 10 bp | flat 30 bp | sqrt (state) |
|---|---:|---|---|---|---|
| equal-weight (all coins) | 0.005 | −9.7 % / −0.42 | −11.0 % / −0.52 | −13.6 % / −0.71 | −11.9 % / −0.59 |
| BTC buy-and-hold | 0.002 | −12.9 % / −0.85 | −13.3 % / −0.88 | −14.0 % / −0.94 | −13.1 % / −0.87 |
| momentum-24h top5 | 0.343 | −23.3 % / −0.98 | −69.2 % / −5.49 | −95.1 % / −14.46 | −95.1 % / −14.29 |
| reversal-1h top5 | 1.394 | **+71.4 % / +3.11** | −95.8 % / −16.09 | −100.0 % / −53.59 | −99.8 % / −30.18 |
| momentum-24h top5 (daily reb.) | 0.061 | +11.9 % / +0.90 | −4.8 % / +0.07 | −31.3 % / −1.56 | −35.8 % / −1.89 |
| momentum-24h top5 (calm bars only) | 0.186 | −6.0 % / −0.03 | −42.7 % / −2.72 | −78.8 % / −8.02 | −71.0 % / −6.34 |
| momentum-24h top5 (stressed bars only) | 0.174 | −19.1 % / −0.81 | −49.1 % / −3.26 | −79.9 % / −8.14 | −89.5 % / −11.26 |
| **forecast-causal top5** | 0.747 | **+37.5 % / +2.07** | −81.2 % / −9.02 | −99.7 % / −30.91 | −98.5 % / −22.60 |
| forecast-causal top5 (daily reb.) | 0.057 | −0.3 % / +0.28 | −14.3 % / −0.57 | −36.7 % / −2.24 | −36.1 % / −2.18 |
| forecast-paper top5 | 1.320 | −51.6 % / −3.42 | −98.6 % / −21.50 | −100.0 % / −56.67 | −99.9 % / −32.63 |
| forecast-paper top5 (daily reb.) | 0.060 | +6.4 % / +0.64 | −9.3 % / −0.19 | −34.2 % / −1.82 | −38.3 % / −2.14 |

**Sharpe rank under each cost model** (1 = best):

| strategy | no cost | flat 10 bp | flat 30 bp | sqrt (state) |
|---|---:|---:|---:|---:|
| equal-weight (all coins) | 7 | 3 | 1 | 1 |
| BTC buy-and-hold | 9 | 5 | 2 | 2 |
| momentum-24h top5 | 10 | 8 | 8 | 8 |
| reversal-1h top5 | 1 | 10 | 10 | 10 |
| momentum-24h top5 (daily reb.) | 3 | 1 | 3 | 3 |
| momentum-24h top5 (calm bars only) | 6 | 6 | 6 | 6 |
| momentum-24h top5 (stressed bars only) | 8 | 7 | 7 | 7 |
| forecast-causal top5 | 2 | 9 | 9 | 9 |
| forecast-causal top5 (daily reb.) | 5 | 4 | 5 | 5 |
| forecast-paper top5 | 11 | 11 | 11 | 11 |
| forecast-paper top5 (daily reb.) | 4 | 2 | 4 | 4 |

Flat 10 bp → flat 30 bp: **5 of 11** move, rank correlation **+0.900**.
Flat 10 bp → sqrt (state): **5 of 11** move, rank correlation **+0.900**.
Flat 30 bp vs sqrt (state): **identical ranking on all 11** (columns 3 and 4 above).

**Effective rate actually charged by the state-dependent model**, bp of turnover:

| strategy | turnover/bar | effective bp |
|---|---:|---:|
| momentum-24h top5 (daily reb.) | 0.061 | **30.2** |
| forecast-paper top5 (daily reb.) | 0.060 | 27.8 |
| forecast-causal top5 (daily reb.) | 0.057 | 25.3 |
| momentum-24h top5 (stressed bars only) | 0.175 | **18.3** |
| equal-weight (all coins) | 0.005 | 17.0 |
| momentum-24h top5 (calm bars only) | 0.186 | **15.1** |
| momentum-24h top5 | 0.343 | 10.3 |
| forecast-causal top5 | 0.749 | 5.7 |
| BTC buy-and-hold | 0.002 | 5.3 |
| reversal-1h top5 | 1.398 | 3.0 |
| forecast-paper top5 | 1.324 | **2.8** |

Span **2.8 → 30.2 bp** (11×), monotone *against* turnover. The calm/stressed pair — built
with near-equal turnover and opposite volatility exposure — is charged **15.1 vs 18.3 bp**,
a 21 % difference that reorders nothing.

## 8. Deflated Sharpe, crypto (gross of cost)

σ_SR = 0.0183 per bar → **SR\*₀ = 2.782** (K=11), 3.820 (K=44), 1.921 (K=11, robust σ).

| strategy (gross) | Sharpe | skew | kurt | PSR(0) | DSR K=11 | DSR K=44 | DSR K=11 robust |
|---|---:|---:|---:|---:|---:|---:|---:|
| reversal-1h top5 | +3.11 | +0.00 | 9.93 | 0.957 | 0.571 | 0.348 | 0.744 |
| **forecast-causal top5** | **+2.07** | −0.18 | 12.13 | 0.872 | **0.347** | **0.168** | 0.533 |
| momentum-24h top5 (daily reb.) | +0.90 | +0.05 | 8.49 | 0.689 | 0.149 | 0.053 | 0.286 |
| forecast-paper top5 (daily reb.) | +0.64 | −0.04 | 11.49 | 0.638 | 0.119 | 0.040 | 0.240 |
| forecast-causal top5 (daily reb.) | +0.28 | −0.09 | 12.87 | 0.560 | 0.083 | 0.025 | 0.182 |
| momentum-24h top5 (calm bars only) | −0.03 | −0.00 | 8.52 | 0.493 | 0.060 | 0.017 | 0.141 |
| equal-weight (all coins) | −0.42 | −0.41 | 11.36 | 0.408 | 0.038 | 0.010 | 0.098 |
| momentum-24h top5 (stressed bars only) | −0.81 | −0.30 | 12.19 | 0.328 | 0.024 | 0.005 | 0.066 |
| BTC buy-and-hold | −0.85 | −0.24 | 11.85 | 0.319 | 0.022 | 0.005 | 0.063 |
| momentum-24h top5 | −0.98 | −0.20 | 10.26 | 0.295 | 0.019 | 0.004 | 0.055 |
| forecast-paper top5 | −3.42 | −0.29 | 11.26 | 0.029 | 0.000 | 0.000 | 0.002 |

## 9. Censoring cost (equity panel, mask applied in reverse; 20 replicates)

**Momentum-20, `gap` mask** (36.8 % retention, 73 blocks)

| view | bars | Sharpe | ann. vol | CAGR as reported | Δ Sharpe | Δ vol |
|---|---:|---:|---:|---:|---:|---:|
| **truth** | 2,414 | **+0.585** | **0.221** | **+11.1 %** | — | — |
| naive | 888 | +1.216 ± 0.110 | 0.340 ± 0.025 | **+42.3 %** | **+0.631** | **+0.119** |
| block | 888 | +0.851 ± 0.239 | 0.231 ± 0.019 | +19.0 % | +0.266 | +0.010 |

**Momentum-5, `gap` mask**

| view | bars | Sharpe | ann. vol | CAGR as reported |
|---|---:|---:|---:|---:|
| **truth** | 2,414 | **+0.433** | 0.227 | +7.5 % |
| naive | 888 | +1.079 ± 0.377 | 0.346 ± 0.027 | **+35.8 %** |
| block | 888 | +0.787 ± 0.181 | 0.228 ± 0.018 | +16.9 % |

**`seasonal` mask** (65.7 % retention, 261 blocks, 1,586 bars)

| strategy | view | Sharpe | ann. vol | CAGR as reported | Δ Sharpe vs truth |
|---|---|---:|---:|---:|---:|
| momentum-20 | truth | +0.585 | 0.221 | +11.1 % | — |
| momentum-20 | naive | +0.786 ± 0.053 | 0.264 ± 0.006 | +18.8 % | **+0.201** |
| momentum-20 | block | +0.753 ± 0.210 | 0.218 ± 0.008 | +15.2 % | +0.168 |
| momentum-5 | truth | +0.433 | 0.227 | +7.5 % | — |
| momentum-5 | naive | +0.533 ± 0.101 | 0.263 ± 0.007 | +11.2 % | **+0.100** |
| momentum-5 | block | +0.501 ± 0.180 | 0.214 ± 0.007 | +8.8 % | +0.068 |

## 10. Prediction markets — population

| | Polymarket-v1 | vendor tape (superseded) |
|---|---:|---:|
| fills | **730,492,126** | 382,119 |
| markets / events / accounts | 838,118 / **704,521** / 2,588,367 | 887 / — / 109,053 |
| taker notional | **$27.9 B** | — |
| span | 2022-11-21 → 2026-04-28 | 2026-05-21 → 2026-09-04 |
| accounts ≥10 fills | **1,819,876** | 1,744 |
| their fills | 727,446,812 | — |
| their realised PnL | **−$58,815,295** = **−0.2156 %** of notional | — |
| markets resolving cleanly | — | 384 of 815 |

Derived tables: `pmv1_bars_1h.parquet` 8.8 M market-hours; `pmv1_markets.parquet` 838,118
rows; **45,715** distinct close-days; **90,381** day × category cells.

## 11. Skill test — calibration ladder (exact over all 1,819,876 accounts)

Null requires **rms(z) = 1.000**. Tails here are normal-approximation.

| randomisation unit | median units/acct | mean z | sd(z) | **rms(z)** | z > 1.645 | z < −1.645 |
|---|---:|---:|---:|---:|---:|---:|
| **trade** (as published) | 38 | −0.256 | 1.628 | **1.648** | 7.27 % | 11.98 % |
| bet = trader × market | 16 | −0.545 | 1.456 | 1.555 | 5.67 % | 18.31 % |
| **event** = trader × neg-risk event | 14 | −0.440 | 1.304 | **1.376** | 5.55 % | 12.38 % |
| trader × close-day × category | 12 | −0.443 | 1.272 | 1.347 | 5.35 % | 12.40 % |
| **trader × close-day** | 11 | −0.433 | **1.238** | 1.312 | 5.14 % | 11.14 % |
| trader (degenerate, one flip) | 1 | −0.368 | 0.930 | **1.000** | 0 % | 0 % |

**Exact Rademacher null** (10,000 sims, 50,000 sampled accounts; repeating the draw moves
these ~0.3 pp):

| unit | "skilled" p < 0.05 | "anti-skilled" p > 0.95 |
|---|---:|---:|
| trade | **8.48 %** | 13.02 % |
| bet | 7.18 % | 28.53 % |
| event | 7.01 % | 27.36 % |
| event, recentred on median z | **18.68 %** (20.18 % exact) | 2.3 % |

Flagged by the trade-level null but **not** the event-level null: **2.54 %** of accounts.
Flagged by all three units: **5.90 %**. Median event-level z: **−0.828**.
Fifteen-week vendor tape, same ladder: sd(z) **2.370 → 1.353 → 1.096**.

## 12. Skill-variance decomposition (V = 2ρ/(1−ρ))

| cut | accounts | Pearson ρ | s.e. | Spearman | V | **sd(z) skill implies** | their own sd(z) | their rms(z) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024-07-01 | 3,594 | +0.1629 | 0.0162 | +0.1474 | 0.389 | **1.179** | 1.234 | 1.299 |
| 2025-01-01 | 88,279 | +0.0246 | 0.0034 | +0.0180 | 0.050 | **1.025** | 1.207 | 1.270 |
| 2025-07-01 | 151,103 | +0.1094 | 0.0025 | +0.0866 | 0.246 | **1.116** | 1.291 | 1.316 |

## 13. Persistence — leak-free, events assigned whole

| cut | accounts (≥10 fills each half) | Spearman(z_in, z_out) | 2 s.e. | flagged before → after | base rate | **lift** |
|---|---:|---:|---:|---:|---:|---:|
| 2024-07-01 | 3,594 | **+0.147** | ±0.033 | 26.2 % | 6.2 % | **4.21×** |
| 2025-01-01 | 88,279 | **+0.018** | ±0.007 | 22.1 % | 11.6 % | **1.92×** |
| 2025-07-01 | 151,103 | **+0.087** | ±0.005 | 43.5 % | 16.5 % | **2.63×** |

Out-of-sample PnL as a fraction of out-of-sample notional:

| cut | bottom decile | all accounts | **top decile** |
|---|---:|---:|---:|
| 2024-07-01 | +0.02 % | +1.20 % | **+3.55 %** |
| 2025-01-01 | −0.35 % | +0.52 % | **+1.30 %** |
| 2025-07-01 | −0.68 % | +0.38 % | **+1.67 %** |

**Straddle confound.** At a 2025-01-01 cut only **855 of 704,521** events straddle, but
dropping them takes the paired sample 164,627 → **88,279** and the rank correlation
**+0.103 → +0.018**.

Full decile detail, 2025-07-01 cut (`pmv1_persistence_2025-07-01.json`):

| decile | n | mean z_in | mean z_out | PnL_out | ROI_out |
|---|---:|---:|---:|---:|---:|
| D1 | 15,111 | −2.135 | +0.020 | −$1,542,999 | −0.68 % |
| D2 | 15,110 | −1.495 | +0.104 | −$444,302 | −0.26 % |
| D3 | 15,110 | −1.221 | +0.147 | −$441,105 | −0.24 % |
| D4 | 15,110 | −1.025 | +0.285 | −$20,499 | −0.01 % |
| D5 | 15,111 | −0.828 | +0.186 | −$1,373,044 | −0.63 % |
| D6 | 15,110 | −0.425 | +0.042 | −$500,578 | −0.09 % |
| D7 | 15,110 | +0.094 | +0.046 | +$1,965,193 | +0.40 % |
| D8 | 15,110 | +0.663 | +0.069 | −$100,468 | −0.02 % |
| D9 | 15,110 | +1.054 | +0.244 | +$4,572,970 | +1.12 % |
| **D10** | 15,111 | +1.965 | **+0.775** | **+$11,791,628** | **+1.67 %** |

Totals: PnL_out $13,906,797 on notional $3,618,246,115.

## 14. Imbalance regressions

**5,244,127 bars / 25,317 markets** at 1 h; **2,323,642 / 14,780** at 4 h.

| specification | 1 hour | 4 hour | *vendor tape, 15 wk* |
|---|---|---|---|
| contemporaneous r_h on imbalance_h — large | **+0.0205 (t +45.6)** | **+0.0378 (t +58.2)** | +0.0238 (t +6.1) |
| contemporaneous — small | +0.0413 (t +91.0) | +0.0590 (t +98.8) | +0.0131 (t +8.1) |
| predictive r_{h+1} on imbalance_h — large | **+0.0033 (t +13.3)** | **+0.0020 (t +5.1)** | −0.0011 (t −0.4) |
| predictive — small | −0.0156 (t −68.3) | −0.0209 (t −57.8) | −0.0060 (t −4.0) |
| predictive, controlling for r_h — large | **+0.0043 (t +17.3)** | **+0.0061 (t +15.7)** | +0.0029 (t +1.0) |
| own-return reversal, r_h coefficient | −0.0501 (t −19.0) | −0.1091 (t −29.1) | −0.141 (t −4.6) |

R² of the predictive specification **0.0017**; mean |r| = 0.0785 log-odds.

**By bar-liquidity quintile (1 h, predictive, large):**

| quintile | median gross | β | t |
|---|---:|---:|---:|
| Q1 | $1 | +0.0001 | +0.24 |
| Q2 | $13 | −0.0005 | −1.84 |
| **Q3** | **$81** | **−0.0012** | **−4.39** |
| Q4 | $394 | +0.0004 | +1.53 |
| **Q5** | **$3,569** | **+0.0056** | **+8.92** |

4 h panel: Q3 −0.0012 (t −2.72), Q5 +0.0024 (t +2.34). Vendor tape median bar: **$69**.

**By category (1 h, predictive, large):**

| category | bars | β | t |
|---|---:|---:|---:|
| Finance | 113,415 | +0.0100 | +6.15 |
| Culture | 206,988 | +0.0095 | +7.58 |
| **Politics** | 844,596 | **+0.0067** | **+11.88** |
| Crypto | 663,056 | +0.0037 | +5.33 |
| Sports | 1,321,309 | +0.0023 | +3.79 |
| Soccer / NBA / United States | 148k / 118k / 269k | ≈ 0 | −0.02 / −0.54 / +0.09 |

## 15. Effective spread (8,822,101 bars, 38,685 markets; both sides trade in 51.4 %)

| quintile | bars | median gross | median p | median spread | mean spread | spread / price |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 906,279 | $3 | 0.007 | 0.00100 | 0.00494 | 10.00 % |
| Q2 | 906,278 | $30 | 0.041 | 0.00179 | 0.00768 | 6.67 % |
| Q3 | 906,278 | $140 | 0.100 | 0.00341 | 0.00845 | 3.85 % |
| Q4 | 906,278 | $580 | 0.172 | 0.00673 | 0.00939 | 3.75 % |
| **Q5** | 906,279 | **$4,729** | 0.260 | **0.00522** | 0.00493 | **2.69 %** |

## 16. Imbalance round trip at taker VWAPs (1-bar hold)

All figures **pp per share**. ROI is value-weighted.

| subset | trips | gross pp | **net pp** | cost pp | half in | half out | net ROI | t(gross) | t(net) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all trips | 1,273,650 | +0.0244 | **−0.6924** | 0.7168 | 0.3922 | 0.3246 | −1.397 % | +5.68 | −148.4 |
| Q1 | 254,730 | −0.0030 | −0.5759 | 0.5728 | 0.2953 | 0.2775 | −0.975 % | −0.97 | −145.3 |
| Q2 | 254,730 | +0.0010 | −0.7786 | 0.7796 | 0.3935 | 0.3861 | −1.621 % | +0.23 | −156.5 |
| Q3 | 254,730 | +0.0044 | −0.9264 | 0.9308 | 0.4752 | 0.4556 | −1.974 % | +0.76 | −141.1 |
| Q4 | 254,730 | +0.0341 | −0.9036 | 0.9377 | 0.4933 | 0.4444 | −1.932 % | +4.09 | −99.5 |
| **Q5** | 254,730 | **+0.0858** | **−0.2776** | **0.3633** | 0.3037 | 0.0596 | −0.591 % | **+4.71** | −14.3 |

**By hold (top liquidity quintile):**

| hold | trips | gross pp | net pp | cost pp | half in | half out | t(gross) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 h | 254,730 | +0.0858 | −0.2776 | 0.3633 | 0.3037 | 0.0596 | +4.71 |
| **2 h** | 242,237 | **+0.1338** | −0.3428 | 0.4767 | 0.3746 | 0.1021 | +5.63 |
| 4 h | 222,409 | +0.1048 | −0.5228 | 0.6275 | 0.4110 | 0.2165 | +4.45 |
| 8 h | 192,682 | +0.0406 | −0.6510 | 0.6916 | 0.4121 | 0.2795 | +1.64 |
| 24 h | 131,695 | +0.0589 | −0.6601 | 0.7190 | 0.4115 | 0.3076 | +1.80 |

**By signal strength (top quintile, 1-bar hold):**

| subset | trips | gross pp | net pp | cost pp |
|---|---:|---:|---:|---:|
| \|oi\| = 1 (fully one-sided) | 82,319 | +0.1361 | −0.4883 | 0.6244 |
| \|oi\| > 0.5 | 167,548 | +0.1262 | −0.3539 | 0.4801 |
| \|oi\| ≤ 0.5 | 87,182 | +0.0080 | −0.1309 | 0.1389 |

**Counterparty.** The passive side of the same round trip earns **+0.2776 pp/share** in Q5:
the two half-spreads (0.3633 pp) less 0.0858 pp of adverse selection.

**Fee fields.** `fee_usdc` identically zero across the whole archive; `taker_base_fee` set
on **3.6 %** of fills.

---

## 17. Known gaps in this table

* ~~**No RL run on a null panel** (C4)~~ — done 2026-09-15, §4.5.
* **No resampling of the *real* equity panel.** §4.5's null side has a dispersion across
  four panels; the real side is one draw of market history, so every real-side *t* there is
  across seeds only. This is now the binding gap in the RL workstream.
* **§4.5's null side is 4 panels / 25 runs**, so per-arm null *t*-statistics are weak and
  the argument rests on the pooled rows and the 17-run reference distribution. One
  leak-free arm (`ppo_causal`, 2022) clears its floor on the null at *t* = +2.79 — with six
  arms × two years, a false positive is expected, and that is §2's multiplicity point
  appearing inside the audit.
* **The null panels carry a single market factor**, so a defect that bites only under
  richer cross-sectional dependence would not show in §4.5.
* **No CPCV or PBO** for any workstream.
* **No equity cost re-simulation** — §3 is a first-order recharge, used only for ordering,
  because policies were not checkpointed.
* **Crypto null has 3 seeds, equity null has 4** — the crypto sd (0.0057) is a 3-draw
  estimate.
* **Crypto test window is 111 days in 4 blocks**, 3 of them one calendar quarter. No crypto
  result has been tested in a February or an August.
* **§9's masks are applied to equities, not crypto** — deliberate, since that is the only
  way to get ground truth.
* **Permanent impact is reported but not netted** (`analysis/costs.py`).
* **§15's spread is observable only in the 51.4 % of bars where both sides trade.**
