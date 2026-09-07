# Results — reproducing the reading list, and moving it to crypto and prediction markets

FYP **H398280** (supervisor: LU Yao), *AI in Financial Data Analytics and Trading*.
Work done 2026-09-07 against the local findata mirror, the live findata API, and the
**Polymarket-v1 on-chain archive** [16]. Papers are cited by their number in
[`PAPERS.md`](PAPERS.md); data provenance is in [`DATA.md`](DATA.md); the prior HRT
reproduction is in [`hrt/README.md`](hrt/README.md).

Everything below is computed in this repo. Commands are in §8; §9 lists the two things
that need a human.

---

## 0. Summary

| # | Claim tested | Source | Outcome |
|---|---|---|---|
| 1 | HRT's reported returns | [1] | **Does not replicate leak-free** (established earlier; re-baselined here) |
| 2 | The reported Sharpes are real, not selection | [8] | **No** — no leak-free equity arm survives deflation (DSR ≤ 0.29 / ≤ 0.006), and **no crypto strategy does either**, gross of cost |
| 3 | The pipeline finds signal only where signal exists | [10] | **Fails under the paper's timing**: test IC **+0.54** on data with provably zero predictability |
| 4 | The cost model changes the algorithm ranking | [2] | **Replicates** — 3–6 of 6 equity arms move rank; 5 of 11 crypto strategies move |
| 5 | ...and *state-dependent* cost changes it further | idea 6 | **No** — the square-root state model ranks identically to a flat 30 bp |
| 6 | A skilled minority drives prediction markets | [13], [15] | **Split verdict.** The published estimator over-rejects at every scale, and correcting the unit is necessary but *not sufficient* (sd(z) 1.63 → 1.30, never 1.00). The claim nevertheless survives a test that needs no null at all: flagged skill **persists out of sample**, 1.9–4.2× lift at three cut dates. The earlier non-replication was a power artefact |
| 7 | Large-trade order imbalance predicts returns | [14] | **Replicates at scale** (t = +13.3), reversing the fifteen-week null. The gradient in bar liquidity is monotone and confirms the mechanism that non-replication conjectured |
| 8 | A censored vendor calendar is a cosmetic problem | new | **No** — measured against ground truth it inflates Sharpe by **+0.63** and reported CAGR by **~4×** |

Four findings are new and transferable beyond this repo:

* **The leakage is measurable, not conjectural.** 65 % of the variance of the label under
  the paper's stated timing is mechanically explained by a single same-bar Alpha158
  feature (`KMID`) that the model is given as an input. Run on synthetic data where
  nothing is predictable, the same pipeline still reports a test IC of +0.54.
* **The published sign-randomisation skill test is mis-calibrated by construction**, and
  correcting the randomisation unit is *necessary but not sufficient*: over 1.8 M
  accounts, sd(z) falls 1.628 → 1.456 → 1.304 as the unit goes trade → bet → event, and
  stops well above the 1.00 the null requires.
* **Skill in prediction markets is better measured out of sample than against a null.**
  Splitting 3.5 years at a cut date and requiring markets not to straddle it, the top
  in-sample decile earns +1.3 % to +3.5 % of notional afterwards against a population
  +0.4 % to +1.2 %, and accounts flagged before the cut are 1.9–4.2× more likely to be
  flagged after it. No null is needed for this and none can be mis-specified.
* **The cost of a censored calendar is quantifiable.** Applying the findata crypto tape's
  own retention pattern to a *complete* equity panel and comparing against the truth it
  hides: a naively concatenated backtest reports Sharpe 1.22 where the truth is 0.59, and
  a 42 % CAGR where the truth is 11 %.

**The correction this round forces.** Sections 3.1 and 3.2 of the previous version
recorded two non-replications on fifteen weeks of vendor tape, each with an explicit
power caveat. Both caveats were right: on 3.5 years of ground-truth on-chain data, [14]
replicates cleanly and [13]'s substantive claim survives. **The vendor tape, not the
papers, was the binding constraint.** What does *not* change is the methodological
finding in each — the mis-calibrated null, and the liquidity-dependence of the
imbalance result — and those are now measured at a scale where they are not arguable.

---

## 1. What data is actually available

`DATA.md` documents the equity mirror. Four things it does not cover were established here.

### 1.1 Crypto exists, undocumented, on the generic price route

findata publishes no `/crypto` family and the OpenAPI spec (165 paths) contains no
crypto route. Crypto pairs are nevertheless served by the ordinary `/ohlc/{symbol}`
endpoint and carry `exchange = "CRYPTO"`, `industry = "cryptocurrency"` in `/symbols`.
Probing 50 candidate tickers found **42 live pairs** (`BTCUSD`, `ETHUSD`, … `ZECUSD`);
`MATICUSD`, `SHIBUSD`, `PEPEUSD`, `ALGOUSD`, `VETUSD`, `GRTUSD`, `SANDUSD`, `SNXUSD`
return nothing. Pulled to `data/crypto.sqlite`: **27,566 daily bars** and
**1,150,609 hourly bars**, 2020-01 → 2026-09.

### 1.2 That crypto tape is *seasonally censored*, and this is the important part

Coverage is not random. Measured per calendar month on the deepest symbol
(`analysis/coverage.py`):

| interval | Apr / May / Jun / Oct / Nov / Dec | Jan / Mar / Jul / Sep | Feb | Aug |
|---|---|---|---|---|
| 1 hour | **96.8 %** every year | ~50 % | **0.1 %** | absent except 2026 |
| 1 day | 30–42 % (100 % only inside one 2025-08→2026-05 run) | 28–40 % | 40 % | 28 % |

Of the 81 months in the hourly span, **43 are ≥90 % complete, 15 are under 50 %, and 13 are
absent from the tape entirely**; on the daily tape only **6 of 81** months clear 90 %.
A backtest that simply concatenates whatever the vendor returns is therefore a backtest
of **Q2 and Q4**, with February and August never sampled, and it will never say so.
The daily tape has the same disease in a different shape: outside one contiguous
299-day run (2025-08-01 → 2026-05-26) it serves runs of a median **9 consecutive days**
separated by median **21-day holes** (74 on-runs, 73 off-runs, 36.8 % overall retention),
so every rolling window longer than nine bars silently spans a gap.

The panel builder (`crypto/panel.py`) therefore cuts the tape into **contiguous blocks**
and confines every feature window, every label and every portfolio return to a single
block. The resulting hourly panel is **30,079 bars × 17 coins × 158 features over 45
blocks**, 2020-11 → 2026-06. The 17-coin universe is what survives requiring ≥98 %
coverage in *every* block; requiring it only in recent blocks would admit more coins at
the cost of a look-ahead-shaped survivorship filter. §5.4 measures what that construction
is worth, against ground truth.

### 1.3 The prediction-market tape moved, and one route died

`findata-pm-constraints` recorded a tape starting 2026-05-21. It now runs
**2026-05-21 → 2026-09-04**. Keyword discovery found **11,505 Polymarket markets**;
pulling the 900 highest-volume markets that both resolve inside the window and clear
$100k volume gives **382,119 trades across 887 markets and 109,053 distinct takers**
(`data/pm.sqlite`).

The route the August analysis used for outcomes,
`/prediction-markets/markets/polymarket/{id}`, now **404s for all 887 markets** and does
not appear in the OpenAPI spec at all; it was undocumented and has been withdrawn.
`/candles` cannot substitute because it aggregates both tokens into one series (a single
day shows open 0.50, high 0.999, low 0.001). Outcomes had to be read off the tape
itself (`pm/outcomes.py`): **384 of 815 markets resolve cleanly** by that rule.

This tape is now superseded for every question in §3, and the reason is §1.4.

### 1.4 Polymarket-v1 [16] — the constraint is gone

`PAPERS.md` calls this "the most actionable item on the list" and the previous version of
§7 called it "the one that would change everything". It is on Hugging Face as
`TimeSeventeen/Polymarket-v1`, **CC-BY-4.0, ungated, no token required**. Four layers:
`OrderFilled/` (the raw nominal tape, 1.20 B rows, 27.4 GB), `daily_aligned/` (cleaned
standard-binary, 13.2 GB), `daily_aligned_multi/` (cleaned negative-risk multi-outcome,
3.6 GB) and `CTF/` (contract lifecycle logs, 8.5 GB).

`pm/pmv1_pull.py` fetches the two **cleaned** layers — **2,105 files, 16.8 GB,
746,110,412 rows**, thirteen minutes on a home connection. `OrderFilled/` is deliberately
not pulled: it is the nominal tape with platform relayer/router records still in it, and
the cleaned layers have already removed them.

Three properties matter for this work, and each removes a specific limitation of §3:

* **Ground-truth aggressor side.** `taker` is the aggressor wallet and `taker_direction`
  comes from blockchain settlement, not from a tick or quote rule. [16]'s own contribution
  is showing that Lee-Ready and its relatives are *near-random* on this venue. A
  mis-signed side attenuates an imbalance regressor towards zero — which is precisely the
  shape of the null result the old §3.2 reported.
* **Ground-truth event grouping.** `neg_risk_market_id` identifies the parent negative-risk
  event, so all 40-odd candidates of a "who wins the tournament" market are one event *by
  contract construction*. The old §3.1 had to recover this by clustering market titles on
  token-set Jaccard ≥ 0.6 and defend the threshold.
* **Pre-normalised event coordinates.** `p_event` puts both legs of a binary market on one
  probability axis and `D ∈ {+1,−1}` is the taker's direction on that axis, so no
  hand-rolled leg alignment is needed.

After keeping resolved markets with a winning label and prices inside (0.0005, 0.9995):

| | |
|---|---:|
| fills | **730,492,126** |
| markets / events / accounts | 838,118 / **704,521** / 2,588,367 |
| taker notional | **$27.9 B** |
| span | 2022-11-21 → 2026-04-28 |
| accounts with ≥10 fills | **1,819,876** (cf. [13]'s 1.72 M) |
| their realised PnL | **−$58,815,295**, i.e. **−0.2156 %** of notional |

Against the vendor tape's 384 markets and 1,744 accounts over fifteen weeks, that is
three orders of magnitude on every axis, and it costs a download.

`pm/pmv1_prep.py` projects the two layers once into the columns the analyses need
(`data/pmv1_trades.parquet`, 730 M rows, 7.9 GB, built in 58 s;
`data/pmv1_bars_1h.parquet`, 8.8 M market-hours). Every question below then runs against
those instead of re-scanning 16.8 GB of repeated string metadata — the first attempt at
the account group-by straight off the wide layers spilled 33 GB to disk.

---

## 2. Reproductions

### 2.1 The baseline being audited — HRT [1]

`hrt/README.md` already records the reproduction: **2021 +23.1 % / 2022 −7.6 %** against a
published **+39.8 % / +2.3 %**, with every RL agent below a passive floor in the bear
year (passive-in-env 2021 +26.5 %; equal-weight buy-and-hold +32.8 %; S&P 500 +28.8 %).
Nothing here overturns that. What follows asks whether the numbers that *were* produced
mean anything, and whether the failure generalises.

Note for the write-up: HRT **v2 (May 2026)** is a different paper — 89-name Nasdaq
universe, 2020–2023 test, Sharpe 1.06 → 1.24, turnover 0.112 → 0.090, LLC penalties on
turnover and drawdown. The reproduction targets v1. That should be stated explicitly
rather than glossed.

### 2.2 Deflated Sharpe Ratio [8] — nothing survives it, on either asset class

`analysis/dsr.py` applies Bailey & López de Prado's correction to every run in the equity
sweep. K counts **all 35 runs ever aggregated** — the 25 in `runs/` *and* the 10 in
`runs_invalid/`, because the surviving configuration was chosen after seeing both.

**2021** (σ_SR across trials 0.0857 daily → SR\*₀ = 2.905 annualised). The Sharpe shown is
the arm's **best seed**, not the seed mean, because the quantity being deflated is a
maximum over a search:

| arm | seeds | Sharpe (repo) | skew | kurt | PSR(0) | **DSR** |
|---|---:|---:|---:|---:|---:|---:|
| hrt_paper_ep *(leaky)* | 4 | 9.35 | +0.17 | 5.55 | 1.000 | **0.999** |
| hrt_paper *(leaky)* | 4 | 6.02 | −0.05 | 4.73 | 1.000 | **0.936** |
| ppo_causal | 4 | 2.51 | −0.20 | 4.13 | 0.987 | 0.268 |
| ddpg_causal | 5 | 2.35 | −0.45 | 4.13 | 0.980 | 0.229 |
| hrt_causal | 4 | 2.11 | −0.29 | 4.20 | 0.971 | 0.175 |
| hrt_causal_ep | 4 | 2.04 | −0.09 | 3.47 | 0.969 | 0.159 |

**2022** (SR\*₀ = 2.533 annualised): `hrt_paper_ep` **0.626**, `hrt_paper` **0.328**, and
every leak-free arm at **≤ 0.006**.

Read plainly: an annualised Sharpe of 2.1 looks impressive and has PSR(0) = 0.97 against
a zero benchmark, but the expected maximum Sharpe from 35 trials of this strategy family
is **2.905** — higher than any leak-free arm achieved. **Every leak-free result in the
sweep is below what the search would be expected to produce from luck alone.**

**The same correction now applies to the crypto sweep** (`crypto/dsr.py`), which §5.3
previously quoted undeflated — the exact error §7 accuses the reading list of. Eleven
strategies × four cost models is a search; the trial count is reported at K = 11 (fix the
cost model, search strategies) and K = 44 (choose the cost model afterwards too). Gross of
cost, where the trials are comparable, σ_SR = 0.0183 per bar → SR\*₀ = **2.78 annualised**:

| strategy (gross) | Sharpe | PSR(0) | DSR K=11 | DSR K=44 | DSR K=11, robust σ |
|---|---:|---:|---:|---:|---:|
| reversal-1h top5 | +3.11 | 0.957 | 0.571 | 0.348 | 0.744 |
| **forecast-causal top5** | **+2.07** | 0.872 | **0.347** | **0.168** | 0.533 |
| momentum-24h top5 (daily reb.) | +0.90 | 0.689 | 0.149 | 0.053 | 0.286 |
| forecast-paper top5 (daily reb.) | +0.64 | 0.638 | 0.119 | 0.040 | 0.240 |

The +2.07 Sharpe §5.3 reports for `forecast-causal top5` has **DSR 0.35**, and 0.17 if the
cost model was also chosen after the fact. **Nothing in the crypto sweep survives
deflation, before costs are charged at all.** Under a cost model the trial dispersion is
set by two strategies that lose almost the whole book (Sharpe −57), so σ_SR explodes and
SR\*₀ becomes uninformative; a robust IQR-based scale is reported alongside and still
clears every strategy. The gross panel is the one that discriminates.

This costs about forty lines of code and it should sit in the dissertation's results table
next to every Sharpe on both asset classes. It also fixes the number of trials as
something to be reported honestly: quoting the best of 35 (or of 44) as though it were the
only one is precisely the error the paper names.

### 2.3 MACE [2] — the cost model does reorder the agents, but not because it is state-dependent

**On equities** (`analysis/cost_rerank.py`). The trained policies were not checkpointed
and each run cost ~2 h, so the agents cannot be re-simulated. The run records do carry
the net value path, mean per-step turnover and total cost paid at the 10 bp the
environment charged, which is enough to recharge the same trade schedule at another flat
rate to first order (`r_net(ρ) = r_net(10bp) + (0.001 − ρ)·turnover`). That is an
approximation in level and is used only for ordering, which is MACE's actual claim.

2021 Sharpe by arm, and the rank each arm holds:

| arm | turnover | gross (0 bp) | 10 bp | 30 bp | 60 bp | rank 0/10/30/60 |
|---|---:|---:|---:|---:|---:|---|
| hrt_paper_ep | 0.377 | +6.33 | +5.57 | +4.04 | +1.75 | 1 / 1 / 1 / **3** |
| hrt_paper | 0.289 | +4.87 | +4.31 | +3.19 | +1.51 | 2 / 2 / 2 / **4** |
| hrt_causal_ep | 0.219 | +2.20 | +1.75 | +0.86 | −0.48 | 3 / 5 / 6 / 6 |
| ppo_causal | 0.004 | +2.15 | +2.14 | +2.12 | +2.09 | 4 / 3 / 3 / **1** |
| hrt_causal | 0.196 | +2.09 | +1.70 | +0.90 | −0.29 | 5 / 6 / 5 / 5 |
| ddpg_causal | 0.004 | +2.04 | +2.03 | +2.01 | +1.99 | 6 / 4 / 4 / **2** |

Moving from the 10 bp default: 4 of 6 arms change rank at 0 bp, 2 of 6 at 30 bp, and
**6 of 6 at 60 bp**, where the ordering inverts completely — the near-zero-turnover flat
baselines (ppo, ddpg) take the top two places and the two leak-free hierarchy arms take
the bottom two. In 2022 the same recharge shows `hrt_causal_ep` at Sharpe **+0.07 gross
and −0.24 net**: the hierarchy earns a real gross edge and hands all of it to the broker.
MACE's claim replicates cleanly on an independent codebase.

**On crypto** (`crypto/strategy.py`), eleven strategies over the clean 2026 out-of-sample
window (2,665 hourly bars, 17 coins, 4 blocks), under four cost models. Five of eleven
strategies move rank between flat 10 bp and flat 30 bp (rank correlation +0.900).

The result that matters for **idea 6** is the one that did not come out as hoped. The
state-dependent square-root model — half-spread plus `Y·σ_t·√(q/V_t)`, so the same trade
is charged against the bar's own realised volatility and traded volume — produces a
ranking **identical to a flat 30 bp on all eleven strategies**. It is not that the model
does nothing: the effective rate it charges ranges from **2.8 bp to 30.2 bp** across
strategies, an 11× spread. But the spread is driven by *participation* (`q/V`), not by
regime. Two variants built specifically to separate the two — the same momentum signal
rebalanced only in calm bars versus only in stressed bars, near-identical total turnover,
opposite volatility exposure — are charged 15.1 bp and 18.3 bp, a 21 % difference that
never reorders anything.

A second, less obvious consequence: under the square-root law the *low*-turnover
strategies pay the *highest* per-unit rate (daily-rebalanced momentum 30.2 bp; hourly
`forecast-paper` 1.32 turnover/bar pays 2.8 bp), because concentrating turnover into
fewer, larger rebalances raises `√(q/V)` on each one. The usual "trade less" prescription
is a statement about total cost, not about the rate, and a model that only sees size
inverts the two.

**Consequence for the FYP.** MACE [2] published "the cost model changes the ranking" in
March 2026 with a static Almgren–Chriss form. The remaining gap idea 6 claimed — that
*state dependence* changes it further — is not visible in this sample. Either the
volatility-surface geometry has to be shown to carry information that σ_t and V_t do not,
or the contribution should be restated as something else.

---

## 3. Prediction markets, at full scale

All three subsections below run on Polymarket-v1 (§1.4). Where the fifteen-week vendor-tape
result differs, both are given, because the difference is itself the finding.

### 3.1 The "informed minority" [13], [15] — the estimator over-rejects at every scale

The estimator [13] uses is **sign randomisation**: hold every trade's market, size, price
and timing fixed, flip the direction by a coin toss, and ask whether realised PnL sits in
the tail of that null. The published null flips **each trade** independently. That is the
step to look at. A trader who buys the same token 60 times in one market has taken *one*
bet, not 60, and a trader who sells forty different countries in one World Cup event has
taken one view, not forty. Independent flips shrink the null standard deviation by roughly
√n, so the test must over-reject. Whether a null is calibrated is checkable: under it, the
z-statistic must have unit standard deviation.

A taker who takes direction `D` on `shares = usdc_amount / price` of a token settling at
`win_event` when the event-normalised price was `p_event` realises
`s = D · shares · (win_event − p_event)`. That is the quantity a coin flip flips. Because
`D = ±1`, the null variance of a unit is just `Σ s²`, so **sd(z) is an exact function of
sums** and is computed over the *whole* population — 1,819,876 accounts, no sampling and
no simulation. The flagged fractions need the exact Rademacher null rather than a normal
approximation, so those are simulated (10,000 draws) on 50,000 randomly sampled accounts.

| randomisation unit | median units/acct | **sd(z)** — must be 1.00 | mean z | "skilled" p<0.05 | "anti-skilled" p>0.95 |
|---|---:|---:|---:|---:|---:|
| **trade** (as published) | 38 | **1.628** | −0.256 | **8.48 %** | 13.02 % |
| bet = (trader × market) | 16 | 1.456 | −0.545 | 7.18 % | 28.53 % |
| event = (trader × neg-risk event) | 14 | **1.304** | −0.440 | 7.01 % | 27.36 % |

Three things follow, and the last two are new.

**1. Correcting the unit is necessary.** sd(z) falls monotonically 1.628 → 1.456 → 1.304
as the unit widens, exactly as on the vendor tape (2.370 → 1.353 → 1.096). The published
trade-level null flags 8.48 % of accounts against a 5 % false-positive rate; **2.54 % of
all accounts are flagged by it and not by the event-level null**, and only 5.90 % are
flagged by all three units. (sd(z) and the mean are exact over all 1,819,876 accounts; the
flagged fractions are Monte Carlo over 50,000 of them, and repeating the draw moves them by
about 0.3 percentage points.)

**2. Correcting the unit is not sufficient — and that is a finding the small sample could
not see.** On fifteen weeks the event-level sd(z) was 1.096, close enough to 1.00 to call
calibrated. Over 1.8 M accounts it is **1.304**, which is not. Grouping by the negative-risk
contract collapses candidates of one tournament, but it does not collapse *structurally
distinct markets driven by one uncertainty*. On 2025-06-17, for instance, **37 distinct
Bitcoin markets close and the negative-risk grouping collapses them to 30 events** — the
`bitcoin-up-or-down-<hour>` series are separate `condition_id`s with separate
`neg_risk_market_id`s and one underlying, so an account that takes the same view on BTC all
day still gets thirty independent coin flips. The residual over-dispersion is the obvious
next thing to model, and the cheap test is whether sd(z) falls further when units are pooled
by resolution date and underlying rather than by contract.

**3. The null is also biased against the trader, and correcting that overshoots.** A taker
crosses the spread, so the expected payoff of a randomly-directed taker trade is negative.
It shows up directly: the median account's event-level z is **−0.828**, mean z is −0.44,
the anti-skilled tail (27 %) is four times the skilled tail, and the 1.82 M active accounts
lose **$58.8 M in aggregate, −0.2156 % of notional**. Recentring the null on the typical
account rather than on zero — shifting by the population median z, so the offset is in the
statistic's own units — moves the flagged fraction to **18.68 %** at z > 1.645 (20.18 %
under the exact Rademacher null), and cuts the anti-skilled tail from 27 % to 2.3 %. That
is an upper bound, not an answer: it credits an account for trading in tighter markets as
readily as for being informed. The zero-centred null under-counts and the recentred one
over-counts, which is why §3.2 stops using a null at all.

### 3.2 Does skill *persist*? — the test the fifteen-week tape could not run

Everything in §3.1 is a statement about an estimator. The economic question — is there a
minority who are actually informed — is better asked without a null: rank accounts on one
period, and look at what they do in the next. That needs calendar the vendor tape did not
have. `pm/pmv1_persistence.py` splits at a cut date, computes each account's event-level z
in each half, and reports the second half conditional on the first.

**One confound has to be closed first.** A market whose trading straddles the cut puts the
*same* resolution outcome on both sides of the split, so an account holding it looks
skilled in both halves for one reason. Events are therefore assigned whole — an event
counts only if all of its fills fall on one side — and the two halves then share no
outcome at all. Only **855 of 704,521 events** straddle a 2025-01-01 cut, but they are the
long-lived ones, and dropping them takes the paired sample from 164,627 accounts to
**88,279**. It also removes most of the apparent effect: at that cut the rank correlation
falls from **+0.103 to +0.018**. Any persistence result on this tape that skips this step
is measuring the leak.

Leak-free, at three cut dates:

| cut | accounts (≥10 fills each half) | Spearman(z_in, z_out) | 2 s.e. | flagged before → flagged after | base rate | **lift** |
|---|---:|---:|---:|---:|---:|---:|
| 2024-07-01 | 3,594 | **+0.147** | ±0.033 | 26.2 % | 6.2 % | **4.21×** |
| 2025-01-01 | 88,279 | **+0.018** | ±0.007 | 22.1 % | 11.6 % | **1.92×** |
| 2025-07-01 | 151,103 | **+0.087** | ±0.005 | 43.5 % | 16.5 % | **2.63×** |

The base rate is well above the 5 % a calibrated null would give, for the reason §3.1 gives
— the population is not the null, and its z distribution is wide and skewed. The **lift** is
therefore the number to read, not the level.

and the economic version, out-of-sample PnL as a fraction of out-of-sample notional:

| cut | bottom in-sample decile | all accounts | **top in-sample decile** |
|---|---:|---:|---:|
| 2024-07-01 | +0.02 % | +1.20 % | **+3.55 %** |
| 2025-01-01 | −0.35 % | +0.52 % | **+1.30 %** |
| 2025-07-01 | −0.68 % | +0.38 % | **+1.67 %** |

**Skill persists.** At every cut the rank correlation is positive and many standard errors
from zero, accounts flagged before the cut are 1.9–4.2× more likely to be flagged after it,
and the top in-sample decile earns two to three times the population rate out of sample
while the bottom decile earns nothing or loses. This is out-of-sample by construction, uses
no null, and cannot be broken by mis-specifying one.

Three honest qualifications:

* **The effect is small and the middle is noise.** Only the extreme deciles order reliably;
  D2–D9 are not monotone in either z or PnL at any cut, and the strongest rank correlation
  (+0.147) comes from the smallest sample (3,594 accounts). "A skilled minority exists" is
  supported; "3 % of traders drive the market" is not tested by this.
* **The persistent accounts are not the average account.** The 1.82 M accounts with ≥10
  fills lose 0.22 % of notional in aggregate (§3.1), but accounts active in *both* halves
  of a split earn +0.4 % to +1.2 %. Surviving to trade in the second period is itself a
  selection, and part of the population-level ROI gap is that, not skill.
* **PnL is marked to resolution, not to a closing trade.** `s` values the fill's position
  at settlement, so a round-trip nets correctly (buy at p₁, sell at p₂ ⇒ shares·(p₂−p₁)),
  but the ratio to `usdc_amount` mixes capital bases: for a buy the notional is the cost,
  for a sell it is the proceeds. The decile *ordering* is unaffected; the levels should be
  read as an index, not as a return on capital.

### 3.3 Large-trade order imbalance [14] — replicates at scale, and the earlier null is explained

[14] reports that net order imbalance from large trades predicts subsequent returns.
The fifteen-week test found contemporaneous impact (t = +6.1) but no prediction (t = −0.4),
and conjectured liquidity: "the median bar here carries $69". `pm/pmv1_imbalance.py` runs
the identical specification on 3.5 years, with `D` replacing the inferred side and
`p_event` replacing the hand-rolled leg alignment: hourly bars, imbalance = signed notional
over gross notional computed separately for large trades (top decile by notional within the
market) and the rest, returns in log-odds, market fixed effects by within-transformation
(a dummy matrix at 25,317 markets does not fit), standard errors clustered two-way on
market and on hour.

**5,244,127 bars over 25,317 markets** at 1 h; 2,323,642 over 14,780 markets at 4 h.

| | 1 hour | 4 hour | *(vendor tape, 15 weeks)* |
|---|---|---|---|
| **contemporaneous** r_h on imbalance_h — large | **+0.0205 (t +45.6)** | **+0.0378 (t +58.2)** | +0.0238 (t +6.1) |
| **contemporaneous** — small | +0.0413 (t +91.0) | +0.0590 (t +98.8) | +0.0131 (t +8.1) |
| **predictive** r_{h+1} on imbalance_h — large | **+0.0033 (t +13.3)** | **+0.0020 (t +5.1)** | −0.0011 (t −0.4) |
| **predictive** — small | −0.0156 (t −68.3) | −0.0209 (t −57.8) | −0.0060 (t −4.0) |
| **predictive**, controlling for r_h — large | **+0.0043 (t +17.3)** | **+0.0061 (t +15.7)** | +0.0029 (t +1.0) |
| own-return reversal, r_h coefficient | −0.0501 (t −19.0) | −0.1091 (t −29.1) | −0.141 (t −4.6) |

**[14]'s predictive claim replicates.** Large-trade imbalance predicts the next bar at both
frequencies, with and without controls, and the sign survives a microstructure bias that
runs against it: the bar price is the last print, so bid-ask bounce mechanically pushes the
next return the *other* way — which is exactly what the small-trade coefficient and the
own-return reversal pick up.

**And the fifteen-week null is explained, by the mechanism it guessed.** Splitting the same
regression by how much notional the bar carries:

| bar liquidity quintile | median gross | large-trade β | t |
|---|---:|---:|---:|
| Q1 | $1 | +0.0001 | +0.24 |
| Q2 | $13 | −0.0005 | −1.84 |
| **Q3** | **$81** | **−0.0012** | **−4.39** |
| Q4 | $394 | +0.0004 | +1.53 |
| **Q5** | **$3,569** | **+0.0056** | **+8.92** |

The gradient is monotone from Q3 up and the effect lives entirely in the thick bars. The
vendor tape's median bar carried **$69** — squarely inside Q3, where the coefficient is
*negative and significant*. On that tape "large trade" meant a few hundred dollars and a
single order walking a thin book, and the reversal dominates. The 4 h panel reproduces the
gradient (Q3 −0.0012 t −2.72, Q5 +0.0024 t +2.34).

By market category at 1 h, the effect is strongest where [14] looked:

| category | bars | large-trade β | t |
|---|---:|---:|---:|
| Finance | 113,415 | +0.0100 | +6.15 |
| Culture | 206,988 | +0.0095 | +7.58 |
| **Politics** | 844,596 | **+0.0067** | **+11.88** |
| Crypto | 663,056 | +0.0037 | +5.33 |
| Sports | 1,321,309 | +0.0023 | +3.79 |
| Soccer / NBA / United States | 148k / 118k / 269k | ≈ 0 | −0.02 / −0.54 / +0.09 |

[14] studies the 2024 US presidential election across four venues; Politics is the largest
category here that carries the effect, and the pure-sports subcategories carry none.

**Size, not just sign.** With 5.2 M bars every t-statistic is large, and the R² of the
predictive specification is 0.0017. β = +0.0056 in Q5 means a fully one-sided large-trade
bar predicts about 0.0056 of a log-odds unit next hour — roughly 7 % of a typical bar move
(mean |r| = 0.0785) and, near p = 0.5, about 0.14 percentage points of probability. It is
real, it is robust, and it is small enough that transaction costs are the next question,
not the last one.

---

## 4. The falsification audit [10] — the sharpest result here

Nikolopoulos proposes testing the **workflow**, not the strategy: run the complete
pipeline against a reference class in which there is provably nothing to find, and treat
any significant walk-forward evidence there as falsifying. That is exactly what a
reproduction study needs, and it is cheap.

`analysis/synth_panel.py` builds synthetic panels matching the real one in shape,
calendar, per-name volatility, market-factor loading and overnight/intraday variance
split, in which **returns are a martingale difference**. Conditional *variance* is left
predictable (stochastic volatility, volume clustering) because that is realistic and
because Alpha158 is largely a volatility/volume feature set — the point is that no
feature can carry information about the **sign** of any future return. Realised lag-1
return autocorrelation is −0.027 (real panel: −0.044). Four independent panels were
generated, and the **entire** pipeline — Alpha158 → Transformer → top-30 long book — was
run on each, under both label timings.

| panel | timing | IC train | IC valid | **IC test** |
|---|---|---:|---:|---:|
| **real S&P 500** | causal | +0.0688 | +0.0156 | **+0.0118** |
| **real S&P 500** | paper | +0.8238 | +0.8275 | **+0.8155** |
| synthetic null ×4 | causal | +0.023…+0.341 | +0.003…+0.034 | **mean +0.0023, sd 0.0043** |
| synthetic null ×4 | paper | +0.551…+0.571 | +0.529…+0.554 | **mean +0.5383, sd 0.0097** |

Three things fall out.

**1. Under the paper's stated timing the pipeline is falsified.** On data containing no
predictability whatsoever it reports a test IC of **+0.538** with ICIR above 4 —
"overwhelmingly significant" by any conventional reading. The real panel's +0.8155 is
therefore not 0.8155 of signal: **two-thirds of it is manufactured by the timing
convention alone**, and the remainder is not separable from it without a further test.

**2. The mechanism is a single feature.** The paper's label is the open-to-open return
from day *T* to *T*+1, predicted from features observed through day *T* — and day *T*'s
close lies inside that window. Alpha158's `KMID = (close − open) / open` is that window.
Regressing the label on `KMID` alone: **R² = 0.653** on equities (and 0.580 on the crypto
hourly panel). The forecaster's job under this timing is to learn the identity map onto
an input it was handed, which is why its IC (0.82) sits close to the mechanical ceiling
(√0.653 = 0.81). Qlib's own Alpha158 handler labels with `Ref($close,-2)/Ref($close,-1)-1`,
skipping exactly this window; that one line of upstream source is the strongest available
third-party evidence for the argument.

**3. Under causal timing the pipeline survives — but the surviving signal is marginal.**
Test IC on the null is +0.0023 ± 0.0043 against a real +0.0118, i.e. **z = +2.20**. The
pipeline is not manufacturing an edge, but the edge it finds is barely two null standard
deviations wide. Worse for the methodology: the real causal run's **validation** IC is
**+0.0156**, and the four null runs' validation ICs span **+0.003 to +0.034** — the real
value sits in the middle of the noise distribution. `forecast.py` checkpoints on best
validation IC. **Model selection is being performed on a statistic that pure noise
reproduces**, which is the concrete, local instance of what [9] means when it reports
that walk-forward is weak at false-discovery prevention.

The audit costs a few dozen lines and one GPU-hour. It converts the HRT non-replication
from an anecdote about one reproduction into a demonstrated property of this class of
pipeline, and it is a genuinely novel thing to have done to HRT. On the evidence here it
should be the methodological spine of the dissertation.

---

## 5. Moving the work to crypto

The reading list's §6 solves the HRT news gap by switching to an 89-name Nasdaq
benchmark. Crypto is the other direction: a 24/7 market with no earnings, no dividends,
no delistings and no index membership — so the survivorship and corporate-action problems
that cost the equity reproduction 130 of 500 names simply do not arise, and what remains
is the pure signal-and-cost question.

Panel: 30,079 hourly bars × 17 coins × 158 Alpha158 features, 45 contiguous blocks,
2020-11 → 2026-06. Train ≤2024-12 (21,961 bars), validate 2025 (5,453), test 2026 (2,665).
Identical model, identical feature set, identical timing ablation as the equity run.

### 5.1 The leakage transfers, and it is worse

| panel | timing | IC train | IC valid | IC test |
|---|---|---:|---:|---:|
| crypto hourly | causal | +0.064 | +0.030 | **+0.027** |
| crypto hourly | paper | +0.862 | +0.796 | **+0.843** |
| crypto hourly, **synthetic null ×3** | causal | +0.018…+0.030 | +0.002…+0.007 | **mean +0.0079, sd 0.0057** |

The paper's timing yields test IC **+0.843** on crypto against +0.816 on equities — the
same failure, slightly larger, for the same reason (R²(`KMID` → label) = 0.580). Anyone
porting HRT to a new asset class inherits the defect intact; it is a property of the label
definition, not of the market.

### 5.2 The leak-free signal is stronger in crypto than in equities, but not by much

On the causal timing the crypto pipeline reaches test IC **+0.0273** against a
synthetic-null floor of **+0.0079 ± 0.0057** over three null panels with the identical
block structure — **z = +3.4**. On equities the same comparison is +0.0118 against
+0.0023 ± 0.0043, **z = +2.2**. Both are real; neither is large; crypto's is the clearer
of the two.

Note also what the null floor itself says. The same workflow that finds IC +0.0079 in a
martingale-difference crypto panel finds +0.0273 in the real one. An unaudited crypto
paper reporting IC ≈ 0.01 as evidence of alpha would be reporting its own noise floor,
and would have no way to know. That is the argument for making the null panel a standard
artefact of any such study rather than an optional check.

### 5.3 And it is still not tradeable, for the reason the reproduction already found

Over the clean 2026 test window, `forecast-causal top5` returns **+37.5 % gross** with a
Sharpe of +2.07 — and **−81.2 %** at a flat 10 bp, **−98.5 %** under the state-dependent
model. Turnover is **0.75 of book per hour**. Rebalancing the same signal daily instead
of hourly cuts turnover to 0.057 and the loss to −14.3 %, but also removes the edge.
At a flat 10 bp exactly two of the eleven strategies beat equal-weight buy-and-hold —
both daily-rebalanced (momentum Sharpe +0.07, `forecast-paper` −0.19, against equal-weight
−0.52). At 30 bp and under the state-dependent model **nothing does**: equal-weight takes
first place on Sharpe in both, while itself returning −13.6 % and −11.9 % over the window.
And per §2.2, even the +2.07 gross Sharpe does not survive deflation (DSR 0.35).

This is the equity finding again, in a market with ~3× the volatility (equal-weight
annualised vol 0.50 against 0.16 for the equity equal-weight book) and a real gross
edge: **the signal is genuine and the turnover eats it**. It is also the strongest
available argument that the interesting object is the execution/cost layer rather than
another forecaster — which is where idea 6 was pointing, and §2.3 is the caution about how
much of that space MACE has already taken.

### 5.4 What the censoring costs, measured against ground truth

§1.2 documents the censored calendar and `crypto/panel.py` responds by blocking. What has
never been measured is the size of the error that response avoids, because on the crypto
tape there is nothing to compare against — the uncensored series does not exist.

So `analysis/censoring.py` runs it the other way round, which is the same move as §4: take
a panel that **is** complete — the 2,414-day × 370-name S&P 500 panel from the HRT
reproduction, 2013-06 → 2022-12 — apply the crypto tape's own censoring pattern to it, and
compare three numbers a researcher could report. **truth**: the complete panel.
**naive**: retained rows concatenated and gaps ignored, which is what pivoting the vendor's
response and calling `.diff()` gives. **block**: retained rows cut into contiguous runs
with every window and every return confined to one run. Two masks, matching the two shapes
of censoring in §1.2 — `gap`, the empirical 9-on/21-off run sequence of the daily tape, and
`seasonal`, the hourly tape's month-of-year retention (Feb 0 %, Aug 7 %, Q2/Q4 ~97 %).
Twenty phase replicates each; the strategy is long top-30 on trailing momentum.

**Momentum-20, gap mask** (retains 36.8 % of trading days, 73 blocks):

| view | bars | Sharpe | ann. vol | CAGR as reported | Δ Sharpe | Δ vol |
|---|---:|---:|---:|---:|---:|---:|
| **truth** | 2,414 | **+0.585** | **0.221** | **+11.1 %** *(real calendar)* | — | — |
| naive | 888 | +1.216 ± 0.110 | 0.340 ± 0.025 | **+42.3 %** | **+0.631** | **+0.119** |
| block | 888 | +0.851 ± 0.239 | 0.231 ± 0.019 | +19.0 % | +0.266 | +0.010 |

Momentum-5 gives the same picture (truth +0.433; naive +1.079, CAGR 35.8 %; block +0.787).
Under the seasonal mask, which retains 65.7 % of days and creates no long holes, the naive
bias collapses to +0.20 and block ≈ naive.

Three things to take from this.

**1. The naive number is not slightly wrong.** It more than doubles the Sharpe and nearly
quadruples the reported growth rate, and it does so with a replicate spread of ±0.11 —
this is systematic bias, not luck. The mechanism is visible in the volatility column: a
return spanning a 21-day hole is booked as one daily return, so vol is inflated 54 %, and
the mean is inflated more, because momentum earns more over a month than over a day.

**2. Blocking fixes the mechanical error and cannot fix the sample.** The block view
reproduces the true volatility to within 0.010 (4.5 % relative) where the naive view is out
by 0.119 (54 %), which is the whole point of the construction. But its Sharpe is still
+0.27 high, because it genuinely observes 37 % of the calendar and that third is not a
random third. **No reconstruction recovers what the vendor did not send.**

**3. The damage is specific to long holes, not to seasonal sampling as such.** The
seasonal mask retains fewer days than a reader might guess is safe (65.7 %) and still costs
only +0.20 of Sharpe naively, because it creates few multi-week gaps. It is the 21-day hole
that does the harm.

**Consequence.** Every crypto result in §5.1–§5.3 is computed block-aware, so it is not
subject to (1); it *is* subject to (2), and the honest reading of §5.2's z = +3.4 is that
it is a within-block statement about Q2 and Q4. Any crypto paper built on a vendor tape
without a coverage audit should be assumed to be reporting the naive column.

---

## 6. Shortfalls and weaknesses

### Of the papers

1. **HRT [1] never states its timing precisely enough to be reproducible**, and the
   reading that matches its numbers is the one that leaks. §4 quantifies the leak at
   R² = 0.65 and shows the pipeline reports IC +0.54 on pure noise under it.
2. **[1], [6] and most of the cluster report raw or risk-adjusted return without a
   passive floor over the same window.** HARLF's 26 % annualised over 2018–2024 needs to
   be read against buy-and-hold on the same universe. Here, in the bear year, all four
   leak-free arms finish below a do-nothing agent inside the same environment (passive
   −5.6 %; hrt_causal_ep −7.6 %, ddpg −9.9 %, ppo −11.0 %, hrt_causal −13.4 %), and in the
   bull year the hierarchy finishes below passive as well (+22.4 % / +23.1 % vs +26.5 %).
3. **Nobody in the cluster reports a deflated Sharpe.** §2.2 shows why that matters on
   both asset classes: the expected maximum Sharpe from 35 equity trials is 2.905, above
   every leak-free arm produced, and the best gross crypto Sharpe has DSR 0.35.
4. **MACE [2] establishes that cost *level* and *size-dependence* reorder algorithms; it
   does not establish that *regime-dependence* does.** §2.3 finds it does not, here.
5. **The sign-randomisation test in [13] is mis-specified in two ways** — trade-level
   independence (sd(z) = 1.63 against a required 1.00 over 1.8 M accounts) and a
   zero-centred null for spread-crossing takers (median z = −0.83). §3.1. Its *substantive*
   claim nevertheless survives an out-of-sample test (§3.2), which is a different thing
   from the estimator being right.
6. **[14] replicates**, with the qualification that the effect is confined to the top
   liquidity quintile and is economically small (R² 0.0017). §3.3.

### Of this work

* **Four seeds, two test years, one universe** for the equity RL results. [7] uses sixteen
  walk-forward folds across three continents; that is the standard to match, and the DSR
  and falsification results here should be read as diagnostics on the existing sweep
  rather than as a new sweep.
* **The equity cost re-ranking is a first-order recharge**, not a re-simulation, because
  the policies were not checkpointed. It is used only for ordering. Checkpointing the
  agents is a one-line change that would make it exact and should be done before this
  goes in a dissertation.
* **The crypto test window is 2,665 hourly bars (~111 days) in four blocks** — 2026-03-08
  to 2026-06-30 — because §1.2's censoring leaves nothing else after the training cut.
  Three of the four blocks are one calendar quarter, so the out-of-sample period is
  effectively a single regime, and §5.4 quantifies what that costs. No crypto result here
  has ever been tested in a February or an August.
* **The crypto null has three seeds where the equity null has four**, so its sd (0.0057)
  is estimated from three draws and the z = +3.4 in §5.2 should be read as indicative.
* **§5.4's masks are applied to equities, not to crypto.** That is deliberate — it is the
  only way to get ground truth — but the magnitude transfers only insofar as crypto
  momentum behaves like equity momentum across horizons. The *direction* and the
  volatility mechanism do not depend on that.
* **The Polymarket-v1 event unit is the negative-risk contract**, which §3.1 shows is not
  enough to calibrate the null. Results at the event level are therefore an improvement on
  the published trade-level test, not a correct test.
* **§3.2's PnL is marked to resolution and divided by notional**, which mixes capital bases
  between buys and sells; the decile ordering is unaffected but the levels are an index.
* **The relayer filter in `daily_aligned/` removes relayer *takers*.** Aggregate taker PnL
  is therefore computed over a filtered side of the book, which is one candidate
  explanation for accounts active in both halves of a split earning positive markouts
  where the full population loses.
* **`analysis/costs.py` reports permanent impact but does not net it**, since a single
  small book does not move a real market. A multi-agent setting would need it charged.

---

## 7. What is now cheap that was not

The single largest change this round is not a result, it is a capability. Every
prediction-market question in §3 previously ran against 384 markets and 1,744 accounts and
came back "underpowered, cannot say". They now run against 730 M fills and 1.8 M accounts
in under an hour on a laptop-class machine, because the archive is free and the compaction
step in `pm/pmv1_prep.py` makes repeated passes cheap. The things that were listed as
future work in the previous version — the corrected null at scale, the imbalance test on a
liquid tape, an out-of-sample persistence test — are done, and each took under a day.

---

## 8. Reproducing these numbers

```bash
set -a; source .env; set +a          # LUMID_TOKEN (findata only; the §3 block needs no key)
source .venv/bin/activate
pip install duckdb pyarrow huggingface_hub      # new dependencies for §3

# --- findata: equities and crypto -------------------------------------------
python crypto/pull.py 1d             # -> data/crypto.sqlite   27,566 daily bars
python crypto/pull.py 1hour          #                       1,150,609 hourly bars
python analysis/coverage.py          # the seasonal-censoring table in §1.2

# --- reproductions on the existing equity sweep ------------------------------
python analysis/dsr.py               # §2.2  deflated Sharpe over all 35 runs
python analysis/cost_rerank.py       # §2.3  MACE ranking test on the RL arms

# --- falsification audit (needs hrt/artifacts/panel.npz; ~1 GPU-hour) --------
python analysis/synth_panel.py --seed 0            # -> panel_synth.npz  (seed 0 keeps
for lab in causal paper; do                        #    the unsuffixed name that
  python hrt/forecast.py --panel hrt/artifacts/panel_synth.npz --label $lab \
    --out hrt/artifacts/fr_synth_$lab.npz          #    analysis/falsify.py pairs it by)
done
for s in 1 2 3; do
  python analysis/synth_panel.py --seed $s --out hrt/artifacts/panel_synth_s$s.npz
  for lab in causal paper; do
    python hrt/forecast.py --panel hrt/artifacts/panel_synth_s$s.npz --label $lab \
      --out hrt/artifacts/fr_synth_${lab}_s$s.npz
  done
done
python analysis/falsify.py           # §4

# --- crypto ------------------------------------------------------------------
python crypto/panel.py --interval 1hour           # 30,079 x 17 x 158, 45 blocks
python crypto/forecast.py --label causal          # §5.1
python crypto/forecast.py --label paper
for s in 0 1 2; do                                # crypto null, same design as §4
  python crypto/synth_panel.py --seed $s
  python crypto/forecast.py --panel crypto/artifacts/panel_1hour_synth_s$s.npz \
         --label causal --out crypto/artifacts/fr_synth_causal_s$s.npz
done
python crypto/strategy.py                         # §2.3, §5.3  (also writes value paths)
python crypto/dsr.py                              # §2.2  deflation of the crypto sweep
python analysis/censoring.py --reps 20            # §5.4  ~10 min, CPU only

# --- prediction markets: Polymarket-v1 [16] ----------------------------------
python pm/pmv1_pull.py               # 16.8 GB, ~13 min, no API key needed
python pm/pmv1_prep.py               # -> data/pmv1_trades.parquet (730M rows, 58 s)
python pm/pmv1_prep.py --only bars --freq 4h      # -> data/pmv1_bars_4h.parquet
python pm/pmv1_skill.py --sims 10000 --sample 50000        # §3.1  ~70 min (MC-bound)
for c in 2024-07-01 2025-01-01 2025-07-01; do              # §3.2  ~6 min each
  python pm/pmv1_persistence.py --cut $c
done
python pm/pmv1_imbalance.py --freq 1h             # §3.3
python pm/pmv1_imbalance.py --freq 4h

# --- prediction markets: the superseded vendor tape (§1.3) -------------------
python pm/pull.py discover ; python pm/pull.py trades 900 ; python pm/outcomes.py
python pm/skill.py --sims 10000 ; python pm/imbalance.py --freq 1h
```

New this round: `analysis/censoring.py`, `crypto/dsr.py`,
`pm/{pmv1_pull,pmv1_prep,pmv1_skill,pmv1_persistence,pmv1_imbalance}.py`.
Existing: `analysis/{dsr,synth_panel,falsify,costs,cost_rerank,coverage}.py`,
`crypto/{pull,panel,forecast,synth_panel,strategy}.py`, `pm/{pull,outcomes,skill,imbalance}.py`.
`data/` is git-ignored and fully re-pullable from the commands above. The Polymarket-v1
material is **25.2 GB** of it: 16.8 GB archive plus 8.4 GB of compacted tables.

---

## 9. What needs a human

Two things, and only two.

* **A findata token** (`LUMID_TOKEN` in `.env`) for anything that re-pulls the equity or
  crypto tape. The existing token still works; `CLUSTER.md` notes it should be rotated,
  since a live PAT was committed to the pushed history before this work started. Nothing
  in §3 needs it — Polymarket-v1 is public and ungated, and no Hugging Face account or
  `HF_TOKEN` is required.
* **Cluster time for the one experiment that is still not run** — §10.1. Training the RL
  sweep on synthetic null panels is ~2 GPU-hours per run and the sweep is 25 runs; the
  local GTX 1080 Ti would take several days. `hrt/slurm/` and `CLUSTER.md` are already set
  up for the H200 cluster. This is the only item on the list that cannot be finished on
  this machine.

Everything else in §8 runs here: the Polymarket-v1 work is CPU-only and finished in about
two hours end to end, and `analysis/censoring.py` takes ten minutes.

---

## 10. What to do next

Ordered by evidence produced per unit of work.

1. **Finish the falsification audit on the RL sweep.** §4 falsifies the *forecaster*;
   the agents have never been trained on a null panel. Show whether they still beat the
   passive floor when there is nothing to learn. This is the natural completion of the
   one result that is both novel and decisive, and it is the only item needing the cluster
   (§9).
2. **Report DSR and K next to every Sharpe** — now available for both asset classes
   (`analysis/dsr.py`, `crypto/dsr.py`) — **and move the equity screen to CPCV [9].**
   Forty lines, and it immunises the results chapter.
3. **Make §3.2 the prediction-market contribution, not §3.1.** The out-of-sample
   persistence design needs no null, cannot be broken by mis-specifying one, and gives a
   clean positive result at three cut dates. §3.1 becomes the methodological companion:
   the published estimator over-rejects, and correcting the unit is necessary but not
   sufficient. The concrete open question is where the residual sd(z) = 1.30 comes from —
   test whether pooling units by resolution date and underlying, rather than by
   negative-risk contract, brings it to 1.00.
4. **Push §3.3 to a tradeable statement or drop it.** The predictive coefficient is real
   and confined to the top liquidity quintile — which is also where a strategy would have
   to trade to exploit it, at exactly the sizes that move the price contemporaneously
   (β = +0.0205, t = +45.6). That is the same gross-versus-net question as §5.3, and it is
   the unclaimed result. Note what it will take: **`fee_usdc` is identically zero across
   the whole archive** (Polymarket v1 charged no explicit taker fee) and `taker_base_fee`
   is set on only 3.6 % of fills, so the cost here *is* the spread and has to be estimated
   rather than read off. §3.1's median event-level z of **−0.83** is the estimate the data
   already supports — the typical taker gives up 0.83 of a payoff standard deviation to
   cross — and the honest version of §3.3 charges the strategy that before claiming an edge.
5. **Reconsider idea 6's framing before committing.** §2.3 says the state-dependent cost
   model does not reorder anything a flat 30 bp does not already reorder. Either
   demonstrate that volatility-surface geometry carries information beyond σ_t and V_t, or
   move the contribution to the frequency separation [3] identifies — HRT's LLC is a
   same-bar sizing policy and cannot express intra-day execution at all, which is a
   sharper and unclaimed gap.
6. **Fold §5.4 into the data chapter as a standing requirement.** A coverage audit and a
   naive-versus-block comparison should precede any result computed on a vendor tape. It
   costs ten minutes and it is worth +0.63 of Sharpe.
7. **Re-baseline against HRT v2, not v1**, and report Sharpe and turnover alongside
   cumulative return so the comparison is like-for-like.
