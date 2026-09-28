# Results — reproducing the reading list, and moving it to crypto and prediction markets

FYP **H398280** (supervisor: LU Yao), *AI in Financial Data Analytics and Trading*.
Work done 2026-09-07, 2026-09-14 and 2026-09-15 against the local findata mirror, the live
findata API, and the **Polymarket-v1 on-chain archive** [16]. Papers are cited by their
number in [`PAPERS.md`](PAPERS.md); data provenance is in [`DATA.md`](DATA.md); the prior
HRT reproduction is in [`hrt/README.md`](hrt/README.md).

Everything below is computed in this repo. Commands are in §8; §9 lists what still needs a
human.

**New on 2026-09-27/28**, from a 24-hour single-GPU sandbox (`SANDBOX.md`):
- **§3.7:** open-weight LLMs "forecast" Polymarket only for markets that resolved before
  their training cutoff, and a blind break test recovers the cutoff.
- **§3.8:** the one surviving text arm fails its pre-registered out-of-sample test.
- **§4.2:** checkpoint selection on validation Sharpe is inert, with or without the
  floor-relative fix.
- **§2.3 addendum:** exact cost re-simulation confirms the recharge's ranks and corrects its
  levels.

**New on 2026-09-15**, from the NUS cluster run: the equity RL sweep went from 4–5 seeds per
arm to **10** (25 → 60 runs), and the experiment the previous version listed as the one thing
it could not run — **the same sweep retrained on synthetic null panels** — is done (§4.1).
Both change conclusions. §2.2's deflation threshold and §2.3's re-ranking counts move with
the seed count, and the null sweep converts §4 from a result about the forecaster into a
result about the whole pipeline.

---

## 0. Summary

| # | Claim tested | Source | Outcome |
|---|---|---|---|
| 1 | HRT's reported returns | [1] | **Does not replicate leak-free** (established earlier; re-baselined here) |
| 2 | The reported Sharpes are real, not selection | [8] | **No** — no leak-free equity arm survives deflation (DSR ≤ 0.093 / ≤ 0.001 at K = 70), and **no crypto strategy does either**, gross of cost |
| 3 | The pipeline finds signal only where signal exists | [10] | **Fails under the paper's timing**: test IC **+0.54** on data with provably zero predictability — and, §4.1, the *agents* trained on the same panels clear their passive floor by **+0.15 / +0.26 of book** |
| 4 | The cost model changes the algorithm ranking | [2] | **Replicates, but weakly on equities at 10 seeds** — 0–4 of 6 arms move rank (was 2–6 at 4 seeds); 5 of 11 crypto strategies move |
| 5 | ...and *state-dependent* cost changes it further | idea 6 | **No** — the square-root state model ranks identically to a flat 30 bp |
| 6 | A skilled minority drives prediction markets | [13], [15] | **Split verdict.** The published estimator over-rejects at every scale, and correcting the unit is necessary but *not sufficient* (sd(z) 1.63 → 1.30, never 1.00). The claim nevertheless survives a test that needs no null at all: flagged skill **persists out of sample**, 1.9–4.2× lift at three cut dates. The earlier non-replication was a power artefact |
| 7 | Large-trade order imbalance predicts returns | [14] | **Replicates at scale** (t = +13.3), reversing the fifteen-week null. The gradient in bar liquidity is monotone and confirms the mechanism that non-replication conjectured |
| 8 | A censored vendor calendar is a cosmetic problem | new | **No** — measured against ground truth it inflates Sharpe by **+0.63** and reported CAGR by **~4×** |
| 9 | The corrected skill null can be calibrated by pooling harder | §3.1's own conjecture | **No** — maximal same-day pooling moves sd(z) only 1.304 → 1.238. Most of the residual is not mis-specification but the skill §3.2 measures |
| 10 | The imbalance edge is tradeable | [14] + §10.4 | **Not by a taker.** Gross +0.086 pp/share against a 0.363 pp round trip, at every hold and threshold tested. The same arithmetic pays the *maker* +0.278 pp |
| 11 | A language model can read a prediction market's own question | new | **No.** No encoder — tf-idf, FinBERT, BERT-base, MiniLM, frozen or fine-tuned — beats a price baseline under a walk-forward event split (dIC −0.022 … +0.003), and in half the folds a fair shrinkage search sets the text weight to *exactly zero*. Under an i.i.d. row split the same text is worth **+0.505 of IC** and a book of **+18.56 pp/share (t = +54.6)** |
| 12 | The RL agents, not just the forecaster, are falsified | §4's own open item | **Under the paper's timing, yes.** Retrained where returns are a martingale difference, the leaky arms still beat their own panel's passive floor by **+0.153** (t = +2.94) and **+0.257** (t = +3.81) — a quarter to two-fifths of the excess they show on the real market. The leak-free arms clear nothing on the null (+0.005, t = +0.15) **and, pooled, nothing on the real market either** (−0.017, t = −2.16; −0.041, t = −4.79); both *hierarchy* arms are below the floor in both years |
| 13 | The statistic the agents are checkpointed on measures learning | §4's conjecture, at agent level | **No.** Validation Sharpe is *higher* on the null panels than on the real one for every leak-free arm: **35 of 40** real runs fall below the null median, and a real run sits at the 36th percentile of the null distribution |
| 14 | An LLM's edge over a prediction market is forecasting | new | **No, for the models that have one.** Gemma-4's edge over the price (β +0.115, t +6.29) exists only for markets that resolved before its training cutoff and is zero after; a blind break search puts the switch-off at Feb 2025 against a documented Jan 2025 (perm. p < 0.003). On the same 1,162 markets the later-cutoff Gemma has the edge and the earlier one does not. Qwen3.5's undocumented cutoff is estimated at Nov 2025 (p = 0.007). OLMo-3 and Llama-3.1 show no break (§3.7) |
| 15 | The one surviving text arm holds on untouched months | §3.6, §10 item 7 | **No — pre-registered, not supported.** On May–Sep 2026, rebuilt from the public API, validation selects γ = 0, so dIC = 0 exactly. The price baseline holds out of sample (IC 0.078), text adds nothing (§3.8) |
| 16 | Checkpointing on validation Sharpe selects a better policy | §10 item 3 | **No, and nor does the proposed fix.** Over 59 probe runs, the best-validation checkpoint beats the final one by −0.002 (t −0.34) on the real panel, the same as on null panels; checkpointing on Sharpe over the passive floor gives −0.003. Validation and test rank a run's checkpoints with correlation −0.05 (§4.2) |
| 17 | §2.3's first-order cost recharge is safe to rank with | §10 item 4 | **Yes for ranks, no for levels.** Exact replay from saved checkpoints ranks arms identically in 20 of 24 cells (one swap in the rest) but moves Sharpe levels by up to 0.34 (§2.3) |

Nine findings are new and transferable beyond this repo:

* **The leakage is measurable, not conjectural, and it survives all the way to the
  traded book.** 65 % of the variance of the label under the paper's stated timing is
  mechanically explained by a single same-bar Alpha158 feature (`KMID`) that the model is
  given as an input. Run on synthetic data where nothing is predictable, the same pipeline
  still reports a test IC of +0.54 — and the RL agents trained on that forecast still beat
  a do-nothing floor measured in their own environment, on the same panel, by 15 and 26
  points of book.
* **A falsification audit is also a calibration instrument, and it is cheaper than
  believing the formula.** The same 25 null runs are a direct Monte Carlo of the quantity
  the Deflated Sharpe Ratio estimates analytically. On the leak-free arms the two agree to
  within a factor of 1.30 (2021) and 0.99 (2022) — so the DSR's closed form is roughly
  right here, which is worth knowing because nothing in [8] establishes it for a strategy
  family like this one. It also delivers something the formula cannot: a non-parametric
  p-value that needs no Gaussian assumption and no zero centre. Every leak-free arm's
  **best** seed has p ≥ 0.06, and its seed mean p ≥ 0.18.
* **The hierarchy — the paper's actual contribution — never beats doing nothing.** Over ten
  seeds, `hrt_causal` and `hrt_causal_ep` finish *below* the in-environment passive floor in
  **both** test years and at both alpha units (2021 −0.046 and −0.054, t = −3.02 and −4.14;
  2022 −0.057 and −0.013). Pooled, the leak-free arms are −0.017 (t = −2.16) and −0.041
  (t = −4.79). The one arm that does clear the floor leak-free is the *flat* PPO baseline,
  and only in 2021 (+0.026, t = +3.48), by a margin the null sweep cannot distinguish from
  noise (p = 0.18 on the seed mean). In 2022 the best leak-free run on a **null** panel
  (Sharpe +2.54) beats the best leak-free run on the **real** panel (+0.20), and 10 of 17
  null runs do.
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
* **A mis-calibrated null and a real effect are separable without a null.** The excess
  dispersion left in the corrected skill test is not a pooling artefact — collapsing every
  market an account touches that closes on one day moves sd(z) only 1.304 → 1.238 — and the
  out-of-sample persistence of §3.2 accounts for most of what remains, via
  V = 2ρ/(1−ρ). The residual over-rejection is largely the signal, not the estimator.
* **An order-flow edge can be real, robust and worth nothing to the side that reads it.**
  Priced at the VWAPs a taker actually transacts at, [14]'s signal earns +0.086 pp per
  share gross and costs 0.363 pp to trade — and hands +0.278 pp to the passive counterparty.
* **On a prediction market the question text is a timestamp, and the split convention is
  the whole result.** A tf-idf ridge given only a market's slug predicts *the date the row
  traded* at **R² = 0.920** (43-day MAE against a 175-day naive baseline), and masking every
  literal date, year and integer reaches only 0.886 because the proper nouns are themselves
  dated. Holding features, target and model fixed, the same text block is worth +0.505 of
  test IC under a random row split and −0.007 under a walk-forward event split. This is §4's
  failure — a defensible-looking convention worth more than everything the model does after
  it — reproduced on *real* data, in a model class the reading list does not cover.

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

`hrt/README.md` records the reproduction at four seeds: **2021 +23.1 % / 2022 −7.6 %**
against a published **+39.8 % / +2.3 %**, with every RL agent below a passive floor in the
bear year (passive-in-env 2021 +26.5 %; equal-weight buy-and-hold +32.8 %; S&P 500
+28.8 %). Nothing here overturns that. What follows asks whether the numbers that *were*
produced mean anything, and whether the failure generalises.

**At ten seeds** (the sweep the cluster finished on 2026-09-15; the four committed seeds are
byte-identical, so this is an extension, not a re-run) the leak-free hierarchy arms are
**2021 +21.90 ± 4.8 % / 2022 −11.31 ± 6.1 %** per-step and **+21.05 ± 4.1 % / −6.98 ± 6.9 %**
per-episode. The headline is unchanged and the dispersion is the story: 2022 spans ~20
points across seeds, which is why four seeds could not settle anything and why §4.1 is
reported on ten.

One thing the extra seeds do change, and it is not in HRT's favour. Against the
**in-environment passive floor** — `hrt/passive.py`, the same integer-lot, cash-sequenced,
10 bp environment the agents trade in, deploying once and sitting still — the leak-free arms
are below the floor in **both** years, not only the bear one: mean excess **−0.017 of book
(t = −2.16) in 2021 and −0.041 (t = −4.79) in 2022** over 40 leak-free runs. At four seeds
the 2021 shortfall was inside the noise. The shortfall sits in exactly the arms the paper is
about: `hrt_causal` and `hrt_causal_ep` are below the floor in both years (2021 −0.046 and
−0.054; 2022 −0.057 and −0.013), while the flat PPO baseline clears it in 2021 (+0.026,
t = +3.48, 10/10) before losing to it in 2022 (−0.057, t = −7.07, 0/10).

Note for the write-up: HRT **v2 (May 2026)** is a different paper — 89-name Nasdaq
universe, 2020–2023 test, Sharpe 1.06 → 1.24, turnover 0.112 → 0.090, LLC penalties on
turnover and drawdown. The reproduction targets v1. That should be stated explicitly
rather than glossed.

### 2.2 Deflated Sharpe Ratio [8] — nothing survives it, on either asset class

`analysis/dsr.py` applies Bailey & López de Prado's correction to every run in the equity
sweep. K counts **all 70 runs ever aggregated** — the 60 in `runs/` *and* the 10 in
`runs_invalid/`, because the surviving configuration was chosen after seeing both. The
threshold and K now travel inside `dsr_test20**.json` rather than being quoted by hand,
because they move every time the sweep grows and a hard-coded 2.905 is how a results
chapter goes stale.

**2021** (σ_SR across trials 0.1004 daily → SR\*₀ = **3.829** annualised). The Sharpe shown
is the arm's **best seed**, not the seed mean, because the quantity being deflated is a
maximum over a search:

| arm | seeds | Sharpe (repo) | skew | kurt | PSR(0) | **DSR** | previously (4–5 seeds, K = 35) |
|---|---:|---:|---:|---:|---:|---:|---:|
| hrt_paper_ep *(leaky)* | 10 | 9.35 | +0.17 | 5.55 | 1.000 | **0.988** | SR 9.35 → 0.999 |
| hrt_paper *(leaky)* | 10 | 8.79 | +0.27 | 5.82 | 1.000 | **0.983** | SR 6.02 → 0.936 |
| ppo_causal | 10 | 2.81 | −0.12 | 4.01 | 0.992 | 0.093 | SR 2.51 → 0.268 |
| ddpg_causal | 10 | 2.47 | −0.16 | 4.00 | 0.985 | 0.058 | SR 2.35 → 0.229 |
| hrt_causal | 10 | 2.11 | −0.29 | 4.20 | 0.971 | 0.033 | SR 2.11 → 0.175 |
| hrt_causal_ep | 10 | 2.04 | −0.09 | 3.47 | 0.969 | 0.028 | SR 2.04 → 0.159 |

The last column is what the previous version reported: the best Sharpe over 4–5 seeds,
deflated at K = 35. Both halves of it moved.

**2022** (SR\*₀ = **3.281** annualised): `hrt_paper` **0.817**, `hrt_paper_ep` **0.461**, and
every leak-free arm at **≤ 0.001**.

Doubling the seed count moved this in the direction the correction exists to capture.
Searching harder raises both the best Sharpe observed (`hrt_paper` 6.02 → 8.79) *and* the
Sharpe the search alone is expected to produce (2.905 → 3.829), and the second rises
faster: every leak-free arm's DSR falls by roughly two-thirds. **More seeds did not firm
up the leak-free results; they buried them**, which is the mechanical point [8] makes and
an unusually clean illustration of it.

Read plainly: an annualised Sharpe of 2.1 looks impressive and has PSR(0) = 0.97 against
a zero benchmark, but the expected maximum Sharpe from 70 trials of this strategy family
is **3.829** — higher than any leak-free arm achieved, in either year. **Every leak-free
result in the sweep is below what the search would be expected to produce from luck
alone**, and §4.1 now checks that threshold against a direct simulation of it rather than
taking the closed form on trust.

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

2021 Sharpe by arm, and the rank each arm holds (ten seeds; `analysis/cost_rerank.py` now
writes `hrt/artifacts/cost_rerank.json`, which the figure and this table both read):

| arm | turnover | gross (0 bp) | 10 bp | 30 bp | 60 bp | rank 0/10/30/60 |
|---|---:|---:|---:|---:|---:|---|
| hrt_paper_ep | 0.375 | +6.47 | +5.70 | +4.17 | +1.86 | 1 / 1 / 1 / **3** |
| hrt_paper | 0.330 | +5.35 | +4.69 | +3.36 | +1.36 | 2 / 2 / 2 / **4** |
| ppo_causal | 0.004 | +2.19 | +2.18 | +2.16 | +2.14 | 3 / 3 / 3 / **1** |
| ddpg_causal | 0.004 | +2.03 | +2.02 | +2.01 | +1.98 | 4 / 4 / 4 / **2** |
| hrt_causal_ep | 0.226 | +2.00 | +1.55 | +0.67 | −0.66 | 5 / 6 / 6 / 6 |
| hrt_causal | 0.195 | +1.97 | +1.59 | +0.84 | −0.30 | 6 / 5 / 5 / 5 |

**This is where the extra seeds cost the paper something.** Moving from the 10 bp default:
**2 of 6** arms change rank at 0 bp (was 4 of 6), **0 of 6** at 30 bp (was 2 of 6) and
**4 of 6** at 60 bp (was 6 of 6). In 2022 it is 3 of 6 at every rate. The qualitative claim
survives — the ordering at 60 bp is not the ordering at 10 bp, and the near-zero-turnover
flat baselines take the top two places there while the two leak-free hierarchy arms take
the bottom two — but the *strength* of the equity replication was inflated by seed noise.
At four seeds `ppo_causal` and `ddpg_causal` sat below the causal hierarchy arms at 0 bp
and crossed them as the rate rose; at ten seeds they start above and simply stay above
until 60 bp, so fewer crossings are left to count. A re-ranking count is itself a statistic
with a standard error, and nothing in [2] reports one.

In 2022 the same recharge shows `hrt_causal_ep` at Sharpe **+0.05 gross and −0.23 net**:
the hierarchy earns a real gross edge and hands all of it to the broker. MACE's claim
replicates on an independent codebase; on this sweep it replicates **weakly on equities and
clearly on crypto**, and the honest version of the sentence names the count at both seed
depths.

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

#### Exact re-simulation, 2026-09-28 — the recharge ranks correctly and misstates levels

The recharge above was first-order because the policies had been thrown away. The
2026-09-27 probe sweep (§4.2) kept every checkpoint, so `analysis/cost_resim.py` rebuilds
the checkpoint `train.py` selects and replays both test years through `TradingEnv` at each
flat rate. The policy stays fixed, but the cash path, the fills that cash permits, and the
compounding all respond to the rate.

- **Scope.** 4 leak-free arms × 5 seeds on the real panel and 2 null panels, 59 runs.
- **Replay check.** At 10 bp the replay reproduces 111 of 118 recorded test returns to 1e-9.
  The rest drift by at most 3.4 pp. The policies were trained on GPU and are replayed on
  CPU, so float differences flip a few discrete trades. Both the exact and the recharged
  paths start from the same CPU replay, so the comparison is like-for-like.

| | cells (panel × year × rate) | identical ranking, exact vs recharge | largest level gap |
|---|---:|---:|---:|
| all | 24 | **20** (the other 4 differ by one swap, Spearman 0.8) | 0.34 Sharpe |

On the real panel at 60 bp, the exact replay moves 2 of 4 arms in 2021 and 4 of 4 in
2022; the recharge moves 2 and 3. So MACE's ranking claim stands and is marginally
stronger exactly, and the recharge's *ordering* was trustworthy. Its *levels* were not:
they are off by up to a third of a Sharpe unit. Quote levels from the exact replay only.

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

### 3.4 Where the residual over-dispersion comes from — §3.1's conjecture is wrong

§3.1 ends with a specific guess. The event-level null still over-rejects (sd(z) = 1.304
against a required 1.00), and the reason offered is that the negative-risk contract does
not collapse *structurally distinct markets driven by one uncertainty* — the worked case
being 2025-06-17, when 37 Bitcoin markets close and the grouping yields 30 events. The
stated cheap test is "whether sd(z) falls further when units are pooled by resolution date
and underlying rather than by contract". `pm/pmv1_pooling.py` runs it.

**First, a correction to the diagnostic itself.** The null fixes the *second* moment:
E[z] = 0 and Var(z) = 1 together give **E[z²] = 1 exactly**, so `rms(z)` is the quantity
that must equal 1.00, and it decomposes as `rms² = mean² + sd²` — a location part, which
is the spread the average taker crosses, plus a dispersion part. §3.1 quotes the centred
`sd(z)`, which sees only the second. Both are given below. The last rung is the degenerate
one — a single coin flip per account, which forces z = ±1 — and it returns
**rms(z) = 1.000** and sd(z) = √(1 − 0.368²) = 0.930. That is not a result; it is the
arithmetic checking itself.

Exact over all **1,819,876 accounts** — no sampling, no simulation:

| randomisation unit | median units/acct | mean z | sd(z) | **rms(z)** — must be 1.000 | z > 1.645 | z < −1.645 |
|---|---:|---:|---:|---:|---:|---:|
| **trade** (as published) | 38 | −0.256 | 1.628 | **1.648** | 7.27 % | 11.98 % |
| bet = trader × market | 16 | −0.545 | 1.456 | 1.555 | 5.67 % | 18.31 % |
| **event** = trader × neg-risk event | 14 | −0.440 | 1.304 | **1.376** | 5.55 % | 12.38 % |
| trader × close-day × category | 12 | −0.443 | 1.272 | 1.347 | 5.35 % | 12.40 % |
| **trader × close-day** | 11 | −0.433 | **1.238** | 1.312 | 5.14 % | 11.14 % |
| trader (degenerate, one flip) | 1 | −0.368 | 0.930 | **1.000** | 0 % | 0 % |

(The tails here are normal-approximation tails, so they differ from §3.1's Monte-Carlo
Rademacher fractions; the sd(z) column reproduces §3.1 exactly.)

**The conjecture does not survive.** Pooling every market that closes on the same day and
sits in the same category — which merges all 37 of the Bitcoin markets, and a great deal
besides — moves sd(z) from 1.304 to **1.272**. Pooling *everything an account touches that
closes on a given day*, regardless of category, reaches **1.238**. That is the maximal
same-day pooling available, it deliberately over-pools, and it buys **+0.066 of the 0.304
that has to be explained**. Whatever is inflating the event-level null, it is not that one
day's markets share an underlying.

**What it is instead, measured without a null.** Over-dispersion above 1 has exactly two
sources: units that are not independent decisions, which is what pooling removes, and
genuine heterogeneity in per-unit edge across accounts, which is skill. The second is
identified by §3.2, because skill is the only component that persists across a time split.
Write an account's statistic as `z = δ_i·√n_i + N(0,1)` — a per-unit edge accumulating over
n units, plus null noise of unit variance. Then Var(z) = 1 + V with V = Var(δ√n). A split
puts about n/2 units on each side, so each half carries V/2 and the two halves share only
δ:

    corr(z_in, z_out) = (V/2) / (1 + V/2)    ⟹    V = 2ρ / (1 − ρ)

ρ is Pearson, not Spearman, since the decomposition is about variance; it is computed on
z winsorised at 0.5 % a side, because a handful of accounts hold one enormous resolved
position and Pearson on that tail is effectively a two-point estimate. The paired accounts
are a selected subpopulation — surviving to trade in both halves is itself a filter (§3.2)
— so the last two columns recompute the event-level ladder **on exactly those accounts**:

| cut | accounts | Pearson ρ | s.e. | Spearman | V = 2ρ/(1−ρ) | **sd(z) skill alone implies** | their own sd(z) | their rms(z) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024-07-01 | 3,594 | +0.163 | 0.016 | +0.147 | 0.389 | **1.179** | 1.234 | 1.299 |
| 2025-01-01 | 88,279 | +0.025 | 0.003 | +0.018 | 0.050 | **1.025** | 1.207 | 1.270 |
| 2025-07-01 | 151,103 | +0.109 | 0.003 | +0.087 | 0.246 | **1.116** | 1.291 | 1.316 |

At the 2024 cut, persistence alone predicts sd(z) = 1.179 against the 1.234 those accounts
actually show — **skill accounts for most of the excess**. At the 2025-07 cut it predicts
1.116 against 1.291, roughly half. At 2025-01 — the cut where dropping straddling events
removed nearly all of the apparent persistence (§3.2) — it predicts almost none of it.

**What this changes.** §3.1's verdict was "correcting the unit is necessary but not
sufficient", with the residual filed as an open mis-specification. It is mostly not a
mis-specification. The published trade-level null genuinely over-rejects, and widening the
unit to the event genuinely fixes most of that (1.628 → 1.304); what remains is largely the
signal §3.2 measures directly, and further pooling cannot remove it because there is
nothing left to pool. The honest statement is narrower and stronger than the old one: **the
estimator's error is the randomisation unit and the zero centre, and once both are
addressed the remaining over-dispersion is mostly the thing the paper was looking for.**

Two qualifications. The decomposition assumes δ_i is stable across the split, so a trader
whose edge decays is counted as noise and V is a lower bound; and the three cuts disagree
by more than their standard errors, which says V is a property of the period as much as of
the population. Neither rescues the pooling conjecture, which is what was being tested.

### 3.5 What the imbalance signal is worth once the spread is paid — §10's item 4, settled

§3.3 leaves [14]'s replication one step short of a claim: the predictive coefficient is
real, confined to the top liquidity quintile, and economically small, and nothing has been
charged for trading it. §10 item 4 also says why charging is awkward — **`fee_usdc` is
identically zero across the whole archive**, because Polymarket v1 levied no explicit taker
fee, so the only cost is the spread and it has to be estimated.

`pm/pmv1_spread.py` estimates it from the tape and then, more usefully, avoids needing the
estimate at all. Because `D` is ground truth, the price at which takers **bought** and the
price at which takers **sold** can be separated inside every bar, which gives both an
effective spread

    spread_p = vwap(p_event | D = +1) − vwap(p_event | D = −1)

and, crucially, **the two prices a strategy could actually transact at**. §3.3 regresses on
the bar's last print, and §3.3 itself flags that the last print carries bid-ask bounce
pushing the next return against the signal. A backtest that enters at the last print and
then subtracts a spread therefore charges the crossing twice. Pricing the round trip at the
VWAPs removes the bounce and the spread together and needs no cost model:

* signal known at the close of bar *h*; position opened in *h+1*, closed in *h+2*;
* long: buy at `vwap_buy(h+1)`, sell at `vwap_sell(h+2)`;
* short: sell at `vwap_sell(h+1)`, buy back at `vwap_buy(h+2)`;
* the same trip priced mid-to-mid is the gross benchmark, so the difference is exactly the
  two half-spreads.

**8,822,101 bars over 38,685 markets**; both sides trade inside the bar in 51.4 % of them,
which is the sample where a spread is observable at all.

| liquidity quintile | bars | median gross | median p | median spread | mean spread | spread as % of price |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 906,279 | $3 | 0.007 | 0.00100 | 0.00494 | 10.00 % |
| Q2 | 906,278 | $30 | 0.041 | 0.00179 | 0.00768 | 6.67 % |
| Q3 | 906,278 | $140 | 0.100 | 0.00341 | 0.00845 | 3.85 % |
| Q4 | 906,278 | $580 | 0.172 | 0.00673 | 0.00939 | 3.75 % |
| **Q5** | 906,279 | **$4,729** | 0.260 | **0.00522** | 0.00493 | **2.69 %** |

Half a probability point — half a cent on a dollar contract — in the thickest bars, and
ten per cent of the price in the thinnest. Now the round trip, in **pp = probability points
per share** (1 pp = 1 cent on a $1 contract):

| subset | trips | gross pp | **net pp** | cost pp | half-spread in | half-spread out | t(gross) | t(net) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all trips | 1,273,650 | +0.0244 | **−0.6924** | 0.7168 | 0.3922 | 0.3246 | +5.68 | −148.4 |
| Q1 | 254,730 | −0.0030 | −0.5759 | 0.5728 | 0.2953 | 0.2775 | −0.97 | −145.3 |
| Q2 | 254,730 | +0.0010 | −0.7786 | 0.7796 | 0.3935 | 0.3861 | +0.23 | −156.5 |
| Q3 | 254,730 | +0.0044 | −0.9264 | 0.9308 | 0.4752 | 0.4556 | +0.76 | −141.1 |
| Q4 | 254,730 | +0.0341 | −0.9036 | 0.9377 | 0.4933 | 0.4444 | +4.09 | −99.5 |
| **Q5** | 254,730 | **+0.0858** | **−0.2776** | **0.3633** | 0.3037 | 0.0596 | **+4.71** | −14.3 |

**The gross edge survives the change of price basis**, which is a real check on §3.3: moving
from last-print log-odds to mid-VWAP probability, and from a continuous regressor with
market fixed effects to a plain sign rule with none, the Q5 effect is still there at
t = +4.71 and still absent in Q1–Q3. **The net edge is not close.** In the only quintile
where the signal exists, the round trip costs **4.2× the gross edge**.

Two attempts to rescue it, both of which fail for instructive reasons:

| top quintile, by hold | trips | gross pp | net pp | cost pp | half in | half out |
|---|---:|---:|---:|---:|---:|---:|
| 1 h | 254,730 | +0.0858 | −0.2776 | 0.3633 | 0.3037 | 0.0596 |
| **2 h** | 242,237 | **+0.1338** | −0.3428 | 0.4767 | 0.3746 | 0.1021 |
| 4 h | 222,409 | +0.1048 | −0.5228 | 0.6275 | 0.4110 | 0.2165 |
| 8 h | 192,682 | +0.0406 | −0.6510 | 0.6916 | 0.4121 | 0.2795 |
| 24 h | 131,695 | +0.0589 | −0.6601 | 0.7190 | 0.4115 | 0.3076 |

A longer hold is the standard way to amortise a fixed round trip. It does not work here
because **the cost is not fixed**: the half-spread paid on the way out rises from 0.06 pp
to 0.31 pp as the hold lengthens. The reason is visible in the column — at a one-bar hold
the exit lands in a bar where the same one-sided flow is still running, so the exiting
side transacts close to mid; by 24 hours the exit is a generic bar and pays a full
half-spread. Meanwhile the gross edge peaks at two hours (+0.134 pp) and decays. The best
ratio available is still 3.6×.

| top quintile, by signal strength | trips | gross pp | net pp | cost pp |
|---|---:|---:|---:|---:|
| \|oi\| = 1 (fully one-sided bar) | 82,319 | +0.1361 | −0.4883 | 0.6244 |
| \|oi\| > 0.5 | 167,548 | +0.1262 | −0.3539 | 0.4801 |
| \|oi\| ≤ 0.5 | 87,182 | +0.0080 | −0.1309 | 0.1389 |

Conditioning harder on the signal raises the gross edge by 59 % and the cost by 72 %, because
a one-sided bar is a bar in which liquidity has already been consumed. The selection that
makes the signal strong is the same selection that makes the spread wide.

**The result, stated as a claim rather than a coefficient.** [14]'s predictive effect is
real, it is confined to thick bars, and **it is not tradeable by a taker at any hold or any
signal threshold tested**. But the arithmetic has a second reading that is a positive
result, and it needs no extra computation. The counterparty of this round trip is passive
on both legs, so its PnL is the negative of the taker's net: it collects the two
half-spreads, 0.3633 pp, and pays the 0.0858 pp of adverse selection that the signal is
measuring, for **+0.2776 pp per share**. The information in large-trade order imbalance is
worth roughly a quarter of a cent a share — and on this venue, in this period, **all of it
accrues to whoever is providing the liquidity, not to whoever is reading the signal.**

That is the honest version of §10's item 4, and it points the same way §5.3 does: the
tradeable object is the execution layer, not the forecast.

---

### 3.6 Can a language model read the question? — FinBERT on the market's own text

Every NLP-in-finance result on the reading list reads text *about* an asset — news,
filings, tweets — and maps a sentiment score onto a ticker that exists independently of
the words. A prediction market has no such underlying. The question **is** the contract:
`will-lando-norris-be-the-2025-drivers-champion` specifies the entire payoff, and two
markets differ only in their sentences. It is the one venue on this reading list where a
language model is reading the instrument rather than commentary about it, which makes it
the right place to ask whether a language model can read an instrument at all.

The answer is that it cannot — and that the pipeline which says otherwise is wrong in a
way that is worth more than the signal it claims. §4 shows a label-timing convention
manufacturing +0.538 of IC on noise. This section shows a **split** convention doing the
same thing on real data, at almost the same magnitude, to a model class the reading list
does not cover.

#### The corpus is mostly not language

`market_slug` is the question lowercased and hyphenated, so the text recovers with one
`replace`. What comes back is not uniformly natural language, and nothing downstream is
safe until that is measured. Over all 746 M fills:

| slug shape | markets | share of fills | example |
|---|---:|---:|---|
| contains a unix timestamp | 220,495 | **57.1 %** | `btc-updown-5m-1776662700` |
| contains an ISO date | 392,975 | 11.4 % | `nba-bos-bkn-2025-11-18-1h-spread-away-5pt5` |
| neither | 237,950 | 31.6 % | `will-arsenal-win-a-trophy-this-season` |

**Fifty-seven per cent of the tape by fill count is priced on a string containing its own
resolution time in epoch seconds.** Those markets are individually tiny and live five
minutes, so they are only 0.5 % of liquid market-days, and `pm/pmv1_text_panel.py` tags
every row `kind ∈ {ts, date, nl}` and keeps them apart. All results below are the `nl`
stratum: **459,722 market-days over 13,456 markets and 9,866 events**, a genuine corpus of
questions.

#### The baseline that text has to beat

Reporting a text model's IC alone would be meaningless here, because the price already
carries a large signal. Binning the `nl` stratum by price and clustering on event:

| price bin | rows | markets | mean p | P(win) | edge (pp) |
|---|---:|---:|---:|---:|---:|
| (0.00, 0.02] | 50,511 | 4,805 | 0.015 | 0.020 | **+0.54** |
| (0.05, 0.10] | 73,063 | 5,693 | 0.074 | 0.066 | −0.88 |
| (0.20, 0.30] | 38,139 | 4,714 | 0.251 | 0.269 | +1.82 |
| (0.40, 0.50] | 22,957 | 4,243 | 0.455 | 0.427 | −2.78 |
| (0.60, 0.70] | 17,856 | 2,777 | 0.652 | 0.611 | **−4.15** |
| (0.70, 0.80] | 16,736 | 2,548 | 0.752 | 0.709 | **−4.31** |
| (0.80, 0.90] | 16,337 | 2,258 | 0.855 | 0.811 | **−4.41** |
| (0.95, 1.00] | 11,135 | 1,540 | 0.972 | 0.973 | +0.12 |

Favourites between 0.6 and 0.9 are overpriced by more than four probability points. This
is the reverse of the racetrack favourite–longshot bias and it is the whole of what any
model on this panel has to work with. **Unconditionally there is nothing**: one
observation per market, clustered on event, gives a mean of +0.61 pp at t = −0.63. The
structure is entirely price-conditional.

So the only honest form of the question is nested:

    M1  price state + category        the baseline that must be beaten
    M2  M1 + text features            the same thing plus the sentence
    dIC = IC(M2) - IC(M1)             what the language is worth

Two design points make that comparison fair rather than rigged. **Block-wise shrinkage:** a
single ridge penalty over 31 price columns and 768 embedding columns cannot shrink the text
without also shrinking the price, so M2 would lose as a fitting artifact rather than as a
finding. The text block is scaled by γ before stacking and (α, γ) are searched jointly,
with γ = 0 reproducing M1 exactly — so M2 can only lose on validation by being genuinely
worse. **Walk-forward folds:** events are assigned on first appearance, each fold trains on
everything before a validation window and tests on the month after it, and a row is kept
only if it falls inside its own event's window, so no event straddles a boundary in either
direction.

#### What the price baseline is worth, and the three things that inflate it

M1 alone runs a real book, and getting its number right took three corrections that each
look like a detail and together account for most of it. Long the top quintile of the
signal, short the bottom, one share each, held to resolution — where settlement pays at
par, so unlike §3.5's two-legged round trip **only the entry crosses the spread**.

| correction | net pp/share | t |
|---|---:|---:|
| all `nl` markets, executing in the signal's own bar | +5.90 | +9.99 |
| … binary markets only (drop negative-risk multi-outcome) | +3.45 | +5.28 |
| … executing next day instead of in the signal's own bar | +1.69 | **+3.29** |

Negative-risk events list each candidate as its own market and their `p_event` values are
not built to sum to one across the event, so a book that shorts the favourite and longs
the field inside one is harvesting the data layout (DATA.md §9); they are 2.3× richer than
binary markets and are dropped. Executing in the same bar whose last print and volume
produced the signal is a look-ahead **worth +1.76 pp — more than the entire surviving
edge**. A fourth correction changes no headline but is required for coherence: on 7.7 % of
market-days the day's buy-VWAP sits below its sell-VWAP, an artifact of volume-weighting a
trending session rather than a crossed book, and taking those at face value pays the
strategy to cross and produces a net edge larger than its own gross. The half-spread is
floored at zero.

What survives — **binary markets, next-day VWAP execution, floored spread, 10 walk-forward
folds — is gross +2.53 pp and net +1.69 pp per share at t = +3.29 over 3,860 events**, and
it is worth saying exactly what it is:

| cut | net pp | t | | cut | net pp | t |
|---|---:|---:|---|---|---:|---:|
| **short** | **+3.65** | **+5.82** | | p ∈ (0.65, 0.85] | +5.60 | +4.19 |
| long | −1.15 | −1.30 | | p ∈ (0.85, 1.00] | +5.77 | +5.74 |
| liquidity Q1 (thinnest) | +4.75 | +4.99 | | p ∈ (0.00, 0.05] | −2.13 | −3.61 |
| liquidity Q5 (thickest) | −0.51 | −0.92 | | p ∈ (0.35, 0.65] | −1.18 | −1.16 |

The edge is **entirely the short side, entirely in favourites above 0.65, and entirely in
the thinnest liquidity quintile**. In the top quintile — the only one with capacity — it is
zero. It is positive in 6 of 10 folds, and it is not a few large resolutions: dropping the
25 largest-contributing events leaves +1.59 pp. This is the same conclusion §3.5 and §5.3
reach by other routes — the effect is real and the execution layer eats it — and it is the
floor a text model must clear.

#### Why the text cannot help: the slug is a clock

Before any model, one measurement explains most of what follows. Fit a tf-idf ridge on the
slug alone and ask it to predict **when the row traded**:

| text variant | R² | MAE | naive MAE |
|---|---:|---:|---:|
| verbatim | **0.920** | 43 days | 175 days |
| every date, year and integer masked | 0.886 | 50 days | 175 days |

The question text places itself in time to within six weeks over a 1,252-day archive. And
**masking the calendar barely helps**, which is the more interesting half: you cannot
scrub the clock out of a prediction-market question, because the entities in the question
*are* the clock. `mamdani`, `hyperliquid` and `2024 election` are dates written in proper
nouns, and no regex reaches them.

The term weights say the same thing directly. Under both an i.i.d. row split and a
walk-forward one, the heaviest tf-idf features are calendar tokens and strike levels —
`august 22`, `december 15`, `november 16`, `between 96000`, `above 3900`, `105k and` —
with dated proper nouns (`epstein`, `biden win`, `genius fdv`) behind them. Nothing in the
list is the kind of feature FinBERT's financial-sentiment pretraining was built to supply.
A model given this string is not being told what the question means; it is being told when
and where it sits.

#### The answer: no encoder clears the price baseline

Eight monthly walk-forward folds, `nl` stratum, terminal target. IC(M1) is +0.1034
throughout — the same baseline in every row, so the column that matters is dIC.

| text block | dIC | t over folds | folds positive | folds where γ = 0 was chosen |
|---|---:|---:|---:|---:|
| tf-idf, 20,047 terms | −0.0129 | −0.97 | 3/8 | 3 |
| **FinBERT** | **−0.0073** | −0.39 | 3/8 | 4 |
| BERT-base | +0.0032 | +0.44 | 3/8 | 4 |
| MiniLM | −0.0215 | −1.56 | 2/8 | 4 |
| tf-idf, calendar masked | −0.0128 | −1.67 | 1/8 | — |
| FinBERT, calendar masked | +0.0027 | +0.60 | 2/8 | — |

Not one is distinguishable from zero. The last column is the sharpest way to say it:
**in half the folds the joint (α, γ) search sets the text weight to exactly zero** — offered
a free parameter that can only help on validation, the fair procedure declines to use the
question at all. FinBERT does not beat BERT-base, so the financial pretraining buys
nothing here; MiniLM, trained for semantic similarity, does worst. Masking the calendar
out of the string changes nothing either way.

#### The same text, under the split an off-the-shelf pipeline would use

Now hold the text block, the features and the target fixed and vary only how train and
test are separated. `kind = all`, terminal target, tf-idf and FinBERT:

| split | what it allows | tf-idf dIC | FinBERT dIC | tf-idf text book, net pp/share |
|---|---|---:|---:|---:|
| **random rows** | same market on both sides | **+0.5054** | +0.1043 | **+18.560** (t = +54.65) |
| random events | different markets, overlapping in time | +0.0378 | −0.0186 | +2.391 (t = +4.87) |
| walk-forward | strictly forward in time | −0.0067 | −0.0059 | +2.501 (t = +6.36) |

The price baseline alone runs +2.492 pp under the random split, +2.286 pp under the event
split and +2.785 pp under walk-forward, so under a careless split the text contributes
+16.07 pp of the +18.56 — and **grouping the split by event removes all of it**: the
walk-forward text book, at +2.501 pp, sits *below* the +2.785 pp its own price
baseline earns without any text at all. The leak is therefore not subtle and not
distributional: it is the same market appearing in training and test, where the cheapest
way to predict a label is to recognise the slug that names it. FinBERT leaks less than
tf-idf (+0.104 against +0.505) for the mechanical reason that 768 dense dimensions cannot
memorise 30,772 individual strings as sharply as 30,772 sparse ones can — being a worse
lookup table is the only advantage the transformer demonstrates in this study.

**+0.5054 of IC, and a book at t = +54.65, from a choice about how to draw the split.**
§4's label-timing convention manufactures +0.538 on data built to contain nothing. The two
numbers are not related, and their closeness is a coincidence — but the shape is the same,
and neither pipeline reports a diagnostic that would catch it.

#### Fine-tuning does not rescue it

The obvious objection to the above is that frozen mean-pooled features are a weak read of a
transformer. `pm/pmv1_text_finetune.py` removes it: FinBERT unfrozen and trained end-to-end
on the target, against an **architecture-matched** control — one head, one optimiser, one
schedule, the text tower simply absent — and a null arm identical to the text arm but with
the questions permuted first. Three test months, training rows capped at 24 per market so
the encoder cannot memorise a 470-day market.

| arm | mean valid IC | mean test IC | mean book, net pp/share |
|---|---:|---:|---:|
| price only | 0.1242 | **0.0930** | +2.45 |
| **text** (fine-tuned FinBERT) | **0.1792** | 0.0718 | +4.35 |
| null (questions shuffled) | 0.1259 | 0.0860 | +2.86 |

**The text arm has the highest validation IC in all three months and the lowest mean test
IC of the three.** It also loses to its own null: a model shown deliberately wrong
questions generalises better than one shown the right ones. Month to month it is not stable
in either direction — test IC +0.1175 / +0.0249 / +0.0730 against the price control's
+0.1137 / +0.1311 / +0.0343 — and the book means are carried by two months out of three.

The validation-versus-test column is the same failure §4 identifies in the HRT forecaster,
reproduced in a different model class on real data: **the statistic used to select the model
is one that the text arm can inflate without generalising.** A practitioner who checkpointed
on validation IC, as `forecast.py` does and as this script does, would ship the worst of the
three arms in every month.

#### The falsification arms

| arm | tf-idf | FinBERT |
|---|---:|---:|
| real questions, walk-forward | −0.0129 | −0.0073 |
| **questions shuffled within (month × category × price decile)** | **+0.0019** | **+0.0012** |

With the language destroyed and everything else held, dIC is +0.0019 and +0.0012 — which is
what the real text also delivers. The walk-forward pipeline is therefore *calibrated*: it
does not manufacture an edge from a corpus that has none, and the null and the signal are
indistinguishable because there is nothing to distinguish. This is the check §4 says should
precede any claim, and it is the reason the negative result above can be stated as a result
rather than as a failure to find something.

#### Two arms that point the other way, and why neither is reported as a finding

Honesty requires naming them. **On the 24-hour drift target** — `y = p(t+24h) − p(t)`
rather than hold-to-resolution — text is positive for both encoders: dIC +0.0115 (tf-idf,
6/8 folds) and +0.0148 (FinBERT, t = +2.02, 6/8). The null does **not** reproduce it
(−0.0017 and −0.0093), so it is not manufactured. And **the residual arm**, where the text
block is shown only what the price model could not explain and its prediction is traded
alone, earns net +1.251 pp/share at t = +2.20 over 2,849 events.

Both fail on multiplicity. This section ran **22 arms**, each summarised by a t over 8
folds. Simulating that matrix under a global null (t with 7 df, 200k draws):

| statistic | value |
|---|---:|
| E[max abs t] over 22 arms | 2.88 |
| median max abs t | 2.69 |
| **P(max abs t ≥ 2.02)** | **0.851** |
| P(max abs t ≥ 2.48) | 0.613 |

**Seeing a t of 2.02 somewhere in this matrix is what pure noise does 85 % of the time.**
Neither arm clears the bar its own section sets, and reporting either as a result would be
the specific error §2.2 exists to prevent. They are leads: the drift finding in particular
is cheap to test properly, and the way to do it is to pre-register that one specification
and run it on the months this study has not touched.

Two further arms are negative for a reason worth recording. On the **machine-templated
`date` stratum** — sports slugs such as `nba-bos-bkn-2025-11-18-1h-spread-away-5pt5` — text
is negative (tf-idf −0.0025, FinBERT −0.0086) and, more to the point, **the price baseline
itself is negative there** (−0.41 pp, t = −0.79). There is no edge on that stratum for text
to add to, which is consistent with those markets being the most mechanically priced on the
venue.

#### What this means, and what it does not

The negative result is narrow and should be stated narrowly. It says that on **this**
corpus — short, templated, English question stubs from one venue, 2022–2026 — a frozen or
fine-tuned FinBERT reading of the question adds nothing to a price baseline on the
hold-to-resolution target, and nothing that clears this study's own multiplicity bar on the
24-hour-drift target. It does **not** say that language models cannot price events. It says that the text available here is a market identifier, not a
description of the world, and that a model given it will learn the identifier.

The positive result is the methodological one, and it is the reason this belongs next to
§4 rather than as a standalone NLP chapter. §4 falsifies a *label-timing* convention: the
HRT pipeline reports +0.538 of test IC on data built to contain nothing. This section
falsifies a *split* convention on data that contains something, and the two failures have
the same shape — a defensible-looking choice, made once, worth more than everything the
model does afterwards. Neither is caught by any diagnostic the original papers report, and
both are caught immediately by running the pipeline against a reference class where the
answer is known.

It also answers, in the negative, the obvious way to fold text into the HRT framework. HRT
consumes a forecast and allocates; the natural extension is an extra feature channel
carrying a text embedding alongside Alpha158. On this evidence there is nothing to carry:
the channel would import the encoder's ability to date a string, which under HRT's own
walk-forward protocol is worth nothing and under a careless one is worth a great deal.
The honest version of the integration is the third item in §10 below.

### 3.7 Does a language model forecast the market, or remember it?

§3.6 tested *encoders*. The question a reader asks next is about generative LLMs, and
there the risk is sharper than a leaky split: a model trained on web text written after a
market resolved may simply know the answer. That is look-ahead through the pretraining
corpus. Nothing in a backtest reveals it, and it would make any LLM-on-prediction-markets
result look like skill.

**Design** (`pm/llm_lookahead.py`, `pm/llm_lookahead_eval.py`; run on the lum.id
sandbox, `SANDBOX.md`):

- **Markets.** 12,619 resolved natural-language markets (9,477 events). Each is observed
  on its first market-day with at least one day to resolution and 0.03 < p < 0.97.
- **Forecast.** Each model is asked once, from the question alone. Its answer is read off
  the next-token distribution as P(Yes) / (P(Yes) + P(No)), with no sampling and nothing to
  tune.
- **Prompts.** Two, both fixed in advance and both reported:
  - `forecast` asks the question as a market would;
  - `recall` says the market has resolved and asks what happened, so it bounds the leak
    from above.
- **Edge.** The LLM's *incremental* information over the price: the LLM coefficient β in
  a logistic regression of the outcome on the market's own logit plus the LLM's log-odds,
  with standard errors clustered by event.
- **Models.** Six open-weight models, four with documented training cutoffs spread across
  the markets' resolution window.

The two readings make opposite predictions. If the model is forecasting, its edge should
not care where a market resolved relative to the model's cutoff. If it is remembering, the
edge should be there before the cutoff and gone after. And the *same* market should look
predictable to a later-cutoff model and not to an earlier one.

**Before the cutoff and after**, `forecast` prompt (`recall` in brackets):

| model | documented cutoff | β before cutoff (t) | n | β after (t) |
|---|---|---:|---:|---:|
| Llama-3.1-8B | Dec 2023 | — (28 markets) | 28 | +0.007 (+0.58) [−0.003] |
| Gemma-3-12B | Aug 2024 | +0.050 (+2.00) [+0.069, t +3.79] | 237 | +0.009 (+2.21) |
| OLMo-3-7B | Dec 2024 | +0.011 (+0.41) [+0.023, t +0.62] | 1,012 | +0.030 (+3.03) |
| **Gemma-4-12B** | **Jan 2025** | **+0.115 (+6.29)** [+0.107, t +7.45] | 1,399 | −0.003 (−0.46) |

**The cutoff, estimated blind.** For every candidate month c, the regression is refit with
separate LLM slopes before and after c; the break statistic is the sup-LR over c.
Andrews' asymptotic 5 % critical value (8.85) turns out to be too lenient here: 12 % of null
draws exceed it, because markets cluster within events. So each model is judged against its
**own permutation null** instead. That null shuffles the model's log-odds across markets
*within the same resolution month*, 400 draws (`pm/llm_lookahead_perm.py`).

| model | documented | estimated break (`forecast`) | sup-LR, perm. p | estimated break (`recall`) | sup-LR, perm. p |
|---|---|---|---|---|---|
| Gemma-4-12B | Jan 2025 | **Feb 2025** (95 % set Feb 2025 only) | 51.6, **p < 0.003** | **Jan 2025** (Jan–Feb 2025) | 45.7, **p < 0.003** |
| Gemma-3-12B | Aug 2024 | — | 7.2 | **Jun 2024** (Jun 2024 only) | 13.7, **p = 0.007** |
| Qwen3.5-9B | not documented | **Nov 2025** | 11.5, **p = 0.007** | Nov 2025 | 12.3, **p = 0.020** |
| Ministral-3-8B | not documented | Jun 2024 (edge of search) | 8.9, p = 0.055 | — | 7.0 |
| OLMo-3-7B | Dec 2024 | — | 6.9 | — | 6.7 |
| Llama-3.1-8B | Dec 2023 | — | 2.1 | — | 3.9 |

**The same markets, two models.** This comparison needs no cutoff estimate and holds every
market-level confound fixed:

| markets resolving… | Gemma-3 (Aug 2024) | Gemma-4 (Jan 2025) |
|---|---:|---:|
| between the two cutoffs (1,162) | β +0.012 (t +1.09) | **β +0.092 (t +4.61)** |
| after both (11,220) | +0.008 (+1.86) | −0.003 (−0.46) |

Against Llama-3.1 on the 1,371 markets between Dec 2023 and Jan 2025, the numbers are
−0.009 (t −0.25) against **+0.111 (t +6.04)**. `recall` gives the same picture more
strongly: +0.107 (t +7.27) for Gemma-4.

What this says:

- **The leak is real, specific and dated.** Two of the four models with a documented cutoff
  show an edge that switches off at their cutoff. The blind search lands within 0–2 months
  of the documented date without being told it. For Gemma-4 the edge before the cutoff is
  of the same order as §3.3's whole order-flow effect.
- **It is a property of the training data, not the architecture.** Gemma-3 and Gemma-4 are
  one family. On the same 1,162 markets, the one that was trained after they resolved has
  the edge, and the one trained before does not.
- **Undocumented cutoffs can be measured.** Qwen3.5-9B shows a significant break at
  **November 2025**, on both prompts. For a model released in February 2026 that is a
  plausible cutoff, and it is the only estimate available.
- **Not every model memorises.** OLMo-3 and Llama-3.1 show no break on either prompt.
  OLMo-3's small edge *after* its cutoff (β +0.030, t +3.03) vanishes under `recall`
  (−0.002), so it is not memory.
- **No model is a forecaster.** Every LLM's Brier score (0.25–0.36) is far worse than the
  market's own (0.189). Instruct models put P(Yes) below 10⁻³ on ~71 % of these questions.
  The edge measured here is ranking information in the log-odds, not calibration.

**Why it matters for this dissertation.** It is §4's failure shape for a third time, now
in the model class people actually reach for. A backtest of Gemma-4 over 2024 would report
a large, highly significant edge over a liquid market. That edge would be worth nothing
after January 2025, and nothing in the backtest would say so. **Any LLM result on event
markets has to be restricted to events that resolve after the model's cutoff.** Where the
cutoff is undocumented, this section's break test is a cheap way to estimate it.

**Caveats.**
- The question text is the market slug (§3.6), which is less informative than the full
  question. That makes these edges, if anything, lower bounds.
- One prompt per arm, a first-token answer and no chain of thought: a stronger elicitation
  could leak more.
- The before-cutoff samples are small for the earliest models (28 and 237 markets), which
  is why the paired design carries the argument there.
- An 8-bit load was used for the three models too large for bf16 beside the concurrent RL
  sweep (Gemma-3, Gemma-4, Ministral-3).

### 3.8 The pre-registered drift arm, on untouched months — not supported

§10 item 7 asked for the one §3.6 arm that survived its own null (FinBERT text on 24-hour
drift, dIC +0.0148, t +2.02) to be tested once, on months the study never touched.
`pm/PREREG_drift.md` fixed the specification, the test statistic and the decision rule on
2026-09-27 at 16:42 UTC, **before any post-archive data was pulled** (sha256 `d35a86df…`).

**Data.** The archive ends on 2026-04-28, so the test months were rebuilt from Polymarket's
public API (`pm/pmlive_drift_panel.py`). It reproduces `pmv1_spread.py`'s hourly bars and
`pmv1_text_panel.py`'s market-day rows from taker fills:

- **Markets.** 9,005 resolved markets, none of them (and none of their events) in the
  archive. Each one's full fill history was fetched, paging backwards past the API's
  10,500-row offset cap; 0 failures.
- **Panel.** 21,995 test rows, 2,005 events, May–September 2026.
- **Sign check.** Signed flow correlates +0.20 with the same-hour price change (+0.09 in
  the archive), and buy VWAP exceeds sell VWAP in 89 % of two-sided bars (93 %). So the
  taker direction is signed correctly.
- **Different conditions.** The median effective spread is wider than in the archive
  (0.0058 against 0.0020).

**Result.** On the archive's own validation window (the last 61 days), the (α, γ) search
chose **γ = 0**: validation IC was 0.0785 with or without the FinBERT block. So the model
with text is the model without it, and **dIC = 0 exactly. H1 is not supported**, under the
rule written before the data existed. The shuffle-null arm also selects γ = 0.

**What survives is the price baseline.** It was not the pre-registered hypothesis, so it is
reported as description only. Price and category alone hold their IC on the untouched months
(test **0.0776** against validation 0.0785). Their book, at next-day VWAPs, nets **+3.82
pp/share (t = +4.24)** over 1,270 events, against §3.6's +1.69 pp in-sample.

§3.6 is therefore complete as written. The +0.0148 is what 22-arm selection produces, which
is what its P = 0.851 predicted.

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

### 4.1 The agents, on the same panels — the experiment that was missing

Everything above audits the **forecaster**. The previous version of this document listed
the obvious completion as the one experiment it could not run: the RL **agents** had never
been trained where there is nothing to learn. `hrt/slurm/05_sweep_null.sbatch` runs it, and
it finished on the cluster on 2026-09-15 — 25 runs, ~111 GPU-hours, at the **identical**
501,760-step budget as the real sweep (the summary asserts this rather than assuming it).
Seed *k* trains on null panel *k* mod 4, so the four independent panels of §4 carry the
dispersion rather than the seed alone, and `hrt/passive.py` measures a do-nothing floor
*inside the same environment* on each panel.

**The floor is the whole design.** `report.py` is deliberately not reused: it scores
against `baselines.json`, which is the real panel's benchmarks. A null panel has no drift
by construction — `synth_panel.py` draws the market factor zero-mean — so its *realised*
drift is pure noise and varies wildly: the four panels' passive floors are −1.1 %, −1.5 %,
+9.2 % and +16.2 % in 2021, and +1.9 %, +55.4 %, +21.0 % and −15.8 % in 2022. An agent that
deploys capital inherits whichever draw it was given. **Every number below is therefore an
excess over the passive floor of the panel that run was trained on**, which differences the
draw out and is in any case the only benchmark an RL agent can honestly be held to.

**2021 test year**

| arm | *n* | **null** excess | t | beats floor | *n* | **real** excess | t | beats floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hrt_paper_ep *(leaky)* | 4 | **+0.203** ± 0.141 | +2.89 | 4/4 | 10 | +0.736 ± 0.117 | +19.91 | 10/10 |
| hrt_paper *(leaky)* | 4 | **+0.102** ± 0.154 | +1.33 | 3/4 | 10 | +0.518 ± 0.114 | +14.35 | 10/10 |
| hrt_causal | 4 | +0.086 ± 0.186 | +0.92 | 3/4 | 10 | −0.046 ± 0.048 | −3.02 | 2/10 |
| ddpg_causal | 5 | +0.035 ± 0.059 | +1.31 | 3/5 | 10 | +0.008 ± 0.020 | +1.32 | 5/10 |
| ppo_causal | 4 | +0.002 ± 0.126 | +0.03 | 2/4 | 10 | +0.026 ± 0.023 | +3.48 | 10/10 |
| hrt_causal_ep | 4 | −0.111 ± 0.077 | −2.86 | 0/4 | 10 | −0.054 ± 0.041 | −4.14 | 0/10 |

**2022 test year**

| arm | *n* | **null** excess | t | beats floor | *n* | **real** excess | t | beats floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hrt_paper *(leaky)* | 4 | **+0.286** ± 0.178 | +3.22 | 4/4 | 10 | +0.599 ± 0.177 | +10.72 | 10/10 |
| ppo_causal | 4 | **+0.248** ± 0.178 | +2.79 | 3/4 | 10 | −0.057 ± 0.026 | −7.07 | 0/10 |
| hrt_paper_ep *(leaky)* | 4 | **+0.227** ± 0.225 | +2.02 | 3/4 | 10 | +0.735 ± 0.077 | +30.00 | 10/10 |
| ddpg_causal | 5 | +0.057 ± 0.109 | +1.16 | 4/5 | 10 | −0.036 ± 0.045 | −2.54 | 1/10 |
| hrt_causal | 4 | +0.008 ± 0.214 | +0.07 | 2/4 | 10 | −0.057 ± 0.061 | −2.92 | 3/10 |
| hrt_causal_ep | 4 | −0.138 ± 0.273 | −1.01 | 1/4 | 10 | −0.013 ± 0.069 | −0.62 | 4/10 |

Pooled by label timing, which is where the *n* is:

| | null excess | t | beats floor | real excess | t | beats floor | null / real |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2021, leaky | **+0.153** | +2.94 | 7/8 | +0.627 | +17.70 | 20/20 | **0.24** |
| 2021, leak-free | +0.005 | +0.15 | 8/17 | **−0.017** | −2.16 | 17/40 | — |
| 2022, leaky | **+0.257** | +3.81 | 7/8 | +0.667 | +19.89 | 20/20 | **0.38** |
| 2022, leak-free | +0.045 | +0.82 | 10/17 | **−0.041** | −4.79 | 8/40 | — |

Five things fall out, and three of them were not predictable from §4.

**1. The audit passes as a control, both ways.** This is the part that makes the rest
readable. On panels where nothing is learnable, the **leak-free** agents beat their floor
by +0.005 (t = +0.15) and +0.045 (t = +0.82) — statistically nothing, 8/17 and 10/17, which
is what a correct workflow must report on a null. The **leaky** agents beat it by +0.153
(t = +2.94) and +0.257 (t = +3.81), 7 of 8 both years. So the timing defect is not confined
to the forecaster's IC: it propagates through the alpha signal into the sizing policy and
out into a traded book, and it manufactures **15 to 26 points of excess return on data
whose returns are a martingale difference**. That is 24 % (2021) and 38 % (2022) of the
excess the same arms show on the real market — a direct, if crude, decomposition of how
much of HRT's flagship arm is the convention rather than the market.

**2. The leak-free arms do not beat the floor on the real market either.** Ten seeds make
this sayable: −0.017 (t = −2.16) in 2021 and −0.041 (t = −4.79) in 2022, 17/40 and 8/40 runs
above the floor. The bear-year finding in `hrt/README.md` was not a bear-year finding. The
shortfall sits in the arms the paper is *about*: both hierarchy arms are below the floor in
both years, at both alpha units. One leak-free arm does clear it — the flat `ppo_causal`
baseline, in 2021 only, by +0.026 (t = +3.48, 10/10), which is a real margin across seeds
but one the null sweep puts at p = 0.18, and which reverses to −0.057 (t = −7.07, 0/10) in
2022. Put beside row 1: **the hierarchy never beats doing nothing unless it can see the
future, and when it can, it beats doing nothing on noise as well.**

**3. In 2022 the pipeline does strictly better on noise than on the market.** The best
leak-free run on a null panel reaches annualised Sharpe **+2.54**; the best of the forty
leak-free runs on the real panel reaches **+0.20**, and **10 of 17** null runs beat it. In
2021 the real panel wins narrowly (+2.48 against +2.14, 0/17) — and §2.2's threshold of
3.829 kills it anyway. There is no reading of this in which the 2022 sweep learned
something about the market.

**4. Model selection is mining noise, and now that is measured rather than argued.** §4
noticed that the *forecaster*'s validation IC on the real panel sat inside the null runs'
range. At agent level it is worse than inside:

| arms | null validation Sharpe | real validation Sharpe | real vs null |
|---|---|---|---|
| leak-free | **+0.933** [−0.403, +2.349], *n* = 17 | +0.431 [+0.116, +0.885], *n* = 40 | t = **−2.38** |
| leaky | +2.384 [+1.280, +4.309], *n* = 8 | +2.840 [+2.160, +3.594], *n* = 20 | t = +1.05 |

`train.py` checkpoints on best validation Sharpe. For the leak-free arms that statistic is
**significantly larger on panels with nothing to learn**: **35 of 40** real runs fall below
the null median of +0.607, and the average real run sits at the **36th percentile** of the
null distribution. The selection criterion is not merely noisy — on this environment it is
*anti*-informative, because a null panel's realised drift is unconstrained and a lucky
panel hands the agent a validation Sharpe the real market never offers. Checkpointing on it
selects for panel luck.

**5. The ranking of arms partly survives the removal of all signal.** Ordering the six arms
by excess over floor, null against real: Spearman **ρ = +0.771** (p = 0.072) in 2021 and
**+0.143** (p = 0.787) in 2022. In 2021 the null reproduces the top two and the bottom one
exactly. So §2's ordering is, at least in the bull year, substantially a property of the
workflow — turnover, action scale, how much of the book each architecture deploys — rather
than of what any of them learned. The 2022 disagreement is the other side of the same coin:
without signal there is nothing to hold the ordering in place, and `ppo_causal` lands second
on the null and last on the real panel.

**The null sweep also calibrates §2.2, which is a free result.** SR\*₀ is an analytic model
of "the best Sharpe K trials of this family produce when there is no skill". These 25 runs
*are* that experiment. On the leak-free null runs (K = 17):

| year | σ_SR (daily) | observed max | analytic E[max] | ratio |
|---|---:|---:|---:|---:|
| 2021 | 0.0569 | **+2.139** ann | +1.650 ann | 1.30× |
| 2022 | 0.0880 | **+2.539** ann | +2.552 ann | 0.99× |

The closed form is roughly right for this strategy family — too lenient by 30 % in one year
(the sweep actually produced a higher maximum than the formula expects), exact in the other
— which is worth stating because [8] establishes it for return series in general, not for a
25-run RL sweep on a 370-name panel. It also supplies the thing the
formula cannot: a **non-parametric p-value with no Gaussian assumption and no zero centre**,
taking the 17 leak-free null runs' excess Sharpe over their own floor as the reference
distribution.

| arm | 2021 best-seed excess SR | p | 2022 best-seed excess SR | p |
|---|---:|---:|---:|---:|
| ppo_causal | +0.767 | 0.059 | −0.026 | 0.471 |
| ddpg_causal | +0.430 | 0.118 | +0.373 | 0.176 |
| hrt_causal | +0.068 | 0.353 | +0.150 | 0.353 |
| hrt_causal_ep | +0.000 | 0.353 | +0.324 | 0.353 |
| hrt_paper *(leaky)* | +6.749 | 0.000 | +6.273 | 0.000 |
| hrt_paper_ep *(leaky)* | +7.305 | 0.000 | +4.490 | 0.000 |

**Not one leak-free arm reaches p < 0.05 on its best seed**, let alone on its seed mean
(p ≥ 0.18 throughout). That agrees with §2.2's verdict by a completely different route, and
it is the version to report, because a reader can object to the DSR's distributional
assumptions and cannot object to this.

**What this does not establish.** Four honest limits. (i) The null side has *n* = 4–5 per
arm and the panel is the unit of dispersion, so the per-arm null t-statistics are weak; the
pooled rows and the 17-run reference distribution are what the argument rests on. (ii) The
real side is a **single panel** — one draw of the market — so its t-statistics describe seed
dispersion only and say nothing about how the excess would vary across market histories.
(iii) `ppo_causal` beating its floor by +0.248 (t = +2.79) on null panels in 2022 is a false
positive in a leak-free arm, and with six arms × two years the sweep will produce one;
that is the multiplicity point §2.2 makes, appearing inside the audit itself. (iv) The null
panels match the real one on volatility, factor loading and the overnight/intraday split but
not on cross-sectional dependence beyond a single market factor, so a defect that only bites
under richer correlation structure would not show here.

The sweep cost ~111 GPU-hours against the real sweep's ~189, and it is the cheapest single
thing in this repo per conclusion changed. **A synthetic-null re-run of the whole sweep
should be a standard artefact of an RL trading paper, not an optional check** — it tests the
agent, the environment, the cost model and the checkpointing rule at once, which no
statistic computed on the real run can do.

### 4.2 Replacing the checkpoint rule — neither rule selects anything

§4.1 found validation Sharpe, the statistic `train.py` keeps the best checkpoint on,
*higher* on null panels than on the real one. §10 item 3 proposed two fixes:
- checkpoint on validation Sharpe *in excess of the passive floor*;
- or keep the final policy.

Both are now tested on the same runs.

**Design** (`hrt/run_probe.sh`, `train.py --probe`, `analysis/ckpt_rules.py`; run on the
lum.id sandbox, `SANDBOX.md`):

- **Runs.** The four leak-free arms × 5 seeds, on the real panel and on null panels s0
  and s1 at the identical 500k-step budget. 59 of 60 runs completed; the sandbox expired
  during the last one.
- **Probe.** Every one of each run's 20 eval points is scored on the validation window
  *and* on both test years, and its weights are kept. So every rule is applied afterwards
  to the same checkpoints.
- **The `excess` rule** takes the Sharpe of the agent's daily return *minus the floor's
  daily return* on validation. The floor's own Sharpe is constant within a run, so
  subtracting it could not change which checkpoint wins.
- **Consistency check.** The `raw` rule reproduces the checkpoint `train.py` itself kept,
  in every run (asserted).

**Gain over simply keeping the final policy**, in test excess return over the same panel's
passive floor, paired by run:

| rule | real panel (40 run-years) | null panels (78 run-years) |
|---|---:|---:|
| raw — best validation Sharpe (what `train.py` does) | −0.002 (t −0.34) | −0.018 (t −1.30) |
| excess — best validation Sharpe over the floor | −0.003 (t −0.43) | −0.019 (t −1.40) |
| *oracle — best test checkpoint (not a rule)* | *+0.051 (t +9.83)* | *+0.178 (t +7.64)* |

**Does validation predict test within a run?** Spearman correlation across a run's 20
checkpoints, between the validation statistic and test excess:

| | raw | excess |
|---|---:|---:|
| real panel | −0.053 (t −0.86) | −0.048 (t −0.79) |
| null panels | +0.015 (t +0.35) | +0.025 (t +0.60) |

- **The selection step is inert.** On the real panel, keeping the best validation
  checkpoint is worth nothing over keeping the last one (−0.002), exactly as on the null
  panels, where there is nothing to select.
- **It is inert because validation says nothing about test.** Checkpoints *do* differ on
  test: the oracle gap is +0.051 on the real panel. But the within-run rank correlation
  between validation and test is zero, or slightly negative, on real data.
- **The floor-relative fix does not help.** It removes the panel-drift term §4.1 blamed,
  and the answer does not change. So drift is not why selection fails. At this step budget
  the policy's validation-to-test generalisation is simply nil.
- **Nothing rescues the leak-free arms.** Every arm on the real panel finishes below its
  passive floor in 2022 under every rule. Even the oracle, which is not implementable,
  reaches only +0.026. This confirms §4.1 from a different direction.

**Recommendation for the dissertation:** drop checkpoint selection, report the final policy
with its seed distribution, and say why. The standard rule costs a validation window and
buys nothing. On null panels it is also exactly as busy as on real ones, which is §4.1's
finding restated as a procedure.

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
   R² = 0.65 and shows the pipeline reports IC +0.54 on pure noise under it; §4.1 shows
   the agents built on that forecast beat a do-nothing floor by 15–26 points of book on
   the same noise, while the leak-free hierarchy beats it in neither year.
2. **[1], [6] and most of the cluster report raw or risk-adjusted return without a
   passive floor over the same window.** HARLF's 26 % annualised over 2018–2024 needs to
   be read against buy-and-hold on the same universe. Here, over ten seeds, the leak-free
   arms pooled finish below a do-nothing agent inside the same environment in **both** test
   years — mean excess −0.017 (t = −2.16) in 2021 and −0.041 (t = −4.79) in 2022, 17/40 and
   8/40 runs above the floor — and the two *hierarchy* arms are below it in both years
   individually (§4.1). A passive floor measured *inside the environment* rather than in
   weight space is the cheapest referee available and almost nobody reports one.
3. **Nobody in the cluster reports a deflated Sharpe, or a trial count.** §2.2 shows why
   that matters on both asset classes: the expected maximum Sharpe from 70 equity trials
   is 3.829, above every leak-free arm produced, and the best gross crypto Sharpe has
   DSR 0.35. The threshold is not a constant — it grew from 2.905 to 3.829 when this sweep
   went from 35 trials to 70 — so a paper that reports a Sharpe without a K has withheld
   the denominator.
4. **MACE [2] establishes that cost *level* and *size-dependence* reorder algorithms; it
   does not establish that *regime-dependence* does.** §2.3 finds it does not, here. It
   also reports its re-ranking counts without a standard error: on this sweep the equity
   count fell from 2–6 of 6 arms to 0–4 of 6 when the seed count doubled.
5. **The sign-randomisation test in [13] is mis-specified in two ways** — trade-level
   independence (rms(z) = 1.65 against a required 1.00 over 1.8 M accounts) and a
   zero-centred null for spread-crossing takers (median z = −0.83). §3.1. Its *substantive*
   claim nevertheless survives an out-of-sample test (§3.2), which is a different thing
   from the estimator being right. §3.4 bounds how much of the remaining over-rejection is
   still the estimator's fault: not much.
6. **[14] replicates and is not tradeable by a taker.** The effect is confined to the top
   liquidity quintile and is economically small (R² 0.0017, §3.3); priced at executable
   prices it is 4.2× under the round-trip cost, and its value accrues to the maker (§3.5).
   The paper does not charge a cost, and on this venue the cost is the whole answer.

### Of this work

* **Ten seeds, two test years, one universe** for the equity RL results — and the third of
  those is now the binding one. [7] uses sixteen walk-forward folds across three
  continents; that is the standard to match. The seed dimension is settled (§2.1, §4.1);
  the *panel* dimension is not, and every real-side t-statistic in §4.1 is seed dispersion
  around a single draw of market history. The DSR and falsification results should still
  be read as diagnostics on this sweep rather than as a new sweep.
* **The null side of §4.1 is four panels and 25 runs**, so per-arm null t-statistics are
  weak and the argument rests on the pooled rows and on the 17-run leak-free reference
  distribution. The null panels also match the real one on volatility, factor loading and
  the overnight/intraday split but carry only a single market factor, so a defect that
  bites only under richer cross-sectional dependence would not appear.
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
  the published trade-level test, not a correct test. §3.4 shows no coarser contract-free
  unit fixes it either, and attributes most of the residual to real heterogeneity — but
  that attribution rests on δ being stable across a split, so it is a lower bound on skill
  and an upper bound on what is left to blame on the estimator.
* **§3.5's spread is observable only where both sides trade inside one bar** (51.4 % of
  bars). Bars with one-way flow are exactly the bars where a taker would fare worst, so the
  net figures are, if anything, generous. The round trip is also priced at VWAP, i.e. at
  the average fill of everyone trading that side that hour, which assumes the strategy is
  small enough not to move it — the same assumption `analysis/costs.py` makes and does not
  net out.
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
liquid tape, an out-of-sample persistence test — are done, and each took under a day. So
are the two questions *this* version's §10 left open: the pooling ladder (§3.4) is a
33-second query over 727 M fills, and pricing the imbalance round trip at executable prices
(§3.5) costs one 90-second pass over the archive. Neither needed a new idea, only the
data.

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
python hrt/report.py                 # §2.1  aggregate runs/ -> summary.json (now 60 runs)
python analysis/dsr.py               # §2.2  deflated Sharpe; K and SR*_0 -> dsr_test20*.json
python analysis/cost_rerank.py       # §2.3  MACE ranking test  -> cost_rerank.json

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

# --- the same audit at agent level (§4.1) -- CLUSTER ONLY, ~111 GPU-hours ----
# Runs on the H100/H200 queue, not here. Prep is skip-if-present and the sweep is
# resumable at run granularity, so a job killed at the wall clock just needs
# resubmitting. Chain it behind 01_prepare if panel.npz is not built yet.
sbatch hrt/slurm/05_sweep_null.sbatch              # 4 null panels, 8 forecasters,
                                                   # 4 passive floors, 25 RL runs
python analysis/null_sweep.py        # §4.1  -> hrt/artifacts/null_sweep.json (CPU, ~2 s)

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
python pm/pmv1_pooling.py                         # §3.4  ~35 s over 727M fills
python pm/pmv1_spread.py --freq 1h                # §3.5  ~100 s incl. the bar build

# --- prediction markets: the text layer (§3.6) -------------------------------
python pm/pmv1_text_panel.py --horizon 24         # slugs + resolution + panel, ~9 s
for m in finbert bert minilm; do                  # frozen embeddings, ~40 s each on one GPU
  for f in text text_scrub; do python pm/pmv1_text_embed.py --model $m --field $f; done
done
./pm/run_text_matrix.sh                           # 20 arms, ~80 min; resumable
./pm/run_finetune.sh                              # 9 fine-tuning arms, ~45 GPU-min
python pm/pmv1_text_probe.py --probe clock        # the R2 = 0.92 date probe
python pm/pmv1_text_probe.py --probe terms        # what the heavy n-grams are
python pm/pmv1_text_book.py --folds 10 --binary-only          # baseline decomposition
python pm/pmv1_text_book.py --folds 10 --binary-only --same-bar   # the look-ahead, quoted

# --- prediction markets: the superseded vendor tape (§1.3) -------------------
python pm/pull.py discover ; python pm/pull.py trades 900 ; python pm/outcomes.py
python pm/skill.py --sims 10000 ; python pm/imbalance.py --freq 1h
```

New this round: `analysis/null_sweep.py` (§4.1) and `hrt/slurm/05_sweep_null.sbatch`, which
runs the sweep behind it. `null_sweep.py` reads `runs_null/`, `runs/`, `runs_invalid/` and
`passive_synth_s*.json` and writes `hrt/artifacts/null_sweep.json`; it recomputes nothing
and takes about two seconds, so the cluster job is the only expensive part. `analysis/dsr.py`
and `analysis/cost_rerank.py` now also **write** their thresholds and counts
(`dsr_test20**.json` gains `K` and `sr_star_ann`; `cost_rerank.json` is new), because §2.2's
2.905 and §2.3's "4 of 6" were hard-coded into `analysis/figures.py` and went stale the
moment the sweep grew a seed.

Previously new: `pm/pmv1_text_{panel,embed,signal,finetune,probe,book}.py` and the two
drivers `pm/run_{text_matrix,finetune}.sh` (§3.6). Both drivers are **resumable** — an arm
whose output JSON exists is skipped — so an interrupted run costs only the arm in flight;
delete the JSON to force a rerun. `pmv1_text_panel` builds `data/nlp/{slugs,resolution,
panel}.parquet` from the cleaned layers in one pass and everything downstream reads those.
Previously new: `pm/pmv1_pooling.py` (§3.4) and `pm/pmv1_spread.py` (§3.5). `pmv1_pooling`
builds `data/pmv1_markets.parquet` (838,118 resolved markets → close date and category) on
first run and reuses it; `pmv1_spread` builds its own `data/pmv1_spreadbars_1h.parquet`,
which is `pmv1_prep`'s bar table plus the two taker-side VWAPs. Both need
`pmv1_persistence` to have run at the three cut dates, since §3.4 reads its per-account
CSVs. Previously new: `analysis/censoring.py`, `crypto/dsr.py`,
`pm/{pmv1_pull,pmv1_prep,pmv1_skill,pmv1_persistence,pmv1_imbalance}.py`.
Existing: `analysis/{dsr,synth_panel,falsify,costs,cost_rerank,coverage,figures}.py`,
`crypto/{pull,panel,forecast,synth_panel,strategy}.py`, `pm/{pull,outcomes,skill,imbalance}.py`.
`data/` is git-ignored and fully re-pullable from the commands above. The Polymarket-v1
material is **25.7 GB** of it: 16.8 GB archive plus 8.9 GB of compacted tables.

---

## 9. What needs a human

One thing, now that the cluster item is done.

* **A findata token** (`LUMID_TOKEN` in `.env`) for anything that re-pulls the equity or
  crypto tape. The existing token still works; `CLUSTER.md` notes it should be rotated,
  since a live PAT was committed to the pushed history before this work started. Nothing
  in §3 needs it — Polymarket-v1 is public and ungated, and no Hugging Face account or
  `HF_TOKEN` is required.

**The cluster item is closed.** The null-panel RL sweep (§4.1) and the extension of the
real sweep to ten seeds both ran on the NUS cluster on 2026-09-15 — 60 real runs (~189
GPU-hours) and 25 null runs (~111 GPU-hours), all at 501,760 steps. `CLUSTER.md` records
the three job-script bugs that had to be fixed first, the site limits that bind
(`gpu` caps wall time at 3 h, `gpu-long` will not allocate more than 48 CPUs and has no
H200), and the silent CPU-fallback trap in `train.py --device` that would have run the
whole sweep on CPU without saying so. Nothing further needs the cluster unless the panel
dimension is opened up (§10.1).

Everything else in §8 runs here: the Polymarket-v1 work is CPU-only and finished in about
two hours end to end, `analysis/censoring.py` takes ten minutes, and `analysis/null_sweep.py`
takes two seconds once the runs exist.

---

## 10. What to do next

Ordered by evidence produced per unit of work. **Item 1 of the previous version is done**
(§4.1) and what it found reshuffles the rest: the RL workstream is now finished as a
*negative* result, and the remaining work is to write it correctly and to spend the
cluster on the one dimension that is still a single draw.

1. ~~**Finish the falsification audit on the RL sweep.**~~ **Done — §4.1**, and it came out
   stronger than the design anticipated: the leaky arms clear their floor on martingale
   panels, the leak-free arms clear nothing anywhere, and the checkpointing statistic is
   *higher* on noise. Treat §4 + §4.1 as one result and make it the methodological spine.
   The follow-on that is worth cluster time is **not more seeds — it is more panels.**
   Every real-side number in §4.1 is one draw of market history, so the honest next
   experiment is a block-bootstrap or CPCV [9] resampling of the *real* panel, run through
   the same sweep, giving a real-side dispersion to set against the null-side dispersion
   that now exists. That is the one comparison the audit cannot currently make, and it is
   the same shape of job that just ran.
2. **Report DSR and K next to every Sharpe** — now available for both asset classes
   (`analysis/dsr.py`, `crypto/dsr.py`) — **and move the equity screen to CPCV [9].**
   Forty lines, and it immunises the results chapter. §2.2 is also now the cleanest
   *demonstration* of why K matters that this repo can offer: doubling the seed count
   raised the best observed Sharpe and raised the selection threshold faster, cutting every
   leak-free DSR by two-thirds. Write it that way — as a before/after on the same sweep —
   rather than as a table of final numbers. And when reporting it, quote the empirical
   p-values from §4.1 alongside, since they make the same point without assuming normality.
3. ~~**Replace the checkpointing rule, or drop it.**~~ **Done — §4.2: drop it; neither rule selects anything.** This is the highest-value new item and
   it is one sweep. §4.1 measures validation Sharpe — the statistic `train.py` selects on —
   to be *anti*-informative for the leak-free arms: 35 of 40 real runs fall below the null
   median. The diagnosis names the culprit, which is that a null panel's realised drift is
   unconstrained, so the rule selects for panel luck. Two fixes are cheap and testable
   against the null sweep that now exists: checkpoint on validation Sharpe **in excess of
   the passive floor computed on the validation window**, which removes exactly that term;
   or keep the final policy and report the seed distribution honestly. Either is a
   publishable finding on its own, because the rule is standard and is not specific to HRT.
4. ~~**Checkpoint the policies on the next sweep.**~~ **Done — §2.3 addendum: exact replay confirms the ranks, corrects the levels.** §2.3's equity re-ranking is a first-order
   recharge because the trained policies were thrown away, and its weakening at ten seeds
   (2–6 of 6 arms moving rank, down to 0–4 of 6) is exactly the case where an exact
   re-simulation would settle the question rather than invite the objection. One line in
   `train.py`. It also makes any arm re-evaluable without retraining, which at ~3 GPU-hours
   per run is the difference between a cheap follow-up and another cluster allocation.
5. **Make §3.2 the prediction-market contribution, and §3.1+§3.4 its companion.** The
   out-of-sample persistence design needs no null, cannot be broken by mis-specifying one,
   and gives a clean positive result at three cut dates. The methodological companion is
   now a sharper story than it was: the published estimator over-rejects, the randomisation
   unit and the zero centre are both wrong, and — §3.4 — once those are fixed most of what
   is left is the effect itself rather than further mis-specification. The remaining open
   item is small and specific: the three cuts imply V between 0.05 and 0.39, so the skill
   variance is a property of the period, and it is worth knowing whether that tracks
   election calendar, market count, or the share of volume in sports.
6. **§3.3 is settled: report it as a maker result.** §3.5 prices the round trip at
   executable prices and the taker version is 4.2× under water at every hold and every
   signal threshold tested, while the passive counterparty earns +0.278 pp/share. Do not
   spend more time trying to make the taker version work; the two things worth writing are
   (i) that the effect is real, robust across price bases, and confined to thick bars, and
   (ii) that its entire economic value is a liquidity-provision return. That framing also
   connects §3 to §5.3 and idea 6 — three independent workstreams now end at the execution
   layer rather than at the forecast.
7. ~~**Pre-register the one text arm that survived its own null, then stop.**~~ **Done — §3.8, not supported.**
   **Pre-register the one text arm that survived its own null, then stop.** §3.6 is a
   negative result with two loose ends: text on the 24-hour drift target (dIC +0.0148,
   t = +2.02, and the shuffle null does *not* reproduce it) and the residual-only text book
   (+1.251 pp, t = +2.20). Both fail the multiplicity bar that section computes for itself —
   P(max |t| ≥ 2.02) = 0.851 across its 22 arms — so the only honest way to resolve them is
   to fix that single specification in advance and run it on months the study has not
   touched. It is an afternoon. If it fails, §3.6 is complete as written; if it holds, it is
   the only text result on the venue and worth its own section.
8. **Reconsider idea 6's framing before committing.** §2.3 says the state-dependent cost
   model does not reorder anything a flat 30 bp does not already reorder — and it now says
   so from a weaker equity base than before, which strengthens rather than weakens the
   conclusion that state dependence is not where the contribution is. Either demonstrate
   that volatility-surface geometry carries information beyond σ_t and V_t, or move the
   contribution to the frequency separation [3] identifies — HRT's LLC is a same-bar sizing
   policy and cannot express intra-day execution at all, which is a sharper and unclaimed
   gap.
9. **Fold §5.4 into the data chapter as a standing requirement.** A coverage audit and a
   naive-versus-block comparison should precede any result computed on a vendor tape. It
   costs ten minutes and it is worth +0.63 of Sharpe.
10. **Port the null sweep to crypto.** §5.1–§5.3 audit the crypto *forecaster* against three
    null panels but have never run the crypto strategy layer against them, and §5.3's
    +2.07 gross Sharpe is precisely the kind of number §4.1 just showed a null panel can
    manufacture. `crypto/synth_panel.py` and `crypto/strategy.py` both exist; this is a
    driver script, not an experiment, and it is CPU-only.
11. **Re-baseline against HRT v2, not v1**, and report Sharpe and turnover alongside
    cumulative return so the comparison is like-for-like.
