# Reading list — AI in quantitative finance

Curated for FYP **H398280** (supervisor: LU Yao), *AI in Financial Data Analytics and
Trading*. Compiled 2026-09-07.

Every entry was checked against the source page (arXiv abstract, SSRN listing, or
publisher record) rather than recalled — titles, authors, dates and headline numbers
below are as published. Where a paper has been revised since it was first cited in
`fyp_ideas.md`, the revision is flagged, because in two cases the revision changes what
the paper claims.

## Why this list looks the way it does

This is not a generic "AI in finance" bibliography. It is shaped by what this repo has
already established:

| Established here | Consequence for the reading |
|---|---|
| **HRT does not replicate leak-free** (`hrt/README.md`): 2021 +23.1% / 2022 −7.6% against a published +39.8% / +2.3%, and every RL agent finishes below a buy-and-hold floor in the bear year | Papers on *evaluation, leakage and overfitting* (§4) matter more than another architecture paper |
| **The hierarchy earns ~4.4 pts of gross edge in 2022, then pays 6.33 pts in commission at ~70x annual turnover** | The *cost model* is the load-bearing component of idea 6, not a refinement — §2 is the core of the contribution |
| **The news/sentiment channel is unreachable from findata** (200 items, trailing ~2 weeks, date params ignored) | §6 exists to name the datasets that actually cover 2013–2023 |
| **The prediction-market non-replication** (pooled ROI +1.33% top-3% vs +2.51% bottom-90% across 32 markets; trader ROI is the wrong estimator) | §5 is chosen for *price-discovery* methodology and for a tape that predates the 2026-05-21 vendor capture boundary |

Read order if time is short: **1 → 3 → 12 → 8 → 20**, then §5 if the project turns
towards prediction markets rather than RL.

---

## 1. The baseline you already reproduced

### [1] Hierarchical Reinforced Trader (HRT)
Zijie Zhao, Roy E. Welsch. *A Bi-Level Approach for Optimizing Stock Selection and
Execution.* [arXiv:2410.14927](https://arxiv.org/abs/2410.14927) — **v1 19 Oct 2024,
v2 11 May 2026**.

**What it does.** Splits trading into a High-Level Controller (HLC) that picks a
per-asset direction (increase / reduce / hold) and a Low-Level Controller (LLC) that
turns those directions into share-level orders. The decomposition exists to avoid
enumerating a joint action space of size 3^N over N assets. HRT-Base uses price
features only; the full model adds a text-derived signal.

**The revision matters.** v1 tested 2021/2022 on an S&P 500 universe and reported
+39.8% / +2.3%. v2 (May 2026) is a substantially different paper: a **fixed 89-stock
Nasdaq universe on an open stock-news benchmark**, train 2013–2018, validate 2019, test
**2020–2023**, and the reported result is now risk-adjusted rather than raw return —
Sharpe **1.06 → 1.24** and daily turnover **0.112 → 0.090**, with an explicit claim of
robustness "under transaction-cost stress". The LLC objective in v2 carries **turnover,
drawdown and text-risk penalties** that v1 did not have.

**Why it matters here.** The authors independently arrived at turnover control between
v1 and v2 — which is exactly the failure mode the reproduction measured (70x annual
turnover eating a real gross edge). That is corroboration, not scoop: v2 adds a
*penalty*, whereas idea 6 proposes a *cost model that varies with market state*. The v2
universe switch to an 89-stock Nasdaq news benchmark is also the FNSPID/FinRL-DeepSeek
setup (see [17]), which resolves the news gap in `hrt/README.md` at the price of
changing the experiment.

**Take.** Re-baseline against v2, not v1. Report Sharpe and turnover alongside cumulative
return so the comparison is like-for-like. State plainly that v1's numbers were not
reproducible leak-free and that v2 is the live target.

---

## 2. Transaction costs and execution realism — the core of idea 6

### [2] Realistic Market Impact Modeling for RL Trading Environments (MACE)
Lucas Riera Abbade, Anna Helena Reali Costa.
[arXiv:2603.29086](https://arxiv.org/abs/2603.29086) — v1 30 Mar 2026, v2 4 Apr 2026.

**What it does.** Three Gymnasium-compatible environments (stock trading, margin
trading, portfolio optimization) that replace the flat-bps assumption with nonlinear
impact grounded in **Almgren–Chriss** and the **square-root impact law**, plus pluggable
cost models, permanent-impact tracking with exponential decay, and trade-level logging.
Five DRL algorithms (A2C, PPO, DDPG, SAC, TD3) on the NASDAQ-100, fixed 10 bps baseline
vs. the AC model, hyperparameters tuned with Optuna.

**Headline results.** (i) The cost model changes both absolute performance *and the
relative ranking of algorithms* in all three environments. (ii) Under AC, daily costs
fall from **$200k to $8k** and turnover from **19% to 1%**. (iii) Hyperparameter tuning
cuts costs by up to **82%**. (iv) Interactions are environment-specific: DDPG's
out-of-sample Sharpe goes **−2.1 → 0.3** in margin trading under AC while SAC's goes
**−0.5 → −1.2**.

**Why it matters here.** This is the closest published statement of idea 6's premise, and
it is *the* paper to position against. It demonstrates the claim ("cost model decides the
ranking") but its cost model is a **static functional form** — AC with a square-root law,
parameterised once. Idea 6's geometry-based, volatility-surface-derived cost is
*state-dependent*: the same trade costs differently in different regimes. That gap is the
contribution, and MACE gives you the experimental protocol (same agents, swap the cost
model, report the rank change) to demonstrate it in.

**Take.** Adopt their reporting convention — always publish the ranking under two cost
models, never one. Consider building on their environments rather than `hrt/env.py`,
or at minimum replicating their fixed-10bps arm so results are commensurable.
Their finding (iii) is a warning for the reproduction: some of what looks like a
cost problem is an untuned-hyperparameter problem.

### [3] Deep Stock Trading (HRPM)
Rundong Wang, Hongxin Wei, Bo An, Zhouyan Feng, Jun Yao. *A Hierarchical Reinforcement
Learning Framework for Portfolio Optimization and Order Execution.*
[arXiv:2012.12620](https://arxiv.org/abs/2012.12620) — Dec 2020, rev. Feb 2021.

**What it does.** The intellectual predecessor to HRT, and a cleaner statement of the
hierarchy. A **low-frequency** high-level policy sets portfolio weights for long-term
profit; it invokes a **high-frequency** low-level policy that buys or sells the implied
shares inside a short window to minimise trading cost. Trained with a pre-training scheme
followed by iterative joint training, for data efficiency. Evaluated on U.S. and China
equities.

**Why it matters here.** HRPM's motivation is stated in one line — existing portfolio-RL
"assume each reallocation can be finished immediately and thus ignore price slippage as
part of the trading cost." That is the same critique the reproduction arrived at
empirically. Crucially, HRPM's two levels run at **different frequencies**, which HRT's
do not: HRT's LLC is a same-bar sizing policy. If the low level is where cost is
modelled, a same-bar LLC cannot express intra-day execution at all.

**Take.** The frequency separation is the design decision to argue about in the
literature review. It also supplies the pre-train-then-iterate curriculum, which is a
plausible fix for the HLC entropy pathology documented in `hrt/README.md`.

### [4] Multi-Level Market Making with Reinforcement Learning
Patrick Cheridito, Moritz Weiss.
[arXiv:2608.18195](https://arxiv.org/abs/2608.18195) — 18 Aug 2026.

**What it does.** Maximises trading revenue by dynamically submitting market *and* limit
orders of varying sizes **across multiple price levels** while controlling inventory.
Two techniques worth stealing: **multivariate logistic-normal distributions** to model
allocation across levels, and a **deep-set encoder** to compress variable-length order
sets into a fixed-dimensional latent. Adds potential-based reward shaping (which
provably leaves the optimal policy unchanged) to speed learning. Tested against three
simulated counterparty populations — noise traders, tactical traders reacting to
instantaneous volume imbalance, and strategic traders following an EW volume-imbalance
signal.

**Why it matters here.** Two ways in. If the project stays on idea 6, the deep-set
encoder solves a real structural problem — HRT's 370-dimensional factorised categorical
is a brittle way to handle a variable-size universe, and the paper's per-level allocation
is a better-posed low-level action space than a scalar share count. If the project goes
to idea 4 (RL market-making on prediction-market books), this is the current
state of the art to port.

### [5] RL-Based Market Making as Stochastic Control on Non-Stationary LOB Dynamics
Rafael Zimmer, Oswaldo Luiz do Valle Costa.
[arXiv:2509.12456](https://arxiv.org/abs/2509.12456) — Sep 2025, rev. 14 Feb 2026.

**What it does.** A PPO market maker on an explicitly modelled LOB that reproduces
stylized facts — clustered order arrivals, non-stationary spreads and return drifts,
stochastic order quantities and volatility. Benchmarked **against a closed-form optimal
solution**, which is the discipline most RL-trading papers lack.

**Why it matters here.** The closed-form comparison is the methodological point: it lets
you say how much of the agent's performance is the RL and how much is the environment.
For idea 4 the paper is also a caution — it needs a *modelled* LOB, and the memory notes
that Polymarket order-book capture stopped 2026-07-29 with one-sided/empty depth, so any
prediction-market market-making project would have to build this simulator first, from a
Kalshi book that only starts ~2026-07-30.

---

## 3. Hierarchical RL for portfolios — the surrounding cluster

### [6] HARLF
Benjamin Coriat, Eric Benhamou. *Hierarchical Reinforcement Learning and Lightweight
LLM-Driven Sentiment Integration for Financial Portfolio Optimization.*
[arXiv:2507.18560](https://arxiv.org/abs/2507.18560) — 24 Jul 2025.

**What it does.** A **three-tier** hierarchy: base RL agents consume hybrid
price+sentiment data, meta-agents aggregate their decisions, and a super-agent merges on
market data and sentiment. Trained 2000–2017, evaluated 2018–2024. Reports **26%
annualised return, Sharpe 1.2**, beating equal-weight and the S&P 500. Open-sourced.

**Why it matters here.** It is the "more levels, lightweight LLM" branch of the same
family as HRT, and it is honest about being an ensemble rather than a control hierarchy.
Read it sceptically against the reproduction's own result: a 2018–2024 test window
starting after a 2000–2017 train window contains one of the strongest equity bull runs on
record, and no transaction-cost stress test is reported. Compare its 26% to the
buy-and-hold floor over the same window before treating it as a target.

**Take.** Useful as a citation for "hierarchy + sentiment has been tried"; useful as an
example of the evaluation weakness §4 is about.

### [7] DRL for Diversified Portfolio Management Across Global Equity Markets
Kamil Kashif, Robert Ślepaczuk.
[arXiv:2605.17307](https://arxiv.org/abs/2605.17307) — 17 May 2026.

**What it does.** Soft Actor-Critic learning continuous portfolio weights with
**transaction costs, turnover penalties and diversification constraints in the reward**.
Five configurations varying reward formulation, **flat vs. hierarchical Dirichlet policy
structure**, constraints, and temporal encoder (LSTM vs. Transformer). Evaluated by
walk-forward optimisation over **sixteen out-of-sample folds, 2003–2026**, on the
Nasdaq-100, Nikkei 225 and Euro Stoxx 50.

**Headline result — and the reason it is on this list.** Competitive risk-adjusted
performance appears **only in the Euro Stoxx 50**, and *no* configuration achieves
statistically significant excess returns over buy-and-hold across markets. The central
hypothesis is only partially confirmed. The authors say so.

**Why it matters here.** This is the most direct external corroboration of the
reproduction's finding that RL agents finish below a passive floor. It also does the
flat-vs-hierarchical ablation that HRT never runs, over sixteen folds and three
continents — so it is the strongest available evidence on whether the hierarchy is worth
anything at all once costs are charged. Sixteen folds is also the evaluation standard to
match; four seeds on two years (the current reproduction) is not enough to make a claim.

---

## 4. Evaluation, leakage and overfitting — where this FYP's differentiator lives

The reproduction's central finding is negative and methodological. That is a
contribution only if it is framed in this literature.

### [8] The Deflated Sharpe Ratio
David H. Bailey, Marcos López de Prado. *Correcting for Selection Bias, Backtest
Overfitting, and Non-Normality.* *Journal of Portfolio Management* 40(5), 2014.
[SSRN:2460551](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) ·
[PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)

**What it does.** Gives a closed-form correction to an observed Sharpe ratio for two
inflation sources: **selection bias under multiple testing** (you tried K strategies and
reported the best) and **non-normal returns** (skew and kurtosis, which the standard
Sharpe estimator ignores). Yields a probability that the true Sharpe exceeds a threshold.

**Why it matters here.** Non-negotiable for the write-up. The sweep in `hrt/artifacts/runs/`
is 4 arms × 4 seeds plus baselines, and `runs_invalid/` holds 10 more — the reported
Sharpe of the best arm is a maximum over a search, and quoting it undeflated is precisely
the error this paper names. It costs about twenty lines of code to apply and it
immunises the results section.

**Take.** Report DSR next to Sharpe for every arm, and state K honestly (count every run
ever aggregated, not just the surviving ones).

### [9] Backtest Overfitting in the Machine Learning Era
Hamid R. Arian, Daniel Norouzi Mobarekeh, Luis A. Seco. *A Comparison of Out-of-Sample
Testing Methods in a Synthetic Controlled Environment.* *Knowledge-Based Systems* 305
(2024). [DOI:10.1016/j.knosys.2024.112477](https://dl.acm.org/doi/10.1016/j.knosys.2024.112477)

**What it does.** Compares out-of-sample testing protocols in a **synthetic environment
where the ground truth is known**, so false discoveries are countable. Judges each
protocol by Probability of Backtest Overfitting (PBO) and Deflated Sharpe. Finds
**Combinatorial Purged Cross-Validation (CPCV) markedly superior**, and — the finding
that stings — **walk-forward is notably weak at false-discovery prevention**, with high
temporal variability and weaker stationarity. Introduces Bagged and Adaptive CPCV
variants; Python implementation released.

**Why it matters here.** `scripts/walkforward.py` and `scripts/walkforward2.py` exist, and
`fyp-portfolio-horizon-effect` records 15 heavily-overlapping annual vintages whose
t-stats the memory itself calls "descriptive, not inferential." This paper is why. CPCV
with purging and embargo is the fix, and it is directly applicable to both the equity
screen and the RL train/validate/test split.

**Take.** Move the equity-screen evaluation to CPCV. For the RL work, at minimum apply
purging and an embargo around the 2019 validation boundary.

### [10] Spurious Predictability in Financial Machine Learning
Sotirios D. Nikolopoulos.
[arXiv:2604.15531](https://arxiv.org/abs/2604.15531) — 16 Apr 2026.

**What it does.** Shows that **adaptive specification search produces statistically
significant backtests even under a martingale-difference null** — i.e. even when there is
provably nothing to find. Proposes a **falsification audit**: run the *complete* workflow
against synthetic reference classes including zero-predictability environments and
microstructure placebos; a workflow that produces significant walk-forward evidence there
is falsified. For workflows that survive, quantifies selection-induced inflation via a
magnitude gap between optimised in-sample evidence and disjoint walk-forward
realisations, adjusted for effective multiplicity.

**Why it matters here.** This is the single most useful methodological upgrade available
to this project. It tests the **pipeline**, not the strategy — which is exactly what a
reproduction study needs. Running `hrt/forecast.py` → `signal_strategy.py` → the RL sweep
on a synthetic zero-predictability panel, and showing whether the pipeline still reports
an edge, would convert the HRT non-replication from an anecdote into a demonstrated
property of that class of pipeline.

**Take.** Strongly consider making this the project's methodological spine. It is
cheap — the panel generator is a few dozen lines — and it is a genuinely novel thing to
do to HRT.

### [11] Summoning the Oracle to Slay It (FinCAD)
Weixian Waylon Li, Mengyu Wang, Tiejun Ma. *Mitigating Look-Ahead Bias in Financial
Backtesting with Large Language Models.*
[arXiv:2605.24564](https://arxiv.org/abs/2605.24564) — 23 May 2026, rev. 29 Aug 2026.

**What it does.** Names and attacks **parametric look-ahead bias**: an LLM pre-trained
through 2024 already encodes how stocks moved in 2018–2020, so backtesting it on that
period is meaningless. FinCAD is an inference-time adaptation of Context-Aware Decoding
that attenuates memorised outcomes without retraining, paired with an adversarial
bias-discovery pipeline that learns a model-specific "memory-activating" prompt and an
entity/date-adaptive rule scaling CAD strength by a per-(entity, date) confidence signal.
Across five 7–14B models and five mega-caps, the largest model-level mean in-sample
return correction is **−67.1%**; on an eleven-model leaderboard it raises in-sample /
out-of-sample Spearman correlation from **+0.779 to +0.846**.

**Why it matters here.** Two reasons. First, if idea 5 or any LLM channel enters the
project, this defines the correct experimental hygiene and gives the correction method.
Second, and more immediately: the reproduction already found that HRT's published numbers
require look-ahead "in some partial degree" via the `KMID`-style same-bar features.
FinCAD is the closest published analogue of that argument — that a headline backtest
number can be *decomposed* into a leakage component and a real component — and gives
language and a magnitude (−67.1%) to cite for it.

### [12] Agentic Trading: When LLM Agents Meet Financial Markets
Yihan Xia, Panpan You, Taotao Wang, Fang Liu, Han Qi, Xiaoxiao Wu, Shengli Zhang.
[arXiv:2605.19337](https://arxiv.org/abs/2605.19337) — 19 May 2026.

**What it does.** A protocol-coded systematic review of **77 studies** (screened through
2026-03-09), of which **19** meet core empirical criteria, scored on a reproducibility
ladder.

**The numbers are the paper.** Of the 19 primary studies: only **2** report an extractable
time-consistent split protocol; **1** documents an explicit transaction-cost model; **1**
addresses survivorship; **11** report execution timing. **No study reaches R3
reproducibility, and 15 of 19 sit at the lowest level (R0).** Conclusion: "architectural
experimentation is expanding rapidly, while comparable evaluation protocols, execution
semantics, and reproducible artifacts remain the field's immediate bottlenecks."

**Why it matters here.** This is the citation that makes the FYP's negative result a
contribution rather than an embarrassment. It establishes, with counts, that failure to
replicate is the field's *modal* outcome and that cost modelling and survivorship — the
two things this repo has explicitly wrestled with (370 of 500 names, `deadjust.py`, the
delisted-price gap) — are the two least-reported items in the literature. Put it in the
introduction.

---

## 5. Prediction-market price signals

### [13] Prediction Market Accuracy: Crowd Wisdom or Informed Minority?
Roberto Gómez-Cram, Yunhan Guo, Theis Ingerslev Jensen, Howard Kung. Posted 20 Apr 2026;
R&R at *Review of Financial Studies*.
[SSRN:6617059](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6617059) ·
[Yale Insights summary](https://insights.som.yale.edu/insights/wisdom-of-the-few-prediction-markets-are-driven-by-small-number-of-skilled-traders)

**Note the retitle.** `fyp_ideas.md` cites this as "…Crowd Wisdom or Informed Traders?" —
the current title is "**Informed Minority?**". Update the citation.

**What it does.** The universe of Polymarket transactions — 1.72M accounts, $13.76B
volume. Argues accuracy comes from neither crowd wisdom nor insider trading, but from a
persistent skilled minority of ~**3%** of accounts. These traders show "depth and
breadth" unlike localised insiders: they react to public news on arrival, eliminate
law-of-one-price violations, and trade against the crowd's behavioural mistakes. The
crowd supplies most volume and little information; its losses fund the minority.
Skill is separated from luck by **simulating each trader's bets 10,000 times with
coin-flip directions**.

**Why it matters here.** This is the premise of idea 1, and `scripts/pm_informed_vs_crowd.py`
already partly reproduced it (top 3% of takers held 93.2% of notional, Gini 0.972 on the
Fed June 2026 market) — and then failed to replicate the ROI ordering across 32 markets.
Reading the paper closely resolves that: **their claim is about price discovery and
sign-randomised skill, not about pooled ROI.** The sign-randomisation test is the
estimator to adopt, and it is directly implementable on the captured tape.

**Tested 2026-09-07 on [16], at the paper's own scale** (1,819,876 accounts with >=10 fills
against its 1.72M; `RESULTS.md` §3.1–3.2). Split verdict. The **estimator over-rejects**:
flipping each trade independently treats one position as many bets, and sd(z) is 1.628
where the null requires 1.00. Widening the unit to the trader-market and then to the
trader-event helps monotonically (1.456, 1.304) but does **not** reach calibration, so the
correction is necessary and not sufficient; the residual is probably distinct contracts
driven by one underlying (dozens of same-day Bitcoin-threshold markets are separate
`condition_id`s and separate negative-risk events). The null is also zero-centred where a
spread-crossing taker's expected payoff is negative — the median account's event-level z is
**−0.83** and the active population loses 0.22% of notional.

**But the substantive claim survives without any null.** Ranking accounts before a cut date
and measuring them after — dropping markets that straddle the cut, which is essential —
the top in-sample decile earns **+1.3% to +3.5%** of notional out of sample against a
population +0.4% to +1.2%, and accounts flagged before the cut are **1.9–4.2× more likely**
to be flagged after it, at three cut dates. That out-of-sample design is the estimator to
adopt, not the randomisation test.

### [14] Price Discovery and Trading in Modern Prediction Markets
Hunter Ng, Lin Peng, Yubo Tao, Dexin Zhou. Posted 27 Apr 2026.
[SSRN:5331995](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5331995)

**What it does.** First evidence on **cross-venue** price discovery, on common contracts
across **Polymarket, Kalshi, PredictIt and Robinhood** around the 2024 U.S. presidential
election. Liquid prediction markets substantially outperform polls; large price
disparities persist across platforms. **Polymarket leads Kalshi in price discovery,
especially when liquidity and activity are high**, implying economically meaningful
arbitrage. **Net order imbalance from large trades strongly predicts subsequent
returns**, and the venue with greater directional large-trade flow tends to lead.

**Why it matters here.** This is the paper idea 3 must be positioned against — it has
already done cross-venue lead-lag on four platforms, so a student project cannot claim
that framing. But it hands over the two estimators the FYP was missing: an
**information-share / lead-lag** measure (the memory's own methodological conclusion:
"trader ROI is the wrong estimator") and **large-trade order imbalance** as a return
predictor. The second is testable on the existing 32-market sample without any new data.

**Replicated 2026-09-07 on [16]** (5,244,127 hourly market-bars over 25,317 markets,
2022-11 → 2026-04; `RESULTS.md` §3.3). Large-trade imbalance predicts the next bar,
β = +0.0033 (t +13.3) at 1h and +0.0043 (t +17.3) with an own-return control, and the sign
survives a bid-ask-bounce bias that runs against it. An earlier null on fifteen weeks of
vendor tape is **explained rather than contradicted**: the effect is monotone in bar
liquidity and confined to the top notional quintile (median $3,569, β +0.0056, t +8.9),
while the vendor tape's median bar carried $69 — inside the quintile where the coefficient
is *negative*. By category it is strongest in Finance, Culture and Politics and absent in
pure sports, consistent with the paper's election setting. It is real, robust and small:
R² 0.0017, and about 7% of a typical bar move. Whether it survives the spread is the open
question — and it has to be estimated, because `fee_usdc` is identically zero across the
archive (Polymarket v1 charged no explicit taker fee).

### [15] Per-Market Information Leakage and Order-Flow Skill
Maksym Nechepurenko. *Two Methodological Lenses on Informed Trading in Decentralized
Prediction Markets.* [arXiv:2605.02287](https://arxiv.org/abs/2605.02287) — 4 May 2026,
rev. 14 May 2026.

**What it does.** Reconciles the three concurrent 2026 literatures — arguing they are
"three distinct layers of detection, not competing methods on a single layer."
Layer 1: **sign-randomisation testing** for account-level persistent directional skill
(Gómez-Cram et al., who classify 3.14% of accounts as skilled winners and flag 1,950 as
insiders). Layer 2: legal/regulatory insider detection (Mitts & Ofir, 210,000+
wallet–market pairs). Layer 3: an **Information Leakage Score (ILS)** measuring per-market
information front-loading relative to an article-derived public-event timestamp. Worked
through the January 2026 U.S.–Venezuela operation.

**Why it matters here.** The most efficient way into this literature — one short paper
that maps all of it and, more usefully, explains *why* an account-level and a
market-level measure can disagree. That is a direct explanation for the repo's own
non-replication: an account-level ROI test and a market-level discovery test are not
measuring the same thing. The ILS construction (front-loading against a public-event
timestamp) is a well-specified, implementable target for idea 2.

### [16] Polymarket-v1 Database
Boka Qin, Rui Yang. [arXiv:2606.04217](https://arxiv.org/abs/2606.04217) — 2 Jun 2026,
rev. 8 Jun 2026. Publicly released on Hugging Face.

**What it does.** An on-chain archive of Polymarket's first-generation exchange on
Polygon: **21 Nov 2022 – 28 Apr 2026, 1.20 billion trade records, 1.30 million markets,
$61B nominal volume**, with **ground-truth aggressor direction taken from blockchain
settlement** rather than inferred. Shows that standard microstructure classifiers (Lee-Ready
and kin) are **near-random** here because prediction-market structure violates their
assumptions, and traces how that transaction-level error propagates into market-quality
and forecasting conclusions.

**Why it matters here — this is the most actionable item on the list.** The
`findata-pm-constraints` memory records that the vendor tape **starts 2026-05-21**, that
Polymarket book capture **stopped 2026-07-29**, and that the leaderboard is stale and
outcome-conditioned. This dataset covers **3.5 years ending April 2026** — it precedes
and dwarfs the captured tape, it is free, and it carries the one field trade-tape
analysis normally has to guess. It converts idea 1 from a 32-market underpowered study
into a properly powered one. The classifier warning also retro-justifies not having
inferred aggressor side by heuristic.

**Take.** ~~Evaluate this dataset before doing any further prediction-market work on the
vendor tape.~~ **Done 2026-09-07** — pulled, compacted and analysed; see `DATA.md` §9 for
provenance and traps, `RESULTS.md` §1.4 and §3 for what it changed. It is free and ungated
(no Hugging Face account needed), the two cleaned layers are 16.8 GB and download in
thirteen minutes, and the whole of §3 then runs on CPU in about two hours. It reversed both
of the FYP's prediction-market non-replications, which is the single largest change this
dataset made to the project. All further prediction-market work should use it; the vendor
tape is superseded.

### [17] Political Shocks and Price Discovery in Prediction Markets
Kwok Ping Tsang, Zichao Yang. *Evidence from the 2024 U.S. Presidential Election.*
[arXiv:2603.03152](https://arxiv.org/abs/2603.03152) — 3 Mar 2026, rev. 18 Jul 2026 (v4).

**What it does.** An event study on Polymarket's on-chain ledger around three shocks: the
Biden–Trump debate, the assassination attempt, and Biden's withdrawal. Trading rises after
every shock, concentrated among incumbents with greater prior activity. Repricing behaviour
differs by event: the debate's largely **reversed**, the assassination attempt's
**persisted**, and Biden's withdrawal produced heavy trading with minimal Trump price
movement. Key conclusion: **price response tracks what the news reveals about linked
candidates and how much was already anticipated — not the amount of trading.**

**Why it matters here.** The template for idea 2. It gives the event-study design, and its
conclusion is a direct warning against the naive version of that idea: convergence
*speed* confounds with how much the event was anticipated, so a speed-based signal needs
a surprise control or it measures predictability of the news, not of the market.

---

## 6. LLM signal channels — closing the HRT news gap

### [18] FinRL-DeepSeek
Mostapha Benhenda. *LLM-Infused Risk-Sensitive Reinforcement Learning for Trading Agents.*
[arXiv:2502.07393](https://arxiv.org/abs/2502.07393) — 11 Feb 2025.
[Code](https://github.com/benstaf/FinRL_DeepSeek)

**What it does.** Extends **CPPO** (Conditional Value-at-Risk PPO) with two LLM-derived
channels from financial news: a trading **recommendation** signal at the action level and
a **risk-assessment score** at the risk level. Backtested on the Nasdaq-100 using the
**FNSPID** news dataset (Nasdaq news, 1999–2023) with DeepSeek V3, Qwen 2.5 and Llama 3.3.
Code, data and trained agents released; basis of FinRL Contest 2025 Task 1.

**Why it matters here.** This is the concrete resolution of the blocking data gap in
`hrt/README.md`. FNSPID plus its precomputed sentiment and risk scores reaches the
2013–2019 training window that findata cannot, and it is the same benchmark HRT's own v2
moved to — so adopting it aligns the reproduction with the target paper's current
experimental setup rather than diverging from it. The risk-level channel (feeding
sentiment into a CVaR constraint rather than into the action) is also a cleaner idea
than HRT's, and pairs naturally with a state-dependent cost model.

**Take.** If the news channel is wanted, this is the path. Budget for the universe change
(S&P 500 → 89 Nasdaq names) as a deliberate change of experiment, and re-run the passive
and equal-weight floors on the new universe.

### [19] Trading-R1
Yijia Xiao, Edward Sun, Tong Chen, Fang Wu, Di Luo, Wei Wang. *Financial Trading with LLM
Reasoning via Reinforcement Learning.*
[arXiv:2509.11420](https://arxiv.org/abs/2509.11420) — 14 Sep 2025.

**What it does.** Supervised fine-tuning plus RL under a **three-stage easy-to-hard
curriculum**, trained on Tauric-TR1-DB (100k samples, 18 months, 14 equities, five data
sources). Emits structured evidence-based investment theses rather than bare actions, and
reports improved risk-adjusted returns and lower drawdowns than both instruction-following
and reasoning baselines.

**Why it matters here.** The reference point for idea 5. Note what it does *not* do: the
reward shapes reasoning quality and trading outcome, but nothing penalises a trade that
contradicts its own stated thesis. That is precisely the gap idea 5 identifies, and it is
still open. Read alongside [11] — an 18-month window on 14 equities is exactly the regime
where parametric look-ahead is most dangerous, and the paper does not address it.

---

## 7. Infrastructure and surveys

### [20] The Evolution of Reinforcement Learning in Quantitative Finance: A Survey
Nikolaos Pippas, Elliot A. Ludvig, Cagatay Turkay.
[arXiv:2408.10932](https://arxiv.org/abs/2408.10932) — Aug 2024, rev. 6 May 2025 (v3).

**What it does.** Critically evaluates **167 publications**, dissecting RL components
through a quantitative-finance lens — state and action design, reward specification,
transfer/meta-learning, multi-agent settings — and explicitly critiques strengths and
weaknesses rather than merely cataloguing.

**Why it matters here.** The literature-review backbone. Its framing of markets as
complex, multi-agent, information-asymmetric and partly random is the standard way to
justify why a reproduction fails, and the critique sections give per-component
vocabulary for the write-up.

### [21] Qlib: An AI-oriented Quantitative Investment Platform
Xiao Yang, Weiqing Liu, Dong Zhou, Jiang Bian, Tie-Yan Liu.
[arXiv:2009.11189](https://arxiv.org/abs/2009.11189) — 22 Sep 2020.

**What it does.** Microsoft's end-to-end quant research platform: point-in-time data
handling, an expression engine, model zoo, and the **Alpha158 / Alpha360** feature
handlers.

**Why it matters here.** Already load-bearing in this repo — `hrt/data.py` builds the 158
Alpha158 features and `hrt/test_alpha158.py` validates them against an independent
implementation. Cite it for the feature set, and cite the specific detail the
reproduction turned on: **Qlib's own Alpha158 handler labels with
`Ref($close,-2)/Ref($close,-1)-1`**, which deliberately skips the same-bar window that
HRT's stated forward-return definition includes. That single line of Qlib source is the
strongest available third-party evidence for the leakage argument.

---

## What this list does not cover, and why

- **Empirical asset pricing / cross-sectional ML** (Gu, Kelly & Xiu and successors) —
  relevant to the equity screen in `scripts/`, not to the RL thread. Add if the portfolio
  work becomes the primary FYP rather than a benchmark.
- **Generative LOB simulators** — TRADES ([arXiv:2502.07071](https://arxiv.org/abs/2502.07071)),
  MarS ([arXiv:2409.07486](https://arxiv.org/abs/2409.07486)), JAX-LOB
  ([arXiv:2308.13289](https://arxiv.org/abs/2308.13289)). Only becomes essential if idea 4
  is chosen, since prediction-market book history is too short to train on directly.
- **Optimal-execution RL as its own field** (Almgren–Chriss extensions, hybrid
  action-space execution). One level down from where idea 6 sits; [2] and [3] carry the
  parts that matter.

## Three things the reading changed

1. **HRT v2 (May 2026) is a different paper from the one reproduced.** Different
   universe, different test window, Sharpe-and-turnover reporting, and turnover/drawdown
   penalties in the LLC. The reproduction targets v1. Say so explicitly, and re-baseline.
2. **The gap idea 6 fills is narrower than assumed but still real.** MACE [2] already
   published "the cost model changes the algorithm ranking" in March 2026 — with static
   Almgren–Chriss. The remaining contribution is the *state-dependent* cost model, and it
   now has a protocol to be demonstrated in.
3. **Polymarket-v1 [16] removes the binding constraint on idea 1.** 1.2B trades from Nov
   2022, with ground-truth aggressor side, against a vendor tape starting 2026-05-21.
   Worth checking before any further work on the 32-market sample.
