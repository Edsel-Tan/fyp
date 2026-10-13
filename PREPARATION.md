# PREPARATION — what to read, in what order, and what to take from each

FYP **H398280** (supervisor: LU Yao), *AI in Financial Data Analytics and Trading*.
Working file, not a deliverable — git-ignored.

This is not `PAPERS.md`. `PAPERS.md` is the annotated bibliography: what each paper says
and why it is on the list. **This file is the reading plan**: the order to read in, the
specific thing to extract from each paper, and which section of the dissertation that
extraction serves. It is ordered by what `RESULTS.md` now needs, which is not the order
the bibliography is in.

Paper numbers `[n]` are `PAPERS.md`'s numbering throughout.

---

## The state of play, in four lines

The repo's results are mostly **negative and methodological**: HRT does not replicate
leak-free, nothing survives deflation, the pipeline reports IC +0.54 on noise, and the two
prediction-market effects that do replicate are either not tradeable or not what the
published estimator says they are. A negative result is a contribution **only if it is
framed in the evaluation literature**. That single fact sets the reading priority: §4 of
`PAPERS.md` ([8], [9], [10], [11], [12]) outranks every architecture paper.

---

## Tier 0 — read before writing a single word of the dissertation

Four papers. Everything in the results chapter depends on getting these right, and three
of them are already load-bearing in code that has been run.

### [10] Nikolopoulos, *Spurious Predictability in Financial Machine Learning* — arXiv:2604.15531
**Read first, read closely, read all of it.** `RESULTS.md` §4 calls the falsification
audit "the methodological spine of the dissertation" and the audit is already implemented
(`analysis/synth_panel.py`, `analysis/falsify.py`) and run on four equity and three crypto
null panels.

Extract, specifically:
1. **The exact definition of a "reference class"** and how many the paper requires. This
   repo built one (martingale-difference returns, conditional variance left predictable).
   The paper also names **microstructure placebos** — check whether the audit as run is
   complete by the paper's own standard, or whether it is one of several required classes.
2. **The "magnitude gap" machinery** — the paper quantifies selection-induced inflation as
   a gap between optimised in-sample evidence and disjoint walk-forward realisations,
   *adjusted for effective multiplicity*. §4 currently reports the gap informally
   (test IC +0.8155 real vs +0.5383 null). Get the formal statistic and report it.
3. **What counts as falsified.** §4 asserts the paper's timing is falsified because the
   pipeline finds IC +0.538 on noise. Make sure that is the paper's criterion and not a
   paraphrase.
4. The wording for the causal-timing case, which is the awkward one: the pipeline
   *survives* at z = +2.20, but model selection happens on a validation IC that the null
   reproduces. Does the paper have a name for that failure mode? If so use it.

> Watch for: this is the one paper whose argument the project is *extending* rather than
> applying. Know it well enough to defend the extension to crypto and to a second label
> timing.

### [8] Bailey & López de Prado, *The Deflated Sharpe Ratio* — SSRN 2460551
Already applied twice (`analysis/dsr.py` on 35 equity runs, `crypto/dsr.py` on 11 crypto
strategies × 4 cost models). Read to check the application, not to learn the idea.

Extract:
1. **The variance estimator for σ_SR across trials.** The crypto application found that
   under a cost model two strategies lose almost the whole book (Sharpe −57), σ_SR
   explodes and SR\*₀ becomes uninformative; a robust IQR-based scale is reported
   alongside. **Is that legitimate under the paper's derivation, or an invention?** If it
   is an invention, say so explicitly in the text.
2. **What K should count.** The repo counts all 35 runs including `runs_invalid/`,
   reasoning that the surviving configuration was chosen after seeing both. Confirm the
   paper supports that reading; it is the most attackable methodological choice in §2.2.
3. The skew/kurtosis correction path, since the leaky arms have kurtosis 4.7–5.6 and the
   correction matters most there.

### [9] Arian, Norouzi Mobarekeh & Seco, *Backtest Overfitting in the ML Era* — Knowledge-Based Systems 305 (2024)
The one Tier-0 paper whose recommendation is **not yet implemented**. `RESULTS.md` §10
item 2 has asked for CPCV twice and it has not been done.

Extract:
1. **The CPCV construction** — number of groups, number of test splits, and the purging
   and embargo rules. Enough detail to implement it.
2. **PBO** (Probability of Backtest Overfitting): how it is computed and what it needs.
   This is a second immunising statistic to sit next to DSR.
3. The evidence that **walk-forward is weak at false-discovery prevention**, with the
   numbers, because the repo's whole equity screen is walk-forward and the memory note
   `fyp-portfolio-horizon-effect` already concedes its t-stats are "descriptive, not
   inferential".

> Concrete task this unlocks: purging and an embargo around the 2019 validation boundary
> in the RL split, and CPCV for the equity screen in `scripts/`.

### [12] Xia et al., *Agentic Trading: When LLM Agents Meet Financial Markets* — arXiv:2605.19337
Short, and it is the introduction's load-bearing citation. Read for the **counts**, which
are the whole point: of 19 primary studies, 2 report a time-consistent split protocol, 1 a
transaction-cost model, 1 survivorship; no study reaches R3 reproducibility and 15 of 19
sit at R0.

Extract: the exact screening protocol and date (through 2026-03-09), the reproducibility
ladder's rungs, and the per-criterion counts. These numbers go in the introduction to
establish that failure to replicate is the field's *modal* outcome — which is what turns
this FYP's negative result from an embarrassment into a finding.

---

## Tier 1 — the papers being reproduced

Read to the level of reproducing the method, because that is what was done.

### [1] Zhao & Welsch, *Hierarchical Reinforced Trader* — arXiv:2410.14927
**Read v1 and v2 side by side.** They are different papers and `RESULTS.md` §2.1 warns
that conflating them would be a real error.

- **v1** (19 Oct 2024): S&P 500, 2021/2022 test, +39.8 % / +2.3 %. This is what was
  reproduced (+23.1 % / −7.6 %).
- **v2** (11 May 2026): 89-name Nasdaq universe, train 2013–2018, validate 2019, test
  2020–2023, reports Sharpe 1.06 → 1.24 and turnover 0.112 → 0.090, with turnover,
  drawdown and text-risk penalties on the LLC.

Extract:
1. **The precise wording of the forward-return definition and the feature timing.** This
   is the single most important sentence in the reproduction. §4 argues the reading that
   matches the published numbers puts day *T*'s close inside the label window, and that
   `KMID` then explains R² = 0.653 of the label. Quote the paper verbatim; do not
   paraphrase a timing claim.
2. **Whether v2 changes the timing.** If v2 silently fixed it, that is a finding and it
   changes how §4 must be written. If it did not, that is a stronger finding.
3. v2's **cost-stress protocol** — it claims robustness "under transaction-cost stress".
   What rates, what model? §2.3 charges 0/10/30/60 bp and the arms reorder completely at
   60 bp.
4. The LLC objective in both versions, since idea 6 is a claim about what the LLC cannot
   express.

### [13] Gómez-Cram, Guo, Jensen & Kung, *Prediction Market Accuracy: Crowd Wisdom or Informed Minority?* — SSRN 6617059
Note the retitle — `fyp_ideas.md` still cites "…Informed Traders?".

Extract, in priority order:
1. **The exact sign-randomisation procedure.** §3.1 and §3.4 turn entirely on what unit
   gets an independent coin flip. Confirm the null flips **each trade** independently, in
   the paper's own words. If the paper pools at any level, the critique changes shape.
2. **Whether the null is centred at zero**, and whether the paper discusses the
   spread-crossing bias. The repo finds median event-level z = −0.83 and an anti-skilled
   tail four times the skilled tail.
3. **How the 3 % figure is produced**, and at what significance level. §3.2 supports "a
   skilled minority exists" but explicitly does not test "3 % of traders drive the
   market".
4. Sample: 1.72 M accounts, $13.76 B volume, and the date range. The repo's comparable
   figure is 1,819,876 accounts and $27.9 B of taker notional over 2022-11 → 2026-04, so
   the two samples need reconciling in the write-up.
5. Whether any **out-of-sample persistence** test appears. §3.2's design is presented as
   the thing the estimator debate does not need; if the paper already does it, §3.2 becomes
   a replication rather than a contribution.

### [14] *Price Discovery and Trading in Modern Prediction Markets*
Extract:
1. **The exact imbalance specification** — bar length, how "large" is defined, the return
   transform, fixed effects, clustering. `pm/pmv1_imbalance.py` uses: hourly bars, top
   decile by notional within market, log-odds returns, market fixed effects by
   within-transformation, two-way clustering on market and hour. Any deviation must be
   declared.
2. **Which venues and which window.** The paper studies the 2024 US presidential election
   across four venues; the repo's Politics category is the largest one carrying the effect.
3. **Whether any cost is charged.** §3.5 finds the effect is 4.2× under the round-trip
   cost and that its value accrues to the maker. If the paper charges nothing, that is the
   gap the FYP fills; if it charges something, compare.
4. The reported effect size, for a like-for-like comparison with β = +0.0033 (1 h) and
   +0.0056 in the top liquidity quintile.

### [2] Riera Abbade & Reali Costa, *MACE* — arXiv:2603.29086
Extract:
1. **The Almgren–Chriss parameterisation and the square-root law as implemented**, with
   the constants. `crypto/strategy.py` uses half-spread + `Y·σ_t·√(q/V_t)`; if MACE's form
   differs the §2.3 comparison needs restating.
2. **How "the ranking changes" is measured** — rank correlation, top-1 change, something
   else. §2.3 reports 4/6 arms changing rank at 0 bp and 6/6 at 60 bp on equities, 5/11
   strategies on crypto (rank correlation +0.900). Match the metric.
3. **Whether permanent impact is netted** in their agents' PnL. `analysis/costs.py` reports
   it but does not net it, and §6 flags that as a weakness.
4. Whether MACE tests **state dependence** as such. §2.3 finds the square-root state model
   ranks identically to a flat 30 bp, which is the result that undercuts idea 6 — so
   whether MACE already claimed or already excluded this is decisive for the contribution
   statement.

---

## Tier 2 — needed to defend the framing

### [16] Qin & Yang, *Polymarket-v1 Database* — arXiv:2606.04217
The dataset paper for the entire §3. Extract the **schema semantics**, not the headline
numbers: what `D`, `p_event`, `neg_risk_market_id` and `outcome_seq` mean precisely, what
the `daily_aligned` cleaning removes (the relayer-taker filter matters — §6 flags it as a
candidate explanation for a PnL asymmetry), and the evidence that Lee-Ready and kin are
near-random here. That last item is what justifies preferring this tape over the vendor
tape, and it is quoted in §1.4.

Also confirm: 1.20 B raw records vs the 746 M in the two cleaned layers vs the
730,492,126 resolved fills the repo analyses. Those three numbers must be explained in the
data chapter.

### [7] Kashif & Ślepaczuk, *DRL for Diversified Portfolio Management* — arXiv:2605.17307
The strongest external corroboration of the repo's central negative result, and the
evaluation standard to match (**sixteen walk-forward folds, three continents, 2003–2026**,
against this repo's four seeds and two test years).

Extract: the **flat-vs-hierarchical Dirichlet ablation** and its result — HRT never runs
this, and it is the best available evidence on whether the hierarchy is worth anything once
costs are charged. Also their statement that no configuration achieves significant excess
returns over buy-and-hold across markets, with the significance test used.

### [11] Li, Wang & Ma, *Summoning the Oracle to Slay It (FinCAD)* — arXiv:2605.24564
Read for **one argument and one number**: that a headline backtest can be *decomposed* into
a leakage component and a real component, and the −67.1 % largest model-level in-sample
correction. §4 makes structurally the same argument about `KMID` (R² = 0.653 of the label,
two-thirds of the reported IC manufactured by the timing convention). FinCAD is the closest
published analogue and supplies the vocabulary.

Skip the CAD implementation detail unless an LLM channel actually enters the project.

### [21] Yang et al., *Qlib* — arXiv:2009.11189
Skim, except for one thing: **the Alpha158 handler's label definition**,
`Ref($close,-2)/Ref($close,-1)-1`. That one line of upstream source is the strongest
third-party evidence for the leakage argument, because it deliberately skips exactly the
window HRT's stated definition includes. Locate it in the released code as well as in the
paper, and cite both.

### [15] Nechepurenko, *Per-Market Information Leakage and Order-Flow Skill* — arXiv:2605.02287
Short, and the most efficient map of the three concurrent 2026 prediction-market
literatures. Read it **before** [13] if time is tight — it explains why an account-level
and a market-level measure can disagree, which is directly the repo's own earlier
non-replication.

Extract: the three-layer framing (sign-randomisation / regulatory detection / ILS), the
reported 3.14 % skilled-winner classification, and the **ILS construction** — front-loading
relative to an article-derived public-event timestamp — which is the only well-specified
implementable target in the prediction-market cluster that this repo has not touched.

---

## Tier 3 — read if the corresponding direction is chosen

| paper | read only if | what to extract |
|---|---|---|
| [3] HRPM, arXiv:2012.12620 | idea 6 is reframed around **frequency separation** (§10 item 5) | the two-level *different-frequency* design, and the pre-train-then-iterate curriculum as a fix for the HLC entropy pathology in `hrt/README.md` |
| [4] Cheridito & Weiss, arXiv:2608.18195 | market making (idea 4), or a better LLC action space | the deep-set encoder for variable-length order sets; multivariate logistic-normal allocation across price levels |
| [5] Zimmer & Costa, arXiv:2509.12456 | market making on a modelled LOB | the **closed-form optimal benchmark** — the discipline that separates the agent's contribution from the environment's |
| [6] HARLF, arXiv:2507.18560 | the hierarchy-plus-sentiment branch is discussed | read *sceptically*: 26 % annualised over a 2018–2024 bull window with no cost stress and no passive floor. It is an example of the evaluation weakness Tier 0 is about |
| [17] Tsang & Yang, arXiv:2603.03152 | idea 2 (event studies on prediction markets) | the event-study design, and the warning that convergence *speed* confounds with how anticipated the event was |
| [18] FinRL-DeepSeek, arXiv:2502.07393 | the news channel is wanted | FNSPID coverage 1999–2023, the CVaR risk-level channel, and the cost of the universe change (S&P 500 → 89 Nasdaq names) |
| [19] Trading-R1, arXiv:2509.11420 | idea 5 (LLM reasoning) | note what it does *not* do: nothing penalises a trade contradicting its own stated thesis. Read with [11] |
| [20] Pippas et al., arXiv:2408.10932 | writing the literature review | the per-component critique vocabulary (state, action, reward, transfer, multi-agent) across 167 papers |

---

## Reading order, with a time budget

If there are **two days**: [10], [8], [12], [1] v1+v2. That is enough to write the
methodology and results chapters honestly.

If there are **five days**, add: [9], [13], [14], [2], [16].

If there are **ten days**, add Tier 2 entire, plus whichever Tier 3 row matches the chosen
direction.

Read [15] before [13] regardless — it is short and it is the map.

---

## Questions the reading has to settle

These are open in `RESULTS.md` and cannot be closed by running more code. Each is a
question to a specific paper.

1. **Does HRT v2 change the label timing?** → [1] v2. If yes, §4's critique applies only to
   v1 and must say so; if no, §4 gets much stronger.
2. **Is a robust (IQR-based) σ_SR admissible in the DSR?** → [8]. §2.2's crypto panel
   depends on it under a cost model.
3. **Should K include the `runs_invalid/` arms?** → [8]. The most attackable choice in §2.2.
4. **Does [13] flip trades or something coarser?** → [13]. §3.1's entire critique rests on
   "each trade, independently".
5. **Does [13] already run an out-of-sample persistence test?** → [13]. Determines whether
   §3.2 is a contribution or a replication.
6. **Does [14] charge any transaction cost?** → [14]. Determines whether §3.5 fills a gap
   or contradicts a result.
7. **Does MACE already test regime dependence?** → [2]. Determines whether §2.3's negative
   result is novel, and whether idea 6 survives in its current form.
8. **What exactly does `daily_aligned` filter out?** → [16]. §6 flags the relayer-taker
   filter as a candidate explanation for a PnL asymmetry and cannot resolve it from data.

---

## Not on the list, and why

Deliberate exclusions, carried from `PAPERS.md` — worth being able to justify aloud:

* **Empirical asset pricing / cross-sectional ML** (Gu, Kelly & Xiu and successors) —
  relevant to the equity screen in `scripts/`, not to the RL thread. Add only if the
  portfolio work becomes the primary FYP rather than a benchmark.
* **Generative LOB simulators** (TRADES, MarS, JAX-LOB) — only needed for idea 4, since
  prediction-market book history is too short to train on directly.
* **Optimal-execution RL as a field** — one level below where idea 6 sits; [2] and [3]
  carry the argument.
