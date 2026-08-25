# FYP Idea Brainstorm — AI in Financial Data Analytics and Trading (H398280)

Supervisor: LU Yao
Focus areas requested: price signals for prediction markets, reinforcement learning for trading

Ideas below were checked against past student project titles under this supervisor to avoid overlap
(spatio-temporal graph arbitrage, LLM-gated cascades, social-signal studies, and generic crypto
portfolio RL are already well covered by prior cohorts).

---

## Prediction market price signals

### 1. Who moves the price? Informed-trader vs. crowd decomposition, live

A recent paper (Gómez-Cram, Guo, Kung, Jensen — 1.72M accounts, $13.8B volume) found that ~3% of
Polymarket traders drive most price discovery, undermining the "wisdom of crowds" story. Past
students already did arbitrage-filtering and social-amplification-vs-novel-information work, but
nobody has built a *live, wallet-level classifier* that scores each incoming trade by
"informativeness" (order size, wallet history, timing relative to news, PnL track record) and uses
that score to reweight the market's implied probability into a better real-time forecast than the
raw price. This sits closer to microstructure/market-quality research than another
arbitrage-detection pipeline.

**Literature**
- Gómez-Cram, Guo, Kung & Jensen, "Prediction Market Accuracy: Crowd Wisdom or Informed Traders?" (2026) — [CoinDesk summary](https://www.coindesk.com/markets/2026/04/26/only-3-of-traders-drive-prediction-markets-accuracy-not-the-crowd-study-finds)
- Saguillo et al., "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets" — [arXiv:2508.03474](https://arxiv.org/abs/2508.03474)
- Mitts & Ofir insider-detection work, referenced in ["Information Leakage at Population Scale"](https://arxiv.org/pdf/2605.00459)

### 2. Calibration and price-signal decay after news shocks

Measure how fast Polymarket/Kalshi prices actually converge to the "true" probability after a
discrete news event (court ruling, poll release, earnings), and whether convergence *speed* itself
is a tradeable signal — markets that reprice slowly may offer short-lived edge. This is a
forecasting-quality / event-study project rather than a graph-learning project, so it avoids the
spatio-temporal-graph cluster the past cohort mined heavily.

**Literature**
- "Price as Focal Point: Prediction Markets, Conditional Reflexivity, and the Politics of Common Knowledge" — [arXiv:2604.24147](https://arxiv.org/pdf/2604.24147)
- "The Anatomy of a Blockchain Prediction Market: Polymarket in the 2024 U.S. Presidential Election" — [arXiv:2603.03136](https://arxiv.org/html/2603.03136v2)

### 3. Cross-platform mispricing between Polymarket and Kalshi (or vs. sportsbook odds)

Rather than intra-Polymarket arbitrage (already done multiple times: filtering framework, NBA
order-book arbitrage, event-pairs graph arbitrage), look at cross-venue divergence where one venue
is CFTC-regulated (Kalshi) and one isn't, creating structurally different participant bases and
latency. A clean, well-scoped data-engineering + signal project.

**Literature**
- "Arbitrage Analysis in Polymarket NBA Markets" — [arXiv:2605.00864](https://arxiv.org/html/2605.00864v1)

---

## Reinforcement learning for trading

### 4. RL market-making / liquidity provision on prediction-market order books

There's a strong recent RL-for-market-making literature (LOB-based, Hawkes-process simulators,
non-stationary stochastic control), almost all applied to equities/crypto CLOBs — a past student
already did DRL for concentrated liquidity on DEXs, so don't repeat that. Applying RL market-making
to a prediction-market order book (very different reward shape: bounded [0,1] price, binary
terminal payoff, no dividend/carry) is close to unexplored and fits naturally with the lab's
Polymarket data access.

**Literature**
- "Market Making with Deep RL from Limit Order Books" — [arXiv:2305.15821](https://arxiv.org/abs/2305.15821)
- "RL-Based Market Making as Stochastic Control on Non-Stationary LOB Dynamics" — [arXiv:2509.12456](https://arxiv.org/html/2509.12456v1)
- "RL for Trade Execution with Market and Limit Orders" — [arXiv:2507.06345](https://arxiv.org/pdf/2507.06345)

### 5. LLM-reasoning-gated RL policy (Trading-R1 style) with thesis-consistency reward

Instead of "LLM alpha signal → black-box RL", use an LLM to produce structured reasoning traces
(thesis, risk, invalidation condition) and train an RL policy with reward shaped partly by whether
the trade respects its own stated thesis — closer to interpretable, auditable trading than pure
alpha-factor mining. Past students already did an "LLM-Gated QUADRAIL Forecaster" and "Agentic
Optimization in Portfolio Management," so the differentiator has to be the specific reward design
(thesis-consistency penalty) and target market (e.g. event-prediction markets, not equities) rather
than the LLM+RL combo itself.

**Literature**
- "Trading-R1: Financial Trading with LLM Reasoning via RL" — [arXiv:2509.11420](https://arxiv.org/pdf/2509.11420)
- "Sentiment-Aware Stock Price Prediction with Transformer and LLM-Generated Formulaic Alpha" — [arXiv:2508.04975](https://arxiv.org/abs/2508.04975)
- "Adaptive Alpha Weighting with PPO" — [arXiv:2509.01393](https://arxiv.org/html/2509.01393v2)

### 6. Hierarchical RL for execution allocation with realistic transaction-cost modeling

A past student did "Integrated Crypto Trading Framework: signal generation + VWAP execution," so
pure execution-RL is taken — but a hierarchical setup (high-level asset/side selection via one
policy, low-level order-placement via another, à la Hierarchical Reinforced Trader) combined with a
*non-constant* transaction-cost model (geometry-based, from the volatility surface rather than a
flat bps assumption) is a meaningfully different angle with good recent literature to build on.

**Literature**
- "Hierarchical Reinforced Trader (HRT): A Bi-Level Approach for Optimizing Stock Selection and Execution" — [arXiv:2410.14927](https://arxiv.org/pdf/2410.14927)
- "Deep Reinforcement Learning for Cryptocurrency Portfolio Management: A Free-Energy Framework with Geometry-Based Transaction Costs" — [MDPI](https://www.mdpi.com/2227-9091/14/5/103)

---

## Recommendation

If the goal is to genuinely combine "price signals for prediction markets" and "RL for trading"
into one coherent thesis rather than picking two separate directions: **idea 4 (RL market-making on
prediction-market order books)** is the strongest candidate — technically meaty (needs a market
simulator, an RL agent, and inventory/spread reward design), under-explored relative to
equity/crypto LOB market-making, plugs directly into the lab's existing Polymarket
data/infrastructure, and doesn't overlap with any past title.

**Idea 1 (informed-trader decomposition)** is the best pure "price signals" option for a more
research/measurement-oriented project rather than a systems-heavy RL build — worth discussing with
Lu Yao given "define your own success" is explicit in the project brief.
