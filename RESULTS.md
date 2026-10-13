# Results — LoL win probability, and whether it beats Polymarket

Executes [`PLAN.md`](PLAN.md): predict League of Legends game outcomes pre-game and in-game, then test whether the
model trades profitably against Polymarket's per-game markets. Work done 2026-09-28/29. The code and data are in
`esports/` (untracked). The working log, with every intermediate number and dead end, is
[`esports/PROGRESS.md`](esports/PROGRESS.md). Figures are in `esports/results/viz/`.

> The FYP's main `RESULTS.md` (reading-list reproductions, prediction-market microstructure) lives on the
> `hrt-reproduction` / `pmlive-deploy` branches. This file covers the LoL study only. When the branches merge, the two
> need joining by hand (this can become one section of that file).

Split: train 2024-04 … 2025-06 (2024-01…03 is rating burn-in only); test 2025-07 onward. Nothing after the split is used
for fitting or for choosing hyperparameters. CIs are 95 % and match-clustered (games in one series are dependent).

---

## 0. Summary

| PLAN item | Outcome |
|---|---|
| Pull data for every input | Done without screenshots. The lolesports livestats feed gives per-player gold, level, K/D/A, CS, health, items, wards and per-team objectives for every pro game back to 2024 (22,523 games, 47 leagues). The feed has no player positions, so there is no minimap bitmap. |
| Pre-game: player vs team ratings, recent form, h2h | **Ratings beat names.** Player Elo > team Elo. One-hot player identity overfits everywhere, **including at international events** (§2.2, the "Faker" test). Best: 14 rating/form features plus league Elo, lane Elo and draft strength, log loss **0.604** (AUC 0.731). |
| In-game: gold-only vs full state | Gold-only (PLAN's simple model) 0.480 per-game log loss; **all stats × time + pre-game prior 0.4255** (−0.054). A tuned GBM is worse, and every published architecture ties or loses. |
| Realistic latency | Measured from the market itself: prices move 5–10 s after the feed's fight timestamps. The model is **more accurate than the last trade for up to 30 s of market lag** (Diebold–Mariano p ≤ 0.012). |
| Trading: fixed clip vs confidence gate | Gate at L=0: **+11.4 pp/share [1.8, 21.3]**, decaying with latency. Clip: +2.5 pp [0.7, 4.4], negative net of fees. |
| Significance / overfitting | Shuffled-outcome nulls ≈ 0. **No backtest cell survives Holm or BH across the 48 cells examined**, and the deflated Sharpe of the primary cell is 0.39–0.56. The *forecasting* edge is significant; the *trading* edge is not established. |
| Baselines (equal weight, informed traders, simple model) | Coin-flip 0.693 and side-only 0.690 pre-game; gold-only as above. Copying the 50 most skilled wallets −2.5 pp (ns). Copying every taker −0.3 pp (≈ half a spread). |

The one-line version: **the model knows the game state better than the market does for the first ~30 s after it changes,
and that is statistically solid. Whether it can be traded is not.** Whether it can be traded hinges on two things this
data cannot settle: how quickly the feed is published in real time (never measured, see §7), and a trading PnL that does
not survive multiple-testing correction.

---

## 1. Data and labels

| Need | Source | Result |
|---|---|---|
| Schedule, results | lolesports `persisted/gw` API | 11,377 matches, 47 leagues, 2024-01 … 2026-09 |
| In-game state | livestats `window/{gid}` (60 s frames; 10 s / full resolution for PM-linked games) | 22,523 games |
| Items, wards, damage | livestats `details/{gid}` | 20,611 games |
| Market prices | Polymarket-v1 on-chain archive | 3.63 M fills, $473 M; 1,759 per-game markets linked |

- **The API has no per-game winner.** `winner.py` infers it from end-of-game structure pushes. It agrees with ground truth
  **99.88 %** of the time (series sweeps, n = 12,709), and 99.85 % of series pass the game-count check.
- **Blue/red side from `getEventDetails` is random** (765 agree vs 713 swapped). It has to come from livestats
  metadata. Using the API's sides would have scrambled every side-dependent feature.
- Market linking: fuzzy team matching with score ≥ 85 gives 99.6 % series-winner agreement. Two markets whose own
  resolution contradicts the game label (inverted outcome labels, `lol-bro1-hle-2026-04-27`) are dropped.
- Taker fees exist only from 2026-03. Results are reported separately for the pre-fee and fee regimes.

## 2. Pre-game model

Test n = 10,584 games. The reference is `all features` (a logistic on 14 rating/form features).

| Model | Log loss | AUC | Δ vs all features |
|---|---|---|---|
| coin flip / blue-side rate | 0.693 / 0.690 | – | +0.080 / +0.077 |
| team Elo (K=64) | 0.634 | 0.694 | +0.020 |
| **player Elo (K=32–64)** | **0.629** | **0.701** | +0.015 |
| form (last 20) / h2h / experience | 0.653 / 0.687 / 0.637 | | |
| **all features** (logistic) | **0.6137** | 0.720 | 0 |
| GBM, same features | 0.616 | 0.716 | +0.002 (ns) |
| one-hot players | 0.662 | 0.637 | +0.049 |
| + league Elo, lane Elo (gold@15), champion strength, player-on-champion | **0.6043** | **0.731** | **−0.0091 [−0.0115, −0.0067]** |

### 2.1 Which constructions matter
Player ratings beat team ratings. Decayed all-time champion win rates beat per-patch ones, because one patch is ~460
games, too few. Lane Elo from gold at 15 minutes adds −0.004. League-strength Elo, updated only on inter-league
games, adds −0.003 overall and −0.018 on international games. Head-to-head residuals are significant but tiny
(dropping them costs 0.0004). Null: side × patch, patch age × Elo, lane champion matchups.

### 2.2 The "Faker" worry: player names at international events (`intl_onehot.py`)
PLAN's concern was that a player-name model would overrate famous players when leagues meet. Test n = 310
international games, 265 of them between leagues.

| Model | all test | international | intl, cross-league |
|---|---|---|---|
| all features | 0.6134 | 0.6295 | 0.6154 |
| prior_new (+ league Elo etc.) | **0.6043** | **0.6035** | **0.5881** |
| one-hot | 0.6622 | 0.6547 | 0.6442 |
| one-hot + Elo | 0.6275 | 0.6610 | 0.6546 |
| one-hot as residual on prior_new | 0.6041 | 0.6065 (+0.003 ns) | 0.5912 (+0.003 ns) |

Names never help, internationally or elsewhere. As a residual on top of the ratings, the player effects take the
smallest penalty offered (C = 0.01), which means they want to be zero. The international problem is real, but it is
about *leagues*, not *players*: league Elo alone moves cross-league log loss by −0.027 [−0.051, −0.005].

## 3. In-game model

Test: 10,582 games, 349,422 60-second frames. The metric is per-game mean log loss, so each game counts once.

| Model | Per-game LL | Δ vs gold-only |
|---|---|---|
| pre-game prior only | 0.627 | +0.147 |
| **gold only, slope varying with time** (PLAN's simple model) | 0.480 | 0 |
| gold + prior | 0.453 | −0.027 |
| all stats × (1, m, m²) + prior (`v1`) | 0.4339 | −0.046 |
| **same with the v2 prior (`v2`, primary)** | **0.4255** | **−0.054 [−0.059, −0.050]** |
| GBM, untuned | 0.4415 | −0.038 |
| GBM, tuned on validation (7 leaves, 742 trees) | 0.4306 | +0.005 vs v2 [+0.004, +0.006] |
| GBM boosting v2's residual | 0.4254 | −0.0000 vs v2 (stops at 1–3 trees) |

- **The gains come from two things**: the pre-game prior (−0.023 with v1's prior, a further −0.008 with v2's) and
  letting every weight vary with game time. A fixed-weight prior hurts after 25 minutes, because long games are
  selected for upsets.
- **Literature recreation** (`lit_compare.py`): LSTM/RNN (Silva 2018), MLP with calibration losses (Kim 2020),
  Hodge per-minute models and the Riot/AWS broadcast feature set, all refit on this split and these frames. Every
  neural net lands within 0.004 of the logistic. Kim et al.'s calibration gain does not reproduce, because the plain
  model is already calibrated. Percent-elapsed-time accuracy (Junior & Campelo) is a leak: it needs the final game length.
- Also tried and null (|Δ| ≤ 0.0004): ward and vision counts, champion scaling × time, team conversion × gold,
  "one player's lead matters more" hinge terms, recency weighting, monthly retraining, and epistemic-uncertainty
  integration.

### 3.1 Calibration by game minute (`ingame_calib_gbm.py`, `fig_calib_minute.png`)
The v2 slope is 0.91–0.98 in every 5-minute bucket: slightly overconfident, most clearly at 15–20 min (0.91 [0.88, 0.94]).
Early buckets have intercept −0.14: the prior slightly overrates blue in the test period. ECE is ≤ 0.03 everywhere.
The tuned GBM is better calibrated before 15 minutes but gets steadily worse late (0.81 at 40+). The slope also drifts
with model age, from 1.0 at the split to 0.85–0.90 a year later. A rolling recalibration fixes the slope but not the
log loss, so retrain monthly for calibration's sake only.

## 4. Model vs market

1,625–1,701 per-game markets. The market price is the last trade at (frame time + lag), at most 60 s old. A positive lag
gives the market extra time, so it is conservative for the model.

| Market lag | v2 − market, per-game LL | Diebold–Mariano t (p) | Model weight beyond the price |
|---|---|---|---|
| 0 s | **−0.026 [−0.037, −0.015]** | −4.13 (<0.001) | 0.57 |
| 5 s | −0.022 [−0.033, −0.012] | −3.83 (<0.001) | 0.55 |
| 15 s | −0.017 [−0.027, −0.008] | −3.45 (<0.001) | 0.50 |
| 30 s | −0.011 [−0.021, −0.002] | −2.51 (0.012) | 0.44 |
| 60 s | −0.002 (ns) | −0.98 (0.33) | 0.36 |
| 120 s | +0.019 | +2.66 | 0.24 |
| 300 s | +0.075 | +10.6 | 0.12 |

DM uses the per-match mean loss differential in time order with a Newey-West SE (6 lags). v1 beats the market only at
0–5 s. **The draft/strength prior is what extends the lead to 30 s.** Even at 300 s the model keeps a significant weight
beyond the price (0.12 [0.04, 0.20]).

**Latency, from the market side** (`event_study.py`, 2,822 fights): around a fight's first kill (feed timestamp), the price
is flat for the prior minute. It moves +0.031 by 5 s, +0.077 by 15 s and +0.147 by 60 s, and trade intensity peaks at
4× at 15 s. The market does not lead the feed. It reacts 5–10 s after it, which matches the backtest edge dying between
L = 5 s and 15 s. In the broadcast VOD, the feed timestamps line up with the on-screen game to about 1 s.

## 5. Execution backtest

Eval months Feb–Apr 2026. Selection (for θ, and for the market+model combination) uses months before 2026-02. A trade
is 1 share at the share-VWAP of the next $50 of same-side aggressor prints after decision time + L, held to resolution.

| Strategy (L = 0) | pp/share | 95 % CI | trades / games | fee regime net |
|---|---|---|---|---|
| **v2, gate θ = 0.30** | **+11.4** | [1.8, 21.3] | 299 / 155 | +7.4 |
| v1, gate θ = 0.25 | +9.2 | [0.9, 17.9] | 732 / 277 | +8.6 |
| v2, clip (every frame) | +2.5 | [0.7, 4.4] | 20,719 | −1.1 |
| gold-only, gate | +1.1 | ns | 1,860 | |
| market+model combination, gate | +6.9 | ns | 280 | |
| copy 50 most skilled wallets | −2.5 | [−8.4, 2.7] | 308 | |
| copy every taker | −0.3 | [−0.6, −0.0] | 1.04 M | |

- The v2 gate decays with latency: 11.4 (0 s), 8.6 (5 s), 5.5 (15 s, θ = 0.15), 7.0 (30 s), 2.9 (60 s), 2.3 (120 s).
- Profit is largest at 20–30 minutes (+10.3) and on underdog buys (+7.3 vs +4.4).
- **Controls.** Five shuffled-outcome nulls give −0.07 … +0.25 pp (v2 gate). A placebo that uses another game's signal
  at the same minute gives +0.48 ± 0.34. Copying every taker costs about half a spread, which shows the fill model is
  not flattering.

## 6. Significance (`significance.py`)

- **Forecasting: significant.** The DM tests in §4 reject at 0–30 s, and the smallest of those four (t = −2.51) still
  passes Holm across them.
- **Trading: not established.**
  - There are 48 cells: 4 signals × 6 latencies × {clip, gate}, with each gate θ chosen on the selection months.
    Using match-clustered one-sided p-values, the best cell is the **v2 clip at L = 0 (t = 2.65, p = 0.004)**, but its
    **Holm-adjusted p is 0.19** and its BH q is 0.17.
  - The primary cell, the v2 gate at L = 0, has t = 2.24, p = 0.013, Holm 0.58.
  - Among the 24 gate cells alone: Holm 0.30, BH 0.21.
  - The **deflated Sharpe** of the primary cell (per-game PnL, T = 155 games, SR 0.17 per game, skew 2.0,
    kurtosis 15) is **0.56** with the 48 cells as trials, and **0.39** with all 264 (cell, θ) combinations.
    PSR against zero is 0.993.
  - So the edge is real *in sample*, and it is too small and too heavy-tailed to separate from what the search
    would produce by chance.
- What survives is the **shape**: the edge is monotone in latency, the model's weight in the combined model is
  monotone, the nulls are flat, and it lines up with the event-study reaction time. A single pre-registered forward
  test would settle it (§8).

## 7. Shortfalls

- **Real-time feed latency is unmeasured.** The edge needs the livestats feed published within ~5 s of the game.
  `live_lag.py` exists, but the live run was cancelled (user decision, 2026-09-28). Every positive backtest cell rests on
  this assumption.
- **The fill model is a proxy.** Execution is at later aggressor prints, not against a reconstructed book, because the
  Polymarket LOB tape stopped. Queue position and our own impact beyond $50 are not modelled.
- **The eval window is three months** (155 games in the primary cell). The fee regime starts inside it.
- **Winner labels for 28 % of games are inferred** (99.88 % accurate). The residual error biases log loss up slightly
  and equally for all models.
- The feed has no positions, so real vision and map control are missing. Ward counts add nothing.

## 8. What to do next

1. Measure the live publication delay of the livestats feed against the broadcast (`live_lag.py`) during a live event.
   This decides whether §5 is tradable at all.
2. Pre-register the v2 gate (θ = 0.30, L = feed delay) and forward-test it on markets after 2026-04-28. The archive ends
   there, so this needs the live capture ([`pm/`](pm/)).
3. Retrain or recalibrate monthly (the slope decays about 0.02 per month).

## 9. Reproducing

```bash
cd esports/src
# data collection (resumable): collect_frames.py, collect_details.py; see esports/PROGRESS.md
bash run_all.sh                            # labels -> pregame v1/v2 -> ingame (v1, v2) -> markets -> backtest -> wallets -> nulls
python market_compare.py 5 15              # extra lags for the DM table
python intl_onehot.py                      # §2.2
python ingame_calib_gbm.py                 # §3 GBM rows, §3.1
python significance.py                     # §4 DM, §6
python eval_ingame2.py; python eval_vision.py; python lit_compare.py; python confidence.py; python recal.py; python event_study.py
python fig_results.py                      # figures
```
Logs: `esports/results/{run_all,intl_onehot,ingame_calib_gbm,significance}.log`. CSVs are alongside them.
