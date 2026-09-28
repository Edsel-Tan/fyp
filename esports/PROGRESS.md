# LoL win-probability study — progress (updated 2026-09-28)

Executing `~/fyp/PLAN.md`. Everything lives in `~/fyp/esports/`, which git does not track. `esports/data/` is also
covered by the `data` pattern in `.gitignore`, so a branch checkout leaves all of it in place.

## Data sources (step 1)

| Need | Source | Status |
|---|---|---|
| Market prices | Polymarket-v1 archive (`~/fyp/data/pmv1`) | **done**: 27,982 LoL markets ($588M). 1,842 series + 2,262 per-game winner markets, 2025Q3 to 2026-04-28 |
| Schedule and results | lolesports `persisted/gw` API (public key in `src/lolapi.py`) | **done**: `data/matches.parquet`, 11,377 matches over 47 leagues, 2024-01 to 2026-09 |
| Game ids | `getEventDetails` | **done**: `data/games.parquet`, 22,808 completed games |
| In-game state | livestats feed `feed.lolesports.com/livestats/v1/window/{gid}?startingTime=` | **running**: 60 s sampling into `data/frames60/{gid}.parquet` + `.meta.json` (players, champions, roles, patch) |
| Oracle's Elixir CSVs | Google Drive folder `1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH` (file ids in the session) | blocked by Drive's "quota exceeded"; retry later |
| Leaguepedia Cargo | lol.fandom.com api | rate-limited from this IP on the first call |

Livestats replaces the plan's screenshot OCR. It gives per-player gold, level, K/D/A, CS and current health (dead or alive), plus
per-team towers, inhibitors, barons and dragon types, at sub-second resolution, for historical games back to 2024.
It has **no player positions**, so a minimap bitmap is not available. Map state has to come from structures, objectives
and alive counts.

## Linking markets to games

- `src/link_pm.py` gives `data/pm_links_raw.parquet`. It fuzzy-matches team names, uses the team codes in the slug, penalises academy teams, and allows the date to differ by up to one day.
  Keep `score >= 85`. At that threshold the series winner agrees with lolesports **99.6 %** of the time (n=1,464). The remaining misses are
  PCIFIC outcome-label spelling. Linked markets cover 91 % of volume.
- `data/pm_fills.parquet`: 3.63 M fills, 3,249 markets, $473 M. `p_event` is the probability of the `outcome_seq=1` team.

## Traps found

1. **Sides from `getEventDetails` are random** (765 agree vs 713 swapped). Take blue and red from livestats
   `gameMetadata.blueTeamMetadata.esportsTeamId`.
2. There is no per-game winner field anywhere in the lolesports API. `src/winner.py` infers it with a logistic on 60 s
   end-of-game features (structure push over the last 60/120/180 s, end-state diffs). It is **validated**: grouped 10-fold CV accuracy
   is **99.83 %** (11 errors in 6,554 games with ground truth), and 99.91 % of series pass the series-count check.
   - Rejected: a second stage that refetches the last 4 min at full resolution for unsure games and takes the side that
     took the last structure. On 7 hand-picked misses it looked better (5/7), but against CV predictions it is worse in every
     confidence band. That was a selection effect.
   - Ground truths are `sweep` (all games of an n–0 series, only when #games = w1+w2), `final` (the last game goes to the series
     winner) and `pm` (per-game market resolution). The priority is sweep > final > pm, because PM labels disagree with the others
     0.3 % of the time (link side flips).
   - `src/label_games.py raw` builds `data/game_table_raw.parquet`. `src/label_games.py` builds `data/game_table.parquet`, with
     `blue_win` from truth or else the rule, `y_src` saying which, and players, champions and roles per side.
3. Fees: `fee_usdc` in `pm_fills` is non-zero only from 2026-03 (64 % of fills from then on); RAW.md's "identically zero" holds only for the earlier archive.
   Report pre-fee and fee regimes separately.
4. Frame timestamps come in mixed formats (`.xxxZ` or `Z`): parse with `format="ISO8601"`.
5. `data/oe/2026.csv` is Drive's quota-exceeded HTML page, not data.
6. The collector queue runs newest to oldest. PM-linked games sit at queue positions ~2.2k–4k and ~15.6k–15.9k.

## Background jobs (still running)

Two `collect_frames.py 60` processes, working the priority list from each end (`data/ids_all.txt`, `data/ids_rev.txt`).
They resume from disk and skip files that already exist. At ~1,833 / 22,808 games and ~110 games/min, the remaining
~21,000 need about 3 h. Logs are in `logs/frames60*.log`. If they die, re-run from `esports/src`:
`WORKERS=48 setsid nohup python collect_frames.py 60 ../data/frames60 ../data/ids_all.txt > ../logs/frames60.log 2>&1 &`

## Results on full data (2026-09-28, 22,523 games with 60 s frames)

Pipeline: `src/run_all.sh` (it stops on the first failure and logs to `results/run_all.log`; the backtest, wallet and null reruns after the pm_map fix are in
`results/run_tail.log`). Split: train 2024-04..2025-06 (the first 3 months are rating burn-in), test 2025-07+. The CSVs are in `results/`.

- **Winner labels**: rule vs truth 99.88 % (sweep, n=12,709), 99.84 % (final), 99.77 % (pm), and 99.85 % of series pass the count check.
- **Pre-game** (test n=10,584 games; log loss / AUC): team Elo K=64 0.634 / 0.694; **player Elo K=32–64 0.629 / 0.701** (beats team
  Elo); experience 0.637; form_20 0.653; h2h 0.687. **All features (logistic) 0.614 / 0.720** is best (GBM 0.616, ns worse).
  **One-hot player identity overfits**: train 0.56 vs test 0.66, worse than any Elo; one-hot + Elo (C=0.01) is 0.628, still worse
  than all features. So encode players through ratings, not names.
- **In-game** (10,582 test games, per-game mean log loss): prior 0.627; gold-only with a time-varying slope 0.480; gold+prior 0.453; **all stats × time
  0.434** (−0.046 [−0.050, −0.042] vs gold); GBM 0.442. A fixed prior weight hurts after 25 min (long games are selected for upsets), so
  the prior's weight has to vary with time.
- **Model vs market** (1,625 PM game markets; the last trade ≤ 60 s old; 2 markets with inverted sides dropped, lol-bro1-hle-2026-04-27):
  at lag 0 the **model beats the market**: per-game log loss −0.013 [−0.026, −0.002]. At lag 30 s they are equal (+0.000); at 60 s +0.009 (ns); at 120 s the market is ahead by
  +0.030. Encompassing weight of the model beyond the price: 0.45 (0 s), 0.32 (30 s), 0.26 (60 s), 0.16 (120 s), 0.07 ns (300 s).
- **Backtest** (eval Feb–Apr 2026; fills at the share-VWAP of the next $50 of our-side prints after decision + L): gated all-stats model at L=0 gives **+9.2 pp/share
  [1.2, 18.1]** (θ=0.25 chosen on selection months, 732 trades, 277 games; pre-fee +4.0, fee regime +11.6 gross / +8.6 net). At L=5 s 8.0 [−0.7, 17.2];
  15 s 6.7 ns; 30 s 6.0 ns; 60 s 2.0; 120 s 1.2. Fixed clip +1.7 [−0.2, 3.5]. Gold-only and market+model are ns everywhere.
  Profit by minute: 2.6 (0–10), 5.6, 9.3 (20–30), 5.1 (30+).
  Controls: **5 shuffled-outcome nulls give −0.07…+0.32 pp** (the gated config); the placebo (another game's signal at the same minute) gives +0.5 ± 0.35.
  **Multiple testing**: 18 (signal, L) cells were examined; the primary cell has t≈2.1, which does **not** survive Bonferroni over 18. The monotone decay in L is the
  stronger evidence.
- **Follow skilled wallets** (top 50 takers by market-clustered t on markets before 2026-02): −2.5 pp [−8.4, 2.7] at L=0 (ns; only 308 copies).
  Copying every taker: −0.31 pp [−0.59, −0.02], about a half-spread, which is a check that the execution costs are realistic.
- **Event study** (`src/event_study.py`, complete full-resolution collection: 1,534 games with market fills; 2,822 fights, 14,887 towers, 1,780 barons,
  2,134 inhibitors, 5,772 drakes): around a fight's first kill (feed ts), the price is flat from −60 s to 0 (+0.021 → +0.024), then +0.031 at 5 s,
  +0.051 at 10 s, +0.077 at 15 s, +0.099 at 20 s, +0.124 at 30 s and +0.147 at 60 s (~half the move by 15–20 s). Trade intensity peaks at 4.0× the pre-event rate at 15 s.
  **The market does not lead the feed**; it reacts ~5–10 s after the feed timestamp (partly because the fight is still unfolding). This matches the backtest edge
  dying between L=5 s and L=15 s. Towers, inhibitors and barons are priced in before their timestamp (they follow fights and are anticipated), so they do not
  identify latency.
- **Live publication delay: not being verified** (2026-09-28, user decision; the L=5 s backtest cells stay). `src/live_lag.py` is kept but was
  killed before its 16:10 UTC run, so the edge's tradability (it needs the feed published within ~5 s) is **untested**. WSCI streams have `statsStatus: disabled`.

## Literature recreation (2026-09-28, `src/lit_compare.py`, `results/lit_compare*.csv`)

Published models refit on our split and 60 s frames. Per-game log loss on the test set (Δ vs `ours lr all` 0.4339, clustered 95 % CI):
- Silva et al. 2018 RNN/LSTM 0.4373/0.4371 (+0.003); Kim et al. 2020 MLP 0.4373, + temperature scaling (T=0.98) 0.4375, DU loss 0.4368
  (+0.003, ECE 1.66 % vs 1.77 %: their calibration gain does not reproduce; on pro data the plain model is already calibrated).
- Riot/AWS broadcast WP feature set (LightGBM standing in for XGBoost, with buff and respawn timers): 0.4613 (+0.027) without team strength, 0.4416 with our prior.
- Hodge et al. per-minute models with 5-min deltas: 0.4576 ≈ ours without prior 0.4573. Time-varying coefficients are the whole gain; separate per-minute models add nothing.
- Snapshot LR at 10 and 15 min (Kaggle/OE style): +0.027 and +0.019 vs ours on the same frames.
- Junior & Campelo percent-elapsed-time slices: our causal model scores **87.6 % accuracy at PET 80 %** (they report 81.6–85 %), so PET accuracy measures
  which frames were picked. Giving LightGBM PET as a feature cuts log loss by 0.036 (60 %) and 0.021 (80 %), which is the leak, because PET needs the final game length.
- Conclusion: architecture does not matter (every NN is within 0.004). The pre-game prior (−0.023) and time-varying weights are what help.
  Our model's validation temperature is 0.95, which is already calibrated.
- New data found: livestats `details/{gid}` has per-player `wardsPlaced`, `wardsDestroyed`, items, damage share and KP back to 2024-01 (not collected yet).
  Lane-opponent history is thin: the median test game has 3 prior meetings per role, 20 % have none. Faker vs Chovy is 69 games (27–42).

## Direction 2: confidence and calibration (2026-09-28, `src/confidence.py`, `src/recal.py`, `results/confidence*.csv`, `results/recal.log`)

- **Epistemic uncertainty doesn't help** (with binary labels, per-state noise can't be identified; see the Kim et al. note).
  - Measures: match-bootstrap sd of the logit (median 0.09) and model-family disagreement (median 0.19, corr 0.57).
  - Integrating either one out: Δ ≈ 0. Family-mean ensemble: +0.0009 (worse).
  - Both sd measures are highest in decided late-game states. There, the model's encompassing weight beyond the market is highest
    (0.53 vs 0.35 in the low-sd tercile), and gated trades earn the most (+13.3 pp vs +6.5).
  - So a "trade only when confident" filter is backwards. The low-sd filter gives 8.7 pp (ns); the z-gate edge/sd gives 0.5 pp (ns). Neither beats the θ gate (9.2).
- **Calibration** (`lr all`, test): slope 0.937 overall.
  - Overconfident on tier-1 leagues (0.894) and PM-linked games (0.889).
  - Teams with little history have intercept −0.22.
  - International 0.99; new patch (<7 days) 0.95, no worse than later in the patch.
- **The slope drifts with model age**: 1.10 on the validation months, 1.0 at the split, then 0.85–0.90 by 2026-05..09.
  - A fixed recalibration fit on validation makes test worse: +0.0017 [+0.0010, +0.0026].
  - A causal 3-month rolling recalibration restores slope 0.98, but log loss is unchanged (+0.0004 ns).
  - Practical rule: retrain or recalibrate monthly for calibration's sake. It won't improve accuracy.

## Directions 3–5: draft, team strength, head-to-head, patch (2026-09-28)

`src/pregame2.py` gives `data/pregame2.parquet`: a single causal pass with 120-day half-life decayed counts.
`src/eval_pregame2.py` chooses C on the last 91 days of train and scores test once. It also writes `data/prior2.parquet`, with out-of-fold train priors.
- **Pre-game** (test log loss vs `all features` 0.6134):
  - Each group on its own:
    - lane Elo from gold difference at 15 min −0.0038 [−0.0054, −0.0024]
    - league-strength Elo (updated on inter-league games only) −0.0028 [−0.0041, −0.0016]; international games −0.018 (ns)
    - champion strength: decayed all-time −0.0027; per-patch only −0.0016; both −0.0027
    - player-on-champion −0.0017
    - team conversion −0.0008
    - h2h residuals (team −0.0010, player pairs −0.0008, lane pairs −0.0016, all significant but tiny)
    - null: side-by-patch and patch-age × Elo (+0.0001), win-based role Elo (+0.0002), lane champion matchups (0.0000)
  - **All new: 0.6039, −0.0095 [−0.0121, −0.0069]; AUC 0.720 → 0.731; international games −0.030 [−0.048, −0.009].**
  - Dropping the h2h features costs only 0.0004.
  - **Per-patch champion win rates lose to decayed all-time rates**: one patch (~460 games) is too little pro data.
  - Carry dependence as a team's gold share by role barely varies between teams (bottom lane 23.3 % ± 1.0 %), so it can't be used as a feature.
- **In-game** (`src/eval_ingame2.py`, per-game log loss vs `lr all` 0.4339):
  - The 14-feature out-of-fold prior alone gives −0.0047. **The new prior gives 0.4255, −0.0084 [−0.0105, −0.0063].**
  - On top of that, these are all null (|Δ| ≤ 0.0004): composition scaling × time and gold, conversion × gold, the player hinge × role gold (direction 4's
    "one player's lead matters more"), 180-day recency weighting, and monthly retraining.
- **Market and backtest with the new model** (`results/backtest_v2.log`):
  - Model minus market log loss: −0.026 [−0.037, −0.014] at 0 s, −0.011 [−0.021, −0.001] at 30 s (the old model was level at 30 s), −0.002 at 60 s (ns).
  - Fixed clip +2.5 pp [0.7, 4.4]; the old one was +1.7 (ns). It is −1.1 net in the fee regime.
  - Gated, θ=0.30 chosen on selection months: +11.4 pp [1.8, 21.3] over 299 trades at L=0; +8.6 (ns) at L=5.

## Direction 1: vision (2026-09-28, `src/collect_details.py`, `src/eval_vision.py`, `results/vision_eval.log`)

- `data/details60/{gid}.parquet`: the livestats `details` feed at the frames60 timestamps (per-player wards placed and destroyed, items, control wards,
  damage share, KP). 20,611 games; every post-burn-in game except about 900 without a feed.
- Added to the `prior_new oof` in-game model (wards placed, wards destroyed, control wards, support and jungle wards placed, 5-min deltas,
  each × (1, m, m²)): **Δ −0.0000 [−0.0004, +0.0003]**, flat in every 10-minute bucket.
  - Partial effects are small (≤ 0.12 logit per SD).
  - Support and jungle wards placed are *negative* given the rest of the state, as expected if the losing side wards to compensate.
  - Ward counts carry nothing that gold, objectives and alive counts don't. Real map vision needs positions (minimap computer vision), which the feed doesn't have.

## Background jobs

None.

## Next steps

0. Directions 1–5 are done (sections above). The only in-game change worth keeping is `prior_new` (out-of-fold, with lane Elo, league Elo, champion strength and
   player-champion features). Wire it into `run_all.sh` / `eval_ingame.py` and rerun `market_compare.py` and the nulls on it; the backtest cell is `results/backtest_v2.log`.
   Retrain or recalibrate monthly (the calibration slope decays about 0.02 per month).

1. Pre-game: repeat the one-hot overfit test on international events specifically (PLAN's "Faker" worry; intl n=310 test games).
2. In-game: per-minute calibration plot; GBM tuning.
3. Significance: a formal deflation (Holm over the 18 cells, or a deflated Sharpe on per-game PnL) and Diebold–Mariano on per-game log loss vs the market.
4. Write `~/fyp/RESULTS.md` (PLAN.md asks for it there). Check first whether one exists on another branch (memory notes refer to RESULTS.md §3).
