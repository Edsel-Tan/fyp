# LoL win-probability study — progress (updated 2026-10-13)

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

## Visualisations (2026-09-29, `src/viz_game.py`, `src/render_replay.py`, `src/fig_results.py`, output in `results/viz/`)

- **Replay videos** of GEN vs KT, Worlds 2025 semifinal game 1 (test period; Polymarket $1.2M; GEN, on blue, peaked at 96 % model / 95 % market around
  38 min and KT came back). The broadcast VOD (YouTube `jEMKzyJElws`, in `data/vod/`) is on top. Below it is a live panel with the model (`prior_new oof`, refit on train
  exactly as eval_ingame2, scored on full-resolution frames10 at 1 s), the last Polymarket trade, $ traded, event glyphs and the state diffs.
  - Sync: VOD second = wall time − the vod's `firstFrameTime` from getEventDetails. Verified on the broadcast: timer 10:00 and 30:08, gold −0.7K and +4.3K vs feed
    −731 and +4,302; at KT's baron the broadcast and panel agree to within ~1 s. So the feed's timestamps line up with the (VOD of the) broadcast.
  - frames10 clock is re-anchored at the first frame with total gold > 5,000 (= 1:05); ingame_data's 60 s anchor would be 65 s off on dense frames.
    It still drifts by ~8 s by 30:00 (an unflagged pause?), so the panel's x-axis is labelled "feed clock".
  - Files: `gen_kt_g1_full.mp4` (54 min, with audio), `gen_kt_g1_comeback.mp4` (36:00 to end, with audio), `gen_kt_g1_timelapse10x.mp4` (5 min).
  - Other game: `python viz_game.py GID` then `python render_replay.py GID VOD.mp4 FIRST_FRAME_ISO OUT.mp4 [--from s --to s --speed k]`.
    Get the VOD with `python -m yt_dlp -f 298+140` (yt-dlp is installed in ~/fyp/.venv).
- **Static figures** `fig_*.png`: market_lag, backtest_lag, event_study, ingame_minute, calibration, pregame, literature, game_traces.
- New numbers from fig_market_lag (v2 `prior_new oof` minus market, per-game log loss): 0 s −0.026, **5 s −0.022 [−0.034, −0.012], 15 s −0.017 [−0.027, −0.007]**,
  30 s −0.011, 60 s −0.002 (ns), 120 s +0.019, 300 s +0.075. v1 at 5 s is −0.010 (ns). So v2's accuracy lead over the price survives 15–30 s of market lag,
  but the v2 trading edge was still only tested at L = 0 and 5 s.

## v2 in the pipeline, and next steps 1–4 (2026-09-29)

- **Pipeline**: `run_all.sh` now runs `pregame2.py` and `eval_pregame2.py` too. `eval_ingame.py` adds `lr v2` (= `prior_new oof`, and it reproduces
  0.4255 exactly). `market_compare.py` and `backtest.py` add `q_lr v2`, and the backtest's `q_comb` is now built on v2. The null grep was fixed (it dropped the L=30 rows).
  The full rerun is in `results/run_all.log`; the v1-only run is kept as `results/run_all_v1.log`. Every earlier number reproduces, and the v2 gate at L=0 is +11.4 [1.8, 21.3].
  v2 nulls: −0.07…+0.25 pp. market_compare was also run at 5 and 15 s (for DM).
- **Step 1, one-hot at international events** (`src/intl_onehot.py`, `results/intl_onehot.log`): names never help.
  - Intl log loss: one-hot 0.655 and one-hot + Elo 0.661, vs all features 0.630 and prior_new 0.604.
  - One-hot as a residual on prior_new: +0.003 (ns). C falls to the smallest value on the grid.
  - The international gap is about leagues: league Elo gives −0.027 on cross-league games.
- **Step 2** (`src/ingame_calib_gbm.py`, `results/ingame_calib_gbm.log`, `results/ingame_calib_minute.csv`, `viz/fig_calib_minute.png`):
  - Tuned GBM (grid over leaves 4–255, min_child 100/1000, colsample; early stopping on the last 91 days of train; picks 7 leaves and 742 trees): **+0.0051 [+0.0040, +0.0063] worse than v2**.
  - GBM boosting v2's residual stops at 1–3 trees (Δ 0.0000).
  - v2 per-minute calibration slope is 0.91–0.98, lowest at 15–20 min (0.913 [0.883, 0.942]). Intercept −0.14 at 0–10 min. ECE ≤ 0.03.
  - The GBM is better calibrated early but slides to 0.81 at 40+.
- **Step 3** (`src/significance.py`, `results/significance.log`, `sig_backtest.csv`, `sig_dm.csv`):
  - **DM v2 − market** (per-match mean differential, Newey-West): t = −4.13 at 0 s, −3.83 at 5 s, −3.45 at 15 s, −2.51 at 30 s (p 0.012), −0.98 at 60 s; +2.66 at 120 s.
    v1 is significant only at 0–5 s.
  - **Backtest family**: 48 cells (4 signals × 6 L × clip/gate). The best is the v2 clip at L=0 (t 2.65, p 0.004), but **Holm 0.19, BH 0.17**. The v2 gate at L=0 has t 2.24 and Holm 0.58.
    Gate-only family: Holm 0.30.
  - **DSR** of the primary cell (per-game PnL, T=155, SR 0.17, skew 2.0, kurt 15): 0.56 with N=48 trials, 0.39 with N=264 (cells × θ). PSR vs 0: 0.993.
  - So the forecasting edge is significant and the trading edge is not.
- **Step 4**: `~/fyp/RESULTS.md` is written (LoL only). The FYP-wide RESULTS.md is on `hrt-reproduction`/`pmlive-deploy`, so a merge will need to join them by hand.

## Team Elo vs roster (2026-10-02, `src/team_resid.py`, `src/eval_team_resid.py`, `results/team_resid.log`)

- How it's used: `elo_team_{16,32,64}` (pregame.py) is keyed on esportsTeamId and never resets on roster changes. It sits in BASE next to `elo_play_*`,
  so it reaches v1 through `elo_team_32` and v2 through `prior_new`. pregame2's own team Elo only sets the expectation for `h2h_res`/`pp_res`.
- The premise holds on its own: when a team has kept ≤ 2 of the players from ~20 games back, team Elo alone is worse than player Elo alone (0.619 vs 0.609).
  With a stable roster it's better. Turnover is concentrated in the regional leagues (hitpoint, roadoflegends, arabian, nlc, ...).
- Literal fix (regress T on the roster's mean player Elo as of each game, trailing 365 days, refit every 250 games): the slope is 0.90–1.05, so the residual is about T − P.
  It adds +0.0000. With both Elos already in the logistic it's a reparameterisation, and T is still stale after a swap.
- Hierarchical Elo (strength = mean player rating + team residual u; one surprise updates both; u shrunk by (kept/5)^γ on roster change, half-life HL):
  - On its own it beats team Elo on its own (test 0.6283 vs 0.6331).
  - On top of base it gives −0.0001 [−0.0004, +0.0002]. Replacing team Elo with it: base +0.0002 [−0.0006, +0.0009], prior_new +0.0001 [−0.0004, +0.0006].
  - Every roster-overlap bucket is ns, except ov = 5 in base (+0.0017, worse).
  - Validation picks γ = 0 (no roster shrink), but only weakly (0.6151 vs 0.6155). Whatever the team term carries survives roster swaps (org/staff?).
- So the logistic already discounts team Elo through player Elo. The stale team Elo costs nothing measurable; no change to v2.
- Coach: not built. There's no coach-by-date source (getTeams lists players only; Leaguepedia Cargo was still rate-limited on 2026-10-02).
  The slot would be strength += c[coach] in build_hier.

## Background jobs

None.

## Next steps

1. Live feed latency (`src/live_lag.py`, during a live event). This decides whether any backtest cell is tradable. It was cancelled on 2026-09-28 by the user's decision, so ask before running it.
2. Pre-registered forward test of the v2 gate (θ=0.30) on markets after 2026-04-28. It needs the Polymarket live capture (`~/fyp/pm/`).
3. Retrain or recalibrate monthly (the slope decays about 0.02 per month).
4. Merge `RESULTS.md` with the FYP-wide one when the branches come together.
5. v4 (below): the validation-selected window set (H, S, G, R) is worth −0.0007 on test and ns live. Swapping it into lollive is the user's decision.
   HUD OCR (priority 1) is the remaining untested input. Any new live feature must pass `v4/test_features.py` (dense-vs-60 s parity).

## v3: broadcast minimap + objective status (2026-10-05, `src/v3/`, results `results/v3_*`)

User request: v3 = v2 + the live broadcast's minimap (champion and ward positions only) + whether drake / baron is up.
Everything is in `src/v3/` (run from `esports/src` as `python v3/X.py`); data in `data/v3/`.

**Objective status (feed only, all 22,372 games; `objectives.py`, `eval_objectives.py`, `results/v3_objectives.log`).**
Spawn rules checked on first-kill times: drake 5:00 (+5:00 respawn), soul -> elder +6:00, baron 20:00 (2024, 2026) / 25:00 (2025), +6:00.
- Added to v2 (up x gold, up x alive, soul point): **Δ −0.0000 [−0.0001, +0.0001]**. No mean-forecast value.
- But it predicts **volatility**: the mean |next-minute change in v2's logit| is 0.53 baseline, 0.64 with drake up, **0.95 with a soul-point drake up, 1.32 with elder up**.
  Baron up: 0.55 (a long state, so it barely moves this). This is the swing the user expected, and it is a risk/gating input, not a direction.

**Minimap pipeline.**
- `vod_index.py`: getEventDetails vods give YouTube VODs with `firstFrameTime` for 11,165 / 22,523 games. VOD second = feed ts − firstFrameTime.
  Checked on the broadcast timer in 4 games: the broadcast clock trails my feed-clock estimate by 12–19 s, consistently (that's the 60 s clock anchor's bias), so the frames are in sync.
- `grab.py` / `extract.py`: one ffmpeg seek per 60 s feed frame on the DASH url (~4 s each); keeps the 360x360 bottom-right corner. Probe 3 frames first.
- `minimap.py locate()`: every 2024–26 Riot broadcast uses the same stylised minimap, bottom right, 225–280 px. Multi-scale NCC vs `ref/background.png`:
  game frames score 0.62–0.71 (25 VODs, 11 leagues), desk/draft/black frames 0.22–0.27. Threshold 0.45. Instant replays are not detected (they look like the game).
- Champion icons: **template matching failed**. Data Dragon icons, even with a decoy check and a foreground gate, put champions on empty map:
  the broadcast minimap has fog-of-war shading that changes every frame, and the encode blurs ~20 px icons.
  A classifier with the feed's dead champions as labels failed too (AUC 0.68), because the "alive" class was itself full of empties.
- **Fix: `synth.py` + `train_det.py`**. A small U-Net heatmap detector (blue / red icon centres) trained on synthetic maps:
  each game's median background, plus fog, camera box, minion/ward dots, ring icons from Data Dragon art, blur and JPEG. GTX 1080 Ti, 8,000 steps, 46 min, `ref/det.pt`.
  On 403 real games (`val2.py`, threshold 0.6), the per-team count matches the feed's alive count exactly 41 % of the time. Over-counts are 12 % (red base structures at p ~0.6). Under-counts are 47 %, mostly icons piled up in fights, and they grow late game.
  Identity (who is who) comes from Hungarian matching of the team's alive champions' icons to the peaks (`parse2.py`). The features don't use it.
- Wards: team-coloured 12–90 px² blobs off structures, against the game's own median background. Detected counts correlate only 0.23–0.36 within game with the feed's
  wards placed in the last 120 s (validate.py, old parser). They're noisy.
- `features.py`: mm_vis, mm_adv (mean progress), mm_deep, mm_drk / mm_bar (champions within 0.12 of the pit), mm_group, wards (total, deep, at the pits),
  and pit presence x objective up.

**Results (400 test-period games, nearly all PM-linked; 5-fold match-grouped CV on top of v2's out-of-sample logit).**
- `eval_v3b.py` (v2 unpenalised, new terms L2): **no arm beats v2**. Pit x up −0.0002 [−0.0004, +0.0001] at best; minimap all ≥ +0.0002; obj + minimap with time
  interactions overfits (+0.0011 to +0.0064). `eval_v3.py` (plain sklearn, C = 1): v3 +0.0068 [+0.0039, +0.0101] worse.
- **But the minimap predicts who takes the objective that is up** (`eval_taker.py`, one pre-set spec):
  next drake/elder (2,327 frames, 399 games): log loss 0.618 -> 0.595, **−0.020 [−0.031, −0.009]**, AUC 0.702 -> 0.732. Champions at the pit are the biggest term (+0.45 per SD).
  Next baron (2,522 frames): −0.0046 [−0.0094, −0.0002], AUC 0.860 -> 0.862 (the feed already knows who is ahead).
  So the minimap sees objective fights before the feed does, but that knowledge is spent within a minute (the feed records the take). It does not move the resolution forecast.
  If it's worth anything, it's in a short-horizon trade ahead of the market's drake repricing. **Not tested yet**: does P_mm(taker) − P_feed(taker) predict the mid's move?

**Blocker: YouTube bot-blocked this IP** (HTTP 429, "Sign in to confirm you're not a bot") after ~400 games / ~14k seeks at 16 workers on 2026-10-05 ~06:30 UTC.
1,918 games were marked `bad` by the block. extract.py now retries `bad` without `reason: probe` and sleeps 60 s after a no-url game. Before resuming,
wait for the block to clear, or pass cookies (`yt-dlp --cookies`), and use fewer workers. The train-period games were never reached (the list puts PM test games first).
This also threatens live capture from this IP.

**Live (`live_minimap.py`, not wired into lollive).** Resolve a getLive stream (youtube / twitch) with yt-dlp, fetch the HLS playlist, take the newest segment,
decode it fully, and keep the last frame (seeking to the end breaks h264 references). The frame's wall time is EXT-X-PROGRAM-DATE-TIME + duration; on a test stream the newest segment was 3 s old, at 2.7 s per frame.
`MinimapStream(...).start().at(feed_ts)` gives features of the snapshot nearest the feed frame (±20 s). Untested on a Riot broadcast: whether a Riot stream's PDT clock
lines up with feed ts, and how far the broadcast trails the ~22 s-old feed.

### v3 round 2 (2026-10-05, after the user's notes)

- **Objective timers came from the feed all along** (objectives.py: cumulative drake/baron counts plus fixed respawn timers; nothing read off the screen).
  Live, kills are seen at feed resolution, so the timers are exact there. Detected icon counts were used only to validate the detector, never as model features.
- **Gold float** (`gold_float.py`, `eval_float.py`, `results/v3_float.log`). The feed has no current-gold field. `window` has totalGold and `details` has item ids,
  so float = totalGold − Data Dragon value of the items held (per patch; all ids resolved; 20,460 games). Float grows steadily from 0.3k to 3k per player
  (consumables and sell-back losses), and blue − red cancels that trend. 0.3 % glitch frames (empty item lists) are flagged.
  Added to v2 on the full split: **−0.0003 [−0.0006, +0.0000]**, mostly after 30 min (−0.0012). Per +1k float lead: −0.03 logit at 10 min vs +0.157 for +1k gold,
  so unspent gold counts for ~80 % of gold in items. A small, borderline gain.
- **Distances to the pit** (features.py DIST: red − blue mean distance of the 3 nearest detected champions, and of the nearest; × objective up).
  Taker test (400 games, CV): next drake, feed + distances −0.017 [−0.028, −0.006] (the nearest champion's distance is the top term); with presence counts too, −0.024 [−0.037, −0.012], AUC 0.70 -> 0.74.
  Baron −0.004 (ns). Resolution log loss (452 games, CV; `results/v3_eval_b2.log`): no arm beats v2 (best −0.0004 ns); interactions with time overfit.
- Extraction resumed with the user's cookies (`~/fyp/cookies.txt`, passed by grab.stream_url; never print it) at 6 workers, train-period games first (`ids_main2.txt`).
  `v3/run_when_ready.sh` (log `logs/v3_run_when_ready.log`) waits for 1,200 train games, then parses and runs eval_v3b in SPLIT mode (λ by CV inside train)
  -> `results/v3_eval_split.log`, `results/v3_taker_all.log`.

### v3 in the live test (2026-10-05)

- Frozen by `src/v3/build_live_v3.py` into `~/fyp-pmlive/lollive/data/v3/`. Stage A (float, 20k games): v2 logit slope 0.949, intercept −0.035
  (a mild recalibration, like v2f's), plus float x (1, m, m^2). Stage B (463 minimap games): the stage-A logit as a fixed offset, plus the minimap terms
  (dist x up, pit x up, dd_miss), L2 0.1, all |coef| <= 0.025 per SD (a first fit with a free intercept recalibrated v2 by −0.17 on these games, so it was dropped).
  Taker models (feed + minimap, 2,784 / 2,811 frames) in v3_takers.pkl, logged only.
- Live capture: yt-dlp is refused on live YouTube even with cookies, and the YouTube embed / watch page fail in headless Chromium from this IP. Twitch and SOOP
  play at 1080p in headless Playwright Chromium, and the minimap crop locates (0.73–0.79 on the Demacia Cup SOOP stream).
  Latency from HLS PDT: Twitch 3.5–6.5 s, SOOP ~9 s.
- lollive: `src/capture.py`, `src/v3live.py`, `src/v3_synccheck.py`, strategy `v3_gate_guard`, tick columns q_v3 / q_take_* / v3_log; see lollive README "v3".
- Demacia Cup VIT–GAM game 2's livestats feed stopped at 08:00 UTC (10 frames), so the first v3 ticks come from a later game.

## Overfitting check: why live disagrees with the backtest (2026-10-05, `src/overfit_check.py`, `src/v3/overfit_v3.py`, `results/overfit_check*.log`, `results/overfit_v3.log`)

- **v2 is not overfit.** 52 params on 290k train frames: train in-sample 0.4333, train 5-fold CV 0.4345, and test 0.4255. Test by quarter: 0.427 / 0.444 / 0.425 / 0.421 / 0.427, with no decay.
- **The backtest edge sits in frames and leagues that the live test doesn't see** (backtest at the live lag, 22 s, v2 minus the market):
  - All PM frames vs the last trade: −0.013 [−0.023, −0.004] (reproduces).
  - Frames with no two-sided trading in the last 60 s (20 %, last trade median 21 s old): **−0.026 [−0.039, −0.015]**. Two-sided frames: **+0.002 [−0.008, +0.012]**.
    On those frames, a mid proxy (mean of the last taker-buy and taker-sell prints, from pm_fills.D) gives v2 minus mid +0.003 ns. Last trade vs mid itself differs by only +0.001.
    So v2 beats a *stale* last-trade price; it doesn't beat a live two-sided price. The live tracker always scores against the book mid.
  - By league, the edge is in thin markets: arabian −0.20, hellenic −0.11, les −0.08, lcp −0.03. In the major leagues the market is ahead: lck +0.010, lec +0.022, worlds +0.025,
    **emea_masters +0.023, lcs +0.037**. The live leagues (emea_masters, lcs, cblol) together give **+0.021 [+0.003, +0.039]** in the backtest, against +0.029 live.
  - Resampling 69-game sets from the backtest's two-sided frames, P(mean ≥ the live +0.029) = 0.11. The live gap is consistent with the backtest.
- **Live, with vs without a real prior** (per-game tick log loss minus the mid; 75 resolved games):
  - All: v2 +0.051 [−0.012, +0.122]. Real prior (team_games_min ≥ 20, 69 games): v2 **+0.029 [−0.019, +0.084]**, v1 +0.025, gold-only (no prior) +0.068 [+0.014, +0.132], prior alone +0.19.
  - No real prior (the 6 Asian Games games): v2 +0.30, v1 +0.33, gold-only +0.38, and the coin-flip prior alone +0.10.
  - EMEA Masters with a prior: v2 +0.029. Other leagues: v2 +0.029, v1 +0.003, gold −0.004 (24 games). Minute ticks only: +0.028. v2f +0.028 vs v2 +0.023 on its 32 games, so v2f doesn't help.
- **v3**: there are no live q_v3 ticks yet (the capture hasn't produced any), so v3 can't be scored live. Historically:
  - Stage A (recal + float, 13 params, 19k games) isn't overfit. On test, v2 is 0.4255, fit-on-train 0.4252 and the live fit-on-all 0.4250.
  - Stage B (minimap, 9 params, 463 games) improves only in sample: stage A 0.4521, in-sample 0.4517, 5-fold CV 0.4522 at L2 0.1, and 0.4528 (= v2) unregularised.
    The minimap terms are noise for resolution (a mild overfit, kept small by the L2), so live v3 ≈ v2 recalibrated + float.

## v2-major-leagues (2026-10-05, `src/v2_major.py`, `results/v2_major.log`, `results/backtest_v2major.csv`, model `data/v2_major_leagues.pkl`)

v2's in-game logistic refit on major-league train games only (lck lpl lec lcs lta_n/s/cross cblol lcp pcs vcs ljl lla + worlds msi first_stand ewc americas_cup;
EMEA Masters is tier 2, so excluded): 3,692 train games instead of 8,783. The prior stays v2's all-league prior_new. Ablation `+prior`: the prior refit on major games too.
**It is no better anywhere.**
- Test split, per-game log loss vs v2: all +0.0011 [+0.0006, +0.0017], major games +0.0008 [−0.0000, +0.0016], non-major +0.0013, EMEA Masters +0.0007 ns.
  With the major prior too: major +0.0002 ns, EMEA Masters +0.0091 (worse).
- Vs the market: identical to v2 within ±0.0012 in every cell (lag 0/22, last trade / two-sided mid, all / major).
  On major PM games at 22 s, v2 is +0.003 and v2-major-leagues +0.004 vs the last trade, both ns.
- Trading backtest (eval from 2026-02): lower in 15 of 16 (L, scope, strategy) cells, e.g. the gate at L=0 on all games 11.5 vs 11.4 pp gross but 6.1 vs 7.4 net in the fee regime;
  major-only gate L=0 12.1 vs 13.2.
- Live (rescored from lollive frames60 with the stored prior_new): the saved feed is coarser than the trader's, and only 697 ticks fall on a saved frame. On those, the recomputed v2 equals the stored q_v2 exactly.
  v2-major-leagues minus v2 is **+0.0023 [−0.006, +0.009]** (real-prior games); on all ticks with as-of features it is −0.0009 [−0.008, +0.006]. Only 11 live games are major-league.
- Reading: in-game dynamics (gold, objectives → win) transfer across leagues, so dropping 58 % of the data costs more than the league match gains.
  The market's edge in major leagues is about price quality (liquidity, stale prints), not model fit.

## Missing inputs and feature construction: ideas (2026-10-05; priorities 2–5 implemented and tested in "v4" below)

Context: live v2 trails the mid by ~0.029 on real-prior games. On two-sided frames the backtest attributes ~0.01 of that to feed latency
(v2 − mid −0.007 at 0 s → +0.003 at 22 s). The rest is information the market has: the live leagues already favour the market in the backtest (+0.021 at 22 s).
Literature: in-play prices absorb big events within seconds (Croxson & Reade 2014). Guan & Tung 2026 (arXiv 2609.08060) find a pre-game model ties Polymarket,
with a modest market edge on cross-region Worlds and series-decider maps.

**Literature inputs vs ours.** Already used: gold, kills, towers, drakes, barons, inhibs, level, CS, gold per role, Elo/skill, draft/champion strength.
Tested and null: sequence models (LSTM/Transformer), per-minute models. Not used:
- per-player K/D/A and damage (Jalovaara 2024; in our feed / details60 but unused)
- voidgrubs, Herald, Atakhan, Feats of Strength (Jalovaara; not in the fields we collect)
- buff/respawn timers (Riot/AWS-style set; only v3 flags)
- schedule-assigned first-pick covariate and side-specific (same-side) EWMA form (Guan & Tung)
- Hodge-style 5-min deltas on top of v2 (only tested standalone)

**Inputs the market has that we lack:**
- A. Fresher state: OCR of the broadcast HUD (Twitch 3.5–6.5 s vs the feed's ~22 s; the v3 Playwright capture already exists). It also gives grubs/Herald/plates.
- B. Fight in progress: p*_currentHealth/maxHealth (collected, used only as alive); a recent kill / HP-drop indicator to widen uncertainty.
- C. Who is dead and for how long: respawn timer from level + clock.
- D. Item power spikes: completed legendaries from details items + Data Dragon (only float used so far).
- E. Comeback mechanics: shutdown bounties from per-player streaks; objective bounties above gold-lead thresholds.
- F. Pre-game:
  - the market's pre-game price as the prior, or as an override when team_games_min is small (it would have fixed the Asian Games games)
  - decider-map flag, map number, fearless-draft pool depletion, first pick vs side, the side the previous loser chose
- G. Market state: momentum and taker imbalance. q_v2 − mid already predicts the mid's 1–5 min move; a market-anchored forecast is a different target.
- H. Broadcast-only: summoner/ult cooldowns, TP flanks (hard).

**Construction issues (x → f(x)) in `design()` / `live.features()`:**
1. The x·(1, m, m²) time basis extrapolates (m² = 16–25 at 40–50 min) → clipped B-spline / piecewise knots at 10/15/20/25/30.
2. Gold is linear in the logit → gold / tot_gold (tot_gold is computed but unused) or log ratio, plus a saturating term.
3. Cumulative counts stand in for transient state → inhibs currently down (5-min respawn), baron/elder buff active + time left, towers by tier / base exposed.
4. Diffs hide races → per-side drake counts / distance to soul, and drake type (already in b_drakes).
5. alive → Σ dead players' respawn seconds left × gold share (computable, no new data).
6. kills duplicate gold → bounty gold at risk.
7. Level sum diff ignores the 6/11/16 thresholds and the 2026 top level cap → per-role level or XP.
8. The prior fades with time, not with decidedness → prior × |gold share| or |state logit| (untested; composition × gold was null).
9. Clock anchor:
   - the broadcast clock trails our estimate by 12–19 s (v3 sync check)
   - 2026 games start ~35 s earlier (minions 0:30), but live.features anchors on passive gold at 1:05
   - v3's fixed objective timers depend on this
   - verify on a 2026 broadcast
10. Rule eras: the live model was trained ≤ 2025-06. 2026 removed Atakhan and Feats of Strength, added role quests (bot 7th item slot, higher top level cap) and changed plates.
    Recent refits moved LL only ±0.0015, but a season × gold/level interaction is untested.

**Priority:**
(1) HUD OCR for fresher state + missing objectives;
(2) transient-state features + spline time + relative gold (cheap, data on disk);
(3) HP fight-in-progress features (check with the event-study window);
(4) market pre-game prior / unknown-team guard + fearless/first-pick/decider/same-side form;
(5) bounties, item spikes, deltas (likely small, given the null wards/float/composition results).
Test each on the same split vs v2, then vs the market at 0 and 22 s, then live on the same ticks.

Sources:
- Guan & Tung 2026 (arxiv.org/pdf/2609.08060)
- Jalovaara 2024 Aalto thesis (sal.aalto.fi/publications/pdf-files/theses/mas/tjal24a_public.pdf)
- Junior & Campelo (arxiv 2309.02449)
- Riot/AWS WP dev diary (lolesports.com)
- Hodge et al. (eprints.whiterose.ac.uk/152931)
- Croxson & Reade (centaur.reading.ac.uk/34884)
- LoL 2026 changes (esports.net, sheepesports.com)

## v4: the missing-input candidates, tested (2026-10-05, `src/v4/`, results `results/v4_*`)

Priorities 2–5 above are implemented. Priority 1 (HUD OCR) is not: it needs broadcast video, which YouTube blocks from this IP.
Protocol as planned: each candidate vs v2 on the same split, then vs the market at 0 and 22 s, then live on the same ticks.
Run from `esports/src` as `python v4/X.py`. Data is in `data/v4/`. `run_all.sh` runs the build and test-split steps.

**Code.**
- `features.py` gives `data/v4/feat.parquet` (762,214 frames, 22,372 games). Groups:
  - T transient: inhibitors down, baron and elder buff time left
  - S soul race: nb² − nr², soul point, six drake types
  - R respawn: seconds left by level × time factor, plain and gold-weighted
  - H fight in progress: alive HP share, low-HP count, last-minute kill/gold change, kills × gold
  - D 5-min deltas
  - B bounties: streak and gold shutdown proxy, objective-bounty hinge
  - I items: completed legendaries from details60 + Data Dragon
  - L levels: 6/11/16 ranks, per-role level
  - G relative gold: gold share, tanh(gold / 5k)
  - P prior × |gold share| (built in eval)
  - All groups except I use `window` fields only.
- `pregame_f.py` gives group F: match point, decider × Elo, side the previous loser chose, same-side form (residual and EWMA),
  fearless pool breadth / pool × map / comfort already spent, and `pm_pre`, the market's pre-game price (VWAP of the last 30 min, 1,520 games).
  Fearless detection: no team repeats a champion within a series in any league from 2025 (and NACL 2024).
- `eval_pregame_f.py` (protocol of eval_pregame2) gives `data/v4/prior4.parquet`.
- `eval_v4.py [12534]`: 1 test split, 5 validation-selected set, 2 market, 3 backtest, 4 live. Writes `data/v4/models.pkl` and
  `data/ingame_test_preds_v4.parquet`, and backs `PREDS_SFX=_v4 backtest.py`.
- `test_features.py`: respawn values, plus train/serve parity (below).

**Two bugs found and fixed on the way.**
1. **Train/serve skew.** The live tracker's saved feeds are dense (median 0.2 s between frames), but training frames are 60 s apart.
   Row-to-row deltas (dg1, dk1, fight, "died since the previous frame") therefore spanned ~0.2 s live: live dg1 had SD 0.08 vs 0.96 historically.
   - Fix: these windows are now time-based (back to the latest frame ≥ 50 s older). Death times are bracketed by the frame that first shows them and the one before.
   - On 60 s data the values are the same as before; live they are exact.
   - `test_features.py` computes dense vs thinned-to-60 s features at shared timestamps: every feature has corr ≥ 0.967 or is identical.
   - Any future live feature must pass this test.
2. **Unmatched tick sets in the live comparisons.** The stored `q_v2` is NaN on 12 real-prior ticks (early, before the prior is ready).
   `pgll` dropped them for v2 and scored them for everything else. `eval_v4.compare` now scores only rows where both columns exist.
   - Effect on earlier claims: live v2 − mid (all real-prior ticks) +0.029 → **+0.030**; on exact-frame ticks +0.0295 → **+0.0332**.
   - v2-major-leagues minus stored v2 on exact-frame ticks: **+0.0023 → −0.0017** [−0.0098, +0.0051]. Still ns, so the v2_major conclusion stands, but the sign flips.
   - `overfit_check.py` and `v2_major.py` still use the unmatched version.

**Pre-game F** (`results/v4_eval_pregame_f.log`; test log loss vs v2's prior set, 0.6043):
- series terms +0.0007 [+0.0004, +0.0010] (worse); same-side form −0.0002 (ns); fearless +0.0004 (ns); **all F +0.0006 (ns)**.
- On fearless game 2+, all F is +0.0017 [+0.0004, +0.0028] (worse).
- Signs are sensible but tiny: pool breadth +0.13 per SD; prev_loser −0.035 (the loser picks the side it wants and still loses slightly more).
- As the in-game prior (`prior_f`): +0.0002 (ns).
- **Market pre-game price** (1,520 Polymarket-linked test games):
  - on its own 0.6019 vs the v2 prior 0.5978 (+0.0041, ns)
  - **50/50 logit blend −0.0071 [−0.0145, +0.0007]**; CV-fitted blend −0.0049 (ns; weights v2 0.62, market 0.45)
  - unknown-team guard (market when n_min_games < 20; 228 games): +0.0013 (ns). On those games the market is no better than the prior: they still have Elo history, unlike the Asian Games teams.

**In-game, test split** (`results/v4_eval.log`, per-game log loss minus v2 0.4255, match-clustered 95 % CI):

| Arm | Δ vs v2 | Notes |
|---|---|---|
| +H fight in progress | **−0.0011 [−0.0014, −0.0009]** | −0.0075 on the 7 % of frames with an HP-share gap > 0.25; −0.0020 after a kill in the last minute vs −0.0006 otherwise; −0.0009 to −0.0015 in every quarter, −0.0016 in the live leagues |
| +D 5-min deltas | −0.0006 [−0.0008, −0.0003] | gold momentum +0.46 per SD at 10 min; kill/tower deltas negative given gold |
| +T transient | −0.0004 [−0.0005, −0.0003] | inhibitors down +0.27 per SD; baron buff +0.11; all of it after 30 min |
| +I items | −0.0002 [−0.0004, −0.0000] | |
| +R, +L, +G, +P, +B | −0.0001…+0.0002 (ns) | gold share's slope grows from +0.35 to +0.97 per SD while tanh(gold) turns negative, so a late lead counts as a share, but the fit gains nothing |
| +S soul race / drake types | **+0.0006 [+0.0002, +0.0010]** | helps on validation (2025Q2), hurts on test, +0.0038 at 30+ min: drake meta drift |
| spline time basis | +0.0002 [+0.0000, +0.0004] | the m² extrapolation does no measurable harm |
| all | −0.0013 [−0.0020, −0.0006] | picked after looking at test |
| all window (no I) | −0.0012 [−0.0018, −0.0006] | |
| **validation-selected** (greedy on the last 91 train days: H, S, G, I, R) | **−0.0009 [−0.0014, −0.0003]** | the honest estimate |
| validation-selected, window only (H, S, G, R) | −0.0007 [−0.0012, −0.0001] | what live can compute |

Most of each gain is after 30 min (per frame: all −0.0047, H −0.0047, T −0.0029, D −0.0029).
Construction 9, clock anchor:
- Fastest first-drake/baron kills on our clock show no 2026 shift at 60 s resolution (1 % quantile 364 s every season; baron 1,265 s in 2024 and 2026).
- At 10 s resolution, 2026 kills come later than 2025 relative to spawn (drake 346 vs 319 s). The feared 35 s-early start would make them earlier. Not a broadcast check.
Construction 10 (season interaction) can't be tested on this split, because train has no 2026 games.

**Vs the market** (`results/v4_market.csv`):
- Arm minus v2 on PM frames is about twice the all-games gain: H −0.0018 [−0.0024, −0.0012] at 0 s and 22 s; all −0.0020 [−0.0039, −0.0002] at 0 s; S +0.0015 (worse).
- Against the mid proxy on two-sided frames at 22 s: v2 +0.0031, H +0.0014, all +0.0014, validation-selected window +0.0027 (all ns).
- So the features close at most ~0.002 of a gap that is ~0 on two-sided frames and +0.030 live.

**Backtest** (`results/backtest_v4.csv`; gate θ chosen on the selection months; eval from 2026-02):

| L | v2 | all | all window | validation-selected |
|---|---|---|---|---|
| 0 | +11.4 [1.8, 21.3] | **+14.4 [5.4, 22.8]**, fee net 11.8 | +12.2 [3.7, 21.2] | +11.2 [3.8, 19.0] (705 trades) |
| 5 | 8.7 (ns) | 11.8 [1.5, 21.6] | 10.4 [2.1, 18.6] | 9.0 (ns) |
| 15 | 5.5 | 7.6 [0.9, 14.0] | | 7.9 [1.6, 14.3] |

- Clip is unchanged (2.5–2.6 pp).
- "all" was picked post hoc and adds 3 signals to the 48-cell family, so this is not new significant evidence. The validation-selected set equals v2 at L=0.

**Live** (`results/v4_live.log`; 69 real-prior games, rescored from the saved feeds with the stored prior, matched ticks):
- Against v2 recomputed on the same as-of features, every window arm is ns: +H +0.0015 [−0.0001, +0.0030], all window +0.0017, selected window +0.0028 [−0.0019, +0.0074].
- On the 697 exact-frame ticks: −0.0005…+0.0013 (ns), spline +0.0025 [+0.0003, +0.0052].
- 69 games can't resolve a ±0.001 effect, so live neither confirms nor refutes the test-split gains.

**Reading.** The cheap feed-side features are real but small:
- fight state, momentum and transient structure state are worth ~0.001 log loss (0.002 on PM games), almost all of it late or mid-fight;
- the draft/series/fearless pre-game additions and the soul/drake-type terms are noise or drift.

None of it moves v2 materially against a live two-sided price. Fresher state (HUD OCR, priority 1) is still the only listed input that addresses the feed's ~22 s lag.
The live tracker is unchanged: swapping its model mid pre-registered test is the user's call.

### v4 in the live tracker (2026-10-05)

- Frozen by `src/v4/build_live_v4.py` into `~/fyp-pmlive/lollive/data/v4/v4_model.pkl` (`sel window`: H, S, G, R; test 0.4248 vs v2 0.4255).
  `features.py` is copied verbatim to lollive `src/v4feat.py` (its `ingame_data` import is now lazy so the copy runs on its own).
  Checks: the test predictions reproduce exactly, and the live code path reproduces eval_v4's rescoring to 1e-15 (`results/v4_build_live.log`).
- Clock caveat for every offline live rescoring (eval_v4 part 4, v2_major part 4): the saved feeds lack the trader's backfilled early frames, so
  live.features on a saved file re-anchors the clock differently in 124 of 157 sampled ticks (games joined mid-game). eval_v4/v2_major got the
  design's clock from the tick (merge suffix) but the window features' clock from the file. The trader itself is unaffected.
- lollive: `src/v4live.py`, ticks `q_v4` / `v4_log`, strategy `v4_gate_guard` (exploratory), replay check `src/v4_replay.py`; see lollive README "v4".
  The trader was restarted at 11:34 UTC. Today's Demacia Cup feeds stop after ~10 frames (no ticks since 10-04 23:58 for any model), so the first live q_v4 ticks will come from the next league with a working feed.

## Reddit "I built a model to predict Worlds 2026" model vs v4 (2026-10-12, `src/reddit/`, results `results/reddit_*`, `results/backtest_reddit.csv`)

User request: test the r/leagueoflegends post's model (text pasted by the user; reddit blocks this IP) against v4 in the backtests.
The post's recipe:
- Layer 1, player ratings: a Kalman win/loss rating adjusted for margin of victory and internationals; lane performance residualised on all 10 champions;
  damage share residualised the same way; team and lane-matchup Elos.
- Layer 2: a Bradley-Terry team rating with domestic weighting.
- Stack: a logistic regression, trained 2023 onward and refit monthly. Pre-draft and post-draft versions.

Their claims: post-draft game log loss 0.5896 (2025-09 onward), beating Polymarket pre-game by ~0.01–0.016, tying Kalshi.

**Rebuild** (`build.py`, all causal):
- Kalman rating: an EKF on a Bradley-Terry logit with team strength = mean of the 5 players. The update is weighted by end-gold margin and internationals, with per-day process noise.
  Tuned on validation (`results/reddit_tune.log`): q = 1e-3, v0 = 2, intl weight 3, MOV on.
- Lane rating: gd15 minus a monthly-refit ridge on 10 champions by role. Damage rating: end damage share minus a ridge on own and enemy champions. Both as decayed shrunk means.
- Team rating: weekly-refit weighted BT on team and home-league indicators (365 days, half-life 120 d, intl ×2).
- Post-draft (the post doesn't say which inputs): a champion-only win logit and the draft-expected gd15.
- Not reproducible here: their 100k games since 2013. Our history starts 2023-10, so the ratings get ~6 months of burn-in before train.

**Pre-game** (`eval.py 1`, `results/reddit_pregame.log`; test 10,584 games, log loss vs v4's prior (= v2 prior_new set) 0.6043):

| Model | Δ vs v4 prior |
|---|---|
| ratings only | +0.0108 [+0.0069, +0.0146] |
| pre-draft | +0.0083 [+0.0046, +0.0119] |
| post-draft | +0.0056 [+0.0017, +0.0093] (0.6099; 2025-09 onward 0.6113 vs v4 0.6060) |
| monthly refit | post-draft +0.0052; v4 prior **−0.0016 [−0.0026, −0.0007]** |
| LightGBM stack | +0.0101 |
| **union (v4 prior set + reddit post-draft)** | **−0.0046 [−0.0069, −0.0023]**; monthly −0.0060; internationals −0.008 (ns) |

- C = 0.003 is an interior validation optimum.
- Largest stacked coefficient: BT team rating 0.26 per SD; Kalman 0.13, draft 0.13.
- On their own the reddit features are worse than ours, but they carry information ours lack (the Kalman/BT ratings with MOV and international weighting are the likely source).
- Vs the Polymarket pre-game price (1,520 games):
  - market 0.6019, reddit post-draft +0.0054 (ns), v4 prior −0.0041 (ns), union −0.0072 (ns)
  - 50/50 blend market + v4 prior −0.0111 [−0.0200, −0.0031]
  - The post's "model beats Polymarket" does not reproduce for the rebuild alone.

**In-game** (`eval.py 2`, `results/reddit_ingame.log`; v4 = `sel window` reproduced exactly, test 0.4248):

| Arm | Δ vs v4 (all test games) | Internationals |
|---|---|---|
| reddit prior in place of prior_new | +0.0012 [−0.0007, +0.0032] | +0.0068 |
| reddit prior, monthly | +0.0013 | |
| **union prior** | **−0.0029 [−0.0042, −0.0016]**; −0.0045 at 0–10 min, −0.0010 at 30+ | −0.0055 (ns) |
| prior_new + reddit prior as a second term | −0.0015 [−0.0023, −0.0006] | |

**Vs the market** (`results/reddit_market.csv`):
- Union prior minus v4 on PM frames: −0.0025 (ns) at 0 s, −0.0023 (ns) at 22 s.
- Two-sided mid at 22 s: v4 +0.0027, union +0.0007, reddit prior +0.0068 (all ns).

**Backtest** (`results/backtest_reddit.csv`, gross pp per share, eval from 2026-02):

| L | v4 gate | reddit-prior gate | union gate | v4 clip | union clip |
|---|---|---|---|---|---|
| 0 | **13.8 [5.2, 22.2]** (332 trades) | 7.4 [0.0, 14.9] | 12.5 [4.8, 20.7] | 2.5 | 2.9 (fee net +0.25 vs −1.12) |
| 5 | 9.8 (ns) | 6.3 | 6.7 | 2.3 | 2.8 |
| 15 | 7.5 | 5.0 | 5.0 | 2.1 | 2.5 |
| 30 | 5.8 | 3.7 | 8.3 | 1.8 | 2.1 |

- The reddit prior alone is lower than v4 in every gated cell.
- The union is +0.3–0.4 pp on clip at every L, and mixed on the gate. CIs overlap entirely, and this adds 4 signals × 4 L × 2 to the family.

**Reading.**
- As published (in our rebuild), the model is a weaker pre-game forecaster than v4's prior and a weaker in-game prior. It does not beat the market.
- Its ratings do add to ours: union −0.0046 pre-game, −0.0029 in-game. That is the largest single in-game gain since prior_new, about 3× v4's feature groups.
- That gain does not reach the market comparison or the trading edge.
- Monthly refit helps v4's pre-game prior (−0.0016). It is untested as an in-game prior.
- Not tested: series-level forecasts, Kalshi, and their T1 Worlds adjustment (one Worlds in our test window).
- Candidates for a v5, the user's call: the union prior, refit monthly; re-scored live before any swap in lollive.

## v5 (2026-10-12, `src/v5/`, results `results/backtest_v5*`, `results/book_sim*`, `results/v5_*.log`, figures `results/viz/fig_v5_*.png`)

v5 = v4 (`sel window`: STATS + H, S, G, R, each × (1, m, m²)) with the union prior refit monthly. The union prior is v2's prior_new set plus the reddit features (C = 0.01).
- Train rows use the out-of-fold prior; test rows use the monthly refit.
- `v5/build.py` writes `data/v5/models.pkl`, `data/ingame_test_preds_v5.parquet` and 5 shuffled-outcome nulls.
- Test per-game log loss: **v5 0.4216, v4 0.4248, v2 0.4255; v5 − v4 −0.0032 [−0.0044, −0.0020]**.

**Historical execution backtest** (`v5/backtest_v5.py`; backtest.py's fills; eval 2026-02..04; θ chosen on the selection months):

| L | v5 clip | v5 gate | v4 gate | v2 gate |
|---|---|---|---|---|
| 0 | +2.4 [0.6, 4.1]; fee net −0.6 | +8.7 [2.2, 15.3] (θ 0.20, 1,284 trades); fee net +4.9 | **+13.8** (θ 0.30) | +11.4 |
| 5 | | +8.3 | 9.8 | |
| 15 | | +7.1 | 7.5 | |
| 30 | | **+8.2** (v5 best) | 5.8 | 7.0 |
| 60 | | +4.5 | | |
| 120 | | +1.8 | | |

- At θ 0.30 v5 is +13.3, so most of the L=0 gap is the selection's θ choice.
- Sharpe, v5 gate: per game 0.13 at L=0 and 0.13 at L=30; annualised daily net 4.2 at L=0 and 3.5 at L=30. v5 holds up best at L ≥ 30 s.
- Clip: net Sharpe ≤ 0 everywhere.
- Controls:
  - nulls −0.13…+0.25 pp (clip) and −0.02…+0.15 (gate)
  - placebo +0.46 ± 0.35
  - comb5 (mid + v5) is worse than v5
- **DSR of v5 gate L=0: 0.44 (N=48), 0.32 (N=96)**. Not significant, like v2/v4.
- Profit by minute: 2.9 (0–10), 9.3, 14.2 (20–30), 7.2 (30+).

**Recorded order books** (`v5/live_period.py`, `v5/book_sim.py`, `results/book_sim.log`):
- **Data:** the self-captured Polymarket minute bars (`~/fyp-pmlive/data/pmlive`; book at each minute's last update), 2026-09-17 to 10-12.
- **Games scored:** all games 2026-09-17..10-11, 527 in total. The 279 after the study's tables were downloaded fresh as 60 s feeds (`data/v5/frames60`) and labelled by lollive's games_new.
  - The causal pregame/pregame2/reddit pass on the extended table reproduces the study's test priors exactly.
  - Live-period log loss: v5 0.4109, v4 0.4155, v2 0.4149.
- **Linking:** 245 games to captured per-game winner markets (lollive's matcher). The resolution check uses the price from game end to +2 h, because the last captured bar is often a frozen post-close book.
  1 contradiction was dropped; 189/189 agree at game end.
- **Clock check:** bars vs lollive's own tick snapshots give median |mid diff| 1.5 c, minimised at offset −1/0 min, so the clocks agree.
- **Live-tradable scope:** WSCI and stats-disabled games have no live feed, so the headline uses 142 games with a two-sided book (median spread 1 c).
- **Execution:** the book ending the decision minute (conservative), 1 c walk past the touch, fee 0.05 p(1−p), $50 per decision, L = 22 s.

| Strategy (thresholds fixed in advance) | v5 net pp [95 % CI] | v4 | v2 | v5 Sharpe per game / annualised |
|---|---|---|---|---|
| clip | −0.6 [−5.0, 4.0] | −1.7 | −1.7 | −0.05 / −1.8 |
| gate θ 0.30 | +6.2 [−18, 30] (152 trades, 50 games) | +11.8 | +14.6 | 0.17 / 3.7 |
| gate θ 0.20 | +7.1 [−8, 23] | +3.0 | +1.2 | 0.17 / 4.5 |
| gate 0.30 + team guard | +1.7 [−13, 18] | +4.6 | +9.1 | 0.17 / 4.6 |
| convergence exit | −2.5 [−3.6, −1.5] | −2.7 | −2.7 | −0.30 / −11.3 |
| endgame (q ≥ 0.99, ask ≤ 0.985) | +5.8 [2.2, 12.6], 71 trades, 100 % hit, only $393 | ≈ | ≈ | 0.31 / 5.1 |
| random (control) | −0.5 [−1.8, 0.9] | | | −0.11 / −5.3 |

- **Gate profit is a few games.**
  - The top 3 games are ≥ 100 % of the v5 gate's net $.
  - By league: EMEA Masters −11 pp (22 games), LEC +59, LPL +39, CBLOL +44.
  - Including the untradable WSCI games inflates v5 gate30 to +16.5 and v2 to +27.3 (scope `all`).
- **Sensitivity:**
  - L=0: v5 gate30 +9.1, clip −0.8. L=60: +5.1, clip −1.2.
  - The optimistic quote (last book before the decision minute) lifts everything, e.g. conv −2.5 → +1.2 and v5 gate30 → +14.1. Minute bars can't pin the fill price within the minute.
- **Forecasting on the same frames:** the captured mid beats every model (per-game log loss mid 0.414, v5 0.435, v4/v2 0.440). This repeats the live tracker's finding.
- **Reading.**
  - v5 is the best forecaster offline and in the live period, but no strategy's PnL is distinguishable from zero except endgame, which is tiny.
  - Endgame is identical across models.
  - The ranking of models flips between strategies and latencies.

Figures:
- historical: `fig_v5_hist_latency`, `_cum`, `_sharpe`, `_null`
- books: `fig_v5_book_cum`, `_sharpe`, `_pp`, `_sens`

Rerun: `python v5/build.py nulls && python v5/backtest_v5.py && python v5/live_period.py && python v5/book_sim.py && python v5/fig_v5.py` (from `esports/src`).
Not done: v5 is not wired into lollive (the user's call).

## Why the market beats the models live (2026-10-12, `src/gap/`, results `results/gap_analysis.log`, `results/gap_event_lag.log`)

User question: the market out-forecasts our models in the live leagues. What is the model missing?
Data:
- `gap/panel.py`: v5/live_period's 60 s frames for the 243 games linked to captured per-game markets, of which 203 are tradable (142 with every offset).
  Each frame gets the captured mid as-of minute offsets k = −5..+3 and `mid_same`, the mid interpolated in logit to the frame's own wall time.
  It also gets the market's pre-game price `pm_pre` (median mid over the 10 min before the first frame).
- `gap/analyze.py`: sections 1–6 below.
- `gap/event_lag.py`: second-resolution event study.
  It uses the tracker's dense saved feeds (122 games, 2,942 events) and the captured pmlive fills.

**1. Timing, not modelling, at equal information.** On the 142 tradable games:
- v5 minus mid_same is **+0.005 [−0.031, +0.038]**, a tie. v2 and v4 are +0.011.
- Each minute of market freshness is worth ~0.013 log loss. mid_−1 (the market 0–60 s older than the state) is +0.010 vs mid_same, and mid_0 is −0.004.
- book_sim's "mid beats every model" (0.414 vs 0.435) quotes the market 22–82 s after the state, so most of that gap is the clock.

**2. By phase, the two win in different halves.**

| Minutes | v5 − mid | Encompassing weight: market | v5 |
|---|---|---|---|
| 0–10 | +0.031 | 1.08 | 0.00 |
| 10–20 | +0.027 | 1.06 | 0.03 |
| 20–30 | −0.019 | 0.43 | 0.79 |
| 30+ | −0.041 | 0.23 | 0.77 |

- Before 20 min, the market fully encompasses v5. After 20 min, v5 carries most of the weight.
- Cross-league games (home leagues differ: EMEA Masters, promotion, Demacia Cup, ...): **+0.045 [+0.012, +0.077]**. Same-league: −0.030 (ns).

**3. The early-game gap is the pre-game prior on cross-league games.**
- Live (135 games): market 0.557 vs v5 prior 0.588.
  - Cross-league **+0.068 [+0.004, +0.136]**; same-league −0.004.
  - Encompassing: market 1.16 (0.45), prior 0.01 (0.44).
  - Where they disagree by > 0.5 logit, the market's side won 70 % (40 games).
- Historical (1,520 study-test games): same-league prior − market −0.016 (the prior is better); cross-league +0.012; difference +0.028 [−0.002, +0.060].
- Reading: same direction, borderline. Team strength *across* leagues (ERLs meeting at EMEA Masters, academy vs main, cross-region) is where our ratings are weakest. league_elo only updates on the rare inter-league games.

**4. Counterfactual: the market's pre-game price as v5's prior** (frozen coefficients):
- Overall: v5 +0.007 → **−0.002** vs mid; 50/50 blend −0.001.
- Cross-league: +0.049 → +0.020. 0–10 min: +0.032 → +0.009.
- 10–20 min stays +0.023 (ns), so not all of the early gap is the prior.
- Same direction as the historical 50/50 blend (−0.011 vs the market, v4 section).

**5. Implied weights** (OLS of logit price on the state, by phase):
- Compared with v5, the market puts much less weight on early gold (0.29 vs 0.74 per k at 0–10 min), drakes, HP share and alive count. It puts more weight on kills.
- The market keeps its own opening line at weight 0.44–0.81 all game. v5's prior fades from 0.91 to 0.30.
- That persistence is right early and wrong late (section 2).
- Who is right per feature can't be resolved on 142 games, except gold (section 6).

**6. 2026 drift in the value of an early gold lead.**
- Offset-logit residual of v5 (trained ≤ 2025-06): gold −0.084 (0.042) + 0.035·m per k on 7.7k 2026 test games, versus +0.146 (0.082) − 0.046·m in 2025H2.
- In the live period: −0.40 (0.24) + 0.13·m.
- A correction fitted on 2026 games before the live period gives −0.003 [−0.007, +0.001] on the live period. It doesn't move the market comparison.
- Real but small: early leads convert less in 2026, as the market already prices.

**Event study** (`gap/event_lag.py`; price = signed blue-equivalent last fill, relative to −90 s; isolated = no event in the prior 90 s):

| Event | Price move by +120 s | Share done before the feed timestamp | Timing of the move |
|---|---|---|---|
| Isolated kills (249) | +10.2 c | 64 % | 10 % at −40 s, 50 % at −6 s, 90 % at +36 s |
| Isolated kills, EMEA Masters / LCS | | | 50 % at −8 s |
| Barons | | | 50 % at −44 s |
| Inhibitors | | | 50 % at −32 s |
| Isolated towers | ~0.5 c | | |

- The tracker receives a frame 22 s after its timestamp (median; feed_x 20–40 s).
- Demacia Cup and LCS promotion have feed_x 60–120 s, a median feed age of 71 and 101 s. That is Riot's publication delay; the tracker probes the minimum.
- So live, ~90 % of a kill's price move is done before our model sees it.
- The broadcast shows a fight developing (engage, HP, a baron being started) 10–40 s before the kill or objective lands in the feed.
  Untested: whether the feed's rfc460Timestamp also trails the broadcast. lollive v3_synccheck can settle it once minimap peaks are reliable.
  This contradicts the 2026-09-28 historical estimate ("market reacts 5–10 s after"), which was timed on 60 s frames.

**Trading split** (book_sim v5, L=22, post hoc, so exploratory):
- clip: cross-league **−6.8 pp [−10.6, −2.7]**, same-league +4.9 (ns).
- gate20: early −4.5, late (≥ 20 min) +18.3 [−1.1, +33.1]. gate30: late +19.7 (ns).

**What the model lacks, ranked:**
1. Cross-league team strength. Use the market's pre-game price in the prior, or improve inter-league ratings.
2. Fresher, pre-event information: the market front-runs the feed by 10–40 s on fights and objectives, plus our 22–110 s publication delay. Only broadcast capture (HUD/minimap) can address it.
3. 2026 recalibration of the gold value (−0.003).

At equal information, v5 already matches the market, and from 20 min on it beats it.

## v6: the "what the model lacks" fixes (2026-10-13, `src/v6/`, results `results/v6_*.log`, `results/backtest_v6*`, `results/book_sim_v6*`)

User request: build the ranked list above into v5, add v6 to the live tracker, and backtest it.
**v6 = v5's frozen in-game logistic with the union prior blended with the market's pre-game price.**
**v6m = v6 combined with the live mid, per game phase.** Item by item:

| Item | What was built | Result | In v6? |
|---|---|---|---|
| 1a cross-league ratings | union prior + `cross` and cross × {elo_team_32, elo_play_32, league_elo, bt, kal_z, form_10}, monthly refit (`v6/prior.py`) | validation 0.6045 vs 0.6038 (worse at every C); test +0.0005 [−0.0002, +0.0013], cross-league +0.0035 (ns) | no |
| 1b market pre-game price | z6 = z_u + (w0 + w1·cross)(logit pm_pre − z_u), fit on the 277 selection-month games (2025-07..2026-01) with an archive price: w0 −0.06 (0.18), **w1 +0.52 (0.57)** | eval months, games with a price −0.0010 [−0.0028, +0.0008]; cross-league −0.0019 [−0.0045, +0.0006]; live period with a captured price (165 games) −0.0023 [−0.0115, +0.0077] | yes |
| 1b' prior persistence | (g0 + g1 m)(z6 − z_u) in-game (the market keeps its opening line) | match-grouped CV on the selection frames 0.4462 vs 0.4417 without | dropped |
| 2 fresher information | v6m = σ(b_p logit(mid) + c_p logit(q_v6)), p = 0–10/10–20/20–30/30+ min, no intercept, per latency L (`v6/backtest_v6.py`) | see below; the only fresher public information is the market itself, since broadcast HUD/minimap capture was not built (YouTube blocks this IP for VOD training data) | as v6m |
| 3 2026 gold drift | offset-logit correction on (gold, kills, towers, alive) × (1, m), refit monthly on the previous 6 months | +0.0009 [−0.0001, +0.0018] on the eval months. 36 variants (`v6/drift_exp.py`, `results/v6_drift.log`: inputs, intercept, 3/6/12 months, ×0.5): the one best on the selection months (gold + intercept, 3 m) is the worst on eval (+0.0015) and live (+0.0028) | dropped |

- The cross-league price weight is where the gap analysis said it would be. w0 ≈ 0 means same-league games keep our prior; w1 ≈ 0.5 is a 50/50 blend on cross-league games.
- The analysis's −0.003 drift gain was in-sample hindsight (a correction fitted on 2026-01..04 and applied to the live period). Refit causally, it does not hold.

**Forecasting** (`v6/build.py`, `results/v6_build.log`; per-game log loss vs v5):

| Scope | v5 | v6 − v5 |
|---|---|---|
| Test split, 10,582 games | 0.4216 | −0.0001 [−0.0004, +0.0001] |
| Eval months 2026-02+, 7,085 games | 0.4199 | −0.0002 [−0.0005, +0.0002] |
| Live period 09-17..10-11, 527 games | 0.4109 | −0.0007 [−0.0036, +0.0025] |

- Only ~15 % of games have a pre-game price, so the overall effect is diluted.

**v6m weights** (archive, selection months, no intercept; mid weight / v6 weight):

| L | 0–10 min | 10–20 | 20–30 | 30+ |
|---|---|---|---|---|
| 0 s | 0.41 / 0.64 | 0.10 / 0.99 | −0.10 / 1.02 | 0.29 / 0.57 |
| 30 s | 0.49 / 0.54 | 0.07 / 1.01 | 0.02 / 0.90 | 0.61 / 0.25 |

- Weights fitted *with* an intercept picked up a −0.4 logit side bias from the selection months and lost out of sample (eval LL 0.5015 vs mid 0.4907), so the intercept is off.
- Archive eval-month frame log loss:
  - L = 0: v6m 0.4874, mid 0.4907, v6 0.4905, v5 0.4912.
  - L = 30: mid 0.4801, v6m 0.4830, v6 0.4902.

**Historical execution backtest** (`v6/backtest_v6.py`, `results/backtest_v6.csv`; backtest.py's fills, eval months 2026-02..04, θ chosen on the selection months; gross pp per share):

| L | v6 gate | v6m gate | v5 gate | v4 gate | v2 gate | v6 clip |
|---|---|---|---|---|---|---|
| 0 | **+12.7** [3.1, 23.0] (θ .30, 283 trades; fee net +9.4) | **+15.2** [3.3, 27.5] (θ .30, 166; fee net +13.4) | +8.7 (θ .20) | +13.8 | +11.4 | +2.5 (net ≈ 0) |
| 5 | +12.5 | +15.2 [4.4, 25.3] (θ .25) | +8.3 | +9.8 | +8.7 | |
| 15 | +7.8 | +9.9 | +7.1 | +7.5 | +5.5 | |
| 30 | +8.4 [0.6, 16.6] | +12.2 [−0.5, 24.7] (θ .25; fee net +14.3) | +8.2 | +5.8 | +7.0 | +1.8 |
| 60 | +3.5 | +3.9 (66 trades) | +4.5 | +2.9 | +2.9 | |
| 120 | +0.9 | −0.4 (52) | +1.8 | +1.5 | +2.3 | |

- The v6_x stage reproduces v5 exactly (check).
- v6's selection-month θ moved from 0.20 to 0.30. Part of the L=0 gain over v5 is that choice (v5 at θ 0.30 was +13.3).
- Annualised net Sharpe at L=0: v6 3.9, v6m 4.2 (v5 4.2). At L=30: v6 3.4, v6m 3.3.
- **DSR**:
  - v6m gate: L=0 0.87 (N=72 cells here), 0.84 (N=168 with the v2/v4/v5 families); L=30 0.75 / 0.70.
  - v6 gate: L=0 0.84 / 0.79; L=30 0.54 / 0.45.
  - The best yet (v5 was 0.44) but not significant.
- Placebo (another game's signal, same minute): +0.5 ± 0.4 pp.
- v6m gate profit by minute (L=0): 12.0 (10–20), 15.5 (20–30), 22.4 (30+). It barely trades before 10 min.

**Captured order books** (`v6/book_sim_v6.py`, `results/book_sim_v6.log`):
- Scored: the 79 tradable games 10-01..10-11 with both books, L = 22 s, next-minute quote. The 93 earlier captured games are the fit set for `q_v6m_cap`.
- v6m's signal mid is the book at the end of the minute *before* the decision minute, so it never sees the book it trades on.
- q_v6m uses the archive weights. That was decided before scoring, because the captured fit set gives implausible negative mid weights at 10–30 min.

Forecasting (per-game log loss minus v5):

| Signal | Δ vs v5 |
|---|---|
| v6 | −0.0064 [−0.0138, +0.0006] |
| **v6m** | **−0.0154 [−0.0257, −0.0053]** |
| v6m_cap | +0.0070 |
| prev-minute mid | −0.0299 |
| decision mid | −0.0438 |

- v6m − decision mid is +0.028 [+0.003, +0.054] (v5 +0.044). v6m closes about a third of the live gap. The market still wins.
- By phase, v6m beats v6 everywhere except 20–30 min (0.376 vs 0.374), most at 0–10 and 30+.

Trading (net pp, 79 games):

| Strategy | v6 | v6m | v5 |
|---|---|---|---|
| gate30 | −16.3 (61 trades) | −9.3 (37) | −17.4 |
| gate_guard | −4.2 | +2.9 | −6.8 |
| clip | −3.4 | −3.3 | −3.6 |
| conv | −2.0 | −2.1 | −2.1 |
| endgame | +2.7 [1.9, 3.7] | | |

- Random: −1.3.
- Every CI spans zero except conv (negative) and endgame.
- gateS (θ chosen on the fit set) picked 0.30 for every signal.
- Same reading as v5: the historical edge does not show on the live-era books. That is 79 games, so the CIs are ±20 pp on the gates.

**Live tracker** (`~/fyp-pmlive/lollive`, restarted 2026-10-13 08:22 UTC; README "v6"):
- `src/v6live.py`:
  - The union prior's rating passes (pregame, pregame2, `src/rbuild.py` = a verbatim copy of reddit/build.py) run in a spawned single-process pool, about 2 min per batch of live games, so they don't hold the trader's GIL.
  - The prior is refit at each month start.
  - pm_pre = median CLOB prices-history over the 10 min before the first frame.
  - Then the blend, v5's in-game model on v4's live feature path, and v6m at the archive L nearest the tick's feed age.
- Strategies (exploratory): `v6_gate_guard` (θ 0.30 + team guard, max 2 open) and `v6m_gate` (θ 0.25 = the backtest's selection at L 5–30 s, same guard).
- Logged as `ticks.q_v6`, `ticks.q_v6m` and `games.v6_prior_u / v6_cross / v6_pm_pre / v6_prior`.
- Checks:
  - `v6/build_live_v6.py check`: the tracker's feature pass and October refit reproduce v6/prior.py's prior_union_m on all 202 October games (max |diff| 5.6e-17).
  - A replay through Game.maybe_decide and App.v6_worker (scratchpad, LCS promotion 117322971012796414): the prior matches the study to 1e-16, pm_pre 0.60, and q_v6 / q_v6m on all 25 ticks.
- `src/backfill_v6.py` writes q_v6 / q_v6m for every quoted tick logged before v6 went live into `data/backfill.sqlite`. The dashboard's Backtest tab replays the v6 strategies over all games since 09-30.
  `backfill_signals.py` now upserts by column, so a rerun keeps the v6 columns.
- Backfill: all 6,291 quoted ticks of 147 games; 144 have a pre-game price.
- Dashboard replay, 2026-10-13: 144 resolved tracker games since 09-30, fills at the logged ask + 0.61 c (all net):

| Strategy | Trades | Net | pp |
|---|---|---|---|
| v6m_gate | 23 | **+$17** | +1.6 |
| v6_gate_guard | 39 | −$805 | −9.9 |
| v4_gate_guard | | −$488 | −5.6 |
| v2_gate_guard | | −$698 | −7.4 |
| v2_endgame | | +$56 | +2.7 |
| random | | −$23.6k | −2.5 |

- v6m_gate is the only gate at or above zero, on 19 games, so nothing is established.

Rerun: `python v6/prior.py && python v6/build.py && python v6/drift_exp.py && python v6/backtest_v6.py && python v6/book_sim_v6.py && python v6/build_live_v6.py check` (from `esports/src`).
