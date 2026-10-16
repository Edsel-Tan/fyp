# The v7 LoL win-probability model: full specification

Written 2026-10-16 from the code in `esports/src/` (study) and `~/fyp-pmlive/lollive/src/` (live tracker).
v7 is the model currently live in the tracker (signals `q_v7`, `q_v7m`; strategies `v7_gate_guard`, `v7m_gate`, plus `_dyn` sizing twins).

Notation used throughout:
- σ(z) = 1 / (1 + e^(−z)), logit(p) = log(p / (1 − p)).
- Every quantity is from the **blue side's view**: a difference "blue minus red". y = 1 if blue wins the game.
- "Causal" means a game's own result (or anything that ends after it starts) never enters its features.
  Every rating pass processes games in start order. Before game i's features are read, every game with `t_end ≤ t_start(i)` is applied to the state.

---

## 0. One-paragraph summary

v7 is a **stack of logistic regressions**:

1. **Rating passes.** About 20 causal rating systems (Elo variants, Kalman filters, decayed win-rate tables, a Bradley–Terry ridge regression, champion models) produce **55 pre-game features**.
2. **Pre-game logistic.** One standardised, L2-penalised logistic regression (C = 0.01), refit at the start of every month on all finished games since 2024-04-01, turns the 55 features into a **pre-game logit z**.
3. **Market blend.** z is blended with the Polymarket pre-game price when one exists (weight ≈ 0.92 for cross-league games, ≈ 0 for same-league games). The result is **z7**.
4. **In-game logistic.** A frozen L2 logistic regression (C = 1, 102 columns) combines z7 with 33 live game-state features. Each enters with time-varying weights (1, m, m²), m = minutes / 10. The output is **q_v7**.
5. **Market combination.** **q_v7m** combines logit q_v7 with the live market mid. It uses one no-intercept logistic per game phase (0–10, 10–20, 20–30, 30+ min) and per feed latency.

All losses are binary log loss.

```
 lolesports schedule / results ─┐
 livestats feed (end states) ───┼─► rating passes (§2) ─► 55 features ─► pre-game logistic, monthly refit (§3) ─► z
 game details (damage shares) ──┘                                                                                │
 Polymarket pre-game price pm_pre ───────────────────────────────────────────► market blend (§4) ◄────────────────┘
                                                                                     │ z7
 livestats feed (live, every frame) ─► 33 state features (§5) ─► in-game logistic, frozen (§5) ─► q_v7
                                                                                                    │
 Polymarket live mid ─────────────────────────────────────────────► per-phase combination (§6) ─► q_v7m
```

---

## 1. Data sources and labels

| Input | Source | Used for |
|---|---|---|
| Schedule, matches, teams, leagues, series number, best-of | lolesports `persisted/gw` API (`getSchedule`, `getEventDetails`; public key in `src/lolapi.py`) | the game table, series state, home leagues |
| Game ids | `getEventDetails` | joining to the feed |
| In-game state | livestats `feed.lolesports.com/livestats/v1/window/{gameId}` | every in-game feature, end-of-game gold, g_area, gd15 |
| Blue / red sides | livestats `gameMetadata.blueTeamMetadata.esportsTeamId` | the side of every feature (getEventDetails sides are random) |
| Players, champions, roles, patch | livestats metadata | player ratings, draft features |
| Per-player damage share | livestats `details` endpoint (`championDamageShare`), last frame | `dmg_k` ratings |
| Historical market prices | Polymarket-v1 on-chain archive (`~/fyp/data/pmv1`), per-game winner markets | pm_pre (study), backtests, the v7m fit |
| Live market prices | Polymarket CLOB: `prices-history` (pm_pre) and the order book (mid, bid, ask) | the blend, q_v7m, trading |

**Historical sampling.**
- The training feed is sampled every 60 s (`data/frames60/{gid}.parquet`).
- The live tracker polls the feed densely (median 0.2 s between frames).
- Every windowed feature is defined in seconds, not rows, so both samplings give the same numbers (`v4/test_features.py`: corr ≥ 0.967).

**Game clock** (`ingame_data.one`):
- Loading frames (no gold) are dropped.
- Clock at the first gold frame: gt₀ = 65 + (gold_total − 5000) / 20.4 if gold_total < 5400, else 90 s.
- After that, the clock advances with wall time, excluding `paused` frames.
- Accuracy is about ±30 s at 60 s sampling.

**Winner label `blue_win`.** The lolesports API has no per-game winner field. The label is, in priority order:
1. **sweep**: every game of an n–0 series.
2. **final**: the last game of a series goes to the series winner.
3. **pm**: the Polymarket per-game market's resolution.
4. Otherwise, a logistic rule on end-of-game features: structures pushed in the last 60/120/180 s, and end-state differences.

The rule has 99.83 % grouped 10-fold CV accuracy (`src/winner.py`).

**Home league.** A team's home league is its most frequent non-international league.
- `cross` = 1 when the two teams' home leagues differ.
- `intl` = the game is at Worlds, MSI, First Stand or EWC.

**Data size.**
- 22,722 games in the extended game table (2023-10-09 → 2026-10-13); the live tracker appends new games.
- Rating passes use all of them.
- Fits use games from **BURN = 2024-04-01**; the months before are rating burn-in.

**Time splits** (`eval_pregame.py`, `v6/build.py`):

| Name | Dates | Role |
|---|---|---|
| burn-in | 2023-10 → 2024-03 | ratings only |
| train | 2024-04-01 → 2025-06-30 | fits the in-game logistic, the OOF training prior and the hyper-parameter grids |
| validation slice | the last 91 days of train (VAL0 = 2025-04-01 → 2025-06-30) | every hyper-parameter / feature-set choice (fit on [BURN, VAL0)) |
| test | 2025-07-01 → | every reported number |
| selection months | 2025-07 → 2026-01 (SEL_END = 2026-02-01) | fit the market blend, the v7m weights and the trading θ |
| eval months | 2026-02 → | the out-of-sample backtest |

---

## 2. Pre-game features (55 columns)

Each subsection gives the update equations. Unless stated otherwise:
- a feature is blue minus red;
- a new entity (team or player) starts at the mean rating of the last 200 entities first seen in its league (1500 for Elo, 0 for logit ratings).

### 2.1 Elo family (`src/pregame.py`)

Expected score: E(a, b) = 1 / (1 + 10^((R_b − R_a)/400)).

**Team Elo** (K ∈ {16, 32, 64}). After a game,

  R_blue ← R_blue + K (y − E(R_blue, R_red)),  R_red ← R_red − K (y − E).

**Player Elo** (K ∈ {16, 32, 64}). A team's rating is the mean of its five players' ratings. Every player on a side moves by the same step d = K (y − E(R̄_blue, R̄_red)).

| Feature | Definition |
|---|---|
| `elo_team_16/32/64` | R_blue − R_red, team Elo |
| `elo_play_16/32/64` | mean player Elo, blue − red |
| `form_5/10/20` | team win rate over its last N games, shrunk: (wins + 1)/(n + 2); blue − red |
| `h2h` | blue's wins − red's wins in earlier meetings |
| `exp` | mean over the 5 players of log(1 + career games); blue − red |
| `roster` | players who also played the team's previous game (0–5); blue − red |
| `rest_d` | days since the team's previous game, clipped to [0, 30]; blue − red |
| `series_diff` | blue's game wins − red's game wins earlier in this series |

### 2.2 Draft, patch and lane features (`src/pregame2.py`)

Decayed counters use an exponential half-life of **HL = 120 days**: on each update, s ← s · 0.5^(Δt/HL) + x and n ← n · 0.5^(Δt/HL) + 1, and reads are decayed the same way.

The patch version `pv` is major.minor. Indices: k = role (top, jungle, mid, bottom, support); ± = +1 for a blue entry, −1 for a red one.

| Feature | Equation |
|---|---|
| `champ_all` | Σ_k ± logit(p_all), p_all = (s + 5)/(n + 10) over the champion-in-role's decayed wins s and games n |
| `champ_patch` | Σ_k ± logit(p_patch), p_patch = (s_pv + 20·p_all)/(n_pv + 20) (current patch, shrunk to p_all) |
| `lane_mu` | ½ Σ_k ± S/(N + 10), where S and N are decayed sums of the matchup residual y − σ(logit p_patch,b − logit p_patch,r) for (role, own champion, enemy champion) |
| `pc_exp` | Σ_k ± log(1 + n_player-champion) / 5 (decayed games of that player on that champion) |
| `pc_wr` | Σ_k ± S/(N + 8) of the player-on-champion residual y − p_patch |
| `side_patch` | logit((s_pv + 50·p_s)/(n_pv + 50)), p_s = (s + 10)/(n + 20): the current patch's blue-side win rate, shrunk to the long-run rate (not a difference) |
| `age_x_elo` | elo_play_32 · exp(−patch_age / 7), patch_age = days since the patch was first seen |
| `role_elo_k` | per-player win Elo by role, K = 32, blue player − red player |
| `lane_elo_k` | per-player lane Elo, K = 32, with the score σ(gd15_k / 1000) instead of the win (gd15_k = role-k gold difference at 15:00) |
| `conv` | S/(N + 10) per team of y − q15, where q15 = σ(a₀ + a₁ · gd15_total / 1000) is fitted on burn-in games; blue − red (converting leads) |
| `league_elo` | league Elo, K = 16, updated **only** on cross-league games; blue's home league − red's |

### 2.3 Reddit-model ratings (`src/reddit/build.py`, rebuilt from the r/leagueoflegends "Worlds 2026" post)

**Player Kalman** (`kal_z`, `kal_0..4`). Each player i has a mean μ_i and a variance v_i.

- Prediction step: v_i ← min(v_i + q · Δdays, v₀), with q = 1e-3 and v₀ = 2.
- Game logit and variance: z = mean(μ_blue) − mean(μ_red), V = Σ_{10 players} v_i / 25, p = σ(z).
- Weight: w = w_int · MOV, where
  - w_int = 3 for internationals, else 1;
  - MOV = clip(log(1 + |end gold diff| / 1000) / log 6, 0.3, 1.6).
- Update, with den = 1 + w V p(1 − p):

  μ_i ← μ_i ± w v_i · 0.2 (y − p) / den,  v_i ← max(v_i − w v_i² · 0.04 · p(1 − p) / den, 1e-4).

- Features: `kal_z` = mean μ_blue − mean μ_red; `kal_k` = μ of blue's role-k player − μ of red's.

**Draft-residual lane rating** (`lane_k`).
- Expected gold at 15 min: ê_gd,k = a monthly-refit ridge regression (α = 30) of gd15_k / 1000 on an antisymmetric champion-by-role one-hot. Blue's role-j champion is +1 and red's −1, over 5 × #champions columns.
- Residual: r = (gd15_k − ê_gd,k) / 1000, credited +r to the blue player and −r to the red one.
- Player rating: S/(N + 5) over decayed sums (HL 120 d).
- Feature: blue player − red player.

**Damage-share rating** (`dmg_k`). The same construction on the last-frame damage share, with a ridge expectation from the own five and enemy five champions by role. The residual is × 100.

**Bradley–Terry** (`bt`, replaced in v7 by `bt7`, §2.4).

**Draft models.**
- `draft_wr` = the logit of a monthly L2 logistic (C = 0.05) of y on the antisymmetric champion-by-role one-hot.
- `draft_gd` = Σ_k ê_gd,k / 1000.

### 2.4 The v7 additions (`src/v7/v7feat.py`, a verbatim copy runs in the tracker)

#### Team + league Kalman (`kt`, `kl`, `kl_sd`)

State:
- every team has a within-league rating θ_t with variance v_t;
- every home league has an offset μ_L with variance u_L.

All start at 0 with variance 1.

Prediction step (per day since the entity's last update): v ← min(v + q_t Δd, 1) with q_t = 0.5; u ← min(u + q_l Δd, 1) with q_l = 1e-4.
So team variance refills within about 2 days (fast), while league offsets move slowly.

Game logit:

  z = θ_b − θ_r + c·(μ_Lb − μ_Lr) + side,  side = 0.12,  c = 1 if cross-league else 0.

Soft result (dominance weighting, λ = 0.5, k_d = 3). With g_area = (1/T) Σ_{minutes} tanh(gold diff / 5000) over a 60 s grid,

  s = λ·y + (1 − λ)·σ(k_d · g_area).

So a stomp counts as ~1 and a lucky comeback as ~0.5–0.6. If g_area is unknown, s = y.

EKF update, with p = σ(z), w = 2 for internationals else 1, V = v_b + v_r + c(u_b + u_r), den = 1 + w V p(1 − p), r = s − p:

  θ_b ← θ_b + w v_b r / den,  v_b ← max(v_b − w v_b² p(1−p) / den, 1e-4)
  θ_r ← θ_r − w v_r r / den  (same variance update)
  if cross: μ_Lb ← μ_Lb + w u_b r / den,  μ_Lr ← μ_Lr − w u_r r / den  (same variance updates)

Features:
- `kt` = θ_b − θ_r
- `kl` = μ_Lb − μ_Lr
- `kl_sd` = √(u_b + u_r)

The settings come from a validation grid (`results/v7_kgrid.log`).

#### Online Bradley–Terry (`bt7`)

**Weekly anchor.** Every 7 days at w₀, fit an L2 logistic with C = 1 on the games that ended in [w₀ − 365 d, w₀):

  min_β ½‖β‖² + C Σ_i ω_i ℓ(y_i, x_i·β + b₀),  ω_i = (2 if intl else 1) · 0.5^((w₀ − t_end,i)/120), normalised to mean 1.

- x_i has +1 / −1 for the blue / red team and, if cross-league, +1 / −1 for their home leagues.
- `bt` = x·β, with no side intercept.

**Per-game Newton step** (the v7 change). Inside the week, every finished game j updates β by one exact Newton step on its own likelihood term. The inverse Hessian is updated by Sherman–Morrison:

  H = I + Xᵀ diag(C ω p(1 − p)) X  (at the anchor fit, over the columns active this week)
  a = C ω_j p_j (1 − p_j),  h = H⁻¹ x_j
  H⁻¹ ← H⁻¹ − h hᵀ · a / (1 + a x_jᵀ h)
  β ← β + H⁻¹ x_j · C ω_j (y_j − p_j)

`bt7` = x·β with the updated β, so it moves after every game. On the weekly grid it equals the study's `bt` exactly.

### 2.5 The 55 columns of the v7 prior (`data/v7/choice.json`)

`elo_team_16/32/64, elo_play_16/32/64, form_5/10/20, h2h, exp, roster, rest_d, series_diff, age_x_elo, champ_all, champ_patch, conv,
lane_elo_0..4, lane_mu, league_elo, pc_exp, pc_wr, role_elo_0..4, side_patch, kal_z, kal_0..4, lane_0..4, dmg_0..4, bt7, draft_wr, draft_gd, kt, kl, kl_sd`.

Seven of these depend on the draft (`champ_*`, `lane_mu`, `pc_*`, `draft_*`), so the prior is a **post-draft** prior.

---

## 3. The pre-game logistic (the "union prior" with v7 columns)

**Model.** The prior is

  z = β₀ + Σ_{j=1}^{55} β_j · x_j / s_j,  P(blue wins | pre-game) = σ(z),

where s_j is the standard deviation of feature j over the fit set.

**It is a regression**: an L2-penalised logistic regression (sklearn `LogisticRegression`, lbfgs, C = 0.01). It minimises

  ½‖β‖² + C Σ_i [−y_i log σ(z_i) − (1 − y_i) log(1 − σ(z_i))]   (the intercept is not penalised).

C = 0.01 was chosen on the validation slice. The penalty is strong, so the 55 highly collinear ratings share weight instead of fighting.

**Training and refit protocol.**
- **Fit set:** every game with t_start ≥ BURN, t_end < the month start, and a known result. All games are **equally weighted**, in an expanding window.
- **Refit:** at the start of every calendar month. The month's games are scored with that fit (`v6/prior.py:monthly`). The live tracker does the same at each month start (`v7live.V7.priors`).
- **Training prior for the in-game model:** the in-game model was trained on 2024-04 → 2025-06 frames. For those frames the prior is **out-of-fold**: 5 folds by match, each fold scored by a fit on the other four (`reddit/eval.py`, column `prior_both_m`). This uses v5's union set (`bt` rather than `bt7`, no Kalman).

**Largest standardised coefficients** (refit for 2026-10, 19,605 games; logit per SD):

| Coefficient | Value |
|---|---|
| β₀ (side) | +0.090 |
| `exp` | +0.32 |
| `bt7` | +0.30 |
| `kal_z` | +0.15 |
| `kt` | +0.14 |
| `kl` | +0.14 |
| `kal_2` (mid) | +0.13 |
| `kal_0` (top) | +0.13 |
| `dmg_3` (bot) | +0.11 |
| `draft_wr` | +0.11 |
| `elo_play_16` | −0.11 (collinear with the Kalman player ratings) |

---

## 4. The market pre-game blend → z7

pm_pre is the market's pre-game P(blue):
- **study:** the Polymarket-v1 archive's last-30-min VWAP;
- **live:** the median CLOB `prices-history` price of the blue token over the 10 min before the game's first feed frame.

The price is clipped to [0.01, 0.99]. The blend is

  z7 = z + 1[pm_pre exists] · (w₀ + w₁ · cross) · (logit pm_pre − z).

- **Regression:** a binomial GLM with offset z and no intercept, on the selection-month games that have an archive price (2025-07 → 2026-01). The regressors are the two terms (logit pm_pre − z) and cross · (logit pm_pre − z).
- **Fitted weights:** w₀ = −0.10 (0.19), w₁ = +0.92 (0.68).
- **Reading:** a same-league game keeps the model's prior (weight ≈ −0.1, i.e. ≈ 0). A cross-league game takes ≈ 82 % of the market (w₀ + w₁ = 0.82).
- **Without a price:** z7 = z.
- **Persistence term:** v6 tested g₀ + g₁m · (z7 − z) in-game. Match-grouped CV dropped it, so it is 0 in v7.

---

## 5. The in-game logistic → q_v7

**Model.** With m = clock_seconds / 600 (tens of minutes),

  logit q_v7 = b₀ + Σ_{c ∈ F} Σ_{k=0}^{2} β_{c,k} · m^k · x_c,

where F = 33 state features plus the prior x_prior = z7. That gives 34 × 3 = **102 columns**.
- Each column is standardised (`StandardScaler`, fitted on train).
- Gold-type inputs are in thousands.
- The (1, m, m²) basis lets every weight change smoothly with game time; for example, the prior's weight fades from 0.94 at 5 min to 0.26 at 40 min.

**Regression:** an L2 logistic with C = 1 (`eval_v4.mk`). It minimises the same penalised log loss as §3, summed over **frames**. Each 60 s frame of each training game is one row, so a 35-min game contributes about 34 rows. Frames with state `finished` are excluded.

**Training:**
- **Once**, on the train split: frames of games with BURN ≤ t_start < 2025-07-01, 290,716 frames.
- The prior column during training is the OOF union prior.
- The model is **frozen** after that (`data/v5/models.pkl`, built 2026-10-12).
- At prediction time z7 is plugged into the prior column.

**Features** (all blue − red; `ingame_data.py`, `v4/features.py`):

| Group | Feature | Definition |
|---|---|---|
| state (STATS) | `gold` | team gold difference (k) |
| | `kills`, `towers`, `inhibs` | cumulative differences |
| | `barons` | cumulative Baron kills |
| | `drakes` | elemental drakes (excluding elder) |
| | `elders` | cumulative elder kills |
| | `soul` | soul obtained (≥ 4 elemental drakes), +1 / −1 |
| | `level`, `cs` | summed levels, summed CS |
| | `alive` | players alive |
| | `gold_r1..r5` | per-role gold difference (k), top → support |
| H fight in progress | `hp_al` | mean current/max HP of alive players |
| | `low_hp` | alive players under 35 % HP |
| | `fight_g` | kills by both teams in the last ≥ 50 s × gold lead (k) |
| | `dk1`, `dg1` | kill / gold (k) lead change over the last ≥ 50 s |
| S soul race | `drk_sq` | n_b² − n_r² elemental drakes |
| | `soul_pt` | (blue on 3) − (red on 3) before either has soul |
| | `dt_<type>` | per-type drake counts: infernal, ocean, mountain, cloud, hextech, chemtech |
| G relative gold | `gsh` | (b_gold − r_gold)/(b_gold + r_gold) |
| | `gsat` | tanh(gold diff / 5000) |
| R respawn | `dead_s` | Σ over dead players of expected respawn seconds left / 60 (base timer by level × the 15/30/45-min time factor; the death time is bracketed by frames) |
| | `dead_g` | the same, weighted by the dead player's total gold (k) |

This set ("`sel window`") was picked by greedy forward selection on the validation slice from 9 candidate groups. Groups left out:
- T: inhibitors down, baron/elder buff time left
- D: 5-min deltas
- B: bounties
- L: levels
- I: items (not computable live)

**Effective weights** (logit per unit, β_c,0 + β_c,1 m + β_c,2 m², from the frozen model):

| Term (unit) | 5 min | 10 | 20 | 30 | 40 |
|---|---|---|---|---|---|
| prior z7 (logit) | 0.94 | 0.87 | 0.69 | 0.49 | 0.26 |
| barons (per Baron) | +0.45 | +0.32 | +0.11 | −0.04 | −0.13 |
| elders (per Elder) | +0.68 | +0.63 | +0.54 | +0.48 | +0.44 |
| soul | +1.06 | +0.99 | +0.86 | +0.73 | +0.60 |
| inhibs | +0.47 | +0.43 | +0.34 | +0.28 | +0.22 |
| towers | +0.05 | +0.04 | +0.05 | +0.09 | +0.14 |
| gold share `gsh` (per unit share) | 4.0 | 7.0 | 13.0 | 18.8 | 24.5 |
| `hp_al` | 0.83 | 0.65 | 0.54 | 0.75 | 1.29 |
| `dead_s` (per respawn-minute) | −0.32 | −0.22 | +0.06 | +0.43 | +0.89 |
| `gold_r2` (jungle, per k) | +0.31 | +0.24 | +0.11 | +0.01 | −0.08 |

- The raw `gold` coefficients are small because `gsh` and `gsat` carry most of the gold effect.
- The coefficients are partial effects, conditional on all the collinear terms, so read them jointly.

---

## 6. q_v7m: combination with the live market

With phase p ∈ {0–10, 10–20, 20–30, 30+ min} and the feed's latency bucket L ∈ {0, 5, 15, 30, 60, 120} s:

  logit q_v7m = b_{p,L} · logit(mid) + c_{p,L} · logit(q_v7)   (no intercept; both inputs clipped to [0.005, 0.995])

- **Regression:** a binomial GLM per (p, L) on the archive's selection-month frames (2025-07 → 2026-01). The mid is the print-based mid proxy at the decision instant minus L (`v7/backtest_v7.py:comb_fit`).
- **Why no intercept:** with one, the fit learned a −0.4 logit side bias from the selection months and lost out of sample.
- **Live:** the tracker uses the L nearest the tick's feed age (typically 22 s).

Weights (mid / v7) at L = 30 s:

| | 0–10 | 10–20 | 20–30 | 30+ |
|---|---|---|---|---|
| mid weight | 0.47 | 0.05 | 0.03 | 0.59 |
| v7 weight | 0.56 | 1.02 | 0.89 | 0.26 |

---

## 7. Loss, evaluation and hyper-parameter selection

**Training loss:** penalised binary log loss, as in §3 and §5. Nothing else is used: no Brier, no AUC, no tree models in production. LightGBM was tried and lost.

**Evaluation metric:** log loss ℓ = −[y log q + (1 − y) log(1 − q)], with q clipped to [1e-6, 1 − 1e-6].
- Pre-game: per game.
- In-game: **per-game mean** = the mean of a game's frame log losses, averaged over games. This weights long and short games equally.

**Uncertainty:** model differences are reported as mean(ℓ_A − ℓ_B) with a **match-clustered bootstrap** 95 % CI (2,000 resamples). Games of one series are resampled together because they are not independent.

**Selection discipline:**
- Every choice is made on the validation slice or the selection months, then scored once on test or the eval months. Choices covered: C values, feature groups, Kalman settings, the arm, the blend weights, the v7m weights and the trading θ.
- Trading backtests report the deflated Sharpe ratio (DSR) over the family of tried cells.

**Nulls:** v5 has five shuffled-outcome nulls (train labels permuted across games). Their test log loss is near log 2, so the pipeline has no leak.

---

## 8. Trading layer (live tracker, exploratory)

**Gate:**
- **Buy blue** at the ask when q − ask > θ.
- **Buy red** (sell blue) at the bid when bid − q > θ.

**Rules:**
- θ = 0.30 for `v7_gate_guard` and 0.25 for `v7m_gate`, both the selection-month choices.
- **Guard:** skip the game if either team has fewer than 20 games in the last 365 days.
- At most 2 open entries per game.
- One decision per minute.
- $50 per entry (the `_dyn` twins scale the stake with |q − mid|).
- Positions are held to resolution.

**Costs:**
- Fees are 0.05 · p(1 − p) per share (from 2026-03).
- 1 c of walk slippage is applied in the book simulations.

---

## 9. Performance (from PROGRESS.md "v7")

**Pre-game, test 2025-07 → 2026-10-13** (10,875 games):

| Scope | v7 prior − v6 union |
|---|---|
| All games | −0.0010 [−0.0022, +0.0001] |
| Cross-league | −0.0039 (ns) |

**In-game, per-game LL, q_v7 − q_v6:**

| Scope | Δ |
|---|---|
| Test | −0.0003 [−0.0009, +0.0004] |
| Live period 09-17 → 10-11 | −0.0022 (ns) |

**Historical gate backtest** (gross pp per share):

| Strategy | L = 0 | L = 30 |
|---|---|---|
| v7 gate | +10.7 [1.5, 20.0] (DSR 0.72) | +8.7 |
| v7m gate | +14.6 (DSR 0.84) | +11.0 |

**Captured live books** (79 games): every gate CI spans zero, and the decision-time market mid still beats v7m by +0.028 LL.

---

## 10. Where each piece lives

| Piece | Study code | Live code |
|---|---|---|
| Elo family | `src/pregame.py` | `lollive/src/pregame.py` |
| draft / lane / league | `src/pregame2.py` | `lollive/src/pregame2.py` |
| reddit ratings | `src/reddit/build.py` | `lollive/src/rbuild.py` (verbatim) |
| v7 Kalman + bt7 | `src/v7/v7feat.py` | `lollive/src/v7feat.py` (verbatim) |
| monthly prior refit | `src/v6/prior.py:monthly`, `src/v7/prior.py` | `lollive/src/v7live.py:V7.priors` |
| market blend | `src/v6/build.py:fit_blend`, `src/v7/build.py` | `v6live.V6.z6` |
| in-game features | `src/ingame_data.py`, `src/v4/features.py` | `lollive/src/live.py`, `v4feat.py` |
| in-game model | `src/v5/build.py` (fit), `data/v5/models.pkl` | `data/v7/v7_model.pkl['v5']` |
| v7m | `src/v7/backtest_v7.py:comb_fit` | `v6live.V6.comb` |
| checks | `src/v7/build_live_v7.py check` (tracker = study to max diff 0.0 on 202 October games) | |

---

## 11. Known limits of v7, measured 2026-10-16 (details in PROGRESS.md "v8 candidate")

- **Frozen in-game fit.** The in-game logistic was last fit on 2025-06 data. A monthly equal-weight refit is −0.0006 per game on the eval months and −0.005 per frame after 30 min.
  Decay-weighted refits (half-life 90–730 d) are no better, and h = 90 d is worse out of sample.
- **Refit cadence and champion decay.** Refitting the pre-game logistic monthly is slower than it needs to be: weekly is −0.0004. The champion tables' 120-day half-life is too slow: 30 d is −0.0005.
  Player / team rating half-lives and decay-weighted regression samples do not help. Per-patch role coefficients show no excess variation (Cochran's Q, Holm min p 0.71).
- **Baron / Elder buffs.** Only the cumulative `barons` / `elders` counts enter the model.
  - It under-credits the holder by ~0.1–0.15 logit during the buff and over-credits slightly afterwards.
  - A buff term is worth +0.35–0.5 logit (Baron) / +0.9–1.0 (Elder) at the kill, decaying to 0 at expiry.
  - That is −0.00015 per game, −0.0008 on buff frames.
- **Calibration.** Test calibration slope is ≈ 0.94 overall and 0.89 in the 3 min after a buff expires, so the model is mildly overconfident.
- **The v8 candidate** (all of the above) is −0.0013 [−0.0021, −0.0006] per game vs v7 on the eval months and −0.0027 on the live period. Trading is indistinguishable.
