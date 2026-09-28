# Pre-registration — the 24-hour drift text arm on untouched months

Written 2026-09-27, **before any Polymarket data after 2026-04-28 was pulled or looked at.**
This file fixes one hypothesis, one test statistic and one decision rule. Nothing below
may be changed once the out-of-sample panel exists; any deviation forced by the data must
be reported as a deviation, next to the pre-registered result, not in place of it.

## Why this one arm

RESULTS.md §3.6 ran 22 arms. One of them — FinBERT text on the 24-hour drift target — is
positive (dIC +0.0148, t = +2.02 over 8 folds, 6/8 positive), and its shuffle null does
not reproduce it. But P(max |t| ≥ 2.02) over 22 arms under a global null is 0.851, so it
is a lead, not a finding. §10 item 7 says: fix that specification and test it once, on
months the study never touched. The Polymarket-v1 archive ends 2026-04-28, so
**2026-05-01 onward is untouched**.

## Hypothesis

H1: on Polymarket natural-language (`kind = nl`) markets, adding frozen FinBERT embeddings
of the question to the price + category baseline raises the information coefficient for
the 24-hour forward price change. H0: it does not (dIC ≤ 0).

## The specification (frozen)

Everything is `pm/pmv1_text_signal.py --text finbert --field text --kind nl --target fwd`
as run for §3.6, except that the walk-forward folds are replaced by one pre-specified cut:

| | window | rows |
|---|---|---|
| train | archive rows with `date < 2026-02-27`, events first seen before then | all of them |
| validation | archive rows in `[2026-02-27, 2026-04-29)`, events first seen in that window | selects (α, γ) |
| **test** | **new rows, 2026-05-01 → the end of the pull**, events **never seen in the archive** | the one test |

`2026-02-27` is the archive's last day minus the script's own `VALID_DAYS = 61`.

- M1 = price block (the script's `price_block`, the 20 most frequent training categories).
  M2 = M1 + the FinBERT block scaled by γ. Ridge. (α, γ) are searched on the same grids
  (`ALPHAS`, `GAMMAS`) by validation IC, exactly as `best_ridge` does. **Neither model is
  refit on validation.** The models fitted on train at the selected (α, γ) are applied
  unchanged to the test rows. Standardisation uses train moments.
- Text is `slug.replace('-', ' ')` as in `pmv1_text_panel.py`, embedded by
  `pm/pmv1_text_embed.py`'s FinBERT procedure (mean-pooled, max length 64).
- Row weights are the script's `w = 1 / rows per market`.

## Rebuilding the panel for the new months

The archive does not reach May 2026, so the test panel is rebuilt from Polymarket's public
APIs (`pm/pmlive_drift_panel.py`, written after this file). It reproduces `pmv1_spread.py`'s
hourly bars and `pmv1_text_panel.py`'s market-day rows from taker fills:

- Markets: Gamma markets with `umaResolutionStatus = resolved`, closed between 2026-05-01
  and the pull date, that are **not** present in the archive, and whose slug is `nl` under
  the panel's own `RE_UNIX` / `RE_ISO` regexes.
- Fills: `data-api /trades?market=…` (taker-only, its default), paged backwards with `end=`
  until exhausted. `p_event` = price on outcome index 0, else 1 − price. `D` = +1 for BUY on
  index 0 or SELL on index 1, −1 otherwise. `usdc_amount = size × price`.
- Bars: the same `min_volume = 100,000` lifetime filter, the same per-market 90th-percentile
  large-trade threshold, the same hourly `p`, signed/absolute flow and taker VWAPs, the same
  market-day aggregation, the same `y_fwd = p(bar + 24h) − p(bar)` (NULL if no print at
  exactly that bar), and the same `0.01 < p < 0.99` and `close_ts > ts` filters.
- Category: the market's Gamma event tags, in order; the first whose label matches one of
  the 20 training categories (case-insensitive) is used, else `Other`.
- A market whose fill history cannot be reconstructed completely (API error after
  retries) is dropped **before** any feature or label is computed, and the drop count
  reported.

## Test statistic and decision rule

- **Primary:** dIC = IC(M2) − IC(M1) on the pooled test rows, Pearson, as `ic()` computes it.
- **Inference:** an event-clustered bootstrap of dIC (10,000 resamples of `event_id` with
  replacement, seed 0). p = share of resamples with dIC ≤ 0.
- **Decision:** H1 is supported if and only if **dIC > 0 and p < 0.05** (one-sided). One
  test, so no multiplicity correction applies.
- **Secondary, reported but not decisive:** dIC per calendar month; the selected γ (if the
  validation search picks γ = 0, M2 = M1 and dIC is exactly 0, and the arm is recorded as
  **not supported**); the M2 − M1 book in pp/share at next-day VWAPs; and the shuffle-null
  dIC on the same test rows.

## What each outcome would mean

- **Supported:** the only text result on the venue that survives its own multiplicity bar.
  It gets its own section, and is still conditional on the rebuilt panel matching the
  archive's construction.
- **Not supported:** §3.6 is complete as written. The +0.0148 is recorded as what 22-arm
  selection produces, which is what P = 0.851 predicted.
