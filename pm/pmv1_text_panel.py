#!/usr/bin/env python3
"""Build the text panel: every liquid Polymarket-v1 market-day, with the market's
question attached. ([16])

RESULTS.md sec.3 reads this archive as numbers -- prices, imbalances, taker skill.
But the tradeable object on a prediction market *is* a natural-language sentence:
the question fully specifies the payoff, and there is no underlying asset with a
separate identity. That makes it the one venue where a language model is reading
the contract itself rather than commentary about it. This script assembles the
panel needed to ask whether that reading is worth anything.

The archive carries no question field. It carries `market_slug`, the URL stub,
which is the question lowercased and hyphenated. Recovering text from it is one
`replace`, but the corpus that comes back is *not* uniformly natural language,
and that fact drives every design choice downstream:

  57.1 % of the 746 M fills sit on markets whose slug contains a unix timestamp
  ("btc-updown-5m-1776662700", "eth-updown-15m-1764523800")

  a further 11.4 % carry an ISO date inside a sports template
  ("nba-bos-bkn-2025-11-18-1h-spread-away-5pt5")

Handing either to a text model hands it the clock. So each row is tagged with a
`kind` in {ts, date, nl} and carries two text columns: `text` (verbatim) and
`text_scrub` (timestamps, dates, years and bare integers masked). The pair is
what separates "the language predicts" from "the string encodes when".

One row per (market, day): the last bar of the day, so an observation is always a
tradeable state -- a price a taker saw. Execution prices come in two sets. The
same-day VWAPs describe the bar the signal was formed in and exist only for
diagnostics; anything a strategy transacts at must be `*_next`, the VWAPs of the
next day this market printed, because a book that enters in the bar whose last
print and volume produced its own signal is trading on information it did not
have. `gap_days` records how far away that next print was. Labels are both of the targets sec.3.5 distinguishes:

  y_term = win_event - p   payoff of buying YES here and holding to resolution.
                           Settlement pays at par, so this trip crosses the
                           spread once, not twice -- the structural reason it is
                           the more promising of the two after sec.3.5.
  y_fwd  = p(t+h) - p(t)   forward drift over h hours; "price direction" proper.

    python pm/pmv1_text_panel.py --freq 1h --horizon 24
"""
import argparse, os, re, sys
import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, os.pardir)
DATA = os.path.join(ROOT, "data")
NLP = os.path.join(DATA, "nlp")
PMV1 = os.path.join(DATA, "pmv1")

# A slug is machine-generated if it carries an absolute clock reading. Both
# patterns are anchored on a separator so that "2pt5" or "1h" cannot match.
RE_UNIX = r'(^|-)1[6-8][0-9]{8}($|-)'      # 2022-09 .. 2027-01 in epoch seconds
RE_ISO = r'(^|-)20[2-3][0-9]-[0-9]{2}-[0-9]{2}($|-)'

LAYERS = "['{pmv1}/daily_aligned/*.parquet','{pmv1}/daily_aligned_multi/*.parquet']"

# One slug and one category per market. Both are constant within a condition_id;
# any_value is a cheap assertion of that rather than a choice.
SLUGS = """
COPY (
  SELECT (hash(condition_id) >> 1)::BIGINT AS market_id,
         any_value(market_slug)      AS slug,
         any_value(category_refined) AS cat,
         any_value(category)         AS cat_raw,
         min(block_timestamp)        AS first_ts,
         max(block_timestamp)        AS last_ts,
         count(*)                    AS fills
  FROM read_parquet({layers}, union_by_name=true)
  GROUP BY 1
) TO '{out}' (FORMAT parquet)
"""

# Resolution on the outcome_seq = 1 axis, matching pmv1_prep.py's p_event convention.
#
# TRAP: `neg_risk` is a VARCHAR carrying a different boolean vocabulary in each
# layer -- the standard binary layer writes 'f' and only 'f', the negative-risk
# layer writes 'true' and only 'true'. A predicate written against one layer
# ("neg_risk <> 'false'") silently labels every row of the other as true, which
# reads as "every market is multi-outcome" and quietly disables any test that
# conditions on it. Compare against 'true' and nothing else.
RESOLUTION = """
COPY (
  SELECT (hash(condition_id) >> 1)::BIGINT AS market_id,
         (hash(COALESCE(neg_risk_market_id, condition_id)) >> 1)::BIGINT AS event_id,
         max(CASE WHEN outcome_seq = 1 AND outcome_label = winning_outcome_label THEN 1.0
                  WHEN outcome_seq = 1 THEN 0.0 END) AS win_event,
         bool_or(neg_risk = 'true')      AS neg_risk,
         epoch(min(close_at))::BIGINT    AS close_ts,
         epoch(min(resolved_at))::BIGINT AS resolved_ts
  FROM read_parquet({layers}, union_by_name=true)
  WHERE resolution_status = 'resolved' AND winning_outcome_label IS NOT NULL
  GROUP BY 1,2
) TO '{out}' (FORMAT parquet)
"""

PANEL = """
WITH slug AS (
  SELECT market_id, slug, cat, fills FROM read_parquet('{nlp}/slugs.parquet')
), res AS (
  SELECT market_id, event_id, win_event, neg_risk, close_ts
  FROM read_parquet('{nlp}/resolution.parquet') WHERE win_event IS NOT NULL
), bars AS (
  SELECT market_id, bar, cat, gross, p, sgn_large, abs_large, sgn_small, abs_small,
         vwap_buy, vwap_sell, buy_notional, sell_notional
  FROM read_parquet('{data}/pmv1_spreadbars_1h.parquet')
), day AS (
  -- last bar of each market-day, plus that day's activity
  SELECT market_id, (bar * 3600 / 86400)::BIGINT AS day,
         arg_max(bar, bar)           AS bar,
         arg_max(p, bar)             AS p,
         arg_max(vwap_buy, bar)      AS vwap_buy,
         arg_max(vwap_sell, bar)     AS vwap_sell,
         sum(gross)                  AS gross_day,
         sum(sgn_large + sgn_small)  AS sgn_day,
         sum(abs_large + abs_small)  AS abs_day,
         count(*)                    AS nbars
  FROM bars GROUP BY 1, 2
), nxt AS (
  -- the next market-day on which this market actually printed. The signal is
  -- formed at the close of day d; everything transactable is in day d+1, so the
  -- book must never touch day d's own VWAPs.
  SELECT market_id, day,
         lead(vwap_buy)  OVER (PARTITION BY market_id ORDER BY day) AS vwap_buy_next,
         lead(vwap_sell) OVER (PARTITION BY market_id ORDER BY day) AS vwap_sell_next,
         lead(p)         OVER (PARTITION BY market_id ORDER BY day) AS p_next,
         lead(day)       OVER (PARTITION BY market_id ORDER BY day) AS day_next
  FROM day
), fwd AS (
  -- forward price h hours after the observation bar, if the market printed then
  SELECT d.market_id, d.day, b.p AS p_fwd
  FROM day d JOIN bars b ON b.market_id = d.market_id AND b.bar = d.bar + {h}
), cum AS (
  SELECT market_id, day, sum(gross_day) OVER (
           PARTITION BY market_id ORDER BY day
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS gross_cum,
         min(day) OVER (PARTITION BY market_id) AS first_day
  FROM day
)
SELECT d.market_id, r.event_id, s.slug, s.cat, r.neg_risk,
       d.bar * 3600                          AS ts,
       c.first_day * 86400                   AS first_ts,
       r.close_ts,
       d.p, d.vwap_buy, d.vwap_sell,
       n.vwap_buy_next, n.vwap_sell_next, n.p_next,
       (n.day_next - d.day)                  AS gap_days,
       d.gross_day, c.gross_cum, d.sgn_day, d.abs_day, d.nbars,
       (d.day - c.first_day)                 AS age_days,
       (r.close_ts - d.bar * 3600) / 3600.0  AS hours_to_close,
       r.win_event,
       r.win_event - d.p                     AS y_term,
       f.p_fwd - d.p                         AS y_fwd
FROM day d
JOIN slug s USING (market_id)
JOIN res r USING (market_id)
JOIN cum c ON c.market_id = d.market_id AND c.day = d.day
LEFT JOIN nxt n ON n.market_id = d.market_id AND n.day = d.day
LEFT JOIN fwd f ON f.market_id = d.market_id AND f.day = d.day
WHERE d.p > 0.01 AND d.p < 0.99
  AND r.close_ts > d.bar * 3600
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=24, help="forward horizon in hours")
    ap.add_argument("--out", default=os.path.join(NLP, "panel.parquet"))
    ap.add_argument("--skip-prep", action="store_true",
                    help="reuse an existing slugs.parquet / resolution.parquet")
    a = ap.parse_args()

    os.makedirs(NLP, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'")
    con.execute(f"SET temp_directory='{os.path.join(DATA, 'duckdb_tmp')}'")

    layers = LAYERS.format(pmv1=PMV1)
    for name, sql in (("slugs", SLUGS), ("resolution", RESOLUTION)):
        path = os.path.join(NLP, f"{name}.parquet")
        if a.skip_prep and os.path.exists(path):
            continue
        con.execute(sql.format(layers=layers, out=path))
        print(f"wrote {path}")

    sql = PANEL.format(nlp=NLP, data=DATA, h=a.horizon)
    con.execute(f"CREATE TEMP VIEW panel AS {sql}")

    # Text reconstruction and the templated/natural split, done in SQL so the
    # panel is written once.
    con.execute(f"""
    COPY (
      SELECT *,
             replace(slug, '-', ' ') AS text,
             CASE WHEN regexp_matches(slug, '{RE_UNIX}') THEN 'ts'
                  WHEN regexp_matches(slug, '{RE_ISO}')  THEN 'date'
                  ELSE 'nl' END AS kind,
             -- scrubbed variant: mask every absolute clock reading, keep the words
             regexp_replace(
               regexp_replace(
                 regexp_replace(
                   regexp_replace(replace(slug, '-', ' '),
                     '\\b1[6-8][0-9]{{8}}\\b', ' <ts> ', 'g'),
                   '\\b20[2-3][0-9] [0-9]{{2}} [0-9]{{2}}\\b', ' <date> ', 'g'),
                 '\\b20[2-3][0-9]\\b', ' <year> ', 'g'),
               '\\b[0-9]+\\b', ' <num> ', 'g') AS text_scrub
      FROM panel
    ) TO '{a.out}' (FORMAT parquet, COMPRESSION zstd)
    """)

    q = lambda s: con.execute(s).df()
    print(f"wrote {a.out}")
    print(q(f"""SELECT count(*) n_rows, count(DISTINCT market_id) mkts,
                       count(DISTINCT event_id) events,
                       count(y_fwd) has_fwd,
                       to_timestamp(min(ts))::DATE t0, to_timestamp(max(ts))::DATE t1
                FROM read_parquet('{a.out}')""").to_string(index=False))
    print()
    print(q(f"""SELECT kind, count(*) n_rows, count(DISTINCT market_id) mkts,
                       round(avg(win_event), 4) base_rate, round(avg(p), 4) mean_p,
                       round(sum(gross_day) / 1e6, 1) vol_musd
                FROM read_parquet('{a.out}') GROUP BY 1 ORDER BY 2 DESC""").to_string(index=False))
    print()
    print(q(f"""SELECT date_trunc('quarter', to_timestamp(ts))::DATE q, count(*) n_rows,
                       count(DISTINCT market_id) mkts
                FROM read_parquet('{a.out}') GROUP BY 1 ORDER BY 1""").to_string(index=False))


if __name__ == "__main__":
    main()
