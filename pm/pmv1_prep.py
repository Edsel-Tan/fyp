#!/usr/bin/env python3
"""Compact the Polymarket-v1 trade layers into the two tables the analyses need.

The cleaned layers are 746M rows (602M standard binary + 144M neg-risk) across
16.8 GB of 24-column Parquet, most of it repeated market metadata stored as
strings. Every analysis in pm/ needs a different four or five of those columns,
and re-scanning the wide layers once per question costs tens of gigabytes of
spill on the taker group-by. So project once:

  data/pmv1_trades.parquet   one row per resolved fill, integer-keyed:
      taker_id, market_id, event_id, ts, s, notional
    where s = D * (usdc_amount / price) * (win_event - p_event) is the realised
    payoff of the fill in event-probability coordinates -- the quantity every
    sign-randomisation null flips. Because D = +-1, the null variance of a unit
    is just the sum of s^2, so no separate unsigned column is needed.

  data/pmv1_bars.parquet     one row per (market, hourly bar), for the imbalance
      regressions: last p_event, gross notional, and signed/unsigned notional
      split into large (top decile within the market) and small trades.

Ids are 63-bit hashes of the on-chain addresses and condition ids: with a few
million distinct takers the collision probability is ~1e-6, and nothing
downstream needs to resolve an id back to an address.

    python pm/pmv1_prep.py --freq 1h
"""
import argparse, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, os.pardir)
PMV1 = os.path.join(ROOT, "data", "pmv1")

# win_event and p_event both live on the outcome_seq = 1 axis, so the payoff of
# one share held long in event coordinates is (win_event - p_event) and D signs it.
TRADES = """
COPY (
  WITH raw AS (
    SELECT taker, condition_id, COALESCE(neg_risk_market_id, condition_id) AS event_id,
           D, p_event, usdc_amount, price, outcome_seq, block_timestamp,
           CASE WHEN outcome_label = winning_outcome_label THEN 1.0 ELSE 0.0 END AS win_tok
    FROM read_parquet({glob}, union_by_name=true)
    WHERE resolution_status = 'resolved' AND winning_outcome_label IS NOT NULL
      AND price > 0.0005 AND price < 0.9995 AND usdc_amount > 0 AND D IS NOT NULL
  )
  SELECT (hash(taker) >> 1)::BIGINT       AS taker_id,
         (hash(condition_id) >> 1)::BIGINT AS market_id,
         (hash(event_id) >> 1)::BIGINT  AS event_id,
         block_timestamp            AS ts,
         D * (usdc_amount / price)
           * ((CASE WHEN outcome_seq = 1 THEN win_tok ELSE 1.0 - win_tok END) - p_event) AS s,
         usdc_amount                AS notional
  FROM raw
) TO '{out}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)
"""

BARS = """
COPY (
  WITH src AS (
    SELECT condition_id, category_refined, block_timestamp, p_event, D, usdc_amount
    FROM read_parquet({glob}, union_by_name=true)
    WHERE p_event > 0 AND p_event < 1 AND usdc_amount > 0 AND D IS NOT NULL
  ), mkt AS (
    SELECT condition_id, sum(usdc_amount) AS vol,
           quantile_cont(usdc_amount, {large_q}) AS thr
    FROM src GROUP BY 1 HAVING sum(usdc_amount) >= {min_volume}
  ), tagged AS (
    SELECT s.condition_id, s.category_refined,
           (s.block_timestamp / {step})::BIGINT AS bar,
           s.block_timestamp, s.p_event, s.D, s.usdc_amount,
           s.usdc_amount >= m.thr AS is_large
    FROM src s JOIN mkt m USING (condition_id)
  )
  SELECT (hash(condition_id) >> 1)::BIGINT AS market_id, bar,
         any_value(category_refined) AS cat,
         count(*) AS n, sum(usdc_amount) AS gross,
         arg_max(p_event, block_timestamp) AS p,
         sum(CASE WHEN is_large THEN D * usdc_amount ELSE 0 END) AS sgn_large,
         sum(CASE WHEN is_large THEN usdc_amount ELSE 0 END)     AS abs_large,
         sum(CASE WHEN NOT is_large THEN D * usdc_amount ELSE 0 END) AS sgn_small,
         sum(CASE WHEN NOT is_large THEN usdc_amount ELSE 0 END)     AS abs_small
  FROM tagged GROUP BY 1, 2
) TO '{out}' (FORMAT parquet, COMPRESSION zstd)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", action="append",
                    choices=["daily_aligned", "daily_aligned_multi"])
    ap.add_argument("--freq", default="1h")
    ap.add_argument("--large-q", type=float, default=0.90)
    ap.add_argument("--min-volume", type=float, default=100_000)
    ap.add_argument("--memory-gb", type=int, default=20)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--only", choices=["trades", "bars"])
    a = ap.parse_args()
    layers = a.layer or ["daily_aligned", "daily_aligned_multi"]

    import duckdb, pandas as pd
    globs = [os.path.join(PMV1, L, "*.parquet") for L in layers
             if os.path.isdir(os.path.join(PMV1, L))]
    if not globs:
        sys.exit(f"no parquet under {PMV1}: run pm/pmv1_pull.py first")
    tmp = os.path.join(ROOT, "data", "duckdb_tmp")
    os.makedirs(tmp, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{a.memory_gb}GB'")
    con.execute(f"PRAGMA threads={a.threads}")
    con.execute(f"PRAGMA temp_directory='{tmp}'")

    step = int(pd.Timedelta(a.freq).total_seconds())
    jobs = []
    if a.only in (None, "trades"):
        jobs.append(("trades", TRADES.format(
            glob=str(globs), out=os.path.join(ROOT, "data", "pmv1_trades.parquet"))))
    if a.only in (None, "bars"):
        jobs.append(("bars", BARS.format(
            glob=str(globs), large_q=a.large_q, min_volume=a.min_volume, step=step,
            out=os.path.join(ROOT, "data", f"pmv1_bars_{a.freq}.parquet"))))
    for name, sql in jobs:
        t0 = time.time()
        print(f"building {name} ...", flush=True)
        con.execute(sql)
        out = sql.split("TO '")[1].split("'")[0]
        print(f"  {name}: {os.path.getsize(out)/1e9:.2f} GB in {time.time()-t0:.0f}s "
              f"-> {out}", flush=True)
        print("  rows:", con.execute(
            f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0], flush=True)


if __name__ == "__main__":
    main()
