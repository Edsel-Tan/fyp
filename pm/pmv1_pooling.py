#!/usr/bin/env python3
"""Where does the residual over-dispersion in the skill null come from? ([13], [16])

RESULTS.md sec.3.1 leaves one question open. Widening the randomisation unit
from the trade to the negative-risk event drops the null's sd(z) from 1.628 to
1.304, but 1.304 is not 1.00, so the event-level null still over-rejects. The
section names the cheap test: "whether sd(z) falls further when units are pooled
by resolution date and underlying rather than by contract", and gives the
motivating case -- on 2025-06-17, 37 distinct Bitcoin markets close and the
negative-risk grouping collapses them to only 30 events, so an account holding
one view on BTC all day still receives thirty independent coin flips.

This script runs that test, and then asks what is left.

Two things can inflate sd(z) above 1. Units that are not independent decisions
(the pooling failure above), and genuine heterogeneity in per-unit edge across
accounts -- i.e. skill, which sec.3.2 shows is really there. They are separable,
because they have different signatures:

  * pooling failure is removed by *coarsening the unit*. So the ladder below
    widens the unit monotonically to its degenerate limit (one flip per
    account, where sd(z) = 1 by construction, which also checks the code).
    Whatever sd(z) survives maximal same-day pooling is not a pooling artefact.

  * skill *persists across a time split*, and its contribution to Var(z) is
    identified by that persistence. Writing an account's standardised payoff as
    z = delta_i * sqrt(n_i) + N(0,1) -- a per-unit edge delta_i accumulating
    over n_i units, plus null noise of unit variance -- gives
    Var(z) = 1 + V where V = Var(delta * sqrt(n)). Splitting the history in
    half puts about n/2 units in each side, so each half carries V/2, and the
    two halves share only delta:
        corr(z_in, z_out) = (V/2) / (1 + V/2)   =>   V = 2*rho / (1 - rho).
    sec.3.2's split-half correlations therefore predict the sd(z) that skill
    alone would produce, with no null and no pooling assumption.

If the ladder stalls above the value skill accounts for, the event-level null is
mis-specified for a third reason and sec.3.1's "necessary but not sufficient"
stands. If it lands on it, the residual over-dispersion *is* the skilled
minority, and the two sections are measuring one thing.

The ladder's coarse rungs deliberately over-pool: (taker, close date, category)
merges every Bitcoin market closing on one day, which is the motivating case,
but also merges unrelated markets that happen to share a day and a category.
That is the right direction for an upper bound -- it cannot leave a pooling
artefact behind.

    python pm/pmv1_prep.py                       # data/pmv1_trades.parquet
    python pm/pmv1_pooling.py
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, os.pardir)
PMV1 = os.path.join(ROOT, "data", "pmv1")
TRADES = os.path.join(ROOT, "data", "pmv1_trades.parquet")
MARKETS = os.path.join(ROOT, "data", "pmv1_markets.parquet")

# One row per resolved market: the close date is when its uncertainty is settled,
# so it is the axis a same-day pooling has to use. resolved_at is null on 63% of
# markets and close_at on 0.05%, so close_at leads and the last print backs it up.
MARKETS_SQL = """
COPY (
  SELECT (hash(condition_id) >> 1)::BIGINT AS market_id,
         any_value(category_refined)       AS cat,
         coalesce(min(epoch(close_at))::BIGINT, max(block_timestamp)) / 86400 AS close_day
  FROM read_parquet({glob}, union_by_name=true)
  WHERE resolution_status = 'resolved' AND winning_outcome_label IS NOT NULL
  GROUP BY 1
) TO '{out}' (FORMAT parquet, COMPRESSION zstd)
"""

# Each rung: the per-account null variance sum(s_u^2) when the coin is flipped
# once per unit u. The unit's payoff is the *sum* of s over its fills, so a
# wider unit lets an account's fills cancel before being squared -- which is
# exactly the arithmetic that shrinks an inflated z.
RUNGS = {
    "bet   = trader x market":          ["market_id"],
    "event = trader x neg-risk event":  ["event_id"],
    "trader x close-day x category":    ["close_day", "cat"],
    "trader x close-day":               ["close_day"],
    "trader (degenerate: one flip)":    [],
}


def build_markets(con, force=False):
    if os.path.exists(MARKETS) and not force:
        return
    globs = [os.path.join(PMV1, L, "*.parquet")
             for L in ("daily_aligned", "daily_aligned_multi")
             if glob.glob(os.path.join(PMV1, L, "*.parquet"))]
    if not globs:
        sys.exit(f"no parquet under {PMV1}: run pm/pmv1_pull.py first")
    print(f"building {MARKETS} ...", flush=True)
    con.execute(MARKETS_SQL.format(glob=str(globs), out=MARKETS))


def variance_of(con, keys):
    """Per-account sum of squared unit payoffs, for the unit keyed by `keys`."""
    if keys:
        inner = (f"SELECT taker_id, sum(s) AS s FROM j GROUP BY taker_id, "
                 + ", ".join(keys))
    else:
        inner = "SELECT taker_id, sum(s) AS s FROM j GROUP BY taker_id"
    return con.execute(f"""
      SELECT taker_id, sum(s * s) AS v, count(*) AS n_units
      FROM ({inner}) GROUP BY 1""").df()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default=TRADES)
    ap.add_argument("--min-trades", type=int, default=10)
    ap.add_argument("--memory-gb", type=int, default=24)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--rebuild-markets", action="store_true")
    ap.add_argument("--persistence", default=os.path.join(ROOT, "data"),
                    help="directory holding pmv1_persistence_<cut>.csv from sec.3.2")
    a = ap.parse_args()
    if not os.path.exists(a.trades):
        sys.exit(f"{a.trades} missing -- run pm/pmv1_prep.py first")

    import duckdb
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{a.memory_gb}GB'")
    con.execute(f"PRAGMA threads={a.threads}")
    con.execute(f"PRAGMA temp_directory='{os.path.join(ROOT, 'data', 'duckdb_tmp')}'")
    build_markets(con, a.rebuild_markets)

    # the active population is sec.3.1's: accounts with >= min_trades resolved fills
    con.execute(f"""
      CREATE OR REPLACE TEMP TABLE act AS
      SELECT taker_id, count(*) AS n_trades, sum(s) AS pnl, sum(s * s) AS var_trade,
             sum(notional) AS notional
      FROM read_parquet('{a.trades}') GROUP BY 1 HAVING count(*) >= {a.min_trades}""")
    con.execute(f"""
      CREATE OR REPLACE TEMP VIEW j AS
      SELECT t.taker_id, t.event_id, t.market_id, t.s, m.close_day, m.cat
      FROM read_parquet('{a.trades}') t
      SEMI JOIN act a ON a.taker_id = t.taker_id
      JOIN read_parquet('{MARKETS}') m ON m.market_id = t.market_id""")

    A = con.execute("SELECT * FROM act").df().set_index("taker_id")
    n_day, n_daycat = con.execute(
        f"SELECT count(DISTINCT close_day), count(DISTINCT (close_day, cat)) "
        f"FROM read_parquet('{MARKETS}')").fetchone()
    print(f"accounts (>= {a.min_trades} fills) {len(A):,}   fills {A.n_trades.sum():,}   "
          f"PnL ${A.pnl.sum():,.0f}   close-days {n_day:,}   day x category cells {n_daycat:,}")

    V = {"trade = one fill (as published)": A.var_trade.to_numpy()}
    med = {"trade = one fill (as published)": float(A.n_trades.median())}
    for name, keys in RUNGS.items():
        d = variance_of(con, keys).set_index("taker_id").reindex(A.index)
        V[name], med[name] = np.nan_to_num(d.v.to_numpy()), float(d.n_units.median())
    pnl = A.pnl.to_numpy()

    def stats(v, mask=None):
        """Calibration of z = PnL / sqrt(sum of squared unit payoffs).

        The null fixes the *second* moment: E[z] = 0 and Var(z) = 1 give
        E[z^2] = 1 exactly, so rms(z) is the diagnostic and it decomposes as
        rms^2 = mean^2 + sd^2 -- a location part, which is the spread the
        average taker crosses, plus a dispersion part. sec.3.1 quotes the
        centred sd, which measures only the second of those.
        """
        sd = np.sqrt(v if mask is None else v[mask])
        q = pnl if mask is None else pnl[mask]
        z = np.divide(q, sd, out=np.zeros(len(q)), where=sd > 0)
        return dict(mean_z=float(z.mean()), sd_z=float(z.std(ddof=1)),
                    rms_z=float(np.sqrt((z ** 2).mean())),
                    hi=float((z > 1.645).mean()), lo=float((z < -1.645).mean()))

    print(f"\n--- calibration by randomisation unit, exact over all {len(A):,} accounts ---")
    print(f"{'randomisation unit':<36}{'units/acct':>11}{'mean z':>9}{'sd(z)':>8}"
          f"{'rms(z)':>9}{'z>1.645':>10}{'z<-1.645':>10}")
    out = []
    for name, v in V.items():
        r = dict(unit=name, units=med[name], **stats(v))
        out.append(r)
        print(f"{name:<36}{r['units']:>11.0f}{r['mean_z']:>+9.3f}{r['sd_z']:>8.3f}"
              f"{r['rms_z']:>9.3f}{r['hi']:>10.2%}{r['lo']:>10.2%}")
    print("the degenerate rung is a check on the arithmetic, not a result: one flip "
          "makes\nz = +-1, so rms(z) must be exactly 1.000 and sd(z) = sqrt(1 - mean^2).")

    # --- what skill alone would produce, from sec.3.2's split-half correlations ---
    # The paired accounts are a selected subpopulation -- surviving to trade in both
    # halves is itself a filter (RESULTS.md sec.3.2) -- so the ladder is recomputed on
    # exactly those accounts, or the comparison is between two different populations.
    print("\n--- Var(z) implied by out-of-sample persistence (no null involved) ---")
    print(f"{'cut':<12}{'accounts':>9}{'Pearson r':>11}{'se':>7}{'Spearman':>10}"
          f"{'V=2r/(1-r)':>12}{'sd(z)|skill':>13}{'their sd(z)':>13}{'their rms':>11}")
    idx = pd.Index(A.index)
    dec = []
    for f in sorted(glob.glob(os.path.join(a.persistence, "pmv1_persistence_2*.csv"))):
        cut = os.path.basename(f)[len("pmv1_persistence_"):-len(".csv")]
        m = pd.read_csv(f)
        # winsorise at 0.5% a side: a few accounts hold one huge resolved position,
        # and Pearson on that tail is effectively a two-point estimate
        zi, zo = (m[c].clip(m[c].quantile(.005), m[c].quantile(.995))
                  for c in ("z_in", "z_out"))
        r = float(np.corrcoef(zi, zo)[0, 1])
        se = (1 - r ** 2) / np.sqrt(len(m) - 3)
        Vskill = 2 * r / (1 - r)
        mask = idx.isin(m.taker_id.to_numpy())
        sub = stats(V["event = trader x neg-risk event"], mask)
        row = dict(cut=cut, n=int(len(m)), pearson=r, se=float(se),
                   spearman=float(m.z_in.rank().corr(m.z_out.rank())),
                   V=float(Vskill), sd_skill=float(np.sqrt(1 + Vskill)),
                   sd_sub=sub["sd_z"], rms_sub=sub["rms_z"], n_sub=int(mask.sum()))
        dec.append(row)
        print(f"{cut:<12}{len(m):>9,}{r:>+11.4f}{se:>7.4f}{row['spearman']:>+10.4f}"
              f"{Vskill:>12.3f}{row['sd_skill']:>13.3f}{sub['sd_z']:>13.3f}"
              f"{sub['rms_z']:>11.3f}")

    ev = [r for r in out if r["unit"].startswith("event")][0]
    pooled = [r for r in out if r["unit"] == "trader x close-day"][0]
    print(f"\nevent-level sd(z) {ev['sd_z']:.3f} -> maximal same-day pooling "
          f"{pooled['sd_z']:.3f}: pooling buys {ev['sd_z'] - pooled['sd_z']:+.3f} of the "
          f"{ev['sd_z'] - 1:.3f} to explain.")
    if dec:
        lo, hi = min(d["sd_skill"] for d in dec), max(d["sd_skill"] for d in dec)
        print(f"persistence implies skill alone would give sd(z) {lo:.3f}-{hi:.3f}, "
              f"against {pooled['sd_z']:.3f} after pooling.")
    json.dump({"n_accounts": int(len(A)), "ladder": out, "persistence": dec},
              open(os.path.join(ROOT, "data", "pmv1_pooling.json"), "w"), indent=1)
    print("wrote data/pmv1_pooling.json")


if __name__ == "__main__":
    main()
