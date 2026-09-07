#!/usr/bin/env python3
"""Sign-randomisation skill test at full scale, on Polymarket-v1 [16].

RESULTS.md sec.3.1 ran this on fifteen weeks of the findata vendor tape: 384
markets, 1,744 accounts, trade side *inferred*, and events recovered by
clustering market titles on token-set Jaccard. It found the published
trade-level null is mis-calibrated -- sd(z) = 2.37 where the null requires 1.00 --
and that correcting the randomisation unit to the event removes the "skilled
minority". Every limitation of that run is removed here:

  * 2022-11-21 -> 2026-04-28 instead of fifteen weeks;
  * `taker` and `D` are ground truth from blockchain settlement, not inferred --
    [16] shows the usual tick/quote classifiers are near-random on this venue,
    so an inferred side would attenuate the statistic towards zero;
  * `neg_risk_market_id` is the ground-truth event grouping: every candidate of
    a "who wins" market belongs to one negative-risk event by construction, so
    the event-level null no longer depends on a title-similarity threshold.

Four nulls are compared. The first three differ only in the unit that gets an
independent coin flip:

  trade  every fill flipped independently                        (as published)
  bet    every (taker, condition_id) position flipped as a unit
  event  every (taker, neg_risk_market_id | condition_id) flipped as a unit

The fourth is sec.3.1's stated next refinement. A taker crosses the spread, so
the expected payoff of a randomly-directed taker trade is negative, not zero;
a null centred at zero therefore reads ordinary transaction costs as negative
skill. `event (recentred)` shifts the null to the *typical* account, so the
question becomes whether an account beats the average taker rather than whether
it beats zero. The shift is measured as the population median of z, i.e. in the
statistic's own units -- recentring by expected cost in dollars instead would
divide a notional-scaled offset by a payoff-scaled deviation, which diverges for
accounts that round-trip a position and end with almost no exposure.

Two passes, for two reasons. The calibration diagnostic sd(z) is an exact
function of aggregates, so it is computed over the whole population with no
sampling and no simulation. The flagged-as-skilled fractions need the exact
Rademacher null rather than a normal approximation, so they are simulated on a
random sample of accounts.

    python pm/pmv1_prep.py            # build data/pmv1_trades.parquet first
    python pm/pmv1_skill.py --sims 10000 --sample 50000
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, os.pardir)
TRADES = os.path.join(ROOT, "data", "pmv1_trades.parquet")
UNIT_KEY = {"trade": None, "bet": "market_id", "event": "event_id"}


def connect(path, memory_gb, threads):
    import duckdb
    if not os.path.exists(path):
        sys.exit(f"{path} missing -- run pm/pmv1_prep.py first")
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{memory_gb}GB'")
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA temp_directory='{os.path.join(ROOT, 'data', 'duckdb_tmp')}'")
    con.execute(f"CREATE OR REPLACE VIEW t AS SELECT * FROM read_parquet('{path}')")
    return con


def population(con, min_trades, cache=None):
    if cache and os.path.exists(cache):
        return pd.read_parquet(cache)
    """Exact per-account aggregates for all three units -- no sampling, no simulation.

    Under a Rademacher null over units, S = sum_u eps_u s_u has mean zero and
    variance sum_u s_u^2, so z = S / sqrt(sum_u s_u^2) needs only sums. At the
    trade level the unit is the row, so sum_u s_u^2 is the plain sum of squares.
    """
    con.execute("""
      CREATE OR REPLACE TEMP TABLE bet AS
      SELECT taker_id, market_id, sum(s) AS s, sum(s * s) AS s2,
             count(*) AS n, sum(notional) AS notional
      FROM t GROUP BY 1, 2""")
    con.execute("""
      CREATE OR REPLACE TEMP TABLE ev AS
      SELECT taker_id, count(*) AS n_events, sum(s * s) AS var_event
      FROM (SELECT taker_id, event_id, sum(s) AS s FROM t GROUP BY 1, 2)
      GROUP BY 1""")
    return con.execute(f"""
      SELECT a.taker_id, a.n_trades, a.n_bets, e.n_events, a.pnl, a.notional,
             a.var_trade, a.var_bet, e.var_event
      FROM (SELECT taker_id, sum(n) AS n_trades, count(*) AS n_bets,
                   sum(s) AS pnl, sum(notional) AS notional,
                   sum(s2) AS var_trade, sum(s * s) AS var_bet
            FROM bet GROUP BY 1 HAVING sum(n) >= {min_trades}) a
      JOIN ev e USING (taker_id)""").df()


def sample_units(con, takers, unit):
    """Per-unit signed payoffs for a set of accounts, for the exact null."""
    con.register("samp_df", pd.DataFrame({"taker_id": np.asarray(takers, np.int64)}))
    key = UNIT_KEY[unit]
    if key is None:
        q = "SELECT taker_id, s FROM t SEMI JOIN samp_df USING (taker_id)"
    else:
        q = (f"SELECT taker_id, sum(s) AS s FROM t SEMI JOIN samp_df USING (taker_id) "
             f"GROUP BY taker_id, {key}")
    return con.execute(q).df()


def mc_pvalues(units, n_sim, rng, offset=None):
    """P(null >= realised) per account under the exact Rademacher null over units.

    `offset` shifts the realised statistic by the account's expected cost, which
    is what recentres the null on the average taker instead of on zero.
    """
    codes, uniq = pd.factorize(units.taker_id.to_numpy())
    q = units.s.to_numpy(np.float64)
    n_t = len(uniq)
    actual = np.bincount(codes, weights=q, minlength=n_t)
    target = actual if offset is None else actual - offset.reindex(uniq).to_numpy()
    ge = np.zeros(n_t)
    for _ in range(n_sim):                     # one draw at a time: a (sims x units)
        eps = rng.integers(0, 2, len(q))       # sign matrix would not fit in memory
        ge += np.bincount(codes, weights=np.where(eps, q, -q), minlength=n_t) >= target
    return pd.DataFrame({"taker_id": uniq, "pnl": actual,
                         "p": (ge + 1) / (n_sim + 1),
                         "n_units": np.bincount(codes, minlength=n_t)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default=TRADES)
    ap.add_argument("--min-trades", type=int, default=10)
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--sample", type=int, default=50000, help="accounts for the MC pass")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--memory-gb", type=int, default=20)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--cache", default=os.path.join(ROOT, "data", "pmv1_accounts.parquet"),
                    help="per-account aggregates; built once, reused on reruns")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    if a.cache and os.path.exists(a.cache):
        print(f"reusing cached account aggregates from {a.cache}")
    con = connect(a.trades, a.memory_gb, a.threads)
    n_rows, n_mkt, n_ev, n_tak, notional, t0, t1 = con.execute(
        "SELECT count(*), approx_count_distinct(market_id), approx_count_distinct(event_id), "
        "approx_count_distinct(taker_id), sum(notional), min(ts), max(ts) FROM t").fetchone()
    print(f"resolved fills {n_rows:,}   markets ~{n_mkt:,}   events ~{n_ev:,}   "
          f"accounts ~{n_tak:,}   taker notional ${notional:,.0f}")
    print(f"span {pd.Timestamp(t0, unit='s')} .. {pd.Timestamp(t1, unit='s')}")

    A = population(con, a.min_trades, a.cache)
    if a.cache and not os.path.exists(a.cache):
        A.to_parquet(a.cache, index=False)
    mu = float(A.pnl.sum() / A.notional.sum())      # average taker payoff per $ traded
    print(f"accounts with >={a.min_trades} fills: {len(A):,}   "
          f"their fills {A.n_trades.sum():,}   notional ${A.notional.sum():,.0f}   "
          f"realised PnL ${A.pnl.sum():,.0f}  ({mu:+.4%} of notional)")

    zs, cal = {}, {}
    variants = [("trade", "n_trades", "var_trade", False),
                ("bet", "n_bets", "var_bet", False),
                ("event", "n_events", "var_event", False),
                ("event (recentred)", "n_events", "var_event", True)]
    sd_ev = np.sqrt(A.var_event.to_numpy())
    z_ev = np.divide(A.pnl.to_numpy(), sd_ev, out=np.zeros(len(A)), where=sd_ev > 0)
    shift = float(np.median(z_ev))                  # the typical account's z
    print(f"\nmedian event-level z across the population: {shift:+.3f} "
          f"-- the spread the average taker pays, in units of its own payoff sd")
    print("\n--- calibration over the whole population (exact, no simulation) ---")
    print(f"{'unit':<24}{'median units/acct':>19}{'mean z':>10}{'sd(z)':>10}"
          f"{'z>1.645':>10}{'z<-1.645':>10}")
    for unit, nc, vc, recentre in variants:
        sd = np.sqrt(A[vc].to_numpy())
        z = np.divide(A.pnl.to_numpy(), sd, out=np.zeros(len(A)), where=sd > 0)
        if recentre:
            z = z - shift
        zs[unit] = z
        cal[unit] = dict(mean_z=float(z.mean()), sd_z=float(z.std(ddof=1)),
                         hi=float((z > 1.645).mean()), lo=float((z < -1.645).mean()),
                         median_units=float(A[nc].median()))
        print(f"{unit:<24}{A[nc].median():>19.0f}{z.mean():>+10.3f}{z.std(ddof=1):>10.3f}"
              f"{(z > 1.645).mean():>10.2%}{(z < -1.645).mean():>10.2%}")

    n = min(a.sample, len(A))
    # sort first: DuckDB's parallel aggregation does not fix row order, so sampling
    # by position would draw a different 50,000 accounts on every run
    samp = A.sort_values("taker_id", kind="stable").sample(n, random_state=0)
    takers = samp.taker_id.to_numpy()
    # the same shift, expressed per account in payoff dollars
    offset = pd.Series(shift * np.sqrt(samp.var_event.to_numpy()),
                       index=samp.taker_id.to_numpy())
    print(f"\n--- exact Rademacher null, {a.sims:,} sims on {n:,} sampled accounts ---")
    res, mc = {}, {}
    for unit, _, _, recentre in variants:
        base = "event" if recentre else unit
        u = res.get(base)
        if u is None:
            u = res[base] = sample_units(con, takers, base)
        d = mc_pvalues(u, a.sims, rng, offset if recentre else None)
        mc[unit] = d
        sk, lo = (d.p < a.alpha).mean(), (d.p > 1 - a.alpha).mean()
        cal[unit].update(skilled=float(sk), anti=float(lo))
        print(f"{unit:<24} units/acct median {d.n_units.median():>5.0f}   "
              f"skilled p<{a.alpha:.2f} {sk:>7.2%}   anti-skilled {lo:>7.2%}   "
              f"excess {sk - a.alpha:+.2%}   "
              f"PnL of flagged ${d[d.p < a.alpha].pnl.sum():,.0f} of ${d.pnl.sum():,.0f}")

    m = mc["trade"].rename(columns={"p": "p_trade", "n_units": "n_trade"})[
        ["taker_id", "pnl", "p_trade", "n_trade"]]
    for unit, col in (("bet", "bet"), ("event", "event"),
                      ("event (recentred)", "event_rc")):
        m = m.merge(mc[unit].rename(columns={"p": f"p_{col}", "n_units": f"n_{col}"})[
            ["taker_id", f"p_{col}", f"n_{col}"]], on="taker_id")
    only_trade = ((m.p_trade < a.alpha) & (m.p_event >= a.alpha)).mean()
    all3 = ((m.p_trade < a.alpha) & (m.p_bet < a.alpha) & (m.p_event < a.alpha)).mean()
    print(f"\nflagged by the published trade-level null but not the event-level null: "
          f"{only_trade:.2%}")
    print(f"flagged by all three units: {all3:.2%}")
    m.to_csv(os.path.join(ROOT, "data", "pmv1_skill.csv"), index=False)
    json.dump({"n_rows": int(n_rows), "n_markets": int(n_mkt), "n_events": int(n_ev),
               "n_takers": int(n_tak), "n_active": int(len(A)),
               "pnl_active": float(A.pnl.sum()), "notional_active": float(A.notional.sum()),
               "mu_per_dollar": mu, "z_shift": shift, "sims": a.sims, "sample": int(n),
               "calibration": cal, "only_trade": float(only_trade), "all3": float(all3)},
              open(os.path.join(ROOT, "data", "pmv1_skill.json"), "w"), indent=1)
    print("wrote data/pmv1_skill.{csv,json}")


if __name__ == "__main__":
    main()
