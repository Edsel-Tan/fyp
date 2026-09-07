#!/usr/bin/env python3
"""Sign-randomisation skill test on the Polymarket tape (PAPERS.md [13], [15] Layer 1).

Gomez-Cram, Guo, Jensen & Kung separate skill from luck by re-simulating each
trader's bets with coin-flip directions and asking whether realised PnL sits in
the tail of that null. `fyp-preliminary-findings` records that pooled *trader ROI*
does not order top-3% against bottom-90% -- which is expected, because ROI is not
the estimator the paper uses. This implements the estimator it does use.

Two nulls are run, and the difference between them is the point:

  trade  -- flip the side of every trade independently, as published
  bet    -- flip the side of every (trader, market) position as a unit
  event  -- flip the side of every (trader, event) position, where an event is a
            cluster of near-duplicate markets (all 32 "Will <country> win the 2026
            FIFA World Cup?" contracts are one event)

A trader who buys the same token 60 times in one market has taken *one* bet, not
60, and a trader who sells 20 different countries in one World Cup event has taken
one view, not 20. Independent per-trade flips shrink the null standard deviation
by ~sqrt(n) and so must over-reject. Whether the null is calibrated is checkable:
under a correct null the z-statistic has unit standard deviation.
"""
import argparse, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from outcomes import resolve                                   # noqa: E402


def cluster_events(titles, thresh=0.6):
    """Connected components over title token-set Jaccard: near-duplicate markets
    (one per candidate/country/team) belong to a single real-world event."""
    import re
    toks = [set(re.findall(r"[a-z0-9]+", str(x).lower())) for x in titles]
    n = len(toks)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a_, b_ = toks[i], toks[j]
            if not a_ or not b_:
                continue
            if len(a_ & b_) / len(a_ | b_) >= thresh:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    return [find(i) for i in range(n)]


def build(min_trades):
    t, r = resolve()
    r = r.set_index("market_id")
    r["event"] = cluster_events(r.title.tolist())
    t = t[t.market_id.isin(r.index)].copy()
    t["win"] = (t.token_id == t.market_id.map(r.win_token)).astype(float)
    t["notional"] = t.price * t["size"]
    # signed edge of the trade if it were a BUY; the side supplies the sign
    t["q"] = (t.win - t.price) * t["size"]
    t["eps"] = np.where(t.side == "BUY", 1.0, -1.0)
    t["event"] = t.market_id.map(r.event)
    t["pnl"] = t.eps * t.q
    n = t.groupby("taker").size()
    return t[t.taker.isin(n[n >= min_trades].index)], r


def randomisation_test(t, unit, n_sim, rng):
    """One-sided p-value per taker: P(null PnL >= realised PnL)."""
    if unit == "trade":
        key = t.index.to_numpy()
    elif unit == "bet":                      # one coin flip per (taker, market) position
        key = t.groupby(["taker", "market_id"]).ngroup().to_numpy()
    else:                                    # one coin flip per (taker, event) position
        key = t.groupby(["taker", "event"]).ngroup().to_numpy()
    uk, kidx = np.unique(key, return_inverse=True)
    # a unit's contribution is the signed sum of its trades' q under the trader's
    # actual sides; flipping the unit flips that whole sum at once
    q_unit = np.bincount(kidx, weights=(t.q * t.eps).to_numpy())
    takers = t.taker.to_numpy()
    unit_taker = pd.Series(takers).groupby(kidx).first().to_numpy()
    codes, uniq = pd.factorize(unit_taker)
    n_t = len(uniq)

    actual = np.bincount(codes, weights=q_unit, minlength=n_t)
    ge = np.zeros(n_t)
    for _ in range(n_sim):
        s = rng.integers(0, 2, len(q_unit)) * 2 - 1
        sim = np.bincount(codes, weights=q_unit * s, minlength=n_t)
        ge += sim >= actual
    p = (ge + 1) / (n_sim + 1)
    sd = np.sqrt(np.bincount(codes, weights=q_unit ** 2, minlength=n_t))
    z = np.divide(actual, sd, out=np.zeros_like(actual), where=sd > 0)
    n_units = np.bincount(codes, minlength=n_t)
    return pd.DataFrame({"taker": uniq, "pnl": actual, "p": p, "z": z, "n_units": n_units})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-trades", type=int, default=10)
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    t, r = build(a.min_trades)
    print(f"resolved markets {len(r)}   distinct events {r.event.nunique()}   "
          f"trades {len(t):,}   takers with >={a.min_trades} trades: "
          f"{t.taker.nunique():,}   notional ${t.notional.sum():,.0f}")
    top = r.groupby("event").size().sort_values(ascending=False).head(4)
    for e, k in top.items():
        print(f"   event {e}: {k} markets   e.g. {r[r.event==e].title.iloc[0][:60]}")

    res = {}
    for unit in ("trade", "bet", "event"):
        d = randomisation_test(t, unit, a.sims, rng)
        res[unit] = d
        sk = (d.p < a.alpha).mean()
        lo = (d.p > 1 - a.alpha).mean()
        print(f"\n--- null: independent {unit}-level sign flips ({a.sims:,} sims) ---")
        print(f"accounts tested            {len(d):,}")
        print(f"units per account (median) {d.n_units.median():.0f}")
        print(f"'skilled' at p<{a.alpha:.2f}       {sk:6.2%}   (null expectation {a.alpha:.0%})")
        print(f"'anti-skilled' at p>{1-a.alpha:.2f}  {lo:6.2%}")
        print(f"excess over null           {sk - a.alpha:+.2%}")
        print(f"PnL held by p<{a.alpha:.2f} group   ${d[d.p<a.alpha].pnl.sum():,.0f} "
              f"of ${d.pnl.sum():,.0f} total")
        print(f"mean z                     {d.z.mean():+.3f}   sd z {d.z.std():.3f} "
              f"(N(0,1) under the null)")

    m = res["trade"].merge(res["bet"], on="taker", suffixes=("_trade", "_bet"))
    m = m.merge(res["event"].rename(columns={"p": "p_event", "z": "z_event",
                                             "pnl": "pnl_event", "n_units": "n_units_event"}),
                on="taker")
    print(f"\nflagged skilled by the published trade-level null but not by the "
          f"event-level null: {((m.p_trade < a.alpha) & (m.p_event >= a.alpha)).mean():.2%}")
    print(f"flagged by all three: {((m.p_trade < a.alpha) & (m.p_bet < a.alpha) & (m.p_event < a.alpha)).mean():.2%}")
    out = os.path.join(HERE, os.pardir, "data", "pm_skill.csv")
    m.to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
