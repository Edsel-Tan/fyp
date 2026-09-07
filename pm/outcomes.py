#!/usr/bin/env python3
"""Derive each market's resolved outcome from the trade tape.

The undocumented /prediction-markets/markets/polymarket/{id} route that the
August analysis used for `outcome_prices` now 404s for every market and is absent
from the OpenAPI spec, and /candles aggregates both tokens into one series (a
single day shows open 0.50, high 0.999, low 0.001), so neither can name a winner.

Resolution is therefore read off the tape: a Polymarket binary settles at 1 and 0,
so in the final trades of a resolved market one token prints at ~1 and the other
at ~0. A market is only accepted when that split is unambiguous.
"""
import os, sqlite3, sys
import numpy as np
import pandas as pd

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data", "pm.sqlite")


def resolve(tail=25, hi=0.95, lo=0.05, min_trades=40):
    con = sqlite3.connect(DB)
    t = pd.read_sql_query("SELECT market_id, token_id, side, price, size, taker, ts FROM trades", con)
    mk = pd.read_sql_query("SELECT market_id, title, volume, end_date FROM markets", con)
    con.close()
    t["ts"] = pd.to_datetime(t.ts)
    t = t.sort_values("ts")

    n = t.groupby("market_id").size()
    keep = set(n[n >= min_trades].index)
    t = t[t.market_id.isin(keep)]

    last = t.groupby(["market_id", "token_id"]).tail(tail)
    px = last.groupby(["market_id", "token_id"]).price.mean().rename("p_end")
    cnt = last.groupby(["market_id", "token_id"]).size().rename("n_end")
    e = pd.concat([px, cnt], axis=1).reset_index()

    rows = []
    for mid, g in e.groupby("market_id"):
        if len(g) != 2:
            continue
        g = g.sort_values("p_end")
        loser, winner = g.iloc[0], g.iloc[1]
        if winner.p_end >= hi and loser.p_end <= lo:
            rows.append((mid, winner.token_id, loser.token_id,
                         float(winner.p_end), float(loser.p_end)))
    out = pd.DataFrame(rows, columns=["market_id", "win_token", "lose_token",
                                      "p_win_end", "p_lose_end"])
    out = out.merge(mk, on="market_id", how="left")
    return t, out


if __name__ == "__main__":
    t, r = resolve()
    print(f"markets with a usable tape : {t.market_id.nunique()}")
    print(f"cleanly resolved on tape   : {len(r)}")
    print(f"trades in resolved markets : {t[t.market_id.isin(set(r.market_id))].shape[0]:,}")
    print(f"distinct takers            : {t[t.market_id.isin(set(r.market_id))].taker.nunique():,}")
    print(f"notional in resolved mkts  : "
          f"${(lambda x: (x.price*x['size']).sum())(t[t.market_id.isin(set(r.market_id))]):,.0f}")
    print()
    print(r.sort_values("volume", ascending=False)[["title", "volume", "p_win_end", "p_lose_end"]]
          .head(12).to_string(index=False))
    r.to_csv(os.path.join(os.path.dirname(DB), "pm_resolved.csv"), index=False)
    print("\nwrote data/pm_resolved.csv")
