#!/usr/bin/env python3
"""The actual mandate: buy once, never trade, hold for years.

Every strategy here is formed on one date using only prior information and then
left completely alone. No rebalancing, no reconstitution, no replacement of
names that die -- a dead position is marked at its last print and carried.

Also reports the survivorship exposure directly, because over a 15-year hold it
is the assumption that matters most: if the tape simply stops for a failing
company rather than printing its way to zero, every long-hold return here is
biased upward.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panel2 import total_return_panel, load_meta
from factors2 import prep, fundamentals_asof, factor_table, universe
from walkforward2 import screen, score_and_pick

END = pd.Timestamp("2026-08-07")


def nav(adj, syms, w, start, end):
    path = adj.loc[start:end, list(syms)].ffill()
    return (path / path.iloc[0] * np.asarray(w)).sum(axis=1)


def stats(nav_series):
    r = nav_series.pct_change().dropna()
    yrs = (nav_series.index[-1] - nav_series.index[0]).days / 365.25
    cagr = nav_series.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    mdd = (nav_series / nav_series.cummax() - 1).min()
    return cagr, vol, mdd, cagr / vol if vol else np.nan


def run(start_year):
    close, adj, vol = total_return_panel()
    meta, inc, bal, cf, rep = load_meta()
    inc, bal, cf = prep(inc), prep(bal), prep(cf)

    cal = adj.index
    start = cal[cal.searchsorted(pd.Timestamp(f"{start_year}-08-01"))]
    yrs = (END - start).days / 365.25

    funds = fundamentals_asof(inc, bal, cf, start)
    ft = factor_table(adj, vol, meta, funds, start)
    u = universe(ft, adj, start)
    pool = screen(u)
    print(f"=== formed {start.date()}, held untouched to {END.date()} ({yrs:.1f} years) ===")
    print(f"investable universe {len(u)}, passing the quality screen {len(pool)}\n")

    strategies = {}
    if len(pool) >= 25:
        p = score_and_pick(pool, 25)
        strategies["Screened 25 (quality/value/growth)"] = p.index
    # no-skill baselines, all formed the same day
    for n in (25, 50, 100, 200):
        big = u.nlargest(n, "mcap")
        strategies[f"{n} largest, equal weight"] = big.index
    strategies["Whole universe, equal weight"] = u.index

    print(f"{'strategy':42} {'CAGR':>7} {'vol':>7} {'maxDD':>8} {'ret/vol':>8} {'x money':>8}")
    out = {}
    for name, syms in strategies.items():
        w = np.repeat(1 / len(syms), len(syms))
        nv = nav(adj, syms, w, start, END)
        c, v, d, s = stats(nv)
        out[name] = nv
        print(f"{name:42} {c:>7.2%} {v:>7.2%} {d:>8.1%} {s:>8.2f} {nv.iloc[-1]:>7.1f}x")

    # cap-weighted universe, weights struck once and left to drift
    cw = (u["mcap"] / u["mcap"].sum()).reindex(u.index)
    nv = nav(adj, u.index, cw.values, start, END)
    c, v, d, s = stats(nv)
    print(f"{'Whole universe, cap weight':42} {c:>7.2%} {v:>7.2%} {d:>8.1%} {s:>8.2f} {nv.iloc[-1]:>7.1f}x")
    out["Whole universe, cap weight"] = nv

    # ---- survivorship exposure ---------------------------------------------
    print("\n--- survivorship exposure over the hold ---")
    win = adj.loc[start:END, u.index]
    lastday = win.apply(lambda s: s.last_valid_index())
    stopped = lastday[lastday < END]
    print(f"names whose tape stops before the end: {len(stopped)} of {len(u)} "
          f"({len(stopped)/len(u):.1%})")
    if len(stopped):
        peak = win[stopped.index].cummax().ffill().iloc[-1]
        finalpx = win[stopped.index].ffill().iloc[-1]
        drop = (finalpx / peak - 1)
        print(f"  their last print vs their own peak: median {drop.median():.1%}, "
              f"{int((drop < -0.8).sum())} ended more than 80% below peak")
        entry = win[stopped.index].iloc[0]
        print(f"  return from entry to last print:    median "
              f"{(finalpx/entry - 1).median():.1%}")
    return out


if __name__ == "__main__":
    yr = int(sys.argv[1]) if len(sys.argv) > 1 else 2011
    run(yr)
