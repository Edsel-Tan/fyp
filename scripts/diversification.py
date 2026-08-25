#!/usr/bin/env python3
"""How many names does an un-rebalanced portfolio need?

Buy-and-hold terminal wealth is LINEAR in the initial weights, so diversifying
cannot raise expected terminal wealth. But CAGR is log of wealth -- concave --
so spreading the bet does raise *expected CAGR* and collapses its dispersion.
This measures both effects on the actual cross-section, and measures how far
the weights drift when you never touch them.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from portfolio_panel import build
from factors import factor_table, universe
from walkforward import prep_fh, FORM, END

RNG = np.random.default_rng(7)
TRIALS = 4000


def main():
    close, dvol, meta, fl, fh = build()
    fh = prep_fh(fh)
    f = factor_table(close, dvol, meta, fh, FORM)
    u = universe(f, close, FORM)

    held = close.loc[FORM:END]
    path = held.ffill()[u.index]
    gross = (path.iloc[-1] / path.iloc[0]).dropna()      # 1 + r over the year
    u = u.loc[gross.index]
    print(f"universe {len(gross)}   mean 1+r {gross.mean():.3f}   "
          f"median {gross.median():.3f}   worst {gross.min():.3f}   best {gross.max():.3f}")

    print("\n--- random equal-weight portfolios, bought once and never touched ---")
    print(f"{'N':>4} {'mean CAGR':>10} {'median':>8} {'5th pct':>9} {'95th pct':>9} "
          f"{'P(lose money)':>14} {'top wt @1y':>11}")
    g = gross.values
    for n in [1, 3, 5, 10, 15, 20, 25, 30, 40, 60, 100, 200]:
        idx = RNG.integers(0, len(g), size=(TRIALS, n))
        picks = g[idx]
        wealth = picks.mean(axis=1)              # equal weight, no rebalancing
        # weight drift: largest end-of-year weight
        top = (picks / picks.sum(axis=1, keepdims=True)).max(axis=1)
        cagr = wealth - 1                        # 1-year hold, so CAGR == return
        print(f"{n:>4} {cagr.mean():>9.2%} {np.median(cagr):>8.2%} "
              f"{np.percentile(cagr, 5):>9.2%} {np.percentile(cagr, 95):>9.2%} "
              f"{(wealth < 1).mean():>14.1%} {top.mean():>11.1%}")

    print("\n--- concentration: what one bad name costs you, un-rebalanced ---")
    for n in [10, 20, 25, 30]:
        print(f"  N={n:<3} equal weight: a single name going to zero costs "
              f"{1/n:.1%} of capital; the other {n-1} keep compounding.")


if __name__ == "__main__":
    main()
