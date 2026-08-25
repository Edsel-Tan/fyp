#!/usr/bin/env python3
"""Does the screen name the same companies month to month?

A rule whose output churns every month is not a buy-and-hold rule -- it just
looks like one because you only ran it once. Re-form at earlier dates and
measure overlap with the live portfolio.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from portfolio_panel import build
from portfolio import build_portfolio
from walkforward import prep_fh

ASOF = pd.Timestamp("2026-08-07")


def main():
    close, dvol, meta, fl, fh = build()
    fh = prep_fh(fh)
    data = (close, dvol, meta, fl, fh)

    # Turnover MUST be measured on the point-in-time path. fundamentals_latest
    # is one snapshot per symbol taken at the pull date, so re-running the full
    # screen at an earlier `asof` filters it to whichever companies had not
    # reported since -- i.e. only stale filers survive, and overlap collapses to
    # zero for reasons that have nothing to do with turnover. The --pit screen
    # reads fundamentals_history, which is genuinely as-of.
    base, _, _ = build_portfolio(ASOF, 25, pit=True, data=data)
    base_set = set(base.index)

    print("re-forming the point-in-time screen at earlier dates:\n")
    print(f"{'formed':12} {'overlap with live 25':>22} {'names dropped':>14}")
    cal = close.loc[:ASOF].index
    for back in [21, 42, 63, 126, 189, 252]:
        d = cal[-1 - back]
        p, _, _ = build_portfolio(d, 25, pit=True, data=data)
        ov = len(base_set & set(p.index))
        print(f"{str(d.date()):12} {ov:>13}/25 ({ov/25:>4.0%}) {25-ov:>14}")

    # N sensitivity: does the top of the list survive widening the portfolio?
    print("\nhow the holding list grows with N (live date):")
    prev = None
    for n in [10, 15, 20, 25, 30, 40]:
        p, _, _ = build_portfolio(ASOF, n, data=data)
        keep = "" if prev is None else f"  keeps {len(set(p.index) & prev)}/{len(prev)} of N={len(prev)}"
        print(f"  N={n:<3} sectors {p['sector'].nunique():>2}"
              f"  max sector weight {p['sector'].value_counts().max()/n:>5.0%}{keep}")
        prev = set(p.index)


if __name__ == "__main__":
    main()
