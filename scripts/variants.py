#!/usr/bin/env python3
"""Sensitivity of the OOS year to each scoring pillar.

Reported in full, including the variants that lose. Any variant chosen AFTER
seeing this table is no longer an out-of-sample result -- see the notes in
report.md for which choices were made ex ante and which were not.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from portfolio_panel import build
from factors import factor_table, universe, zs
from portfolio import screen, score, pick, add_balance_sheet
from walkforward import prep_fh
from backtest import path_stats, nav_of, FORM, END


def main():
    close, dvol, meta, fl, fh = build()
    fh = prep_fh(fh)
    f = factor_table(close, dvol, meta, fh, FORM)
    u = universe(f, close, FORM)
    c = score(screen(u, pit=True), pit=True)

    path = close.loc[FORM:END, u.index].ffill()
    gross = (path.iloc[-1] / path.iloc[0]).dropna()
    print(f"universe {len(gross)}  equal-weight {gross.mean()-1:.2%}   "
          f"screened pool {len(c)}  equal-weight {gross.reindex(c.index).dropna().mean()-1:.2%}")

    combos = {
        "quality+growth+value+risk": ["quality", "growth", "value", "risk"],
        "quality+growth+value": ["quality", "growth", "value"],
        "quality+growth": ["quality", "growth"],
        "quality only": ["quality"],
        "growth only": ["growth"],
        "value only": ["value"],
        "risk only (low vol)": ["risk"],
    }
    print(f"\n{'variant':30} {'ret':>8} {'vol':>7} {'maxDD':>8} {'ret/vol':>8}  top sectors")
    for name, cols in combos.items():
        p = pick(c, 25, cols=cols)
        nav = nav_of(close, p.index, np.repeat(1 / len(p), len(p)), FORM, END)
        r, v, d, s = path_stats(nav)
        top = ", ".join(p["sector"].value_counts().head(3).index)
        print(f"{name:30} {r:>8.2%} {v:>7.2%} {d:>8.2%} {s:>8.2f}  {top}")

    # how much of the loss is the bond-proxy sleeve?
    p = pick(c, 25)
    defensive = p["sector"].isin(["Utilities", "Real Estate"])
    g = gross.reindex(p.index)
    print(f"\nin the full-score 25: {int(defensive.sum())} utility/REIT names "
          f"returned {g[defensive].mean()-1:.2%} (price only), the other "
          f"{int((~defensive).sum())} returned {g[~defensive].mean()-1:.2%}")


if __name__ == "__main__":
    main()
