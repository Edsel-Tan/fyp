#!/usr/bin/env python3
"""Emit the final buy-and-hold portfolio and its diagnostics."""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from portfolio_panel import build
from portfolio import build_portfolio
from walkforward import prep_fh

ASOF = pd.Timestamp("2026-08-07")
N = 25


def main():
    port, cand, data = build_portfolio(ASOF, N)
    close = data[0]

    w = 100.0 / len(port)
    out = port[["name" if "name" in port else "sector"]].copy() if False else pd.DataFrame(index=port.index)
    meta = data[2].set_index("symbol")
    out["company"] = meta["name"].reindex(port.index).str.slice(0, 34)
    out["sector"] = port["sector"]
    out["price"] = port["price"].round(2)
    out["weight%"] = round(w, 2)
    out["shares/$100k"] = (100_000 * w / 100 / port["price"]).round(0).astype(int)
    out["mcap$b"] = (port["mcap"] / 1e9).round(1)
    out["ROE"] = port["roe"].round(3)
    out["opmargin"] = port["om"].round(3)
    out["rev_g"] = port["rev_g"].round(3)
    out["earn_yld"] = port["ey"].round(3)
    out["buyback"] = port["buyback_yield"].round(3)
    out["netdebt/ebitda"] = port["nd_ebitda"].round(2)
    print(f"=== BUY-AND-HOLD PORTFOLIO, formed {ASOF.date()}, {len(port)} names, equal weight ===")
    print(out.to_string())

    print("\nsector mix:")
    sm = port["sector"].value_counts()
    for s, k in sm.items():
        print(f"  {s:24s} {k:2d} names  {k*w:5.1f}%")

    # data-hygiene checks on the names actually held
    print("\n--- data hygiene on held names ---")
    print(f"balance sheets that needed the x1000 unit repair: "
          f"{int(port['unit_fixed'].sum())}")
    px = close[port.index]
    r = px.pct_change()
    print(f"names with a >45% single-day drop (possible unadjusted split): "
          f"{int(((r < -0.45).sum() > 0).sum())}")
    print(f"names with a full 494-day tape: {int(px.notna().all().sum())} of {len(port)}")
    print(f"median daily $ volume, smallest name: ${port['mdvol'].min()/1e6:.1f}m")

    # what the same 25 names did over the whole 2y tape -- IN SAMPLE, for colour only
    path = close.loc[:ASOF, port.index].ffill()
    rel = (path.iloc[-1] / path.iloc[0])
    nav = (path / path.iloc[0]).mean(axis=1)
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    print(f"\n--- IN-SAMPLE ONLY (these names were chosen using data through {ASOF.date()}; "
          "this is not a forecast) ---")
    print(f"2y buy-and-hold CAGR of the held names: {nav.iloc[-1]**(1/yrs)-1:.2%}  "
          f"vol {nav.pct_change().std()*np.sqrt(252):.2%}  "
          f"maxDD {(nav/nav.cummax()-1).min():.2%}")
    print(f"best {rel.max()-1:.1%} ({rel.idxmax()}), worst {rel.min()-1:.1%} ({rel.idxmin()})")

    out.to_csv("portfolio.csv")
    print("\nwritten to portfolio.csv")


if __name__ == "__main__":
    main()
