#!/usr/bin/env python3
"""Final portfolio on the full data, plus the position-count decision.

Position count is re-derived at the horizon that actually applies. The earlier
1-year answer (25) understates it: dispersion between individual stocks widens
with the square root of time, so a basket that is diversified enough for twelve
months is not necessarily diversified enough for ten years.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panel2 import total_return_panel, load_meta, dividend_yield
from factors2 import prep, fundamentals_asof, factor_table, universe
from walkforward2 import screen, score_and_pick

ASOF = pd.Timestamp("2026-08-07")
RNG = np.random.default_rng(11)


def size_study(adj, univs, horizon_years=10, trials=3000):
    """Spread of an un-rebalanced equal-weight basket by size, at a long horizon."""
    print(f"=== how many positions, at a {horizon_years}-year hold ===")
    print(f"{'N':>4} {'median CAGR':>12} {'5th pct':>9} {'95th pct':>9} "
          f"{'P(<0)':>7} {'P(<5%/yr)':>10} {'top wt at end':>14}")
    rows = []
    for year, u in univs.items():
        start = u.attrs["start"]
        stop = start + pd.Timedelta(days=int(365.25 * horizon_years))
        if stop > ASOF:
            continue
        path = adj.loc[start:stop, u.index].ffill()
        g = (path.iloc[-1] / path.iloc[0]).dropna()
        yrs = (path.index[-1] - path.index[0]).days / 365.25
        rows.append((g.values, yrs))
    if not rows:
        return
    for n in (5, 10, 15, 20, 25, 30, 40, 60, 100):
        med, p5, p95, ploss, plow, topw = [], [], [], [], [], []
        for gv, yrs in rows:
            idx = RNG.integers(0, len(gv), size=(trials, n))
            picks = gv[idx]
            tot = picks.mean(axis=1)
            c = tot ** (1 / yrs) - 1
            med.append(np.median(c)); p5.append(np.percentile(c, 5))
            p95.append(np.percentile(c, 95)); ploss.append((c < 0).mean())
            plow.append((c < 0.05).mean())
            topw.append((picks / picks.sum(axis=1, keepdims=True)).max(axis=1).mean())
        print(f"{n:>4} {np.mean(med):>11.2%} {np.mean(p5):>9.2%} {np.mean(p95):>9.2%} "
              f"{np.mean(ploss):>7.1%} {np.mean(plow):>10.1%} {np.mean(topw):>14.1%}")


def main():
    close, adj, vol = total_return_panel()
    meta, inc, bal, cf, rep = load_meta()
    inc, bal, cf = prep(inc), prep(bal), prep(cf)
    cal = adj.index

    univs = {}
    for year in range(2011, 2017):
        start = cal[cal.searchsorted(pd.Timestamp(f"{year}-08-01"))]
        funds = fundamentals_asof(inc, bal, cf, start)
        ft = factor_table(adj, vol, meta, funds, start)
        u = universe(ft, adj, start)
        u.attrs["start"] = start
        univs[year] = u
    size_study(adj, univs, 10)

    # ---- the live portfolio -------------------------------------------------
    funds = fundamentals_asof(inc, bal, cf, ASOF)
    ft = factor_table(adj, vol, meta, funds, ASOF)
    u = universe(ft, adj, ASOF)
    pool = screen(u)
    n = int(os.environ.get("PORT_N", "30"))
    p = score_and_pick(pool, n)

    dy = dividend_yield(ASOF)
    p = p.assign(div_yield=(dy.reindex(p.index) / p["price"]).fillna(0.0))

    print(f"\n=== PORTFOLIO, formed {ASOF.date()} — {len(p)} names, equal weight ===")
    print(f"investable universe {len(u)}, passed the screen {len(pool)}\n")
    out = pd.DataFrame({
        "company": meta.set_index("symbol")["name"].reindex(p.index).str.slice(0, 30),
        "sector": p["sector"].str.slice(0, 20),
        "price": p["price"].round(2),
        "wt%": round(100.0 / len(p), 2),
        "sh/$100k": (100_000 / len(p) / p["price"]).round(0).astype(int),
        "mcap$b": (p["mcap"] / 1e9).round(1),
        "ROE": p["roe"].round(2),
        "opmgn": p["om"].round(2),
        "revg": p["rev_g"].round(2),
        "FCFyld": p["fcf_yield"].round(3),
        "ND/EB": p["nd_ebitda"].round(1),
        "divyld": p["div_yield"].round(3),
    })
    print(out.to_string())
    print("\nsector mix:")
    for s, k in p["sector"].value_counts().items():
        print(f"  {s:24s} {k:2d}  {k*100.0/len(p):5.1f}%")

    print("\n--- hygiene ---")
    px = adj[p.index]
    print(f"  full 20y tape: {int(px.notna().all().sum())}/{len(p)}   "
          f"smallest median $vol ${p['mdvol'].min()/1e6:.0f}m   "
          f"stalest filing {int(p['fund_age_d'].max())}d old")

    p.index.name = "symbol"
    p.assign(weight=1.0 / len(p)).to_csv("portfolio.csv")
    out.to_csv("portfolio_display.csv")
    print(f"\nwritten to portfolio.csv ({len(p)} names)")
    return p


if __name__ == "__main__":
    main()
