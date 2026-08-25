#!/usr/bin/env python3
"""Is the long-hold result skill, or the dispersion of holding only 25 names?

A single 15-year hold of 25 stocks is one draw from a very wide distribution.
This runs the same buy-once-and-hold test from every start year, and against
each one compares the screened portfolio to 2,000 random 25-name portfolios
drawn from the identical investable universe on the identical date. The
percentile is the answer: 50% means the screen did nothing.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panel2 import total_return_panel, load_meta
from factors2 import prep, fundamentals_asof, factor_table, universe
from walkforward2 import screen, score_and_pick

END = pd.Timestamp("2026-08-07")
TRIALS = 2000
RNG = np.random.default_rng(20260817)


def main():
    close, adj, vol = total_return_panel()
    meta, inc, bal, cf, rep = load_meta()
    inc, bal, cf = prep(inc), prep(bal), prep(cf)
    cal = adj.index

    rows = []
    for year in range(2011, 2022):
        start = cal[cal.searchsorted(pd.Timestamp(f"{year}-08-01"))]
        yrs = (END - start).days / 365.25
        funds = fundamentals_asof(inc, bal, cf, start)
        ft = factor_table(adj, vol, meta, funds, start)
        u = universe(ft, adj, start)
        pool = screen(u)
        if len(pool) < 25:
            continue
        p = score_and_pick(pool, 25)

        path = adj.loc[start:END, u.index].ffill()
        g = (path.iloc[-1] / path.iloc[0])          # total growth per name
        g = g[g.notna()]

        def cagr(total):
            return total ** (1 / yrs) - 1

        port = cagr(g.reindex(p.index).dropna().mean())
        ew = cagr(g.mean())
        cw_w = (u["mcap"] / u["mcap"].sum()).reindex(g.index)
        cw = cagr((g * cw_w).sum() / cw_w.sum())

        gv = g.values
        idx = RNG.integers(0, len(gv), size=(TRIALS, 25))
        rnd = gv[idx].mean(axis=1) ** (1 / yrs) - 1
        pct = (rnd < port).mean()

        # how wide is the 25-name draw over this horizon?
        rows.append({
            "formed": start.date(), "yrs": round(yrs, 1), "univ": len(u), "pool": len(pool),
            "screen": port, "ew": ew, "cw": cw, "rand_med": np.median(rnd),
            "rand_p5": np.percentile(rnd, 5), "rand_p95": np.percentile(rnd, 95),
            "pctile": pct,
        })

    r = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    d = r.copy()
    for c in ["screen", "ew", "cw", "rand_med", "rand_p5", "rand_p95"]:
        d[c] = (d[c] * 100).round(1)
    d["pctile"] = (d["pctile"] * 100).round(0)
    print("=== buy once on `formed`, hold untouched to 2026-08-07 (CAGR %) ===")
    print(d.to_string(index=False))

    print(f"\nscreen beat equal-weight in {int((r['screen']>r['ew']).sum())}/{len(r)} start years, "
          f"cap-weight in {int((r['screen']>r['cw']).sum())}/{len(r)}")
    print(f"mean percentile against random 25-name portfolios: {r['pctile'].mean():.0%} "
          f"(50% = no skill)")
    print(f"mean CAGR  screen {r['screen'].mean():.2%}   equal-wt {r['ew'].mean():.2%}   "
          f"cap-wt {r['cw'].mean():.2%}   random-25 median {r['rand_med'].mean():.2%}")
    print(f"\nspread of a random 25-name buy-and-hold, averaged over start years: "
          f"5th {r['rand_p5'].mean():.2%} -> 95th {r['rand_p95'].mean():.2%}  "
          f"(width {r['rand_p95'].mean()-r['rand_p5'].mean():.1%} of CAGR)")
    r.to_csv("/tmp/claude-1000/-home-aimer-fyp/18d6b6c9-a1f0-42ac-bfd5-68b3ae7f7096/scratchpad/longhold.csv", index=False)


if __name__ == "__main__":
    main()
