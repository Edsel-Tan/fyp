#!/usr/bin/env python3
"""Fixed-length holds, so the result does not hang on one terminal date.

longhold_dist.py runs every start year to the same 2026 endpoint, which makes
all of its observations share a terminal regime -- they are 11 overlapping
windows, not 11 independent trials. This holds a fixed number of years from each
start instead, so each vintage ends somewhere different, and reports the horizon
at which any edge appears.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panel2 import total_return_panel, load_meta
from factors2 import prep, fundamentals_asof, factor_table, universe
from walkforward2 import screen, score_and_pick

END = pd.Timestamp("2026-08-07")
TRIALS = 1000
RNG = np.random.default_rng(7)


def main():
    close, adj, vol = total_return_panel()
    meta, inc, bal, cf, rep = load_meta()
    inc, bal, cf = prep(inc), prep(bal), prep(cf)
    cal = adj.index

    # cache the formation work once per start year
    picks, univs = {}, {}
    for year in range(2011, 2026):
        start = cal[cal.searchsorted(pd.Timestamp(f"{year}-08-01"))]
        funds = fundamentals_asof(inc, bal, cf, start)
        ft = factor_table(adj, vol, meta, funds, start)
        u = universe(ft, adj, start)
        pool = screen(u)
        if len(pool) < 25:
            continue
        picks[year] = (start, score_and_pick(pool, 25).index)
        univs[year] = u

    print(f"{'horizon':>8} {'n':>3} {'screen':>8} {'equal-wt':>9} {'cap-wt':>8} "
          f"{'excess vs EW':>13} {'t':>6} {'beat EW':>9} {'pctile':>7}")
    for h in (1, 2, 3, 5, 7, 10):
        rows = []
        for year, (start, syms) in picks.items():
            stop = start + pd.Timedelta(days=int(365.25 * h))
            if stop > END:
                continue
            u = univs[year]
            path = adj.loc[start:stop, u.index].ffill()
            if len(path) < 100:
                continue
            g = (path.iloc[-1] / path.iloc[0]).dropna()
            yrs = (path.index[-1] - path.index[0]).days / 365.25
            port = g.reindex(syms).dropna().mean() ** (1 / yrs) - 1
            ew = g.mean() ** (1 / yrs) - 1
            w = (u["mcap"] / u["mcap"].sum()).reindex(g.index)
            cw = ((g * w).sum() / w.sum()) ** (1 / yrs) - 1
            gv = g.values
            rnd = gv[RNG.integers(0, len(gv), size=(TRIALS, 25))].mean(axis=1) ** (1 / yrs) - 1
            rows.append((port, ew, cw, (rnd < port).mean()))
        if len(rows) < 3:
            continue
        a = np.array(rows)
        ex = a[:, 0] - a[:, 1]
        t = ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex))) if len(ex) > 1 else np.nan
        print(f"{h:>6}y  {len(a):>3} {a[:,0].mean():>8.2%} {a[:,1].mean():>9.2%} "
              f"{a[:,2].mean():>8.2%} {ex.mean():>+13.2%} {t:>6.2f} "
              f"{int((ex>0).sum()):>5}/{len(a):<3} {a[:,3].mean():>7.0%}")

    print("\nNote: vintages overlap at every horizon (annual starts, multi-year holds),")
    print("so these t-statistics overstate significance. Treat them as descriptive.")


if __name__ == "__main__":
    main()
