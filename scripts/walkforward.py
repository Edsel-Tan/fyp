#!/usr/bin/env python3
"""Out-of-sample check: rank the universe at 2025-08-07 on information available
then, hold un-rebalanced for 12 months, measure realised buy-and-hold return."""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from portfolio_panel import build
from factors import factor_table, universe, zs, LAG_DAYS

FORM = pd.Timestamp("2025-08-07")
END = pd.Timestamp("2026-08-07")


def prep_fh(fh):
    fh = fh.copy()
    fh["pe"] = pd.to_datetime(fh["period_end_date"], errors="coerce")
    fh = fh[fh["pe"].notna() & fh["period_type"].isin(["Q1", "Q2", "Q3", "Q4"])]
    fh["avail"] = fh["pe"] + pd.Timedelta(days=LAG_DAYS)
    return fh


def decile_report(f, fwd, col, n=10, minobs=200):
    d = pd.DataFrame({"x": pd.to_numeric(f[col], errors="coerce"), "r": fwd}).dropna()
    if len(d) < minobs:
        return None
    d["q"] = pd.qcut(d["x"].rank(method="first"), n, labels=False)
    g = d.groupby("q")["r"]
    out = pd.DataFrame({"mean": g.mean(), "median": g.median(), "n": g.size()})
    spread = out["mean"].iloc[-1] - out["mean"].iloc[0]
    ic = d["x"].rank().corr(d["r"].rank())
    return out, spread, ic


def main():
    close, dvol, meta, fl, fh = build()
    fh = prep_fh(fh)

    f = factor_table(close, dvol, meta, fh, FORM)
    u = universe(f, close, FORM)
    print(f"formation {FORM.date()}  investable universe: {len(u)}")

    # Realised buy-and-hold return. If a name's tape stops mid-hold (delisting,
    # acquisition, or an upstream coverage gap) the position is marked at its
    # last traded price and held flat -- no reinvestment, no silent dropping.
    held = close.loc[FORM:END]
    p0 = held.iloc[0].reindex(u.index)
    p1 = held.ffill().iloc[-1].reindex(u.index)
    fwd = (p1 / p0 - 1)
    stopped = held.apply(lambda s: s.last_valid_index()) .reindex(u.index) < END
    print(f"names whose tape stops before {END.date()}: {int(stopped.sum())} "
          f"({stopped.mean():.1%} of the universe)")
    u = u[fwd.notna()]
    fwd = fwd.dropna()
    print(f"universe with a priceable holding year: {len(u)}")
    print(f"equal-weight universe 12m return: {fwd.mean():.2%}   median: {fwd.median():.2%}")
    cap_w = u["mcap"] / u["mcap"].sum()
    print(f"cap-weighted universe 12m return:  {(cap_w * fwd).sum():.2%}")

    print("\n--- single-factor rank IC and top-minus-bottom decile spread (OOS year) ---")
    cands = ["mom12_1", "mom6", "trend", "vol", "dd1y", "rev_g", "eps_g", "gm", "om",
             "nm", "om_chg", "ey", "sales_y", "mcap"]
    rows = []
    for c in cands:
        res = decile_report(u, fwd, c)
        if res is None:
            continue
        out, spread, ic = res
        rows.append({"factor": c, "IC": ic, "D10-D1": spread,
                     "D1": out["mean"].iloc[0], "D10": out["mean"].iloc[-1],
                     "n": int(out["n"].sum())})
    r = pd.DataFrame(rows).set_index("factor").sort_values("IC")
    pd.set_option("display.width", 200)
    print((r * 100).round(1).to_string())
    return u, fwd, close, meta, fh, dvol


if __name__ == "__main__":
    main()
