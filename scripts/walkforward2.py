#!/usr/bin/env python3
"""Multi-period walk-forward on 20 years of tape.

The first analysis had one testable year, which is one regime and no evidence.
This re-forms the portfolio every year from 2011 to 2025 using only information
available on each formation date, holds each vintage un-rebalanced for twelve
months, and reports every vintage -- including the bad ones.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panel2 import total_return_panel, load_meta
from factors2 import prep, fundamentals_asof, factor_table, universe, zs

END = pd.Timestamp("2026-08-07")
N = 25
SECTOR_CAP = 4


def score_and_pick(u, n=N, cap=SECTOR_CAP, cols=None):
    c = u.copy()
    c["quality"] = zs(c["om"]) + zs(c["gm"]) + zs(c["roe"]) + zs(c["fcf_margin"])
    c["growth"] = zs(c["rev_g"]) + zs(c["eps_g"].fillna(c["rev_g"])) + zs(c["om_chg"])
    c["value"] = zs(c["ey"]) + zs(c["fcf_yield"]) + zs(c["sales_y"])
    c["compounding"] = zs(-c["accruals"]) + zs(c["cash_conv"])
    cols = cols or ["quality", "growth", "value", "compounding"]
    for k in cols:
        c[k] = c[k] / c[k].std()
    c["total"] = c[cols].mean(axis=1)
    c = c.sort_values("total", ascending=False)
    out, used = [], {}
    for sym, row in c.iterrows():
        s = row["sector"]
        if used.get(s, 0) >= cap:
            continue
        used[s] = used.get(s, 0) + 1
        out.append(sym)
        if len(out) == n:
            break
    return c.loc[out]


def screen(u):
    ok = (
        (u["mcap"] >= 2e9) & (u["mdvol"] >= 2e7)
        & (u["net_income_ttm"] > 0) & (u["operating_income_ttm"] > 0)
        & (u["free_cash_flow_ttm"] > 0)
        & (u["om"] > 0.05) & (u["gm"] > 0.20)
        & (u["rev_g"] > 0.02) & (u["om_chg"] > -0.05)
        & (u["roe"] > 0.10) & (u["roe"] < 2.0)
        & (u["equity_ratio"] > 0.15)
        & ~(u["nd_ebitda"] > 3.0)
        & (u["ey"] > 0.02)
        & (u["dd1y"] > -0.60)
    )
    return u[ok.fillna(False)].copy()


def hold(adj, syms, w, start, end):
    """Un-rebalanced NAV. A stopped tape is marked at its last print and held."""
    path = adj.loc[start:end, list(syms)].ffill()
    if not len(path):
        return None
    return (path / path.iloc[0] * np.asarray(w)).sum(axis=1)


def stats(nav):
    r = nav.pct_change().dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = nav.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    return cagr, vol, (nav / nav.cummax() - 1).min()


def main():
    close, adj, vol = total_return_panel()
    meta, inc, bal, cf, rep = load_meta()
    inc, bal, cf = prep(inc), prep(bal), prep(cf)

    forms = [pd.Timestamp(f"{y}-08-01") for y in range(2011, 2026)]
    cal = adj.index
    forms = [cal[cal.searchsorted(f)] for f in forms]

    rows = []
    port_by_year = {}
    for f in forms:
        nxt = cal[min(cal.searchsorted(f + pd.Timedelta(days=365)), len(cal) - 1)]
        funds = fundamentals_asof(inc, bal, cf, f)
        ft = factor_table(adj, vol, meta, funds, f)
        u = universe(ft, adj, f)
        c = screen(u)
        if len(c) < N:
            rows.append({"formed": f.date(), "pool": len(c), "port": np.nan,
                         "ew": np.nan, "cw": np.nan, "n": len(u)})
            continue
        p = score_and_pick(c)
        port_by_year[f] = p

        nav = hold(adj, p.index, np.repeat(1 / len(p), len(p)), f, nxt)
        path = adj.loc[f:nxt, u.index].ffill()
        g = (path.iloc[-1] / path.iloc[0]).dropna()
        cw = ((u["mcap"] / u["mcap"].sum()) * (g - 1)).sum()
        rows.append({"formed": f.date(), "n": len(u), "pool": len(c),
                     "port": nav.iloc[-1] - 1, "ew": g.mean() - 1, "cw": cw})

    r = pd.DataFrame(rows)
    r["excess_ew"] = r["port"] - r["ew"]
    r["excess_cw"] = r["port"] - r["cw"]
    pd.set_option("display.width", 200)
    print("=== annual vintages: form on 1 Aug, hold 12 months un-rebalanced ===")
    disp = r.copy()
    for c_ in ["port", "ew", "cw", "excess_ew", "excess_cw"]:
        disp[c_] = (disp[c_] * 100).round(1)
    print(disp.to_string(index=False))

    ok = r.dropna(subset=["port"])
    print(f"\nvintages: {len(ok)}")
    print(f"  mean   portfolio {ok['port'].mean():>7.2%}  equal-wt {ok['ew'].mean():>7.2%}  "
          f"cap-wt {ok['cw'].mean():>7.2%}")
    print(f"  median portfolio {ok['port'].median():>7.2%}  equal-wt {ok['ew'].median():>7.2%}  "
          f"cap-wt {ok['cw'].median():>7.2%}")
    print(f"  beat equal-weight in {int((ok['excess_ew'] > 0).sum())}/{len(ok)} years, "
          f"cap-weight in {int((ok['excess_cw'] > 0).sum())}/{len(ok)} years")
    print(f"  mean excess vs equal-wt {ok['excess_ew'].mean():>+7.2%}  "
          f"(t = {ok['excess_ew'].mean()/(ok['excess_ew'].std()/np.sqrt(len(ok))):.2f})")
    print(f"  mean excess vs cap-wt   {ok['excess_cw'].mean():>+7.2%}  "
          f"(t = {ok['excess_cw'].mean()/(ok['excess_cw'].std()/np.sqrt(len(ok))):.2f})")

    # compounded: roll each year's vintage into the next (annual rebalance)
    comp = (1 + ok["port"]).prod() ** (1 / len(ok)) - 1
    compew = (1 + ok["ew"]).prod() ** (1 / len(ok)) - 1
    compcw = (1 + ok["cw"]).prod() ** (1 / len(ok)) - 1
    print(f"\ncompounded across vintages (a fresh 25 every year): "
          f"portfolio {comp:.2%}  equal-wt {compew:.2%}  cap-wt {compcw:.2%}")

    r.to_csv("/tmp/claude-1000/-home-aimer-fyp/18d6b6c9-a1f0-42ac-bfd5-68b3ae7f7096/scratchpad/vintages.csv", index=False)
    return port_by_year


if __name__ == "__main__":
    main()
