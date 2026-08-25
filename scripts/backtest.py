#!/usr/bin/env python3
"""Out-of-sample test of the selection rule.

Form the portfolio at 2025-08-07 using ONLY information available then
(--pit: income-statement history lagged 75 days, plus price), hold it
un-rebalanced for 12 months, and compare against the alternatives.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from portfolio_panel import build
from factors import factor_table, universe
from portfolio import build_portfolio, screen, score, pick
from walkforward import prep_fh

FORM = pd.Timestamp("2025-08-07")
END = pd.Timestamp("2026-08-07")
RNG = np.random.default_rng(11)


def path_stats(nav):
    r = nav.pct_change().dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = nav.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    mdd = (nav / nav.cummax() - 1).min()
    return cagr, vol, mdd, (cagr / vol if vol else np.nan)


def nav_of(close, syms, w, form, end):
    path = close.loc[form:end, list(syms)].ffill()
    rel = path / path.iloc[0]
    return (rel * np.asarray(w)).sum(axis=1)


def main():
    data = build()
    close, dvol, meta, fl, fh = data
    fh = prep_fh(fh)
    data = (close, dvol, meta, fl, fh)

    port, cand, _ = build_portfolio(FORM, 25, pit=True, data=data)
    print(f"=== OOS TEST: formed {FORM.date()} on point-in-time data, held to {END.date()} ===")
    print(f"passed the screen: {len(cand)}   held: {len(port)}")
    print("holdings:", ", ".join(port.index))

    w = np.repeat(1 / len(port), len(port))
    nav = nav_of(close, port.index, w, FORM, END)
    cagr, vol, mdd, sharpe = path_stats(nav)

    # benchmarks over the same window
    f = factor_table(close, dvol, meta, fh, FORM)
    u = universe(f, close, FORM)
    path = close.loc[FORM:END, u.index].ffill()
    gross = (path.iloc[-1] / path.iloc[0]).dropna()
    ew = gross.mean() - 1
    cw = ((u["mcap"] / u["mcap"].sum()) * (gross - 1)).sum()

    ew_nav = nav_of(close, gross.index, np.repeat(1 / len(gross), len(gross)), FORM, END)
    e_cagr, e_vol, e_mdd, e_sh = path_stats(ew_nav)

    print(f"\n{'':28} {'return':>8} {'vol':>7} {'maxDD':>8} {'ret/vol':>8}")
    print(f"{'screened 25, un-rebalanced':28} {cagr:>8.2%} {vol:>7.2%} {mdd:>8.2%} {sharpe:>8.2f}")
    print(f"{'equal-weight universe (1882)':28} {e_cagr:>8.2%} {e_vol:>7.2%} {e_mdd:>8.2%} {e_sh:>8.2f}")
    print(f"{'cap-weight universe':28} {cw:>8.2%}")

    # where does it sit in the distribution of random 25-name portfolios?
    g = gross.values
    idx = RNG.integers(0, len(g), size=(20000, 25))
    rnd = g[idx].mean(axis=1) - 1
    pct = (rnd < (nav.iloc[-1] - 1)).mean()
    print(f"\npercentile vs 20,000 random equal-weight 25-name portfolios: {pct:.1%}")
    print(f"  random 25 -> mean {rnd.mean():.2%}, 5th {np.percentile(rnd,5):.2%}, "
          f"95th {np.percentile(rnd,95):.2%}")

    # per-name contribution
    p = close.loc[FORM:END, port.index].ffill()
    nm = (p.iloc[-1] / p.iloc[0] - 1).sort_values(ascending=False)
    print("\nbest/worst holdings over the year:")
    print(pd.concat([nm.head(5), nm.tail(5)]).map(lambda x: f"{x:.1%}").to_string())

    # weight drift with no rebalancing
    endw = (p.iloc[-1] / p.iloc[0]) / (p.iloc[-1] / p.iloc[0]).sum()
    print(f"\nweight drift after 1y with zero trades: start 4.00% each -> "
          f"end max {endw.max():.2%}, min {endw.min():.2%}")


if __name__ == "__main__":
    main()
