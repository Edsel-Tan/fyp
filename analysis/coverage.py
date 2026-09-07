#!/usr/bin/env python3
"""Coverage audit of the findata crypto tape.

Motivation: any backtest is only as honest as the calendar it runs on. This
script measures what fraction of each calendar month the vendor actually serves,
per interval, and reports the pattern by month-of-year -- because if missingness
is seasonal, a naive backtest silently samples only part of the year.
"""
import calendar, collections, os, sqlite3, sys
import numpy as np
import pandas as pd

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data", "crypto.sqlite")


def report(interval, per_day):
    con = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT symbol, ts FROM ohlc WHERE interval=?", con, params=(interval,))
    con.close()
    df["ts"] = pd.to_datetime(df.ts)
    df["ym"] = df.ts.dt.strftime("%Y-%m")
    df["m"] = df.ts.dt.month

    n_sym = df.symbol.nunique()
    print(f"\n=== interval {interval}: {n_sym} symbols, {len(df):,} bars, "
          f"{df.ts.min().date()} .. {df.ts.max().date()} ===")

    # coverage of the deepest symbol, so symbol listing dates do not confound
    deep = df.symbol.value_counts().index[0]
    d = df[df.symbol == deep]
    got = d.groupby("ym").size()
    rows = []
    for ym, n in got.items():
        y, m = int(ym[:4]), int(ym[5:])
        exp = calendar.monthrange(y, m)[1] * per_day
        rows.append((ym, m, n, exp, n / exp))
    R = pd.DataFrame(rows, columns=["ym", "month", "bars", "expected", "frac"])

    by_m = R.groupby("month").frac.agg(["mean", "min", "max", "count"])
    print(f"reference symbol: {deep}")
    print(f"{'month':>6}{'mean cov':>10}{'min':>8}{'max':>8}{'n_years':>9}")
    for m, r in by_m.iterrows():
        print(f"{calendar.month_abbr[m]:>6}{r['mean']:>10.1%}{r['min']:>8.1%}"
              f"{r['max']:>8.1%}{int(r['count']):>9}")
    span = pd.period_range(d.ts.min(), d.ts.max(), freq="M")
    absent = len(span) - len(R)
    full = R[R.frac >= 0.90]
    print(f"months in span: {len(span)}   >=90% complete: {len(full)}   "
          f"<50% complete: {(R.frac < 0.5).sum()}   absent from tape entirely: {absent}")
    return R


if __name__ == "__main__":
    report("1d", 1)
    report("1hour", 24)
