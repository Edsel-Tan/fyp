#!/usr/bin/env python3
"""Deep price panel built from the second-pass pull.

`ohlc` carries ~20 years instead of 2, so a walk-forward test can span several
regimes instead of one.

On the adjustment question -- `close` is ALREADY a total-return series, split
and dividend adjusted. The `adj_close` column being NULL for every row is a
red herring that cost the first analysis a wrong conclusion. The evidence:

  * AT&T's close in June 2009 is $5.53 while it paid $0.41 a quarter, which
    would be a 28% dividend yield if the series were raw prices.
  * Its only post-2009 split is the 1.324 WBD spin, which explains a 1.3x
    gap, not the 4.3x actually present. The residual 3.3x is exactly what
    ~16 years of reinvested 5-7% dividends compounds to.
  * In the most recent year, where the back-adjustment factor is ~1, the
    implied yield is a sane 4.3% -- so recent bars sit at the raw scale and
    only history is scaled, which is what back-adjustment looks like.
  * Realty Income's 20y "price" CAGR here is 10.9%, matching its real-world
    TOTAL return rather than its ~5% price return.

So dividends must NOT be applied on top; doing so double-counts them and
manufactured a 30%/yr yield for NVDA. The dividends table is still loaded, but
for computing yield as a fundamental signal, never for adjusting prices.
"""
import json, os, sqlite3, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "findata.sqlite")
CACHE = os.environ.get(
    "PANEL_CACHE",
    "/tmp/claude-1000/-home-aimer-fyp/18d6b6c9-a1f0-42ac-bfd5-68b3ae7f7096/scratchpad")
os.makedirs(CACHE, exist_ok=True)


def _con():
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=120)


def total_return_panel(rebuild=False):
    """(close, adjclose, volume) wide frames. close is already total return, so
    adjclose is the same series -- the tuple shape is kept so callers that ask
    for the return series explicitly keep reading correctly."""
    f = os.path.join(CACHE, "tr_panel.pkl")
    if os.path.exists(f) and not rebuild:
        return pd.read_pickle(f)

    con = _con()
    px = pd.read_sql("SELECT symbol, substr(ts,1,10) d, close, volume FROM ohlc "
                     "WHERE close > 0", con)
    con.close()

    px["d"] = pd.to_datetime(px["d"])
    close = px.pivot_table(index="d", columns="symbol", values="close", aggfunc="last").astype(float)
    vol = px.pivot_table(index="d", columns="symbol", values="volume", aggfunc="last").astype(float)

    # keep only real trading days: those where a broad slice of the tape printed
    n = close.notna().sum(axis=1)
    close = close.loc[n > 0.30 * n.max()]
    vol = vol.reindex(close.index)

    out = (close, close.copy(), vol)
    pd.to_pickle(out, f)
    return out


def dividend_yield(asof, window_days=365):
    """Trailing cash yield per symbol, for use as a fundamental signal only."""
    con = _con()
    d = pd.read_sql("SELECT symbol, ex_date, COALESCE(adj_amount, amount) amt "
                    "FROM dividends WHERE COALESCE(adj_amount, amount) > 0", con)
    con.close()
    d["ex_date"] = pd.to_datetime(d["ex_date"], errors="coerce")
    asof = pd.Timestamp(asof)
    d = d[(d["ex_date"] <= asof) & (d["ex_date"] > asof - pd.Timedelta(days=window_days))]
    return d.groupby("symbol")["amt"].sum()


def load_meta():
    con = _con()
    meta = pd.DataFrame([json.loads(r[0]) for r in con.execute("SELECT payload FROM symbols")])
    act = {r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM actively_trading WHERE snapshot_date='2026-08-09'")}
    inc = pd.read_sql("SELECT symbol, period_end_date, period_type, payload "
                      "FROM fundamentals_history", con)
    bal = pd.read_sql("SELECT symbol, period_end_date, period_type, payload FROM fund_balance", con)
    cf = pd.read_sql("SELECT symbol, period_end_date, period_type, payload FROM fund_cashflow", con)
    rep = pd.read_sql("SELECT symbol, period_end_date, report_date FROM fundamentals_latest", con)
    con.close()
    meta["active"] = meta["symbol"].isin(act)

    def explode(df):
        if not len(df):
            return df
        p = pd.DataFrame([json.loads(x) for x in df["payload"]])
        p = p.drop(columns=[c for c in ("symbol", "period_end_date", "period_type",
                                        "report_date", "currency") if c in p.columns])
        out = pd.concat([df.drop(columns=["payload"]).reset_index(drop=True),
                         p.reset_index(drop=True)], axis=1)
        out["pe"] = pd.to_datetime(out["period_end_date"], errors="coerce")
        return out[out["pe"].notna()]

    return meta, explode(inc), explode(bal), explode(cf), rep


if __name__ == "__main__":
    close, adj, vol = total_return_panel(rebuild="--rebuild" in sys.argv)
    print(f"calendar {close.index.min().date()} -> {close.index.max().date()}  "
          f"{len(close)} trading days, {close.shape[1]} symbols")
    for yrs in (2, 5, 10, 15, 20):
        cut = close.index.max() - pd.Timedelta(days=365 * yrs)
        n = (close.loc[cut:].notna().all() & close.loc[:cut].notna().any()).sum()
        print(f"  symbols with a continuous {yrs:>2}y history: {n}")
    # aggregate CAGR is the cheapest check that the series really is total
    # return: a price-only broad universe lands near 6-7%, total return 9-11%
    cut = close.index[close.index >= close.index.max() - pd.Timedelta(days=365*20)][0]
    sub = close.loc[cut:]
    full = sub.columns[sub.notna().all()]
    g = (sub[full].iloc[-1] / sub[full].iloc[0])
    yrs = (sub.index[-1] - sub.index[0]).days / 365.25
    print(f"\n{len(full)} names with a full {yrs:.1f}y history")
    print(f"  median CAGR {g.median()**(1/yrs)-1:.2%}   equal-weight CAGR "
          f"{g.mean()**(1/yrs)-1:.2%}")
