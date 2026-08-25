#!/usr/bin/env python3
"""Export the slice of findata the reproduction actually needs, as one npz.

The SQLite mirror is ~2 GB and the feature panel ~440 MB; neither ports well.
Everything downstream only touches a 370-name OHLCV panel, the closes that run
past it (for the dividend unwind), the dividend record, and the S&P 500 index.
That fits in ~20 MB, which makes the pipeline runnable on a fresh machine with
no re-pull.
"""
import json, os, sqlite3, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
DB = os.path.join(os.path.dirname(HERE), "data", "findata.sqlite")
OUT = os.path.join(ART, "prices_bundle.npz")

PANEL_START, PANEL_END = "2013-06-01", "2023-01-01"


def main():
    syms = json.load(open(os.path.join(ART, "universe.json")))["asof_2015-01-01_usable"]
    con = sqlite3.connect(DB)
    q = ",".join("?" * len(syms))

    # 1. the panel window, all five fields
    df = pd.read_sql_query(
        f"select symbol, substr(ts,1,10) d, open, high, low, close, volume from ohlc "
        f"where ts>=? and ts<? and symbol in ({q})",
        con, params=[PANEL_START, PANEL_END] + syms)
    counts = df.groupby("d").size()
    cal = sorted(counts[counts >= 0.5 * len(syms)].index)
    df = df[df.d.isin(set(cal))]
    fields = ["open", "high", "low", "close", "volume"]
    wide = {f: df.pivot(index="d", columns="symbol", values=f)
                 .reindex(cal)[sorted(syms)].ffill().bfill() for f in fields}
    keep = sorted(syms)

    # 2. closes after the panel, needed to unwind the dividend factor from the
    #    vendor's own 2026 adjustment reference back into the test years
    ext = pd.read_sql_query(
        f"select symbol, substr(ts,1,10) d, close from ohlc "
        f"where ts>=? and symbol in ({q})", con, params=[PANEL_END] + syms)
    ext_w = ext.pivot(index="d", columns="symbol", values="close")
    ext_w = ext_w.reindex(columns=keep)
    ext_cal = sorted(ext_w.index)

    # 3. dividends and 4. the index
    dv = pd.read_sql_query(
        f"select symbol, substr(ex_date,1,10) d, adj_amount, amount from dividends "
        f"where ex_date>=? and symbol in ({q})", con, params=[PANEL_START] + syms)
    dv["amt"] = dv.adj_amount.fillna(dv.amount).astype(float)
    dv = dv[dv.amt > 0]
    gs = pd.read_sql_query(
        "select substr(ts,1,10) d, close from ohlc where symbol='^GSPC' order by ts", con)
    con.close()

    np.savez_compressed(
        OUT,
        symbols=np.array(keep), dates=np.array(cal),
        # float64 throughout: the features are computed from these, and float32
        # rounding shifts them by ~1e-4, which would stop the committed run
        # results from being exactly reproducible on another machine
        **{f: wide[f].to_numpy(np.float64) for f in fields},
        ext_dates=np.array(ext_cal), ext_close=ext_w.loc[ext_cal].to_numpy(np.float64),
        div_symbol=dv.symbol.to_numpy(str), div_date=dv.d.to_numpy(str),
        div_amount=dv.amt.to_numpy(np.float64),
        gspc_dates=gs.d.to_numpy(str), gspc_close=gs.close.to_numpy(np.float64))
    mb = os.path.getsize(OUT) / 1e6
    print(f"wrote {OUT}  {mb:.1f} MB")
    print(f"  panel {len(cal)} days x {len(keep)} symbols  {cal[0]}..{cal[-1]}")
    print(f"  ext closes {len(ext_cal)} days  {ext_cal[0]}..{ext_cal[-1]}")
    print(f"  dividends {len(dv):,} rows   ^GSPC {len(gs):,} rows")


if __name__ == "__main__":
    main()
