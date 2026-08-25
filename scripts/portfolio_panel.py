#!/usr/bin/env python3
"""Build a clean, investable price + fundamentals panel from data/findata.sqlite.

Writes cached parquet-ish pickles into scratch so downstream scripts are fast.
"""
import json, os, sqlite3, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "findata.sqlite")
CACHE = os.environ.get("PANEL_CACHE", "/tmp/claude-1000/-home-aimer-fyp/18d6b6c9-a1f0-42ac-bfd5-68b3ae7f7096/scratchpad")
os.makedirs(CACHE, exist_ok=True)

END = "2026-08-07"          # last day with broad cross-sectional coverage
START = "2024-08-19"        # first day in the tape


def load_prices():
    f = os.path.join(CACHE, "px.pkl")
    if os.path.exists(f):
        return pd.read_pickle(f)
    con = sqlite3.connect(DB)
    df = pd.read_sql(
        "SELECT symbol, substr(ts,1,10) AS d, close, adj_close, volume, vwap "
        "FROM ohlc_daily WHERE ts <= '2026-08-07T99'", con)
    con.close()
    df["d"] = pd.to_datetime(df["d"])
    df.to_pickle(f)
    return df


def load_meta():
    con = sqlite3.connect(DB)
    rows = con.execute("SELECT payload FROM symbols").fetchall()
    meta = pd.DataFrame([json.loads(r[0]) for r in rows])
    act = pd.read_sql("SELECT DISTINCT symbol FROM actively_trading "
                      "WHERE snapshot_date='2026-08-09'", con)
    fl = pd.read_sql("SELECT symbol, period_end_date, period_type, report_date, payload "
                     "FROM fundamentals_latest", con)
    fh = pd.read_sql("SELECT symbol, period_end_date, period_type, payload "
                     "FROM fundamentals_history", con)
    con.close()
    meta["active"] = meta["symbol"].isin(set(act["symbol"]))
    fl = pd.concat([fl.drop(columns=["payload"]),
                    pd.DataFrame([json.loads(p) for p in fl["payload"]]).drop(
                        columns=["symbol", "period_end_date", "period_type",
                                 "report_date", "currency"], errors="ignore")], axis=1)
    fh = pd.concat([fh.drop(columns=["payload"]),
                    pd.DataFrame([json.loads(p) for p in fh["payload"]]).drop(
                        columns=["symbol", "period_end_date", "period_type",
                                 "report_date", "currency"], errors="ignore")], axis=1)
    return meta, fl, fh


def build():
    px = load_prices()
    meta, fl, fh = load_meta()

    # adj_close is the return series; fall back to close when missing
    px["p"] = px["adj_close"].where(px["adj_close"].notna(), px["close"])
    px = px[px["p"] > 0]
    px["dv"] = px["p"] * px["volume"]

    close = px.pivot_table(index="d", columns="symbol", values="p", aggfunc="last")
    dvol = px.pivot_table(index="d", columns="symbol", values="dv", aggfunc="last")
    # trading calendar = days where the broad market traded
    n_per_day = close.notna().sum(axis=1)
    cal = n_per_day[n_per_day > 0.5 * n_per_day.max()].index
    close = close.loc[cal].astype(float)
    dvol = dvol.loc[cal].astype(float)
    return close, dvol, meta, fl, fh


if __name__ == "__main__":
    close, dvol, meta, fl, fh = build()
    print("calendar", close.index.min().date(), "->", close.index.max().date(), len(close), "days")
    print("symbols with full history:", int(close.notna().all().sum()))
    print("median daily $vol > 5M:", int((dvol.median() > 5e6).sum()))
    print("meta cols:", list(meta.columns))
    print("fundamentals_latest cols:", list(fl.columns))
    print("active symbols:", int(meta['active'].sum()))
