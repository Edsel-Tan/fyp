#!/usr/bin/env python3
"""Crypto feature panel from the findata hourly tape.

The tape is not continuous. Retention is deterministic by calendar month:
February and August are always empty, Jan/Mar/Jul/Sep are partial, and
Apr/May/Jun/Oct/Nov/Dec are ~97% complete (see analysis/coverage.py). Rather
than forward-fill across a month-long hole -- which would fabricate returns and
corrupt every rolling feature -- the panel is cut into contiguous *blocks* of
hourly bars and every downstream window is confined to a block.

Output: crypto/artifacts/panel_1h.npz with a `block` index per row.
"""
import argparse, os, sqlite3, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "hrt"))
from data import alpha158                                     # noqa: E402

DB = os.path.join(HERE, os.pardir, "data", "crypto.sqlite")
ART = os.path.join(HERE, "artifacts")
FIELDS = ["open", "high", "low", "close", "volume"]


def blocks_of(index, step_h, min_len):
    """Split a DatetimeIndex into runs of consecutive bars."""
    gap = index.to_series().diff().dt.total_seconds().div(3600).fillna(step_h)
    cut = np.flatnonzero((gap != step_h).to_numpy())
    out, start = [], 0
    for c in list(cut) + [len(index)]:
        if c - start >= min_len:
            out.append((start, c))
        start = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1hour")
    ap.add_argument("--min-block", type=int, default=400)
    ap.add_argument("--min-coverage", type=float, default=0.98)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    step = {"1hour": 1, "4hour": 4, "1d": 24}[a.interval]
    out = a.out or os.path.join(ART, f"panel_{a.interval}.npz")
    os.makedirs(ART, exist_ok=True)

    con = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT symbol, ts, open, high, low, close, volume "
                           "FROM ohlc WHERE interval=?", con, params=(a.interval,))
    con.close()
    df["ts"] = pd.to_datetime(df.ts)
    wide = {f: df.pivot_table(index="ts", columns="symbol", values=f) for f in FIELDS}
    idx = wide["close"].index.sort_values()
    wide = {f: w.reindex(idx) for f, w in wide.items()}
    print(f"{a.interval}: {len(idx)} timestamps, {wide['close'].shape[1]} symbols, "
          f"{idx[0]} .. {idx[-1]}")

    bl = blocks_of(idx, step, a.min_block)
    print(f"contiguous blocks >= {a.min_block} bars: {len(bl)}  "
          f"(covering {sum(hi-lo for lo,hi in bl)} of {len(idx)} bars)")

    # universe: symbols present in essentially every retained block
    cov = pd.concat([wide["close"].iloc[lo:hi].notna().mean() for lo, hi in bl], axis=1)
    keep = sorted(cov.index[(cov >= a.min_coverage).all(axis=1)])
    if len(keep) < 8:                       # relax to symbols good in >=80% of blocks
        keep = sorted(cov.index[(cov >= a.min_coverage).mean(axis=1) >= 0.8])
    print(f"universe: {len(keep)} symbols -> {keep}")

    Xs, ts_all, blk_all = [], [], []
    px = {f: [] for f in FIELDS}
    for b, (lo, hi) in enumerate(bl):
        sub = {f: wide[f].iloc[lo:hi][keep].ffill().bfill() for f in FIELDS}
        if sub["close"].isna().any().any():
            print(f"  block {b}: still NaN after fill, skipped"); continue
        o, h, l, c, v = (sub[f].to_numpy(np.float64) for f in FIELDS)
        with np.errstate(all="ignore"):
            X, names = alpha158(o, h, l, c, v, (h + l + c) / 3.0)
        Xs.append(X); ts_all.append(idx[lo:hi]); blk_all.append(np.full(hi - lo, b))
        for f, arr in zip(FIELDS, (o, h, l, c, v)):
            px[f].append(arr)
        print(f"  block {b:2d}: {hi-lo:5d} bars {idx[lo]} .. {idx[hi-1]}")

    X = np.concatenate(Xs, 0)
    d = dict(X=X, names=np.array(names),
             dates=np.array([str(t) for t in np.concatenate(ts_all)]),
             symbols=np.array(keep), block=np.concatenate(blk_all))
    for f in FIELDS:
        d[f] = np.concatenate(px[f], 0).astype(np.float32)
    np.savez_compressed(out, **d)
    print(f"panel {X.shape}  nan-frac={np.isnan(X).mean():.4f}")
    print("wrote", out, f"{os.path.getsize(out)/1e6:.0f} MB")


if __name__ == "__main__":
    main()
