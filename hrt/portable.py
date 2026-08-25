#!/usr/bin/env python3
"""Read the reproduction's inputs from the portable bundle instead of SQLite.

`export_bundle.py` writes artifacts/prices_bundle.npz on a machine that has the
full findata mirror; every script here prefers the SQLite database when it is
present and falls back to the bundle when it is not, so a fresh checkout runs
without a 2 GB re-pull.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
BUNDLE = os.path.join(ART, "prices_bundle.npz")
DB = os.path.join(os.path.dirname(HERE), "data", "findata.sqlite")

_cache = None


def have_db():
    return os.path.exists(DB)


def load():
    global _cache
    if _cache is None:
        if not os.path.exists(BUNDLE):
            raise FileNotFoundError(
                f"neither {DB} nor {BUNDLE} exists; run export_bundle.py on a machine "
                f"with the findata mirror, or re-pull with scripts/findata_pull.py")
        _cache = dict(np.load(BUNDLE, allow_pickle=True))
    return _cache


def panel(symbols, start, end):
    """-> (calendar, kept symbols, {field: [T,N] float64}) restricted to `symbols`."""
    z = load()
    cal = z["dates"].astype(str)
    syms = z["symbols"].astype(str)
    keep_mask = np.isin(syms, list(symbols))
    day_mask = (cal >= start) & (cal < end)
    out = {f: z[f][day_mask][:, keep_mask].astype(np.float64)
           for f in ("open", "high", "low", "close", "volume")}
    return list(cal[day_mask]), list(syms[keep_mask]), out


def gspc(lo, hi):
    z = load()
    d = z["gspc_dates"].astype(str)
    m = (d >= lo) & (d <= hi)
    return list(d[m]), list(z["gspc_close"][m])


def close_history(symbol, start):
    """Full close series for one symbol from `start` onward, panel + extension."""
    z = load()
    syms = list(z["symbols"].astype(str))
    if symbol not in syms:
        return [], np.array([])
    j = syms.index(symbol)
    d1, c1 = z["dates"].astype(str), z["close"][:, j]
    d2, c2 = z["ext_dates"].astype(str), z["ext_close"][:, j]
    d = np.concatenate([d1, d2])
    c = np.concatenate([c1, c2]).astype(np.float64)
    m = (d >= start) & np.isfinite(c)
    return list(d[m]), c[m]


def dividends(symbol, start):
    z = load()
    m = (z["div_symbol"].astype(str) == symbol) & (z["div_date"].astype(str) >= start)
    return list(zip(z["div_date"].astype(str)[m], z["div_amount"][m]))
