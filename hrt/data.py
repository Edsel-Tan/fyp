#!/usr/bin/env python3
"""Panel construction + Qlib Alpha158 feature computation for the HRT reproduction.

Reads the local findata SQLite mirror, aligns a fixed universe onto a common
trading calendar, and computes the 158 features of Qlib's Alpha158 handler
(9 kbar + 4 price + 29 rolling x 5 windows).

Output: hrt/artifacts/panel.npz
"""
import argparse, json, os, sqlite3, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
DB = os.path.join(os.path.dirname(HERE), "data", "findata.sqlite")
WINDOWS = [5, 10, 20, 30, 60]
EPS = 1e-12


# ---------------------------------------------------------------- panel load
def load_panel(symbols, start, end, min_coverage=0.98):
    if not os.path.exists(DB):
        import portable
        cal, keep, px = portable.panel(symbols, start, end)
        dropped = sorted(set(symbols) - set(keep))
        return cal, keep, dropped, px
    con = sqlite3.connect(DB)
    q = ("select symbol, substr(ts,1,10) as d, open, high, low, close, volume "
         "from ohlc where ts>=? and ts<? and symbol in (%s)"
         % ",".join("?" * len(symbols)))
    df = pd.read_sql_query(q, con, params=[start, end] + list(symbols))
    con.close()

    # trading calendar: dates carried by at least half the universe
    counts = df.groupby("d").size()
    cal = sorted(counts[counts >= 0.5 * len(symbols)].index)
    df = df[df.d.isin(set(cal))]

    fields = ["open", "high", "low", "close", "volume"]
    wide = {f: df.pivot(index="d", columns="symbol", values=f).reindex(cal)
            for f in fields}

    # drop names with thin coverage, then forward-fill the few remaining holes
    cover = wide["close"].notna().mean()
    keep = sorted(cover[cover >= min_coverage].index)
    dropped = sorted(set(symbols) - set(keep))
    for f in fields:
        wide[f] = wide[f][keep].ffill().bfill()

    return cal, keep, dropped, {f: wide[f].to_numpy(np.float64) for f in fields}


# ------------------------------------------------------- rolling primitives
def _roll(a, d):
    """[T,N] -> [T,N,d] sliding windows, oldest first, NaN-padded at the front."""
    T, N = a.shape
    pad = np.full((d - 1, N), np.nan)
    p = np.concatenate([pad, a], 0)
    return np.lib.stride_tricks.sliding_window_view(p, d, axis=0)


def r_mean(a, d):  return np.nanmean(_roll(a, d), -1)
def r_std(a, d):   return np.nanstd(_roll(a, d), -1, ddof=1)
def r_max(a, d):   return np.nanmax(_roll(a, d), -1)
def r_min(a, d):   return np.nanmin(_roll(a, d), -1)
def r_sum(a, d):   return np.nansum(_roll(a, d), -1)


def r_quantile(a, d, q):
    return np.nanquantile(_roll(a, d), q, axis=-1)


def r_rank(a, d):
    """Qlib Rank: percentile of the current value inside the trailing window."""
    w = _roll(a, d)
    return (w <= w[..., -1:]).sum(-1) / d


def r_idxmax(a, d):
    return np.nanargmax(np.nan_to_num(_roll(a, d), nan=-np.inf), -1).astype(np.float64)


def r_idxmin(a, d):
    return np.nanargmin(np.nan_to_num(_roll(a, d), nan=np.inf), -1).astype(np.float64)


def r_regress(a, d):
    """Rolling OLS of the window on t=0..d-1. Returns (slope, rsquare, resid)."""
    w = _roll(a, d)
    t = np.arange(d, dtype=np.float64)
    tm, tv = t.mean(), ((t - t.mean()) ** 2).sum()
    ym = w.mean(-1, keepdims=True)
    cov = ((t - tm) * (w - ym)).sum(-1)
    slope = cov / tv
    sst = ((w - ym) ** 2).sum(-1)
    ssr = slope ** 2 * tv
    rsq = np.where(sst > EPS, ssr / (sst + EPS), np.nan)
    fitted_last = ym[..., 0] + slope * (d - 1 - tm)
    resid = w[..., -1] - fitted_last
    return slope, rsq, resid


def r_corr(x, y, d):
    wx, wy = _roll(x, d), _roll(y, d)
    xm, ym = wx.mean(-1, keepdims=True), wy.mean(-1, keepdims=True)
    dx, dy = wx - xm, wy - ym
    num = (dx * dy).sum(-1)
    den = np.sqrt((dx ** 2).sum(-1) * (dy ** 2).sum(-1))
    return np.where(den > EPS, num / (den + EPS), np.nan)


# ------------------------------------------------------------- Alpha158 set
def alpha158(o, h, l, c, v, vwap):
    """All arrays [T,N]. Returns (features [T,N,158], names)."""
    feats, names = [], []
    add = lambda n, x: (names.append(n), feats.append(x))

    hl = h - l
    add("KMID", (c - o) / (o + EPS))
    add("KLEN", hl / (o + EPS))
    add("KMID2", (c - o) / (hl + EPS))
    add("KUP", (h - np.maximum(o, c)) / (o + EPS))
    add("KUP2", (h - np.maximum(o, c)) / (hl + EPS))
    add("KLOW", (np.minimum(o, c) - l) / (o + EPS))
    add("KLOW2", (np.minimum(o, c) - l) / (hl + EPS))
    add("KSFT", (2 * c - h - l) / (o + EPS))
    add("KSFT2", (2 * c - h - l) / (hl + EPS))

    add("OPEN0", o / c)
    add("HIGH0", h / c)
    add("LOW0", l / c)
    add("VWAP0", vwap / c)

    ref = lambda a, k: np.concatenate([np.full((k,) + a.shape[1:], np.nan), a[:-k]], 0)
    c1 = ref(c, 1)
    v1 = ref(v, 1)
    dc = c - c1
    dv = v - v1
    logv = np.log(v + 1.0)
    cret = c / (c1 + EPS)
    logvr = np.log(v / (v1 + EPS) + 1.0)
    absdc_amt = np.abs(cret - 1.0) * v

    for d in WINDOWS:
        slope, rsq, resid = r_regress(c, d)
        add(f"ROC{d}", ref(c, d) / (c + EPS))
        add(f"MA{d}", r_mean(c, d) / (c + EPS))
        add(f"STD{d}", r_std(c, d) / (c + EPS))
        add(f"BETA{d}", slope / (c + EPS))
        add(f"RSQR{d}", rsq)
        add(f"RESI{d}", resid / (c + EPS))
        mx, mn = r_max(h, d), r_min(l, d)
        add(f"MAX{d}", mx / (c + EPS))
        add(f"MIN{d}", mn / (c + EPS))
        add(f"QTLU{d}", r_quantile(c, d, 0.8) / (c + EPS))
        add(f"QTLD{d}", r_quantile(c, d, 0.2) / (c + EPS))
        add(f"RANK{d}", r_rank(c, d))
        add(f"RSV{d}", (c - mn) / (mx - mn + EPS))
        add(f"IMAX{d}", r_idxmax(h, d) / d)
        add(f"IMIN{d}", r_idxmin(l, d) / d)
        add(f"IMXD{d}", (r_idxmax(h, d) - r_idxmin(l, d)) / d)
        add(f"CORR{d}", r_corr(c, logv, d))
        add(f"CORD{d}", r_corr(cret, logvr, d))
        cntp, cntn = r_mean((dc > 0).astype(np.float64), d), r_mean((dc < 0).astype(np.float64), d)
        add(f"CNTP{d}", cntp)
        add(f"CNTN{d}", cntn)
        add(f"CNTD{d}", cntp - cntn)
        absd = r_sum(np.abs(dc), d)
        sump = r_sum(np.maximum(dc, 0), d) / (absd + EPS)
        sumn = r_sum(np.maximum(-dc, 0), d) / (absd + EPS)
        add(f"SUMP{d}", sump)
        add(f"SUMN{d}", sumn)
        add(f"SUMD{d}", sump - sumn)
        add(f"VMA{d}", r_mean(v, d) / (v + EPS))
        add(f"VSTD{d}", r_std(v, d) / (v + EPS))
        add(f"WVMA{d}", r_std(absdc_amt, d) / (r_mean(absdc_amt, d) + EPS))
        vabsd = r_sum(np.abs(dv), d)
        vsump = r_sum(np.maximum(dv, 0), d) / (vabsd + EPS)
        vsumn = r_sum(np.maximum(-dv, 0), d) / (vabsd + EPS)
        add(f"VSUMP{d}", vsump)
        add(f"VSUMN{d}", vsumn)
        add(f"VSUMD{d}", vsump - vsumn)

    X = np.stack(feats, -1).astype(np.float32)
    return X, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="asof_2015-01-01_usable")
    ap.add_argument("--start", default="2013-06-01")   # warm-up for 60d windows
    ap.add_argument("--end", default="2023-01-01")
    ap.add_argument("--out", default=os.path.join(ART, "panel.npz"))
    a = ap.parse_args()

    syms = json.load(open(os.path.join(ART, "universe.json")))[a.universe]
    print(f"universe {a.universe}: {len(syms)} symbols")
    cal, keep, dropped, px = load_panel(syms, a.start, a.end)
    print(f"calendar {cal[0]}..{cal[-1]}  T={len(cal)}  kept N={len(keep)}  dropped={len(dropped)}")

    o, h, l, c, v = (px[f] for f in ["open", "high", "low", "close", "volume"])
    vwap = (h + l + c) / 3.0            # no vendor VWAP; paper derives it from OHLCV too
    with np.errstate(all="ignore"):
        X, names = alpha158(o, h, l, c, v, vwap)
    print(f"features {X.shape}  ({len(names)} names)  nan-frac={np.isnan(X).mean():.4f}")
    assert len(names) == 158, len(names)

    np.savez_compressed(a.out, X=X, names=np.array(names), dates=np.array(cal),
                        symbols=np.array(keep), open=o.astype(np.float32),
                        high=h.astype(np.float32), low=l.astype(np.float32),
                        close=c.astype(np.float32), volume=v.astype(np.float32))
    print("wrote", a.out, f"{os.path.getsize(a.out)/1e6:.0f} MB")


if __name__ == "__main__":
    main()
