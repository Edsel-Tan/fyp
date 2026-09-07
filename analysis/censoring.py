#!/usr/bin/env python3
"""What the crypto tape's censoring costs, measured against ground truth.

RESULTS.md sec.1.2 establishes that the findata crypto tape is censored by
calendar -- hourly bars are ~97% complete in Apr/May/Jun/Oct/Nov/Dec, ~50% in
Jan/Mar/Jul/Sep and absent in Feb/Aug; daily bars arrive as ~9-day runs separated
by ~21-day holes.  crypto/panel.py responds by cutting the tape into contiguous
blocks.  What has never been measured is the size of the error that response
avoids, because on the crypto tape there is nothing to compare against: the
uncensored series does not exist.

So run it the other way round.  Take a panel that *is* complete -- the S&P 500
daily panel from the HRT reproduction -- apply the crypto tape's own censoring
pattern to it, and compare three numbers that a researcher could report:

  truth   the complete panel                       (unobservable in crypto)
  naive   retained rows concatenated, gaps ignored (what you get by pivoting the
          vendor's response and calling .diff())
  block   retained rows cut into contiguous runs, every window and every return
          confined to one run                      (what crypto/panel.py does)

Two masks are used, matching the two shapes of censoring in sec.1.2:

  gap      the empirical on/off run-length sequence of the daily crypto tape
           (median 9 on, 21 off), tiled over the equity trading-day axis --
           isolates the cost of splicing across holes
  seasonal the hourly tape's month-of-year retention applied by calendar month --
           isolates the cost of never sampling February or August

This is the same move as analysis/falsify.py: build a reference class where the
answer is known, then ask what the pipeline reports there.

    python analysis/censoring.py
    python analysis/censoring.py --reps 40 --panel hrt/artifacts/panel.npz
"""
import argparse, json, os, sqlite3
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, os.pardir)
DB = os.path.join(ROOT, "data", "crypto.sqlite")
TRADING_DAYS = 252.0


# ---------------------------------------------------------------- masks

def crypto_runs(interval="1d"):
    """Empirical on/off run lengths of the deepest symbol's calendar."""
    con = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT symbol, ts FROM ohlc WHERE interval=?", con,
                           params=(interval,))
    con.close()
    df["ts"] = pd.to_datetime(df.ts, utc=True).dt.tz_localize(None).dt.normalize()
    deep = df.symbol.value_counts().index[0]
    d = pd.Index(sorted(df[df.symbol == deep].ts.unique()))
    full = pd.date_range(d[0], d[-1], freq="D")
    have = pd.Series(full.isin(d).astype(int), index=full)
    g = (have != have.shift()).cumsum()
    runs = have.groupby(g).agg(on=("first"), n=("size"))
    return [(int(r.on), int(r.n)) for r in runs.itertuples()], float(have.mean()), deep


def crypto_month_retention(interval="1hour"):
    """Retention fraction per calendar month-of-year on the deepest symbol."""
    import calendar
    con = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT symbol, ts FROM ohlc WHERE interval=?", con,
                           params=(interval,))
    con.close()
    df["ts"] = pd.to_datetime(df.ts, utc=True).dt.tz_localize(None)
    deep = df.symbol.value_counts().index[0]
    d = df[df.symbol == deep]
    per_day = 24 if interval == "1hour" else 1
    got = d.groupby(d.ts.dt.strftime("%Y-%m")).size()
    span = pd.period_range(d.ts.min(), d.ts.max(), freq="M")
    frac = {}
    for p in span:                       # months absent from the tape count as zero
        ym = str(p)
        exp = calendar.monthrange(p.year, p.month)[1] * per_day
        frac.setdefault(p.month, []).append(got.get(ym, 0) / exp)
    return {m: float(np.mean(v)) for m, v in frac.items()}


def mask_gap(dates, runs, phase, rng):
    """Tile the empirical on/off run sequence over the trading-day axis."""
    keep = np.zeros(len(dates), bool)
    i, k = 0, phase % len(runs)
    while i < len(dates):
        on, n = runs[k % len(runs)]
        keep[i:i + n] = bool(on)
        i += n
        k += 1
    return keep


def mask_seasonal(dates, month_frac, phase, rng):
    """Draw each day independently at its month's empirical retention rate."""
    mon = pd.to_datetime(pd.Series(dates)).dt.month.to_numpy()
    p = np.array([month_frac.get(int(m), 0.0) for m in mon])
    return rng.random(len(dates)) < p


# ---------------------------------------------------------------- backtest

def blocks_from(keep):
    """Block id per retained row: runs of consecutive retained rows."""
    idx = np.flatnonzero(keep)
    if len(idx) == 0:
        return idx, np.zeros(0, int)
    new = np.r_[True, np.diff(idx) != 1]
    return idx, np.cumsum(new) - 1


def roll_sum(x, w, block):
    """Trailing w-bar sum of x, never crossing a block boundary."""
    T = len(x)
    out = np.zeros_like(x)
    for t in range(T):
        lo = max(0, t - w + 1)
        m = block[lo:t + 1] == block[t]
        out[t] = x[lo:t + 1][m].sum(0)
    return out


def backtest(close, block, sig, k, gap_aware):
    """Long top-k on the lagged signal, equal weight, close-to-close.

    gap_aware=False is the naive reading: consecutive retained rows are treated
    as consecutive bars, so a return spanning a 21-day hole is booked as one
    daily return. gap_aware=True drops it.
    """
    T, N = close.shape
    r = np.zeros_like(close)
    r[1:] = close[1:] / close[:-1] - 1.0
    if gap_aware:
        r[1:][block[1:] != block[:-1]] = 0.0
    lag = np.vstack([np.full((1, N), -np.inf), sig[:-1]])        # tradeable at t
    port = np.zeros(T)
    for t in range(1, T):
        if gap_aware and block[t] != block[t - 1]:
            continue
        s = lag[t]
        if not np.isfinite(s).any():
            continue
        w = np.argsort(-np.where(np.isfinite(s), s, -np.inf))[:k]
        port[t] = float(np.nanmean(r[t][w]))
    return port


def stats(port, n_years_true=None):
    r = port[np.isfinite(port)]
    r = r[r != 0.0] if (r == 0).mean() > 0.5 else r
    sd = r.std(ddof=1)
    cum = float(np.prod(1 + r) - 1)
    out = dict(bars=int(len(r)), mean_bp=float(r.mean() * 1e4),
               vol=float(sd * np.sqrt(TRADING_DAYS)),
               sharpe=float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0,
               cum=cum)
    # what the researcher reports: bars counted as trading days, 252 to a year
    out["cagr_reported"] = float((1 + cum) ** (TRADING_DAYS / max(len(r), 1)) - 1)
    if n_years_true:                     # the same track record on the real calendar
        out["cagr_true"] = float((1 + cum) ** (1 / n_years_true) - 1)
    return out


def ic(sig, close, block, gap_aware):
    """Cross-sectional rank IC of the lagged signal against the next return."""
    T, N = close.shape
    r = np.zeros_like(close)
    r[1:] = close[1:] / close[:-1] - 1.0
    lag = np.vstack([np.full((1, N), np.nan), sig[:-1]])
    vals = []
    for t in range(1, T):
        if gap_aware and block[t] != block[t - 1]:
            continue
        a, b = lag[t], r[t]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 20:
            continue
        ra = pd.Series(a[m]).rank().to_numpy()
        rb = pd.Series(b[m]).rank().to_numpy()
        if ra.std() > 0 and rb.std() > 0:
            vals.append(np.corrcoef(ra, rb)[0, 1])
    v = np.array(vals)
    return float(v.mean()), float(v.mean() / v.std(ddof=1) * np.sqrt(len(v))) if len(v) > 2 else np.nan


# ---------------------------------------------------------------- driver

def evaluate(close, dates, keep, mom_w, k):
    """Return (naive, block) stats for the censored view of a complete panel."""
    idx, blk = blocks_from(keep)
    c = close[idx]
    years = (pd.Timestamp(str(dates[idx][-1])) - pd.Timestamp(str(dates[idx][0]))).days / 365.25
    out = {}
    for name, gap_aware in (("naive", False), ("block", True)):
        b = blk if gap_aware else np.zeros(len(idx), int)
        r1 = np.zeros_like(c)
        r1[1:] = c[1:] / c[:-1] - 1.0
        if gap_aware:
            r1[1:][b[1:] != b[:-1]] = 0.0
        mom = roll_sum(r1, mom_w, b)
        s = stats(backtest(c, b, mom, k, gap_aware), years)
        s["ic"], s["icir"] = ic(mom, c, b, gap_aware)
        s["blocks"] = int(b.max() + 1)
        out[name] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=os.path.join(ROOT, "hrt", "artifacts", "panel.npz"))
    ap.add_argument("--mom", type=int, action="append",
                    help="momentum lookback in bars (repeatable; default 5 and 20)")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    moms = a.mom or [5, 20]

    z = np.load(a.panel, allow_pickle=True)
    close = z["close"].astype(np.float64)
    dates = z["dates"].astype(str)
    # drop names with holes so the censoring is the only source of missingness
    ok = np.isfinite(close).all(0) & (close > 0).all(0)
    close = close[:, ok]
    T, N = close.shape
    years = (pd.Timestamp(str(dates[-1])) - pd.Timestamp(str(dates[0]))).days / 365.25
    print(f"ground-truth panel: {T} trading days x {N} names, "
          f"{dates[0]} .. {dates[-1]} ({years:.1f}y)")

    runs, ret_d, deep = crypto_runs("1d")
    mfrac = crypto_month_retention("1hour")
    on = [n for o, n in runs if o]
    off = [n for o, n in runs if not o]
    print(f"crypto daily calendar ({deep}): retention {ret_d:.1%}, "
          f"{len(on)} on-runs (median {int(np.median(on))} d), "
          f"{len(off)} off-runs (median {int(np.median(off))} d)")
    print("crypto hourly retention by month: " +
          "  ".join(f"{m:02d}:{mfrac.get(m,0):.0%}" for m in range(1, 13)))

    blk0 = np.zeros(T, int)
    r1_true = np.vstack([np.zeros((1, N)), close[1:] / close[:-1] - 1.0])
    results = {"panel": os.path.basename(a.panel), "n_days": T, "n_names": N,
               "k": a.k, "reps": a.reps, "runs": {}}

    masks = (("gap geometry (9 on / 21 off)", mask_gap, runs),
             ("seasonal (hourly month retention)", mask_seasonal, mfrac))
    keeps = {}
    for mname, fn, arg in masks:
        keeps[mname] = []
        for rep in range(a.reps):
            k = fn(dates, arg, rep, np.random.default_rng(rep))
            if k.sum() >= 200:
                keeps[mname].append(k)

    for mom in moms:
        mom_true = roll_sum(r1_true, mom, blk0)
        truth = stats(backtest(close, blk0, mom_true, a.k, True), years)
        truth["ic"], truth["icir"] = ic(mom_true, close, blk0, True)
        truth["blocks"] = 1
        print(f"\n########## momentum-{mom} top-{a.k} ##########")
        print(f"TRUTH (complete panel): bars {truth['bars']}  "
              f"Sharpe {truth['sharpe']:+.3f}  vol {truth['vol']:.3f}  "
              f"cum {truth['cum']:+.3f}  CAGR {truth['cagr_true']:+.2%}  "
              f"IC {truth['ic']:+.4f} (ICIR {truth['icir']:+.1f})")
        results["runs"][f"mom{mom}"] = {"truth": truth}

        for mname, _, _ in masks:
            acc = {"naive": [], "block": []}
            for keep in keeps[mname]:
                e = evaluate(close, dates, keep, mom, a.k)
                for v in ("naive", "block"):
                    acc[v].append(e[v])
            kf = float(np.mean([k.mean() for k in keeps[mname]]))
            print(f"\n=== mask: {mname} === retained {kf:.1%} of trading days, "
                  f"{len(acc['naive'])} replicates")
            print(f"{'view':<8}{'bars':>7}{'blocks':>8}{'Sharpe':>16}{'vol':>14}"
                  f"{'CAGR reported':>17}{'dSharpe':>10}{'dvol':>9}")
            row = {}
            for v in ("naive", "block"):
                d = pd.DataFrame(acc[v])
                row[v] = {c: [float(d[c].mean()), float(d[c].std(ddof=1))] for c in d.columns}
                f = lambda c, p=3: f"{d[c].mean():+.{p}f}+-{d[c].std(ddof=1):.{p}f}"
                print(f"{v:<8}{d.bars.mean():>7.0f}{d.blocks.mean():>8.0f}"
                      f"{f('sharpe'):>16}{f('vol'):>14}{f('cagr_reported'):>17}"
                      f"{d.sharpe.mean()-truth['sharpe']:>+10.3f}"
                      f"{d.vol.mean()-truth['vol']:>+9.3f}")
            results["runs"][f"mom{mom}"][mname] = row

    out = os.path.join(ROOT, "analysis", "censoring.json")
    json.dump(results, open(out, "w"), indent=1)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
