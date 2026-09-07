#!/usr/bin/env python3
"""Large-trade order imbalance as a return predictor, at full scale ([14], [16]).

RESULTS.md sec.3.2 tested Ng, Peng, Tao & Zhou's claim on fifteen weeks of the
findata vendor tape: the *contemporaneous* impact of large trades replicated
(t = +6.1) but the *predictive* claim did not (t = -0.4). It attributed the gap
to liquidity -- "the median bar here carries $69" -- and could not test that,
because the tape contained no thick bars.

Polymarket-v1 does. It spans 2022-11 to 2026-04, including the 2024 US
presidential election that [14] studies, and it carries the taker side as ground
truth rather than inferred, which matters: [16] shows the usual tick and quote
classifiers are near-random on this venue, and a mis-signed imbalance regressor
attenuates towards zero -- which would fake exactly the null sec.3.2 reports.

So the conjecture becomes testable. Stratify the identical regression by how
much notional the bar actually carries and by market category: if sec.3.2's
explanation is right, the predictive coefficient appears in the thick bars.

Construction follows sec.3.2 exactly, except that `D` replaces the inferred side
and `p_event` replaces the hand-rolled leg alignment:
  * bars are hourly, bar price is the last p_event print;
  * imbalance = sum(D * notional) / sum(notional), computed separately for large
    trades (top decile by notional within the market) and the rest;
  * returns in log-odds; market fixed effects by within-transformation, since a
    dummy matrix at this market count would not fit in memory;
  * standard errors clustered two-way on market and on hour (Cameron-Gelbach-Miller).

    python pm/pmv1_prep.py --freq 1h      # build data/pmv1_bars_1h.parquet first
    python pm/pmv1_imbalance.py --freq 1h
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, os.pardir)


def two_way_cluster_ols(y, X, g1, g2):
    """OLS with Cameron-Gelbach-Miller two-way clustered covariance."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta

    def meat(g):
        M = np.zeros((X.shape[1], X.shape[1]))
        order = np.argsort(g, kind="stable")
        gs, Xs, es = g[order], X[order], e[order]
        bounds = np.flatnonzero(np.diff(gs)) + 1
        for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(gs)]):
            s = Xs[lo:hi].T @ es[lo:hi]
            M += np.outer(s, s)
        return M

    g12 = pd.factorize(pd.Series(g1).astype(str) + "|" + pd.Series(g2).astype(str))[0]
    V = XtX_inv @ (meat(g1) + meat(g2) - meat(g12)) @ XtX_inv
    return beta, np.sqrt(np.clip(np.diag(V), 0, None))


def demean(A, g):
    """Within-transformation on g -- market fixed effects without a dummy matrix."""
    A = np.asarray(A, float)
    n = np.bincount(g)
    if A.ndim == 1:
        return A - (np.bincount(g, weights=A) / n)[g]
    out = np.empty_like(A)
    for j in range(A.shape[1]):
        out[:, j] = A[:, j] - (np.bincount(g, weights=A[:, j]) / n)[g]
    return out


def load(path, min_bars):
    bars = pd.read_parquet(path)
    eps = 1e-9
    bars["oi_large"] = bars.sgn_large / (bars.abs_large + eps)
    bars["oi_small"] = bars.sgn_small / (bars.abs_small + eps)
    pc = bars.p.clip(0.01, 0.99)
    bars["logit"] = np.log(pc / (1 - pc))
    bars = bars.sort_values(["market_id", "bar"], kind="stable")
    g = bars.groupby("market_id", sort=False)
    bars["r"] = g.logit.diff()
    bars["r_next"] = g.r.shift(-1)
    d_prev, d_next = g.bar.diff(), g.bar.diff(-1).abs()
    bars = bars[(d_prev == 1) & (d_next == 1)]           # consecutive bars only
    bars = bars.dropna(subset=["r", "r_next", "oi_large", "oi_small"])
    keep = bars.groupby("market_id").size()
    return bars[bars.market_id.isin(keep[keep >= min_bars].index)].reset_index(drop=True)


def regress(bars, y_name, cols, label, results=None, key=None):
    mid = pd.factorize(bars.market_id)[0]
    hid = pd.factorize(bars.bar)[0]
    y = demean(bars[y_name].to_numpy(float), mid)        # market fixed effects
    X = demean(bars[cols].to_numpy(float), mid)
    beta, se = two_way_cluster_ols(y, X, mid, hid)
    r2 = 1 - ((y - X @ beta) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    out = []
    for i, c in enumerate(cols):
        t = beta[i] / se[i] if se[i] > 0 else np.nan
        out.append((c, float(beta[i]), float(se[i]), float(t)))
        print(f"{label if i == 0 else '':<34}{c:<10}{beta[i]:>+11.4f}{se[i]:>10.4f}"
              f"{t:>+8.2f}{r2 if i == 0 else np.nan:>9.4f}{len(bars) if i == 0 else 0:>11,d}")
    if results is not None:
        results[key or label] = {c: [b, s, t] for c, b, s, t in out}
    return beta, se


HDR = (f"\n{'spec':<34}{'regressor':<10}{'beta':>11}{'se':>10}{'t':>8}{'R2':>9}{'bars':>11}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", default="1h")
    ap.add_argument("--bars", default=None)
    ap.add_argument("--min-bars", type=int, default=20)
    a = ap.parse_args()
    path = a.bars or os.path.join(ROOT, "data", f"pmv1_bars_{a.freq}.parquet")
    if not os.path.exists(path):
        sys.exit(f"{path} missing -- run pm/pmv1_prep.py --freq {a.freq} first")

    bars = load(path, a.min_bars)
    step = int(pd.Timedelta(a.freq).total_seconds())
    span = pd.to_datetime(bars.bar * step, unit="s")
    print(f"freq {a.freq}   bars {len(bars):,}   markets {bars.market_id.nunique():,}   "
          f"distinct clock bars {bars.bar.nunique():,}")
    print(f"span {span.min()} .. {span.max()}")
    print(f"median gross notional/bar ${bars.gross.median():,.0f}   "
          f"mean ${bars.gross.mean():,.0f}   mean |r| (log-odds) {bars.r.abs().mean():.4f}")

    results = {"n_bars": int(len(bars)), "n_markets": int(bars.market_id.nunique()),
               "freq": a.freq, "median_gross": float(bars.gross.median()),
               "mean_gross": float(bars.gross.mean())}
    print(HDR); print("-" * 93)
    regress(bars, "r", ["oi_large", "oi_small"], "contemporaneous r_h", results, "contemp")
    regress(bars, "r_next", ["oi_large", "oi_small"], "predictive r_h+1", results, "pred")
    regress(bars, "r_next", ["oi_large", "oi_small", "r"],
            "predictive + own-return control", results, "pred_ctrl")

    # sec.3.2's conjecture: the null predictive result is a thin-tape artefact.
    print("\n=== predictive r_h+1 by bar-liquidity quintile (gross notional) ===")
    print(HDR); print("-" * 93)
    q = pd.qcut(bars.gross, 5, labels=False, duplicates="drop")
    for i in sorted(pd.unique(q.dropna())):
        sub = bars[q == i]
        if len(sub) < 500:
            continue
        regress(sub, "r_next", ["oi_large", "oi_small"],
                f"Q{int(i)+1}  median ${sub.gross.median():,.0f}", results, f"liq_Q{int(i)+1}")

    print("\n=== predictive r_h+1 by market category ===")
    print(HDR); print("-" * 93)
    for c in bars.cat.value_counts().head(8).index:
        sub = bars[bars.cat == c]
        if len(sub) < 2000:
            continue
        regress(sub, "r_next", ["oi_large", "oi_small"],
                f"{str(c)[:24]} (n={len(sub):,})", results, f"cat_{c}")

    out = os.path.join(ROOT, "data", f"pmv1_imbalance_{a.freq}.json")
    json.dump(results, open(out, "w"), indent=1)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
