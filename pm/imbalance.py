#!/usr/bin/env python3
"""Large-trade order imbalance as a return predictor (PAPERS.md [14]).

Ng, Peng, Tao & Zhou report that net order imbalance from large trades strongly
predicts subsequent returns in prediction markets. That is testable on the
captured tape without any new data, and unlike the account-level skill test it is
a market-level statement -- which is exactly the distinction [15] draws between
its detection layers.

Construction
  * both tokens of a market are mapped to one YES probability series: a trade in
    the NO token at price p is a trade in YES at 1-p with the side flipped;
  * bars are hourly; the bar price is the last YES print;
  * imbalance is signed notional over gross notional inside the bar, computed
    separately for large trades (>= the market's 90th-percentile notional) and
    for the rest;
  * the regression is r_{h+1} on imbalance_h with market fixed effects, standard
    errors clustered two-way on market and on hour.

Prices are bounded in [0,1], so returns are taken in log-odds, which is the
scale on which prediction-market price changes are additive.
"""
import argparse, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from outcomes import resolve                                   # noqa: E402


def two_way_cluster_ols(y, X, g1, g2):
    """OLS with Cameron-Gelbach-Miller two-way clustered covariance."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta

    def meat(g):
        M = np.zeros((X.shape[1], X.shape[1]))
        order = np.argsort(g)
        gs, Xs, es = g[order], X[order], e[order]
        bounds = np.flatnonzero(np.diff(gs)) + 1
        for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(gs)]):
            s = Xs[lo:hi].T @ es[lo:hi]
            M += np.outer(s, s)
        return M

    g12 = pd.factorize(pd.Series(g1).astype(str) + "|" + pd.Series(g2).astype(str))[0]
    V = XtX_inv @ (meat(g1) + meat(g2) - meat(g12)) @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    return beta, se


def build_bars(t, r, freq="1h", large_q=0.90):
    r = r.set_index("market_id")
    t = t[t.market_id.isin(r.index)].copy()
    # the reference leg must not be chosen by the outcome: take the lexicographically
    # smaller token id. (The regression is invariant to the choice -- flipping the leg
    # flips both r and the imbalance -- but picking it by the winner would still read
    # as look-ahead.)
    ref = t.groupby("market_id").token_id.transform("min")
    is_yes = t.token_id == ref
    t["p_yes"] = np.where(is_yes, t.price, 1.0 - t.price)
    buy = t.side == "BUY"
    t["sgn"] = np.where(buy == is_yes, 1.0, -1.0)           # +1 = increases YES exposure
    t["notional"] = t.price * t["size"]
    thr = t.groupby("market_id").notional.transform(lambda x: x.quantile(large_q))
    t["large"] = t.notional >= thr
    t["bar"] = t.ts.dt.floor(freq)

    g = t.groupby(["market_id", "bar"])
    bars = pd.DataFrame({
        "p": g.p_yes.last(),
        "n": g.size(),
        "gross": g.notional.sum(),
        "sgn_large": g.apply(lambda d: (d.sgn * d.notional)[d.large].sum(), include_groups=False),
        "abs_large": g.apply(lambda d: d.notional[d.large].sum(), include_groups=False),
        "sgn_small": g.apply(lambda d: (d.sgn * d.notional)[~d.large].sum(), include_groups=False),
        "abs_small": g.apply(lambda d: d.notional[~d.large].sum(), include_groups=False),
    }).reset_index()

    eps = 1e-9
    bars["oi_large"] = bars.sgn_large / (bars.abs_large + eps)
    bars["oi_small"] = bars.sgn_small / (bars.abs_small + eps)
    # log-odds return, clipped away from the absorbing barriers
    pc = bars.p.clip(0.01, 0.99)
    bars["logit"] = np.log(pc / (1 - pc))
    bars = bars.sort_values(["market_id", "bar"])
    bars["r"] = bars.groupby("market_id").logit.diff()
    bars["r_next"] = bars.groupby("market_id").r.shift(-1)
    # only keep consecutive bars, so a gap in trading is not read as a return
    dt_prev = bars.groupby("market_id").bar.diff().dt.total_seconds()
    dt_next = bars.groupby("market_id").bar.diff(-1).dt.total_seconds().abs()
    step = pd.Timedelta(freq).total_seconds()
    bars = bars[(dt_prev == step) & (dt_next == step)]
    return bars.dropna(subset=["r", "r_next", "oi_large", "oi_small"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", default="1h")
    ap.add_argument("--large-q", type=float, default=0.90)
    ap.add_argument("--min-bars", type=int, default=20)
    a = ap.parse_args()

    t, r = resolve()
    bars = build_bars(t, r, a.freq, a.large_q)
    keep = bars.groupby("market_id").size()
    bars = bars[bars.market_id.isin(keep[keep >= a.min_bars].index)]
    print(f"bars {len(bars):,} over {bars.market_id.nunique()} markets, "
          f"{bars.bar.nunique()} distinct hours, freq={a.freq}, "
          f"large = top {(1-a.large_q)*100:.0f}% of trades by notional")
    print(f"mean |r| per bar (log-odds) {bars.r.abs().mean():.4f}   "
          f"median gross notional/bar ${bars.gross.median():,.0f}")

    mid = pd.factorize(bars.market_id)[0]
    hid = pd.factorize(bars.bar)[0]
    y = bars.r_next.to_numpy()

    specs = {
        "large only":        ["oi_large"],
        "small only":        ["oi_small"],
        "large + small":     ["oi_large", "oi_small"],
        "large + small + r": ["oi_large", "oi_small", "r"],
    }
    print(f"\n{'spec':<20}{'regressor':<12}{'beta':>10}{'se':>10}{'t':>8}{'R2':>8}")
    print("-" * 68)
    for name, cols in specs.items():
        Z = bars[cols].to_numpy(float)
        D = pd.get_dummies(bars.market_id, drop_first=True).to_numpy(float)   # market FE
        X = np.column_stack([np.ones(len(Z)), Z, D])
        beta, se = two_way_cluster_ols(y, X, mid, hid)
        r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        for i, c in enumerate(cols, start=1):
            tstat = beta[i] / se[i] if se[i] > 0 else np.nan
            print(f"{name if i == 1 else '':<20}{c:<12}{beta[i]:>+10.4f}{se[i]:>10.4f}"
                  f"{tstat:>+8.2f}{r2 if i == 1 else np.nan:>8.4f}")

    # contemporaneous check: imbalance should move price within the same bar
    Z = bars[["oi_large", "oi_small"]].to_numpy(float)
    D = pd.get_dummies(bars.market_id, drop_first=True).to_numpy(float)
    X = np.column_stack([np.ones(len(Z)), Z, D])
    beta, se = two_way_cluster_ols(bars.r.to_numpy(), X, mid, hid)
    print(f"\ncontemporaneous r_h on imbalance_h:  large {beta[1]:+.4f} "
          f"(t {beta[1]/se[1]:+.2f})   small {beta[2]:+.4f} (t {beta[2]/se[2]:+.2f})")
    bars.to_csv(os.path.join(HERE, os.pardir, "data", "pm_bars.csv"), index=False)


if __name__ == "__main__":
    main()
