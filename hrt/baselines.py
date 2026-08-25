#!/usr/bin/env python3
"""Non-RL benchmarks: minimum-variance portfolio, equal-weight buy-and-hold,
and the S&P 500 index.

The min-variance portfolio is re-optimised every trading day on a trailing
covariance window, long-only and fully invested -- the setup the paper describes
for its PyPortfolioOpt benchmark.
"""
import argparse, json, os, sqlite3, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import metrics, COST

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
DB = os.path.join(os.path.dirname(HERE), "data", "findata.sqlite")
PERIODS = {"test2021": ("2021-01-01", "2021-12-31"), "test2022": ("2022-01-01", "2022-12-31")}


def project_simplex(v):
    """Euclidean projection onto {w >= 0, sum w = 1} (Duchi et al., 2008)."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    rho = np.nonzero(u - css / np.arange(1, len(v) + 1) > 0)[0][-1]
    return np.maximum(v - css[rho] / (rho + 1.0), 0.0)


def min_var_weights(cov, w0, iters=300):
    """Long-only fully-invested minimum variance by accelerated projected
    gradient. Warm-started from the previous day's weights."""
    cov = cov + np.eye(len(cov)) * 1e-8
    step = 1.0 / (2.0 * np.linalg.eigvalsh(cov)[-1] + 1e-12)
    w = y = project_simplex(w0)
    t = 1.0
    for _ in range(iters):
        w_new = project_simplex(y - step * 2.0 * (cov @ y))
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = w_new + ((t - 1.0) / t_new) * (w_new - w)
        if np.abs(w_new - w).sum() < 1e-9:
            w = w_new
            break
        w, t = w_new, t_new
    return w


def run_min_variance(prices, lookback=252, cost=COST):
    """prices [T,N] open prices, with `lookback` rows of history prepended."""
    rets = prices[1:] / prices[:-1] - 1.0
    T = prices.shape[0]
    n = prices.shape[1]
    w = np.full(n, 1.0 / n)
    vals_gross, vals_net = [1.0], [1.0]
    for t in range(lookback, T - 1):
        cov = np.cov(rets[t - lookback:t].T)
        w_new = min_var_weights(cov, w)
        turn = np.abs(w_new - w).sum()
        r = float(w_new @ (prices[t + 1] / prices[t] - 1.0))
        vals_gross.append(vals_gross[-1] * (1 + r))
        vals_net.append(vals_net[-1] * (1 + r - cost * turn))
        w = w_new * (prices[t + 1] / prices[t])
        w /= w.sum()
    return vals_gross, vals_net


def run_equal_weight(prices):
    """Equal-weight buy-and-hold from the first day of the window."""
    rel = prices / prices[0]
    return list(rel.mean(1))


def sp500_series(lo, hi):
    if not os.path.exists(DB):
        import portable
        return portable.gspc(lo, hi)
    con = sqlite3.connect(DB)
    rows = con.execute(
        "select substr(ts,1,10), close from ohlc where symbol='^GSPC' "
        "and ts>=? and ts<=? order by ts", (lo, hi + "T23:59:59Z")).fetchall()
    con.close()
    return [r[0] for r in rows], [r[1] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=os.path.join(ART, "panel.npz"))
    ap.add_argument("--lookback", type=int, default=252)
    a = ap.parse_args()

    z = np.load(a.panel, allow_pickle=True)
    dates, op = z["dates"].astype(str), z["open"].astype(np.float64)
    out = {}
    for key, (lo, hi) in PERIODS.items():
        idx = np.where((dates >= lo) & (dates <= hi))[0]
        hist = np.arange(idx[0] - a.lookback - 1, idx[0])
        assert hist[0] >= 0
        full = np.concatenate([hist, idx])

        gross, net = run_min_variance(op[full], a.lookback)
        ew = run_equal_weight(op[idx])
        d, c = sp500_series(lo, hi)

        out[key] = {
            "min_variance": metrics(gross) | {"n": len(gross)},
            "min_variance_net": metrics(net) | {"n": len(net)},
            "equal_weight_bh": metrics(ew) | {"n": len(ew)},
            "sp500": metrics(c) | {"n": len(c)},
            "series": {"min_variance": gross, "equal_weight_bh": ew, "sp500": c,
                       "sp500_dates": d, "dates": list(dates[idx])},
        }
        print(f"\n{key}  ({len(idx)} trading days)")
        for k in ("min_variance", "min_variance_net", "equal_weight_bh", "sp500"):
            m = out[key][k]
            print(f"  {k:18s} cum {m['cum_return']:+.4f}  ann {m['ann_return']:+.4f}  "
                  f"vol {m['ann_vol']:.4f}  sharpe {m['sharpe']:+.4f}  mdd {m['max_drawdown']:+.4f}")
    json.dump(out, open(os.path.join(ART, "baselines.json"), "w"))
    print("\nwrote", os.path.join(ART, "baselines.json"))


if __name__ == "__main__":
    main()
