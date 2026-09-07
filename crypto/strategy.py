#!/usr/bin/env python3
"""Cross-sectional crypto strategies under competing cost models.

This is the MACE experiment (PAPERS.md [2]) run on a second asset class: hold the
strategies fixed, swap only the cost model, and report whether the *ranking*
changes. MACE showed this for five DRL algorithms on the NASDAQ-100 with a static
Almgren-Chriss form; here the comparison adds a state-dependent charge, in which
the same trade is priced against the bar's own realised volatility and traded
volume, so cost varies with regime rather than only with size.

Every window and every return is confined to a contiguous block of the tape.
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))
from analysis.costs import FlatCost, SquareRootCost, realised_vol    # noqa: E402

ART = os.path.join(HERE, "artifacts")
BARS_PER_YEAR = 24 * 365


def run(weights_fn, prices, block, cost, capital=1_000_000.0):
    """weights_fn(t, w) -> target weights at bar t given current holdings w;
    execution at the bar's open."""
    T, N = prices.shape
    w = np.zeros(N)
    v = capital
    vals, turns, costs = [v], [], 0.0
    for t in range(T - 1):
        if block[t + 1] != block[t]:                 # gap: flatten, no return taken
            vals.append(v); turns.append(0.0)
            w = np.zeros(N)
            continue
        tgt = weights_fn(t, w)
        traded = np.abs(tgt - w) * v
        c = cost.charge(traded, t=t)
        costs += c
        v -= c
        r = prices[t + 1] / prices[t] - 1.0
        v *= float(1.0 + tgt @ r)
        w = tgt * (1 + r)
        s = w.sum()
        w = w / s if s > 1e-12 else w
        vals.append(v)
        turns.append(float(np.abs(traded).sum() / max(v, 1e-9)))
    return np.array(vals), float(np.mean(turns)), costs


def metrics(vals, n_bars):
    """Sharpe is mean/sd of per-bar returns scaled by sqrt(bars per year).

    The repo's equity convention (annualised return / annualised vol) is not used
    here: over a 111-day window it compounds a large cumulative return to an
    absurd annual figure and the ratio inherits it. Cumulative return over the
    window is reported instead of an annualised one for the same reason."""
    r = vals[1:] / vals[:-1] - 1.0
    r = r[np.isfinite(r)]
    cum = vals[-1] / vals[0] - 1.0
    sd = r.std(ddof=1)
    peak = np.maximum.accumulate(vals)
    return dict(cum=float(cum), vol=float(sd * np.sqrt(BARS_PER_YEAR)),
                sharpe=float(r.mean() / sd * np.sqrt(BARS_PER_YEAR)) if sd > 1e-12 else 0.0,
                mdd=float((vals / peak - 1).min()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=os.path.join(ART, "panel_1hour.npz"))
    ap.add_argument("--start", default="2026-01-01", help="evaluation window start")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()

    z = np.load(a.panel, allow_pickle=True)
    dates = z["dates"].astype(str)
    sel = dates >= a.start
    op = z["open"].astype(np.float64)[sel]
    cl = z["close"].astype(np.float64)[sel]
    vol_bar = z["volume"].astype(np.float64)[sel]
    block = z["block"][sel]
    syms = z["symbols"].astype(str)
    T, N = op.shape
    print(f"evaluation window {dates[sel][0]} .. {dates[sel][-1]}  "
          f"{T} bars, {N} coins, {len(np.unique(block))} blocks")

    fr = {}
    for mode in ("causal", "paper"):
        f = os.path.join(ART, f"fr_{mode}.npz")
        if os.path.exists(f):
            d = np.load(f, allow_pickle=True)
            s = np.nan_to_num(d["fr"].astype(np.float64))[sel]
            # the forecast formed at bar t is tradeable at the open of bar t+1
            s = np.vstack([np.zeros((1, N)), s[:-1]])
            fr[mode] = s

    # ret1[t] uses close_t, which is not observable when the bar-t open is traded.
    # Every signal is therefore shifted so that the weights applied at open_t use
    # information through close_{t-1} only.
    ret1 = np.zeros_like(op)
    ret1[1:] = np.where((block[1:] == block[:-1])[:, None], cl[1:] / cl[:-1] - 1.0, 0.0)
    mom_raw = np.zeros_like(op)
    for t in range(T):
        lo = max(0, t - 23)
        m = block[lo:t + 1] == block[t]
        mom_raw[t] = ret1[lo:t + 1][m].sum(0)
    shift = lambda a: np.vstack([np.zeros((1, N)), a[:-1]])
    mom, rev = shift(mom_raw), shift(ret1)

    def topk(sig, k, t):
        w = np.zeros(N)
        w[np.argsort(-sig[t])[:k]] = 1.0 / k
        return w

    # market-wide stress state, lagged one bar so it is observable at the open
    sig_bar = realised_vol(cl, 24, block)
    stress_raw = sig_bar.mean(1)
    stress = np.concatenate([[stress_raw[0]], stress_raw[:-1]])
    med = np.median(stress)

    def gated(sig, k, t, w, want_high):
        """Rebalance only in the wanted volatility regime, otherwise hold.

        The two gated variants carry near-identical total turnover but place it in
        opposite regimes, so a flat cost charges them the same and a state-dependent
        cost cannot. This is the discriminating test for a regime-varying cost model:
        a cost that only scales with size can never separate them."""
        trade = (stress[t] > med) if want_high else (stress[t] <= med)
        return topk(sig, k, t) if trade else w

    strategies = {
        "equal-weight (all coins)": lambda t, w: np.full(N, 1.0 / N),
        "BTC buy-and-hold":         lambda t, w: (syms == "BTCUSD").astype(float),
        f"momentum-24h top{a.k}":   lambda t, w: topk(mom, a.k, t),
        f"reversal-1h top{a.k}":    lambda t, w: topk(-rev, a.k, t),
        f"momentum-24h top{a.k} (daily reb.)":
                                    lambda t, w: topk(mom, a.k, t - t % 24),
        f"momentum-24h top{a.k} (calm bars only)":
                                    lambda t, w: gated(mom, a.k, t, w, False),
        f"momentum-24h top{a.k} (stressed bars only)":
                                    lambda t, w: gated(mom, a.k, t, w, True),
    }
    for mode, s in fr.items():
        strategies[f"forecast-{mode} top{a.k}"] = (lambda s: lambda t, w: topk(s, a.k, t))(s)
        strategies[f"forecast-{mode} top{a.k} (daily reb.)"] = \
            (lambda s: lambda t, w: topk(s, a.k, t - t % 24))(s)

    sigma = sig_bar
    models = {
        "no cost":        FlatCost(0.0),
        "flat 10bp":      FlatCost(10.0),
        "flat 30bp":      FlatCost(30.0),
        "sqrt (state)":   SquareRootCost(sigma, vol_bar, half_spread_bps=5.0, Y=1.0),
    }

    out, paths = {}, {}
    for cname, cost in models.items():
        print(f"\n=== cost model: {cname} ===")
        print(f"{'strategy':<38}{'cum':>10}{'vol':>9}{'sharpe':>9}"
              f"{'maxDD':>9}{'turn/bar':>10}{'cost$':>12}")
        rows = []
        for sname, fn in strategies.items():
            vals, turn, costs = run(fn, op, block, cost)
            m = metrics(vals, T)
            rows.append((sname, m, turn, costs))
            out[f"{cname}|{sname}"] = dict(m, turnover=turn, cost=costs)
            paths[f"{cname}|{sname}"] = vals.astype(np.float32)   # for crypto/dsr.py
        for sname, m, turn, costs in sorted(rows, key=lambda r: -r[1]["sharpe"]):
            print(f"{sname:<38}{m['cum']:>+10.3f}{m['vol']:>9.3f}"
                  f"{m['sharpe']:>+9.2f}{m['mdd']:>+9.3f}{turn:>10.4f}{costs:>12,.0f}")

    # MACE's headline: does the cost model reorder the strategies?
    print("\n=== ranking by Sharpe under each cost model ===")
    names = list(strategies)
    ranks = {}
    for cname in models:
        order = sorted(names, key=lambda s: -out[f"{cname}|{s}"]["sharpe"])
        ranks[cname] = {s: i + 1 for i, s in enumerate(order)}
    print(f"{'strategy':<38}" + "".join(f"{c:>15}" for c in models))
    for s in names:
        print(f"{s:<38}" + "".join(f"{ranks[c][s]:>15d}" for c in models))
    # how state-dependent is the state-dependent model, in basis points actually paid?
    eff = []
    for sname in names:
        c_sqrt = out[f"sqrt (state)|{sname}"]["cost"]
        turn = out[f"sqrt (state)|{sname}"]["turnover"]
        if turn > 1e-6:
            eff.append((sname, 1e4 * c_sqrt / (turn * T * 1_000_000.0)))
    print("\neffective cost actually paid under the state-dependent model (bp of turnover):")
    for sname, bp in sorted(eff, key=lambda r: -r[1]):
        print(f"  {sname:<40}{bp:>8.1f} bp")

    base = list(models)[1]                                  # flat 10bp, the FinRL default
    for c in list(models)[2:]:
        moved = sum(ranks[c][s] != ranks[base][s] for s in names)
        rho = np.corrcoef([ranks[base][s] for s in names], [ranks[c][s] for s in names])[0, 1]
        print(f"\nvs '{base}': '{c}' moves {moved}/{len(names)} strategies, "
              f"rank correlation {rho:+.3f}")
    json.dump(out, open(os.path.join(ART, "strategy_costs.json"), "w"), indent=1)
    # the value paths are what analysis/dsr.py needs: the reported Sharpe is a
    # maximum over this search, so it has to be deflated by the number of trials
    np.savez_compressed(os.path.join(ART, "strategy_paths.npz"),
                        dates=dates[sel].astype(str), **paths)


if __name__ == "__main__":
    main()
