#!/usr/bin/env python3
"""Reconstruct price-only (split-adjusted, dividend-unadjusted) prices.

findata's OHLC is a total-return series: splits *and* dividends are already
folded in. The paper's Yahoo convention is unstated, so this recovers the
price-only panel to bound how much of any result is dividend tailwind.

For an ex-date d with cash dividend D, the total-return series satisfies
    tr_{d-1} = raw_{d-1} * cum_d * (1 - D / raw_{d-1})
Solving for the raw close one bar before the ex-date gives the closed form
    raw_{d-1} = tr_{d-1} / cum_d + D
which lets the cumulative factor be unwound backwards in one pass.
"""
import json, os, sqlite3, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import metrics

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
DB = os.path.join(os.path.dirname(HERE), "data", "findata.sqlite")


def factors_for(symbol, dates, con):
    """Cumulative dividend factor per date: price_only = tr_close / factor.

    The unwind must start from the vendor's own adjustment reference -- the last
    bar it carries, in 2026 -- not from the end of the panel, or every dividend
    paid after the test period is left folded into the price.
    """
    if con is None:
        import portable
        hdates, hclose = portable.close_history(symbol, dates[0])
        rows = [(d, a, a) for d, a in portable.dividends(symbol, dates[0])]
    else:
        hist = con.execute(
            "select substr(ts,1,10), close from ohlc where symbol=? and ts>=? order by ts",
            (symbol, dates[0])).fetchall()
        hdates = [r[0] for r in hist]
        hclose = np.array([r[1] for r in hist], float)
        rows = con.execute(
            "select substr(ex_date,1,10), adj_amount, amount from dividends "
            "where symbol=? and ex_date>=? order by ex_date",
            (symbol, dates[0])).fetchall()
    divs = {}
    for d, adj, amt in rows:
        v = adj if adj is not None else amt
        if v:
            divs[d] = divs.get(d, 0.0) + float(v)

    cum = np.ones(len(hdates))
    k = 1.0
    for i in range(len(hdates) - 1, -1, -1):
        cum[i] = k
        d = hdates[i]
        if d in divs and i > 0:
            raw_prev = hclose[i - 1] / k + divs[d]
            if raw_prev > 0:
                k *= max(1e-6, 1.0 - divs[d] / raw_prev)

    by_date = dict(zip(hdates, cum))
    return np.array([by_date.get(d, np.nan) for d in dates])


def main():
    z = np.load(os.path.join(ART, "panel.npz"), allow_pickle=True)
    dates = list(z["dates"].astype(str))
    syms = list(z["symbols"].astype(str))
    op, cl = z["open"].astype(np.float64), z["close"].astype(np.float64)
    con = sqlite3.connect(DB) if os.path.exists(DB) else None

    F = np.ones_like(cl)
    for j, s in enumerate(syms):
        F[:, j] = factors_for(s, dates, con)
    F = np.where(np.isfinite(F), F, 1.0)
    op_px = op / F

    np.savez_compressed(os.path.join(ART, "price_only.npz"),
                        open=op_px.astype(np.float32), factor=F.astype(np.float32),
                        dates=np.array(dates), symbols=np.array(syms))

    d = np.array(dates)
    print(f"{'period':>8}{'equal-wt TR':>14}{'equal-wt price':>16}{'dividend drag':>16}")
    print("-" * 54)
    out = {}
    for yr in ("2021", "2022"):
        i = np.where((d >= yr + "-01-01") & (d <= yr + "-12-31"))[0]
        tr = metrics(list((op[i] / op[i][0]).mean(1)))
        px = metrics(list((op_px[i] / op_px[i][0]).mean(1)))
        out[yr] = {"total_return": tr, "price_only": px,
                   "drag": tr["cum_return"] - px["cum_return"]}
        print(f"{yr:>8}{tr['cum_return']:>+14.4f}{px['cum_return']:>+16.4f}"
              f"{tr['cum_return']-px['cum_return']:>+16.4f}")
    json.dump(out, open(os.path.join(ART, "price_only.json"), "w"), indent=1)

    # sanity: the factor should only ever deflate, and by a plausible amount
    print(f"\nfactor range {F.min():.4f}..{F.max():.4f}  "
          f"(median over 2021-01-04: {np.median(F[list(d).index('2021-01-04')]):.4f})")
    print("wrote", os.path.join(ART, "price_only.npz"))


if __name__ == "__main__":
    main()
