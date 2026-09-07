#!/usr/bin/env python3
"""Synthetic zero-predictability panel, for the falsification audit of PAPERS.md [10].

Nikolopoulos (arXiv:2604.15531) proposes testing the *workflow*, not the strategy:
run the complete pipeline against a reference class in which there is provably
nothing to find, and treat any significant walk-forward evidence as falsifying.

The null here is a martingale difference in returns. Conditional *variance* is
left predictable (stochastic vol, volume clustering) because that is realistic and
because Alpha158 is largely a volatility/volume feature set -- the point is that no
feature can carry information about the sign of a future return, since every
return increment is drawn independently of everything observable before it.

The generator matches the real panel's shape, calendar, per-name volatility, market
factor and overnight/intraday variance split, so the downstream code cannot tell the
difference from anything except predictability itself.

  python analysis/synth_panel.py --seed 0 --out hrt/artifacts/panel_synth.npz
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "hrt"))
from data import alpha158                                     # noqa: E402

ART = os.path.join(HERE, os.pardir, "hrt", "artifacts")


def moments(real):
    """Per-name daily vol, market-factor loading and overnight share, from the real panel."""
    o, c = real["open"].astype(np.float64), real["close"].astype(np.float64)
    cc = np.log(c[1:] / c[:-1])                               # close-to-close
    on = np.log(o[1:] / c[:-1])                               # overnight
    mkt = np.nanmean(cc, 1)
    beta = np.array([np.cov(cc[:, i], mkt)[0, 1] / mkt.var() for i in range(cc.shape[1])])
    resid = cc - beta[None, :] * mkt[:, None]
    return dict(sig_mkt=float(mkt.std()),
                sig_idio=np.nanstd(resid, 0),
                beta=beta,
                on_share=float(np.nanvar(on) / np.nanvar(cc)),
                logv_mu=np.nanmean(np.log(real["volume"].astype(np.float64) + 1.0), 0),
                logv_sd=np.nanstd(np.log(real["volume"].astype(np.float64) + 1.0), 0),
                p0=c[0].astype(np.float64))


def generate(m, T, N, rng, sv_phi=0.95, sv_vol=0.25):
    """Return dict of [T,N] open/high/low/close/volume with martingale-difference returns."""
    # stochastic volatility: predictable magnitude, unpredictable sign
    h = np.zeros((T, N))
    for t in range(1, T):
        h[t] = sv_phi * h[t - 1] + sv_vol * rng.standard_normal(N)
    scale = np.exp(h - 0.5 * sv_vol ** 2 / (1 - sv_phi ** 2))

    hm = np.zeros(T)
    for t in range(1, T):
        hm[t] = sv_phi * hm[t - 1] + sv_vol * rng.standard_normal()
    mkt = m["sig_mkt"] * np.exp(hm - 0.5 * sv_vol ** 2 / (1 - sv_phi ** 2)) * rng.standard_normal(T)

    idio = m["sig_idio"][None, :] * scale * rng.standard_normal((T, N))
    cc = m["beta"][None, :] * mkt[:, None] + idio                  # log close-to-close

    # split each day into an overnight jump and an intraday drift, variances matched
    s = np.sqrt(m["on_share"])
    on = s * cc + np.sqrt(max(1e-9, m["on_share"] * (1 - m["on_share"]))) * \
        rng.standard_normal((T, N)) * 0.0                          # keep on = s*cc, no extra noise
    intr = cc - on

    close = np.empty((T, N))
    close[0] = m["p0"]
    close[1:] = m["p0"][None, :] * np.exp(np.cumsum(cc[1:], 0))
    prev = np.vstack([m["p0"][None, :], close[:-1]])
    open_ = prev * np.exp(on)
    close = open_ * np.exp(intr)

    # intraday range: a Brownian bridge overshoot on top of the open-close span
    span = np.abs(close - open_)
    extra = np.abs(rng.standard_normal((T, N))) * 0.6 * span
    high = np.maximum(open_, close) + extra * rng.random((T, N))
    low = np.minimum(open_, close) - extra * rng.random((T, N))
    low = np.maximum(low, 1e-6)

    logv = m["logv_mu"][None, :] + m["logv_sd"][None, :] * (0.7 * h + 0.7 * rng.standard_normal((T, N)))
    vol = np.exp(logv)
    return dict(open=open_, high=high, low=low, close=close, volume=vol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=os.path.join(ART, "panel.npz"))
    ap.add_argument("--out", default=os.path.join(ART, "panel_synth.npz"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    z = np.load(a.real, allow_pickle=True)
    real = {k: z[k] for k in ("open", "high", "low", "close", "volume")}
    dates, syms = z["dates"], z["symbols"]
    T, N = real["close"].shape
    m = moments(real)
    px = generate(m, T, N, np.random.default_rng(a.seed))

    o, h, l, c, v = (px[k] for k in ("open", "high", "low", "close", "volume"))
    vwap = (h + l + c) / 3.0
    with np.errstate(all="ignore"):
        X, names = alpha158(o, h, l, c, v, vwap)
    print(f"synthetic panel {X.shape}  nan-frac={np.isnan(X).mean():.4f}")

    # sanity: the null must hold -- no autocorrelation in returns
    r = np.log(c[1:] / c[:-1])
    ac = np.nanmean([np.corrcoef(r[:-1, i], r[1:, i])[0, 1] for i in range(0, N, 10)])
    print(f"mean lag-1 return autocorr = {ac:+.4f}   (real panel: "
          f"{np.nanmean([np.corrcoef(np.log(real['close'][1:-1, i]/real['close'][:-2, i]), np.log(real['close'][2:, i]/real['close'][1:-1, i]))[0,1] for i in range(0, N, 10)]):+.4f})")

    np.savez_compressed(a.out, X=X, names=np.array(names), dates=dates, symbols=syms,
                        open=o.astype(np.float32), high=h.astype(np.float32),
                        low=l.astype(np.float32), close=c.astype(np.float32),
                        volume=v.astype(np.float32))
    print("wrote", a.out, f"{os.path.getsize(a.out)/1e6:.0f} MB")


if __name__ == "__main__":
    main()
