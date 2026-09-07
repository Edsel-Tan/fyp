#!/usr/bin/env python3
"""Block-aware synthetic zero-predictability crypto panel.

Same null as analysis/synth_panel.py -- returns are a martingale difference, with
predictable variance left in -- but the block structure of the real tape is
preserved exactly, so the forecaster sees the same sample sizes, the same gaps
and the same train/valid/test boundaries as it does on real data. Any test IC it
reports here is what the workflow extracts from nothing.
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "hrt"))
from data import alpha158                                     # noqa: E402

ART = os.path.join(HERE, "artifacts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=os.path.join(ART, "panel_1hour.npz"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = a.out or os.path.join(ART, f"panel_1hour_synth_s{a.seed}.npz")
    rng = np.random.default_rng(a.seed)

    z = np.load(a.real, allow_pickle=True)
    block = z["block"]
    o0, c0 = z["open"].astype(np.float64), z["close"].astype(np.float64)
    v0 = z["volume"].astype(np.float64)
    T, N = c0.shape

    same = np.zeros(T, bool); same[1:] = block[1:] == block[:-1]
    cc = np.zeros((T, N))
    cc[1:] = np.where(same[1:, None], np.log(c0[1:] / c0[:-1]), 0.0)
    on = np.zeros((T, N))
    on[1:] = np.where(same[1:, None], np.log(o0[1:] / c0[:-1]), 0.0)
    sig = cc[same].std(0)
    mkt_sig = cc[same].mean(1).std()
    beta = np.array([np.cov(cc[same][:, i], cc[same].mean(1))[0, 1] /
                     cc[same].mean(1).var() for i in range(N)])
    on_sig = on[same].std(0)

    # stochastic volatility: magnitude predictable, sign not
    phi, sv = 0.97, 0.2
    h = np.zeros((T, N)); hm = np.zeros(T)
    for t in range(1, T):
        if same[t]:
            h[t] = phi * h[t - 1] + sv * rng.standard_normal(N)
            hm[t] = phi * hm[t - 1] + sv * rng.standard_normal()
    scale = np.exp(h - 0.5 * sv ** 2 / (1 - phi ** 2))
    mkt = mkt_sig * np.exp(hm - 0.5 * sv ** 2 / (1 - phi ** 2)) * rng.standard_normal(T)
    r = beta[None, :] * mkt[:, None] + sig[None, :] * scale * rng.standard_normal((T, N))
    r[~same] = 0.0
    on_r = on_sig[None, :] * rng.standard_normal((T, N)); on_r[~same] = 0.0

    close = np.empty((T, N)); open_ = np.empty((T, N))
    close[0] = c0[0]; open_[0] = o0[0]
    for t in range(1, T):
        base = close[t - 1] if same[t] else c0[t]
        open_[t] = base * np.exp(on_r[t])
        close[t] = open_[t] * np.exp(r[t] - on_r[t])
    span = np.abs(close - open_)
    extra = np.abs(rng.standard_normal((T, N))) * 0.6 * span
    high = np.maximum(open_, close) + extra * rng.random((T, N))
    low = np.maximum(np.minimum(open_, close) - extra * rng.random((T, N)), 1e-8)
    lv = np.log(v0 + 1.0)
    vol = np.exp(lv.mean(0)[None, :] + lv.std(0)[None, :] *
                 (0.7 * h + 0.7 * rng.standard_normal((T, N))))

    with np.errstate(all="ignore"):
        X, names = alpha158(open_, high, low, close, vol, (high + low + close) / 3.0)
    ac = np.nanmean([np.corrcoef(r[same][:-1, i], r[same][1:, i])[0, 1] for i in range(N)])
    print(f"synthetic crypto panel {X.shape}  mean lag-1 return autocorr {ac:+.4f}")

    np.savez_compressed(out, X=X, names=np.array(names), dates=z["dates"],
                        symbols=z["symbols"], block=block,
                        open=open_.astype(np.float32), high=high.astype(np.float32),
                        low=low.astype(np.float32), close=close.astype(np.float32),
                        volume=vol.astype(np.float32))
    print("wrote", out)


if __name__ == "__main__":
    main()
