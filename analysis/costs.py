#!/usr/bin/env python3
"""Cost models for the MACE reproduction (PAPERS.md [2]).

MACE's claim is that replacing the flat-bps assumption with nonlinear impact
changes not just absolute performance but the *ranking* of strategies. Two
families are implemented so that claim can be tested directly:

  flat(bps)            cost = bps * notional                 -- FinRL / HRT default
  square_root(...)     Almgren-Chriss temporary impact under the square-root law,
                       cost = (spread/2) * notional + Y * sigma_t * sqrt(q / V_t) * notional

The second is *state-dependent*: sigma_t is the trailing realised volatility of
the bar and V_t its traded volume, so the identical trade is charged differently
in a calm high-volume regime than in a stressed thin one. That state dependence
is the whole point -- MACE parameterises a static functional form once, which
prices a trade the same way in every regime.

Permanent impact is tracked with exponential decay, as in MACE, but is charged
against the executing strategy only through the price path it leaves behind; for
the single-agent backtests here it is reported rather than netted, since a
single small book does not move a real market.
"""
import numpy as np


class FlatCost:
    name = "flat"

    def __init__(self, bps=10.0):
        self.bps = bps
        self.rate = bps * 1e-4

    def charge(self, notional, **_):
        return float(np.sum(np.abs(notional)) * self.rate)

    def __repr__(self):
        return f"flat({self.bps:.0f}bp)"


class SquareRootCost:
    """Half-spread plus Almgren-Chriss temporary impact under the square-root law.

    sigma: per-bar realised volatility [T,N]; volume: per-bar traded notional [T,N].
    `participation` caps q/V so a degenerate thin bar cannot produce an infinite charge.
    """
    name = "sqrt"

    def __init__(self, sigma, volume, half_spread_bps=5.0, Y=1.0, cap=0.5):
        self.sigma = np.asarray(sigma, float)
        self.volume = np.maximum(np.asarray(volume, float), 1.0)
        self.hs = half_spread_bps * 1e-4
        self.Y = Y
        self.cap = cap

    def charge(self, notional, t=None, **_):
        n = np.abs(np.asarray(notional, float))
        part = np.minimum(n / self.volume[t], self.cap)
        impact = self.Y * self.sigma[t] * np.sqrt(part)
        return float(np.sum(n * (self.hs + impact)))

    def __repr__(self):
        return f"sqrt(hs={self.hs*1e4:.0f}bp,Y={self.Y})"


def realised_vol(close, window=24, block=None):
    """Trailing realised volatility per bar, computed within blocks only."""
    c = np.asarray(close, float)
    r = np.zeros_like(c)
    r[1:] = np.log(c[1:] / c[:-1])
    if block is not None:
        b = np.asarray(block)
        r[1:][b[1:] != b[:-1]] = 0.0
    out = np.zeros_like(c)
    for t in range(len(c)):
        lo = max(0, t - window + 1)
        seg = r[lo:t + 1]
        if block is not None:
            seg = seg[np.asarray(block)[lo:t + 1] == block[t]]
        out[t] = seg.std(0) if len(seg) > 2 else 0.0
    return np.maximum(out, 1e-5)
