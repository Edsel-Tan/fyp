#!/usr/bin/env python3
"""FinRL-style multi-stock trading environment for the HRT reproduction.

Follows Liu et al. (2018) as the HRT paper states: state [p, h, b], continuous
share actions bounded by hmax, 0.1% proportional cost, $1M initial capital.
Trades execute at the open, so step i prices are the open of calendar day i and
the reward is realised against the open of day i+1.

Two action interfaces on the same dynamics:
  step(shares)                  -- flat agents (PPO / DDPG baselines)
  step_hier(direction, magnitude) -- HRT: HLC gives sign, LLC gives size
"""
import numpy as np

INITIAL_CAPITAL = 1_000_000.0
HMAX = 100
COST = 0.001
REWARD_SCALING = 1e-4


class TradingEnv:
    def __init__(self, prices, fr=None, initial_capital=INITIAL_CAPITAL,
                 hmax=HMAX, cost=COST, reward_scaling=REWARD_SCALING, seed=0):
        """prices [T,N] execution (open) prices; fr [T,N] HLC signal (optional)."""
        self.p = np.asarray(prices, np.float64)
        self.fr = None if fr is None else np.nan_to_num(np.asarray(fr, np.float64))
        self.T, self.N = self.p.shape
        self.initial_capital = initial_capital
        self.hmax, self.cost, self.reward_scaling = hmax, cost, reward_scaling
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------- lifecycle
    def reset(self):
        self.t = 0
        self.h = np.zeros(self.N)
        self.b = self.initial_capital
        self.trades = 0
        self.cost_paid = 0.0
        self.values = [self.portfolio_value()]
        self.turnover = []
        self.last_trade = np.zeros(self.N, np.int64)
        self.trade_log = []
        return self.state()

    def portfolio_value(self, t=None):
        t = self.t if t is None else t
        return float(self.p[t] @ self.h + self.b)

    # ---------------------------------------------------------------- states
    def state(self):
        """LLC / flat-agent state: [p, h, b] normalised. a^h appended by caller."""
        p = self.p[self.t]
        return np.concatenate([p / 100.0, self.h / self.hmax,
                               [self.b / self.initial_capital]]).astype(np.float32)

    def hlc_state(self):
        """HRT-FR high-level state: the predicted forward-return vector."""
        return self.fr[self.t].astype(np.float32)

    # ------------------------------------------------------------- execution
    def _execute(self, shares):
        """shares: signed integer trade sizes. Sells settle first, then buys are
        filled against remaining cash in a randomised order (vectorised)."""
        p = self.p[self.t]

        sell = np.minimum(np.maximum(-shares, 0), self.h.astype(np.int64))
        gross_sell = float((p * sell).sum())
        self.b += gross_sell * (1 - self.cost)
        self.cost_paid += gross_sell * self.cost
        self.h -= sell

        buy = np.maximum(shares, 0)
        gross_buy = 0.0
        idx = np.flatnonzero(buy > 0)
        if idx.size:
            # cash is finite; randomise priority so no name is starved by index
            idx = idx[self.rng.permutation(idx.size)]
            unit = p[idx] * (1 + self.cost)
            need = buy[idx] * unit
            cum = np.cumsum(need)
            full = cum <= self.b
            k = buy[idx] * full
            j = int(full.sum())                       # first order that cannot fill
            if j < idx.size and unit[j] > 0:
                left = self.b - (cum[j - 1] if j else 0.0)
                k[j] = min(buy[idx][j], int(left // unit[j]))
            spent_gross = float((k * p[idx]).sum())
            self.b -= spent_gross * (1 + self.cost)
            self.cost_paid += spent_gross * self.cost
            np.add.at(self.h, idx, k)
            gross_buy = spent_gross

        self.trades += int((sell > 0).sum() + (buy > 0).sum())
        self.last_trade = np.zeros(self.N, np.int64)
        self.last_trade -= sell
        if idx.size:
            np.add.at(self.last_trade, idx, k)
        return gross_sell + gross_buy

    def _advance(self, notional):
        self.trade_log.append(self.last_trade.copy())
        v0 = self.portfolio_value()
        self.t += 1
        v1 = self.portfolio_value()
        self.values.append(v1)
        self.turnover.append(notional / max(v0, 1e-9))
        reward = (v1 - v0) * self.reward_scaling
        done = self.t >= self.T - 1
        return reward, done

    # --------------------------------------------------------------- stepping
    def step(self, action):
        """Flat interface: action in [-1,1]^N -> signed shares."""
        shares = np.rint(np.clip(action, -1, 1) * self.hmax).astype(np.int64)
        notional = self._execute(shares)
        reward, done = self._advance(notional)
        return self.state(), reward, done, {}

    def step_hier(self, direction, magnitude):
        """HRT interface: direction in {-1,0,1}^N, magnitude in [-1,1]^N.

        The LLC's [-1,1] output maps to the paper's {0..k<=hmax} share count;
        the HLC's direction supplies the sign. Held names never reach the LLC.
        """
        size = np.rint((np.clip(magnitude, -1, 1) + 1.0) / 2.0 * self.hmax).astype(np.int64)
        shares = direction.astype(np.int64) * size
        p_now = self.p[self.t].copy()
        notional = self._execute(shares)
        reward, done = self._advance(notional)
        # alignment reward: did each direction match the realised open-to-open move?
        dp = np.sign(self.p[self.t] - p_now)
        align = np.where(direction == 0, 0.0, np.sign(direction) * dp)
        return self.state(), reward, done, {"align": align, "n_active": int((direction != 0).sum())}


def metrics(values, periods=252):
    """Paper's conventions: Sharpe = annualised return / annualised vol, rf = 0."""
    v = np.asarray(values, float)
    r = v[1:] / v[:-1] - 1.0
    n = len(r)
    cum = v[-1] / v[0] - 1.0
    ann = (1.0 + cum) ** (periods / n) - 1.0
    vol = r.std(ddof=1) * np.sqrt(periods)
    peak = np.maximum.accumulate(v)
    mdd = float((v / peak - 1.0).min())
    return {"cum_return": float(cum), "ann_return": float(ann),
            "ann_vol": float(vol), "sharpe": float(ann / vol) if vol > 1e-12 else 0.0,
            "max_drawdown": mdd}
