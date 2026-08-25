#!/usr/bin/env python3
"""Passive floor measured inside the RL environment itself.

The equal-weight benchmark in baselines.py is a weight-space calculation. This
runs a do-nothing strategy through the same TradingEnv the agents use -- integer
share lots, hmax cap, cash-sequenced fills, 0.1% cost -- so any RL result can be
compared against "deploy the cash once and sit still" under identical mechanics.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import TradingEnv, metrics, HMAX
from train import build_data

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def buy_and_hold(prices, fr, seed=0):
    e = TradingEnv(prices, fr, seed=seed)
    N = e.N
    # day 1: spread the cash evenly, subject to the same hmax share cap
    budget = e.b / N
    shares = np.minimum(np.floor(budget / (prices[0] * (1 + e.cost))), HMAX)
    action = shares / HMAX                       # env multiplies back up by hmax
    done = e.step(action)[2]
    while not done:
        done = e.step(np.zeros(N))[2]
    m = metrics(e.values)
    m["turnover"] = float(np.mean(e.turnover))
    m["deployed_frac"] = float(1.0 - e.b / e.initial_capital)
    m["values"] = [float(v) for v in e.values]
    return m


def main():
    data = build_data(os.path.join(ART, "panel.npz"),
                      os.path.join(ART, "fr_causal.npz"), "causal")
    out = {}
    print(f"{'period':>10}{'cum':>12}{'ann':>12}{'vol':>10}{'sharpe':>10}{'maxDD':>10}{'deployed':>10}")
    print("-" * 74)
    for period in ("test2021", "test2022"):
        m = buy_and_hold(data[period]["prices"], data[period]["fr"])
        out[period] = m
        print(f"{period[-4:]:>10}{m['cum_return']:>+12.4f}{m['ann_return']:>+12.4f}"
              f"{m['ann_vol']:>10.4f}{m['sharpe']:>+10.4f}{m['max_drawdown']:>+10.4f}"
              f"{m['deployed_frac']:>10.3f}")
    json.dump(out, open(os.path.join(ART, "passive.json"), "w"), indent=1)
    print("\nwrote", os.path.join(ART, "passive.json"))


if __name__ == "__main__":
    main()
