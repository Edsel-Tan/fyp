#!/usr/bin/env python3
"""Signal-only benchmark: how much return does the HLC forecast itself carry,
with no RL in the loop?

Each day, hold an equal-weighted long book of the top-`k` names by forecast,
rebalanced at the open with the same 0.1% cost the RL agents pay. Run under both
signal timings to price the cost of the paper's look-ahead.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import metrics, COST
from train import build_data

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def top_k_long(prices, fr, k, cost=COST):
    T, N = prices.shape
    w = np.zeros(N)
    vals = [1.0]
    for t in range(T - 1):
        rank = np.argsort(-fr[t])[:k]
        w_new = np.zeros(N)
        w_new[rank] = 1.0 / k
        turn = np.abs(w_new - w).sum()
        r = float(w_new @ (prices[t + 1] / prices[t] - 1.0))
        vals.append(vals[-1] * (1 + r - cost * turn))
        w = w_new * (prices[t + 1] / prices[t])
        s = w.sum()
        w = w / s if s > 0 else w
    return vals


def main():
    out = {}
    print(f"{'signal':<10}{'k':>5}{'period':>10}{'cum':>12}{'ann':>12}{'vol':>10}{'sharpe':>10}{'maxDD':>10}")
    print("-" * 79)
    for mode in ("causal", "paper", "shuffle"):
        f = os.path.join(ART, "fr_paper.npz" if mode == "paper" else "fr_causal.npz")
        data = build_data(os.path.join(ART, "panel.npz"), f, mode)
        for k in (30, 100):
            for period in ("test2021", "test2022"):
                v = top_k_long(data[period]["prices"], data[period]["fr"], k)
                m = metrics(v)
                out[f"{mode}_k{k}_{period}"] = m
                print(f"{mode:<10}{k:>5}{period[-4:]:>10}{m['cum_return']:>+12.4f}"
                      f"{m['ann_return']:>+12.4f}{m['ann_vol']:>10.4f}"
                      f"{m['sharpe']:>+10.4f}{m['max_drawdown']:>+10.4f}")
    json.dump(out, open(os.path.join(ART, "signal_strategy.json"), "w"), indent=1)
    print("\nwrote", os.path.join(ART, "signal_strategy.json"))


if __name__ == "__main__":
    main()
