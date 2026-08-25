#!/usr/bin/env python3
"""Correctness checks on the trading environment.

Trading must not create value. At each step the portfolio should change only by
(a) the transaction cost paid and (b) the price move on whatever is held --
never by the act of trading itself.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import TradingEnv

rng = np.random.default_rng(3)
T, N = 400, 40
px = np.cumprod(1 + rng.normal(2e-4, 0.02, (T, N)), 0) * 60

fails = 0

# 1. trading is value-neutral up to cost, evaluated at constant prices
e = TradingEnv(px, fr=np.zeros((T, N)), seed=1)
for _ in range(200):
    p = e.p[e.t]
    v_before = float(p @ e.h + e.b)
    cost_before = e.cost_paid
    shares = rng.integers(-100, 101, N)
    e._execute(shares)
    v_after = float(p @ e.h + e.b)
    spent = e.cost_paid - cost_before
    if abs((v_before - v_after) - spent) > 1e-6 * max(1.0, v_before):
        fails += 1
        print(f"  value not conserved: before {v_before:.4f} after {v_after:.4f} cost {spent:.4f}")
    e.t += 1
print(f"[1] trading value-neutral up to cost: {'PASS' if fails == 0 else 'FAIL'} (200 steps)")

# 2. no shorting, no negative cash
e = TradingEnv(px, fr=np.zeros((T, N)), seed=2)
bad_h = bad_b = 0
done = False
while not done:
    a = rng.uniform(-1, 1, N)
    _, _, done, _ = e.step(a)
    bad_h += int((e.h < -1e-9).sum() > 0)
    bad_b += int(e.b < -1e-6)
print(f"[2] holdings stay long-only:  {'PASS' if bad_h == 0 else f'FAIL ({bad_h})'}")
print(f"[3] cash never goes negative: {'PASS' if bad_b == 0 else f'FAIL ({bad_b})'}")

# 4. a do-nothing agent's return must equal the buy-and-hold of what it holds
e = TradingEnv(px, fr=np.zeros((T, N)), seed=4)
e.step(np.full(N, 0.2))
h0, b0 = e.h.copy(), e.b
done = False
while not done:
    _, _, done, _ = e.step(np.zeros(N))
expect = float(px[e.t] @ h0 + b0)
got = e.portfolio_value()
print(f"[4] buy-and-hold matches held basket: "
      f"{'PASS' if abs(expect-got) < 1e-6*expect else 'FAIL'} ({expect:,.2f} vs {got:,.2f})")

# 5. hierarchical path: a hold direction must never move holdings
e = TradingEnv(px, fr=np.zeros((T, N)), seed=5)
moved = 0
for _ in range(150):
    d = rng.integers(-1, 2, N)
    h_before = e.h.copy()
    _, _, done, _ = e.step_hier(d, rng.uniform(-1, 1, N))
    moved += int(((d == 0) & (e.h != h_before)).sum())
    if done:
        break
print(f"[5] HLC 'hold' leaves position untouched: {'PASS' if moved == 0 else f'FAIL ({moved})'}")

# 6. cost actually bites: higher cost must not produce a higher return
outs = []
for c in (0.0, 0.001, 0.01):
    e = TradingEnv(px, fr=np.zeros((T, N)), cost=c, seed=6)
    r = np.random.default_rng(11)
    done = False
    while not done:
        _, _, done, _ = e.step(r.uniform(-1, 1, N))
    outs.append(e.portfolio_value())
mono = outs[0] >= outs[1] >= outs[2]
print(f"[6] higher cost never helps: {'PASS' if mono else 'FAIL'} "
      f"({outs[0]:,.0f} >= {outs[1]:,.0f} >= {outs[2]:,.0f})")
