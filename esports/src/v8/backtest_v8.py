"""v8 on the historical execution backtest (v7/backtest_v7.py's protocol: backtest.py fills, eval months 2026-02..04, theta
chosen on the selection months; v8m = v8 combined with the archive mid per phase, fit on the selection months).
Usage: python v8/backtest_v8.py   (results/backtest_v8.csv, results/backtest_v8.log)"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path[:0] = [SRC, os.path.join(SRC, 'v5'), os.path.join(SRC, 'v7')]
os.chdir(SRC)
import backtest as B
import backtest_v5 as B5
import backtest_v7 as B7
from eval_pregame import ll

B.SFX = '_v8'
SIGS = ['q_v8', 'q_v8m', 'q_v8_n', 'q_v7', 'q_v7m']


def main():
    fr, fills = B.load()
    rows = []
    for L in (0, 5, 15, 30, 60):
        q = B.quotes(fr, fills, L)
        sel = q[q.t_start < B.SEL_END]
        for s in ('q_v7', 'q_v8'):
            q[s + 'm'] = B7.comb_apply(q, B7.comb_fit(sel, s), s)
        sel, ev = q[q.t_start < B.SEL_END], q[q.t_start >= B.SEL_END]
        y = ev.blue_win.astype(int).to_numpy()
        print(f'L={L:3d}s eval-month frame LL: ' + '  '.join(f'{c} {ll(y, ev[c].clip(1e-4, 1 - 1e-4).to_numpy()).mean():.4f}'
                                                       for c in ['mid'] + SIGS), flush=True)
        for sig in SIGS:
            for strat, th in (('clip', None), ('gate', B5.choose(sel, sig))):
                r = B.run(ev, sig, 'A') if strat == 'clip' else B.run(ev, sig, 'B', th)
                rows.append(dict(L=L, sig=sig, strat=strat, theta=th, **B.summarize(r, 'eval'), **B5.sharpe(r)))
    R = pd.DataFrame(rows).drop(columns='name', errors='ignore')
    pd.set_option('display.width', 250)
    print(R[['L', 'sig', 'strat', 'theta', 'n', 'games', 'gross_pp', 'ci_lo', 'ci_hi', 'fee_regime_net_pp', 'sr_ann_net']].to_string(
        index=False, float_format=lambda x: f'{x:.2f}'))
    R.to_csv('../results/backtest_v8.csv', index=False)


if __name__ == '__main__':
    main()
