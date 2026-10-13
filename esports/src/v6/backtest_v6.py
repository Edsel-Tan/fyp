"""v6 on the historical execution backtest (backtest.py's fills and rules; Polymarket-v1 prints, eval months 2026-02..04),
the same protocol as v5/backtest_v5.py.

Signals: q_v6 (market-free), q_v6m (v6 combined with the market's mid at the decision instant, per game phase), q_v5, q_v4, v2.
  q_v6m = sigma(b_p logit(mid) + c_p logit(q_v6)), p = 0-10 / 10-20 / 20-30 / 30+ min, fit at each L on the selection
  months only (archive mid = the print-based proxy). The live tracker's weights come from the captured books instead
  (v6/book_sim_v6.py), since its mid is a book mid.
Strategies: clip, and gate with theta chosen on the selection months. Sharpe per game / annualised daily, PSR, DSR.
Also the v6 stages (q_v6_x, q_v6_pm, q_v6_pers) at L=0 and 30 s, and the placebo and minute breakdown for v6 / v6m.
Writes results/backtest_v6.csv, data/v6/bt_trades.parquet, data/v6/comb_archive.pkl.
"""
import os, sys, pickle
import numpy as np, pandas as pd
import statsmodels.api as sm
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path[:0] = [SRC, os.path.join(SRC, 'v5')]
import backtest as B
import backtest_v5 as B5

B.SFX = '_v6'
D = '../data'
SIGS = ['q_v6', 'q_v6m', 'q_v5', 'q_v4', 'q_lr v2']
STAGES = ['q_v6_x', 'q_v6_pm', 'q_v6_pers']
LAGS = (0, 5, 15, 30, 60, 120)
PH = [0, 10, 20, 30, 1000]


def phase(d):
    return np.clip(np.searchsorted(PH, d.clock.to_numpy() / 60, side='right') - 1, 0, len(PH) - 2)


def comb_fit(sel, sig='q_v6'):
    out = {}
    ph = phase(sel)
    for p in range(len(PH) - 1):
        x = sel[ph == p]
        X = np.c_[B.logit(x.mid), B.logit(x[sig])]  # no intercept: with one, the fit picks up the selection months' side
        out[p] = np.r_[0.0, sm.GLM(x.blue_win.astype(int).to_numpy(), X, family=sm.families.Binomial()).fit().params]  # balance (-0.4 logit) and loses out of sample
    return out


def comb_apply(d, co, sig='q_v6'):
    ph = phase(d)
    z = np.zeros(len(d))
    for p, c in co.items():
        m = ph == p
        z[m] = c[0] + c[1] * B.logit(d.mid.to_numpy()[m]) + c[2] * B.logit(d[sig].to_numpy()[m])
    return 1 / (1 + np.exp(-z))


def main():
    fr, fills = B.load()
    rows, trades, combs = [], [], {}
    for L in LAGS:
        q = B.quotes(fr, fills, L)
        sel = q[q.t_start < B.SEL_END]
        co = comb_fit(sel)
        combs[L] = co
        q['q_v6m'] = comb_apply(q, co)
        print(f'L={L:3d}s comb weights by phase (const, mid, v6): ' + '; '.join(f'{PH[p]}-{PH[p + 1] if PH[p + 1] < 1000 else "end"} '
              + ' '.join(f'{v:+.2f}' for v in c) for p, c in co.items()), flush=True)
        sel, ev = q[q.t_start < B.SEL_END], q[q.t_start >= B.SEL_END]
        for sig in SIGS + (STAGES if L in (0, 30) else []):
            for strat, th in (('clip', None), ('gate', B5.choose(sel, sig))):
                r = B.run(ev, sig, 'A') if strat == 'clip' else B.run(ev, sig, 'B', th)
                rows.append(dict(L=L, sig=sig, strat=strat, theta=th, **B.summarize(r, 'eval'), **B5.sharpe(r)))
                trades.append(r.assign(L=L, sig=sig, strat=strat, theta=th))
        if L in (0, 30):
            from eval_pregame import ll
            y = ev.blue_win.astype(int).to_numpy()
            print(f'  eval-month frame log loss at L={L}: ' + '  '.join(
                f'{c} {ll(y, ev[c].clip(1e-4, 1 - 1e-4).to_numpy()).mean():.4f}' for c in ['mid', 'q_v6m', 'q_v6', 'q_v5']))
            for sig in ('q_v6', 'q_v6m'):
                th = B5.choose(sel, sig)
                pl = [B.summarize(B.run(B.placebo(ev, sig, k), sig + '_placebo', 'B', th), 'p').get('gross_pp', np.nan) for k in range(20)]
                r0 = B.run(ev, sig, 'B', th)
                r0['minute'] = pd.cut(r0.clock / 60, [0, 10, 20, 30, 60])
                print(f'  {sig} gate theta {th}: placebo mean {np.nanmean(pl):+.2f} sd {np.nanstd(pl):.2f} pp; by minute ' +
                      str(r0.groupby('minute', observed=True).gross.agg(n='size', pp=lambda v: round(100 * v.mean(), 1)).to_dict('index')))
    R = pd.DataFrame(rows).drop(columns='name', errors='ignore')
    T = pd.concat(trades, ignore_index=True)
    srs = T.groupby(['L', 'sig', 'strat']).apply(lambda r: (lambda g: g.mean() / g.std())(r.groupby('game_id').gross.sum()))
    for sig in ('q_v6', 'q_v6m'):
        for L in (0, 30):
            cell = T[(T.L == L) & (T.sig == sig) & (T.strat == 'gate')]
            if len(cell) > 3:
                print(f'DSR {sig} gate L={L}: N={len(srs)} {B5.dsr(cell, len(srs), srs.var()):.3f}; N={len(srs) + 96} '
                      f'(+ the v2/v4/v5 families) {B5.dsr(cell, len(srs) + 96, srs.var()):.3f}')
    pd.set_option('display.width', 320)
    cols = ['L', 'sig', 'strat', 'theta', 'n', 'games', 'gross_pp', 'ci_lo', 'ci_hi', 'fee_regime_gross_pp',
            'fee_regime_net_pp', 'sr_game_gross', 'sr_game_net', 'sr_ann_gross', 'sr_ann_net', 'psr0']
    print('\n' + R[cols].to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    R.to_csv(f'{D}/../results/backtest_v6.csv', index=False)
    T.to_parquet(f'{D}/v6/bt_trades.parquet')
    pickle.dump(dict(phases=PH, comb=combs, note='q_v6m = sigma(c0 + c1 logit(mid) + c2 logit(q_v6)) per phase (c0 = 0); keys = latency L (s); '
                     'fit on the archive selection months (< 2026-02-01), mid = (last buy print + last sell print) / 2'),
                open(f'{D}/v6/comb_archive.pkl', 'wb'))


if __name__ == '__main__':
    main()
