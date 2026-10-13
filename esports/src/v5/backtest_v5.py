"""v5 on the historical execution backtest (backtest.py's fills and rules; Polymarket-v1 prints, eval months 2026-02..04).

For L in 0/5/15/30/60/120 s and signals v5, v4, v2 and comb5 (sigma(a logit(mid) + b logit(q_v5) + c), fit on the selection months):
  clip  (strategy A: every frame, 1 share toward q)
  gate  (strategy B: trade when q - ask > theta or bid - q > theta; theta chosen on the selection months)
Sharpe:
  per game  mean / sd of per-game PnL (pp per share summed over the game's trades), as significance.py's SR
  daily     calendar days of the eval window (zero on days without trades), x sqrt(365)
  net uses each fill's observed fee (fees exist from 2026-03)
plus PSR vs 0 and the deflated Sharpe ratio (DSR) over the cells examined.
At L=0 for v5: theta curve (sensitivity only), placebo (another game's signal at the same minute), breakdowns, and the five
shuffled-outcome nulls (v5/build.py nulls) through the same selection.
Writes results/backtest_v5.csv, results/backtest_v5_sharpe.csv, data/v5/bt_trades.parquet.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import norm, skew, kurtosis
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ['PREDS_SFX'] = '_v5'
import backtest as B

D = '../data'
SIGS = ['q_v5', 'q_v4', 'q_lr v2', 'q_comb5']
LAGS = (0, 5, 15, 30, 60, 120)
EV_DAYS = pd.date_range(B.SEL_END, pd.Timestamp('2026-04-28', tz='UTC'), freq='D')


def choose(sel, sig):
    best = max(((B.summarize(B.run(sel, sig, 'B', th), 'sel').get('gross_pp', -99), th) for th in B.THETAS),
               key=lambda z: z[0] if not np.isnan(z[0]) else -99)
    return best[1]


def sharpe(r):
    """Per-game and daily-annualised Sharpe of a trade list (gross and net)."""
    out = {}
    if not len(r):
        return out
    r = r.assign(net=r.gross - r.fee.fillna(0), day=r.t_start.dt.floor('D'))
    for k in ('gross', 'net'):
        g = r.groupby('game_id')[k].sum()
        out[f'sr_game_{k}'] = g.mean() / g.std() if len(g) > 2 else np.nan
        d = r.groupby('day')[k].sum().reindex(EV_DAYS, fill_value=0.0)
        out[f'sr_ann_{k}'] = d.mean() / d.std() * np.sqrt(365) if d.std() > 0 else np.nan
        out[f'total_{k}'] = r[k].sum()
    g = r.groupby('game_id').gross.sum().to_numpy()
    sr, T = g.mean() / g.std(), len(g)
    out['psr0'] = norm.cdf(sr * np.sqrt(T - 1) / np.sqrt(1 - skew(g) * sr + (kurtosis(g, fisher=False) - 1) / 4 * sr ** 2))
    out['n_games_sr'] = T
    return out


def dsr(r, n_trials, sr_var):
    g = r.groupby('game_id').gross.sum().to_numpy()
    sr, T = g.mean() / g.std(), len(g)
    e = 0.5772156649
    sr0 = np.sqrt(sr_var) * ((1 - e) * norm.ppf(1 - 1 / n_trials) + e * norm.ppf(1 - 1 / (n_trials * np.e)))
    return norm.cdf((sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - skew(g) * sr + (kurtosis(g, fisher=False) - 1) / 4 * sr ** 2))


def main():
    fr, fills = B.load()
    rows, trades = [], []
    for L in LAGS:
        q = B.quotes(fr, fills, L)
        sel = q[q.t_start < B.SEL_END]
        X = lambda d: np.c_[B.logit(d.mid), B.logit(d.q_v5)]
        cm = LogisticRegression(C=1e6, max_iter=1000).fit(X(sel), sel.blue_win.astype(int))
        q['q_comb5'] = cm.predict_proba(X(q))[:, 1]
        sel, ev = q[q.t_start < B.SEL_END], q[q.t_start >= B.SEL_END]
        for sig in SIGS:
            for strat, th in (('clip', None), ('gate', choose(sel, sig))):
                r = B.run(ev, sig, 'A') if strat == 'clip' else B.run(ev, sig, 'B', th)
                rows.append(dict(L=L, sig=sig, strat=strat, theta=th, **B.summarize(r, 'eval'), **sharpe(r)))
                trades.append(r.assign(L=L, sig=sig, strat=strat, theta=th))
        if L == 0:
            th = choose(sel, 'q_v5')
            print(f'== L=0, v5: theta {th} chosen on the selection months; eval theta curve (sensitivity only):')
            curve = [dict(theta=t, **B.summarize(B.run(ev, 'q_v5', 'B', t), 'eval')) for t in B.THETAS]
            print(pd.DataFrame(curve)[['theta', 'n', 'games', 'gross_pp', 'ci_lo', 'ci_hi', 'fee_regime_net_pp']].to_string(
                index=False, float_format=lambda x: f'{x:.2f}'))
            pl = [B.summarize(B.run(B.placebo(ev, 'q_v5', k), 'q_v5_placebo', 'B', th), 'p').get('gross_pp', np.nan) for k in range(20)]
            print(f'placebo (another game\'s v5 at the same minute, theta {th}, 20 draws): mean {np.nanmean(pl):.2f} pp, sd {np.nanstd(pl):.2f}')
            r0 = B.run(ev, 'q_v5', 'B', th)
            r0['price'] = pd.cut(r0.px, [0, .2, .4, .6, .8, 1])
            r0['fav'] = np.where((r0.side == 1) == (r0.mid > .5), 'buy favourite', 'buy underdog')
            r0['minute'] = pd.cut(r0.clock / 60, [0, 10, 20, 30, 60])
            r0['regime'] = np.where(r0.t_start >= B.FEE_START, 'fee (2026-03+)', 'pre-fee')
            for c in ['price', 'fav', 'minute', 'regime']:
                print(r0.groupby(c, observed=True).agg(n=('gross', 'size'), games=('game_id', 'nunique'),
                                                       gross_pp=('gross', lambda x: 100 * x.mean())).to_string(float_format=lambda x: f'{x:.2f}'))
        print(f'L={L}s done: frames sel {len(sel)} eval {len(ev)}; comb5 weights mid/v5 {cm.coef_[0].round(3)}', flush=True)
    R = pd.DataFrame(rows).drop(columns='name')
    T = pd.concat(trades, ignore_index=True)

    # deflated Sharpe of v5's gate at L=0: trials = every (signal, L, strategy) cell here plus the 48-cell family of significance.py
    cell = T[(T.L == 0) & (T.sig == 'q_v5') & (T.strat == 'gate')]
    srs = T.groupby(['L', 'sig', 'strat']).apply(lambda r: (lambda g: g.mean() / g.std())(r.groupby('game_id').gross.sum()))
    for n in (len(srs), len(srs) + 48):
        print(f'DSR of v5 gate L=0 with N={n} trials (SR variance across this file\'s cells {srs.var():.4f}): {dsr(cell, n, srs.var()):.3f}')

    print('\n== nulls (v5 refit on shuffled train outcomes), L=0 ==')
    for k in range(1, 6):
        B.SFX = f'_v5_null{k}'
        frn, _ = B.load()
        qn = B.quotes(frn, fills, 0)
        seln, evn = qn[qn.t_start < B.SEL_END], qn[qn.t_start >= B.SEL_END]
        thn = choose(seln, 'q_v5')
        for strat, r in (('clip', B.run(evn, 'q_v5', 'A')), ('gate', B.run(evn, 'q_v5', 'B', thn))):
            s = B.summarize(r, 'null')
            rows.append(dict(L=0, sig=f'q_v5 null{k}', strat=strat, theta=thn if strat == 'gate' else None, **{a: b for a, b in s.items() if a != 'name'}, **sharpe(r)))
            print(f'  null {k} {strat:5s} theta {thn if strat == "gate" else "-"}: {s.get("gross_pp", np.nan):+.2f} pp over {s.get("n", 0)} trades')
    R = pd.DataFrame(rows).drop(columns='name', errors='ignore')
    pd.set_option('display.width', 320)
    cols = ['L', 'sig', 'strat', 'theta', 'n', 'games', 'gross_pp', 'ci_lo', 'ci_hi', 'pre_fee_gross_pp', 'fee_regime_gross_pp',
            'fee_regime_net_pp', 'sr_game_gross', 'sr_game_net', 'sr_ann_gross', 'sr_ann_net', 'psr0']
    print('\n' + R[cols].to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    R.to_csv(f'{D}/../results/backtest_v5.csv', index=False)
    T.to_parquet(f'{D}/v5/bt_trades.parquet')


if __name__ == '__main__':
    main()
