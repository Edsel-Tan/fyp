"""Next step 3: formal deflation of the backtest and Diebold-Mariano tests of model vs market.

1. Every backtest cell (signal x latency L x strategy, eval months, theta for the gate chosen on selection months exactly
   as backtest.py) gets a match-clustered t on mean gross pp/share (ratio estimator: sum over matches / shares, with a
   cluster-robust SE) and a one-sided p (H1: > 0). Holm and Benjamini-Hochberg over the whole family and over the
   gate-only sub-family. `family` is printed so the count is explicit.
2. Deflated Sharpe (Bailey & Lopez de Prado 2014) for the primary cell (q_lr v2, gate, L=0): per-game PnL (sum over the
   game's 1-share trades), games in time order; SR0 from the cross-trial variance of per-game Sharpe over (a) the cells
   and (b) every (cell, theta) combination evaluated on the eval months (the widest honest count of what was looked at).
3. Diebold-Mariano on per-game log loss, model minus market, from market_frames_lag{L}.parquet (market_compare output):
   the loss differential is averaged per match (games within a series are dependent), matches ordered by start time,
   Newey-West SE with floor(4 (n/100)^(2/9)) lags. Two-sided p.
"""
import sys
import numpy as np, pandas as pd
from scipy import stats
import backtest as B
from eval_pregame import ll

D = '../data'
LAGS = [0, 5, 15, 30, 60, 120]
SIGS = ['q_lr v2', 'q_lr all', 'q_gold', 'q_comb']


def clustered_t(r):
    g = r.groupby('match_id').gross.agg(['sum', 'size'])
    s, n = g['sum'].to_numpy(), g['size'].to_numpy()
    mu = s.sum() / n.sum()
    e = s - mu * n
    G = len(g)
    se = np.sqrt(G / (G - 1) * (e ** 2).sum()) / n.sum()
    return mu, se, mu / se, G


def holm(p):
    p = np.asarray(p)
    o = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o])
    out = np.empty_like(adj); out[o] = np.minimum(adj, 1)
    return out


def bh(p):
    p = np.asarray(p)
    o = np.argsort(p)[::-1]
    m = len(p)
    adj = np.minimum.accumulate(p[o] * m / (m - np.arange(m)))
    out = np.empty_like(adj); out[o] = np.minimum(adj, 1)
    return out


def per_game(r):
    return r.sort_values('t_start').groupby('game_id', sort=False).agg(pnl=('gross', 'sum'), t=('t_start', 'first')).sort_values('t').pnl.to_numpy()


def sharpe(x):
    return x.mean() / x.std(ddof=1) if len(x) > 2 and x.std() > 0 else np.nan


def dsr(x, sr_trials):
    """Deflated Sharpe ratio: P(true SR > SR0), SR0 = expected max of N trial Sharpes under the null."""
    sr, T = sharpe(x), len(x)
    N = np.isfinite(sr_trials).sum()
    v = np.nanvar(sr_trials, ddof=1)
    em = 0.5772156649
    sr0 = np.sqrt(v) * ((1 - em) * stats.norm.ppf(1 - 1 / N) + em * stats.norm.ppf(1 - 1 / (N * np.e)))
    g3, g4 = stats.skew(x), stats.kurtosis(x, fisher=False)
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    psr0 = stats.norm.cdf(sr * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return dict(sr=sr, T=T, N=N, sd_trials=np.sqrt(v), sr0=sr0, psr_vs_0=psr0, dsr=stats.norm.cdf(z), skew=g3, kurt=g4)


def backtest_family():
    fr, fills = B.load()
    from sklearn.linear_model import LogisticRegression
    rows, combos, primary = [], [], None
    for L in LAGS:
        q = B.quotes(fr, fills, L)
        sel = q[q.t_start < B.SEL_END]
        X = lambda d: np.c_[B.logit(d.mid), B.logit(d['q_lr v2'])]
        cm = LogisticRegression(C=1e6, max_iter=1000).fit(X(sel), sel.blue_win.astype(int))
        q['q_comb'] = cm.predict_proba(X(q))[:, 1]
        sel, ev = q[q.t_start < B.SEL_END], q[q.t_start >= B.SEL_END]
        for sig in SIGS:
            cells = [('A', 0.0)]
            ss = [(B.summarize(B.run(sel, sig, 'B', th), 'sel').get('gross_pp', np.nan), th) for th in B.THETAS]
            th = max(ss, key=lambda z: z[0] if np.isfinite(z[0]) else -99)[1]
            cells.append(('B', th))
            for strat, t in cells:
                r = B.run(ev, sig, strat, t)
                mu, se, tt, G = clustered_t(r)
                rows.append(dict(L=L, sig=sig, strat=strat, theta=t, n=len(r), matches=G, pp=100 * mu, se_pp=100 * se, t=tt,
                                 p=stats.norm.sf(tt), sr_game=sharpe(per_game(r))))
                if (sig, strat, L) == ('q_lr v2', 'B', 0):
                    primary = r
            for t in B.THETAS:  # every theta on the eval months: the widest count of configurations looked at
                combos.append(sharpe(per_game(B.run(ev, sig, 'B', t))))
            combos.append(rows[-2]['sr_game'])
        print(f'L={L} done', flush=True)
    R = pd.DataFrame(rows)
    R['holm_all'] = holm(R.p); R['bh_all'] = bh(R.p)
    gm = R.strat == 'B'
    R.loc[gm, 'holm_gate'] = holm(R.p[gm]); R.loc[gm, 'bh_gate'] = bh(R.p[gm])
    return R, np.array(combos), primary


def dm(lags):
    out = []
    for L in lags:
        fr = pd.read_parquet(f'{D}/market_frames_lag{L}.parquet')
        y = fr.blue_win.astype(int).to_numpy()
        for c in ('q_lr v2', 'q_lr all'):
            g = fr.assign(d=ll(y, fr[c].to_numpy()) - ll(y, fr.mkt.to_numpy())).groupby('game_id') \
                  .agg(d=('d', 'mean'), match_id=('match_id', 'first'), t=('ts', 'min'))
            m = g.groupby('match_id').agg(d=('d', 'mean'), t=('t', 'min')).sort_values('t').d.to_numpy()
            n = len(m)
            e = m - m.mean()
            K = int(np.floor(4 * (n / 100) ** (2 / 9)))
            lrv = (e @ e) / n + 2 * sum((1 - k / (K + 1)) * (e[k:] @ e[:-k]) / n for k in range(1, K + 1))
            t = m.mean() / np.sqrt(lrv / n)
            out.append(dict(lag=L, model=c, matches=n, games=len(g), d_game=g.d.mean(), d_match=m.mean(), nw_lags=K, dm_t=t,
                            p_two=2 * stats.norm.sf(abs(t))))
    return pd.DataFrame(out)


def main():
    pd.set_option('display.width', 250)
    fm = lambda x: f'{x:.4f}'
    D_ = dm([0, 5, 15, 30, 60, 120, 300] if len(sys.argv) < 2 else [int(a) for a in sys.argv[1:]])
    print('== Diebold-Mariano, per-game log loss model minus market (negative = model better), match-level, Newey-West')
    print(D_.to_string(index=False, float_format=fm))
    D_.to_csv(f'{D}/../results/sig_dm.csv', index=False)

    R, combos, primary = backtest_family()
    print(f'\n== backtest family: {len(R)} cells ({R.strat.eq("B").sum()} gate cells), eval months, one-sided p, match-clustered')
    print(R.to_string(index=False, float_format=fm))
    R.to_csv(f'{D}/../results/sig_backtest.csv', index=False)
    x = per_game(primary)
    print('\n== deflated Sharpe, primary cell q_lr v2 gate L=0 (per-game PnL, 1 share per trade)')
    for name, tr in (('cells', R.sr_game.to_numpy()), ('cells x thetas', combos)):
        d = dsr(x, tr)
        print(f'  trials = {name}: ' + '  '.join(f'{k} {v:.4f}' if isinstance(v, float) else f'{k} {v}' for k, v in d.items()))


if __name__ == '__main__':
    main()
