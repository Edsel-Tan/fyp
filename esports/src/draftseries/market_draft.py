"""Does the draft move the market, and does the market price it correctly? (2026-10-16)

Two market sources, both per-game winner markets linked to our games:
  archive  Polymarket-v1 fills (data/pm_fills.parquet, pm_game_markets.parquet), 2025-07 .. 2026-04: per-minute VWAP of P(blue)
  books    the self-captured minute bars (~/fyp-pmlive/data/pmlive), 2026-09-17 .. 10-13, linked by gap/panel (data/gap/games.parquet):
           two-sided mid at each minute's last bar
Minute k is relative to the game's first livestats frame t0. The break between games is ~21 min (median, prev end -> t0).
1 timing: mean |change in logit price| per minute and its correlation with the model's draft shift d = un_full - un_pre
  (draftseries/priors.py), to find where champion select lands in the price.
2 the draft move: pre = price at t0 - 15 min (after the previous game's end), post = price at t0 - 1. Regress the market's
  move on d (does the market move with the draft?), and the outcome on both (is the move right? does d add to the post price?).
Usage: python draftseries/market_draft.py
"""
import os, sys
import numpy as np, pandas as pd
import statsmodels.api as sm
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path[:0] = [SRC, os.path.join(SRC, 'v5')]
from eval_pregame import ll, boot_diff

D = os.path.join(SRC, '../data')
K0, K1 = -40, 3
lg = lambda p: np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))
sig = lambda z: 1 / (1 + np.exp(-z))


def priors():
    Q = pd.read_parquet(f'{D}/draftseries/priors.parquet').sort_values(['match_id', 'number'])
    Q['prev_end'] = Q.groupby('match_id').t_end.shift()
    Q['d'] = Q.un_full - Q.un_pre
    return Q


def archive_grid(Q):
    mk = pd.read_parquet(f'{D}/pm_game_markets.parquet').sort_values('vol', ascending=False).drop_duplicates('game_id')
    fl = pd.read_parquet(f'{D}/pm_fills.parquet', columns=['condition_id', 'ts', 'p_event', 'usd'])
    fl = fl[fl.condition_id.isin(mk.condition_id)].merge(mk[['condition_id', 'game_id', 'seq1_blue', 'blue_win']], on='condition_id')
    fl['p'] = np.where(fl.seq1_blue == 1, fl.p_event, 1 - fl.p_event)
    fl['w'] = fl.usd.abs().clip(lower=1e-6)
    fl['ts'] = pd.to_datetime(fl.ts, unit='s', utc=True)
    fl = fl.merge(Q[['game_id', 't_start', 'prev_end']], on='game_id')
    fl['k'] = np.floor((fl.ts - fl.t_start).dt.total_seconds() / 60).astype(int)
    fl = fl[(fl.k >= K0) & (fl.k < K1) & (fl.prev_end.isna() | (fl.ts > fl.prev_end + pd.Timedelta('1min')))]
    fl['pw'] = fl.p * fl.w
    g = fl.groupby(['game_id', 'k'])[['pw', 'w']].sum()
    v = (g.pw / g.w).unstack()
    # forward-fill a minute's VWAP for up to 5 minutes (sparse fills)
    v = v.reindex(columns=range(K0, K1)).ffill(axis=1, limit=5)
    return v


def books_grid(Q):
    import book_sim as B
    G = pd.read_parquet(f'{D}/gap/games.parquet')
    bars = B.load_bars(set(G.tok_b))
    t0 = Q.set_index('game_id').t_start
    pe = Q.set_index('game_id').prev_end
    out = {}
    for r in G.itertuples():
        x = bars.get(r.tok_b)
        if x is None or r.game_id not in t0.index:
            continue
        x = x[(x.bb > 0) & (x.ba < 1) & (x.ba > x.bb)]
        m0 = int((t0[r.game_id] - pd.Timestamp(0, tz='UTC')).total_seconds() // 60)
        lo = m0 + K0
        if pd.notna(pe.get(r.game_id)):
            lo = max(lo, int((pe[r.game_id] - pd.Timestamp(0, tz='UTC')).total_seconds() // 60) + 2)
        mid = ((x.bb + x.ba) / 2)
        row = {}
        for k in range(K0, K1):
            mm = m0 + k
            if mm < lo:
                continue
            s = mid.loc[:mm]
            if len(s) and mm - s.index[-1] <= 5:
                row[k] = float(s.iloc[-1])
        out[r.game_id] = row
    return pd.DataFrame.from_dict(out, orient='index').reindex(columns=range(K0, K1))


def timing(V, Q, name):
    Z = lg(V)
    dZ = Z.diff(axis=1)
    d = Q.set_index('game_id').d.reindex(V.index)
    print(f'  [{name}] {len(V)} games; per-minute |d logit price| (mean) and corr(change, draft shift d):')
    row = []
    for k in range(-30, K1):
        x = dZ[k]
        m = x.notna() & d.notna()
        if m.sum() < 30:
            continue
        row.append(f'{k:+d}: {x[m].abs().mean():.3f}/{np.corrcoef(x[m], d[m])[0, 1]:+.2f}')
    for i in range(0, len(row), 11):
        print('     ' + '  '.join(row[i:i + 11]))


def move(V, Q, name, pre_k=-15, post_k=-1):
    W = Q.set_index('game_id').reindex(V.index)
    ok = V[pre_k].notna() & V[post_k].notna() & W.blue_win.notna()
    X = W[ok].assign(pre=lg(V.loc[ok, pre_k]), post=lg(V.loc[ok, post_k]))
    X['dm'] = X.post - X.pre
    X['y'] = X.blue_win.astype(int)
    grp = pd.factorize(X.match_id)[0]
    ols = sm.OLS(X.dm, sm.add_constant(X[['d']])).fit(cov_type='cluster', cov_kwds={'groups': grp})
    print(f'  [{name}] {len(X)} games with a price at t0{pre_k} and t0{post_k}; game 1: {(X.number == 1).sum()}')
    print(f'    market move SD {X.dm.std():.3f} logit (median |move| {np.median(np.abs(sig(X.post) - sig(X.pre))) * 100:.1f} pp); '
          f'model draft shift SD {X.d.std():.3f}')
    print(f'    move ~ d: slope {ols.params.d:.2f} ({ols.bse.d:.2f}), R2 {ols.rsquared:.3f}; corr {np.corrcoef(X.dm, X.d)[0, 1]:+.3f}')
    for nm, m in (('game 1', X.number == 1), ('game 2+', X.number > 1)):
        if m.sum() > 30:
            o = sm.OLS(X.dm[m], sm.add_constant(X.d[m])).fit()
            print(f'      {nm}: n {m.sum()} slope {o.params.d:.2f} ({o.bse.d:.2f}) corr {np.corrcoef(X.dm[m], X.d[m])[0, 1]:+.3f}')

    def glm(cols, off=None):
        Xm = sm.add_constant(X[cols]) if cols else np.ones((len(X), 1))
        r = sm.GLM(X.y, Xm, family=sm.families.Binomial(), offset=None if off is None else X[off]).fit(cov_type='cluster', cov_kwds={'groups': grp})
        return r
    print('    outcome regressions (logit; cluster-robust by match):')
    for cols, off in ((['pre'], None), (['post'], None), (['pre', 'dm'], None), (['post', 'd'], None), (['d'], 'post'), (['d'], 'pre'),
                      (['post', 'un_full'], None)):
        r = glm(cols, off)
        print(f'      y ~ {" + ".join(cols)}{" + offset(" + off + ")" if off else ""}: ' +
              ', '.join(f'{c} {r.params[c]:.2f} ({r.bse[c]:.2f})' for c in cols))
    y = X.y.to_numpy()
    a, b = ll(y, sig(X.post)), ll(y, sig(X.pre))
    c = ll(y, sig(X.post + X.d))
    e = ll(y, sig(X.pre + X.d))
    f = ll(y, sig(X.un_full))
    for nm, u, v in (('post - pre', a, b), ('(pre + d) - pre', e, b), ('(post + d) - post', c, a), ('model full prior - post', f, a)):
        dd = boot_diff(u, v, X.match_id.to_numpy())
        print(f'      LL {nm:24s} {dd[0]:+.4f} [{dd[1]:+.4f}, {dd[2]:+.4f}]')
    return X


def main():
    Q = priors()
    print('== timing')
    A = archive_grid(Q)
    timing(A, Q, 'archive')
    Bk = books_grid(Q)
    timing(Bk, Q, 'books')
    print('== the draft move, pre = t0-15, post = t0-1')
    xa = move(A, Q, 'archive')
    xb = move(Bk, Q, 'books')
    print('== sensitivity: pre = t0-12 / t0-20, post = t0-2')
    for pk, qk in ((-12, -1), (-20, -1), (-15, -2)):
        move(A, Q, f'archive {pk}/{qk}', pk, qk)
    pd.concat([xa.assign(src='archive'), xb.assign(src='books')]).to_parquet(f'{D}/draftseries/market_draft.parquet')


if __name__ == '__main__':
    main()
