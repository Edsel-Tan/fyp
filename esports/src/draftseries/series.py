"""Series updating and momentum (2026-10-16): do the models and the market update enough between games of a series?

The model priors already update after every game: team/player Elo (K 16/32/64), form_5/10/20, h2h, the Kalman ratings
and series_diff (wins so far) are all recomputed causally before each game (pregame.py, reddit/build.py), and the union
logistic is refit monthly. The question is whether anything is left:
  y_k ~ offset(z_k) + beta * x,   z_k = a pre-game logit (v5/v6 union prior, or the market's pre-game price),
for games k >= 2 of a series, from game k's blue team (A) view. Candidate x (all from earlier games of the same series):
  sd       A's game wins minus B's (the model's own series_diff)
  res      sum over earlier games of (A won - P_model(A won))      series surprise vs the model
  res_mkt  the same vs the market's pre-game price
  last     A won the previous game (+1 / -1)
  gold     mean tanh(end gold margin / 5k) for A (dominance)
  dom      mean of v5's in-game P(A) over the earlier games, minus their prior (on-the-day play, test-split frames only)
plus the team's form outside the series:
  day      A's minus B's residual sum over their other games in the 12 h before (other series the same day)
  event    A's minus B's residual sum over their games in the same league in the last 21 days (tournament form)
Fit on all test-period games with match-clustered SEs; out of sample: fit 2025-07..2026-01, score 2026-02..10.
Also: how far the model prior and the market price actually move between games, against what the outcome regression implies.
Usage: python draftseries/series.py
"""
import os, sys
from collections import defaultdict
import numpy as np, pandas as pd
import statsmodels.api as sm
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path[:0] = [SRC, HERE, os.path.join(SRC, 'v6')]
from eval_pregame import SPLIT, ll, boot_diff

D = os.path.join(SRC, '../data')
sig = lambda z: 1 / (1 + np.exp(-z))
lg = lambda p: np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))
OOS0 = pd.Timestamp('2026-02-01', tz='UTC')


def pm_pre():
    import prior as V6
    return V6.pm_pre_all()


def dominance():
    """Per game: mean frame q_v5 (blue) over the game, from the test-split predictions plus the live-period ones."""
    a = pd.read_parquet(f'{D}/ingame_test_preds_v5.parquet', columns=['game_id', 'q_v5'])
    out = a.groupby('game_id').q_v5.mean()
    fp = f'{D}/v5/live_period_preds.parquet'
    if os.path.exists(fp):
        b = pd.read_parquet(fp)
        if 'q_v5' in b:
            b = b.groupby('game_id').q_v5.mean()
            out = pd.concat([out, b[~b.index.isin(out.index)]])
    return out.rename('qbar')


def table():
    Q = pd.read_parquet(f'{D}/draftseries/priors.parquet')
    Q = Q[Q.blue_win.notna()].merge(pm_pre(), on='game_id', how='left')
    Q = Q.merge(dominance(), on='game_id', how='left')
    Q = Q.sort_values('t_start').reset_index(drop=True)
    Q['z'] = Q.un_full
    Q['zp'] = Q.un_pre
    Q['mz'] = lg(Q.pm_pre)
    # per-team game log (for day / event form): residual of each finished game vs the model prior
    teamlog = defaultdict(list)  # team -> [(t_end, league, resid)]
    rows = []
    bym = defaultdict(list)
    for r in Q.itertuples():
        A, B = r.blue_id, r.red_id
        prev = bym[r.match_id]
        x = dict(game_id=r.game_id)
        if prev:
            sA = [(1 if p['A'] == A else -1) for p in prev]  # +1 when A was blue in that game
            yA = [p['y'] if s > 0 else 1 - p['y'] for p, s in zip(prev, sA)]
            pA = [sig(p['z']) if s > 0 else 1 - sig(p['z']) for p, s in zip(prev, sA)]
            mA = [(p['m'] if s > 0 else 1 - p['m']) for p, s in zip(prev, sA)]
            gA = [np.tanh(s * p['g'] / 5000) for p, s in zip(prev, sA)]
            dA = [((p['qbar'] if s > 0 else 1 - p['qbar']) - pa) for p, s, pa in zip(prev, sA, pA)]
            x.update(k=len(prev) + 1, sd=sum(2 * y - 1 for y in yA), res=sum(y - p for y, p in zip(yA, pA)),
                     res_mkt=sum(y - m for y, m in zip(yA, mA)) if all(np.isfinite(mA)) else np.nan,
                     last=2 * yA[-1] - 1, last_res=yA[-1] - pA[-1], gold=np.nanmean(gA) if np.isfinite(gA).any() else 0.0,
                     dom=np.mean(dA) if all(np.isfinite(dA)) else np.nan,
                     prevA_p=pA[-1], prevA_m=mA[-1], prevA_zp=(prev[-1]['zp'] if sA[-1] > 0 else -prev[-1]['zp']),
                     prev_sideA=sA[-1])
        else:
            x.update(k=1)
        t = r.t_start
        for side, tm, op in (('A', A, B), ('B', B, A)):
            lgx = [e for e in teamlog[tm] if e[0] < t and e[3] != r.match_id]
            x[f'day{side}'] = sum(e[2] for e in lgx if (t - e[0]) < pd.Timedelta('12h'))
            x[f'ev{side}'] = sum(e[2] for e in lgx if (t - e[0]) < pd.Timedelta('21D') and e[1] == r.league)
        x['day'] = x['dayA'] - x['dayB']
        x['event'] = x['evA'] - x['evB']
        rows.append(x)
        bym[r.match_id].append(dict(A=A, y=r.blue_win, z=r.z, zp=r.zp, m=r.pm_pre if pd.notna(r.pm_pre) else np.nan,
                                    g=r.e_gold if pd.notna(r.e_gold) else 0.0, qbar=r.qbar))
        p = sig(r.z) if np.isfinite(r.z) else 0.5
        teamlog[A].append((r.t_end, r.league, r.blue_win - p, r.match_id))
        teamlog[B].append((r.t_end, r.league, (1 - r.blue_win) - (1 - p), r.match_id))
    return Q.merge(pd.DataFrame(rows), on='game_id')


def glm(X, cols, off):
    Xm = sm.add_constant(X[cols], has_constant='add')
    return sm.GLM(X.blue_win.astype(int), Xm, family=sm.families.Binomial(), offset=X[off]).fit(
        cov_type='cluster', cov_kwds={'groups': pd.factorize(X.match_id)[0]})


def oos(X, cols, off):
    a = X[X.t_start < OOS0]
    b = X[X.t_start >= OOS0]
    r = sm.GLM(a.blue_win.astype(int), sm.add_constant(a[cols], has_constant='add'), family=sm.families.Binomial(), offset=a[off]).fit()
    zb = b[off] + r.params['const'] + sum(r.params[c] * b[c] for c in cols)
    zb0 = b[off] + sm.GLM(a.blue_win.astype(int), np.ones((len(a), 1)), family=sm.families.Binomial(), offset=a[off]).fit().params.iloc[0]
    y = b.blue_win.astype(int).to_numpy()
    return boot_diff(ll(y, sig(zb)), ll(y, sig(zb0)), b.match_id.to_numpy())


def report(X, off, label):
    print(f'  [{label}] games {len(X)} (matches {X.match_id.nunique()}); OOS = fit < 2026-02, score >= 2026-02 ({(X.t_start >= OOS0).sum()} games)')
    for cols in (['sd'], ['res'], ['last'], ['last_res'], ['gold'], ['dom'], ['res', 'gold'], ['day'], ['event'], ['res', 'gold', 'event']):
        Xc = X.dropna(subset=cols)
        if len(Xc) < 200:
            continue
        r = glm(Xc, cols, off)
        o = oos(Xc, cols, off)
        print(f'    + {" + ".join(cols):20s} n {len(Xc):5d}  ' + ', '.join(f'{c} {r.params[c]:+.3f} (t {r.tvalues[c]:+.1f})' for c in cols) +
              f';  OOS LL {o[0]:+.4f} [{o[1]:+.4f}, {o[2]:+.4f}]')


def main():
    X = table()
    X.to_parquet(f'{D}/draftseries/series.parquet')
    T = X[(X.t_start >= SPLIT) & X.z.notna()].copy()
    S = T[T.k >= 2].copy()
    print(f'== series games k >= 2 in the test period: {len(S)} (Bo3 {(S.bo == 3).sum()}, Bo5 {(S.bo == 5).sum()}), '
          f'with a market price {S.mz.notna().sum()}')
    # what the union prior has learned about series_diff
    import priors as PR, eval as RE
    from sklearn.linear_model import LogisticRegression
    P = PR.load()
    UN = RE.V4P + RE.POST
    fit = (P.t_end < pd.Timestamp('2026-10-01', tz='UTC')) & P.blue_win.notna()
    sc = P.loc[fit, UN].std().replace(0, 1)
    m = LogisticRegression(C=0.01, max_iter=5000).fit(P.loc[fit, UN] / sc, P.blue_win[fit].astype(int))
    co = pd.Series(m.coef_[0] / sc.to_numpy(), index=UN)
    print('  union prior (Oct refit), per-unit coefficients: series_diff %+.3f, form_5 %+.3f, elo_team_32 %+.5f/pt, h2h %+.3f' %
          (co.series_diff, co.form_5, co.elo_team_32, co.h2h))
    print('== 1. leftover momentum on top of the v5/v6 union prior (offset z)')
    report(S, 'z', 'all')
    for nm, m in (('cross-league', S.cross == 1), ('same-league', S.cross == 0), ('international', S.intl == 1), ('Bo5', S.bo == 5),
                  ('Bo3', S.bo == 3), ('low-info (n_min_games < 40)', S.n_min_games < 40)):
        Sm = S[m]
        if len(Sm) > 150:
            r = glm(Sm, ['res'], 'z')
            r2 = glm(Sm.dropna(subset=['gold']), ['gold'], 'z')
            print(f'    {nm:28s} n {len(Sm):5d}  res {r.params.res:+.3f} (t {r.tvalues.res:+.1f})   gold {r2.params.gold:+.3f} (t {r2.tvalues.gold:+.1f})')
    print('== 2. leftover momentum on top of the market\'s pre-game price (offset logit pm_pre)')
    M = S[S.mz.notna()].copy()
    M['res_m'] = M.res_mkt
    print(f'  market-priced games k >= 2: {len(M)}')
    for cols in (['sd'], ['res_m'], ['res'], ['last'], ['gold'], ['dom'], ['event']):
        Mc = M.dropna(subset=cols)
        r = glm(Mc, cols, 'mz')
        print(f'    + {" + ".join(cols):10s} n {len(Mc):5d}  ' + ', '.join(f'{c} {r.params[c]:+.3f} ({r.bse[c]:.3f}, t {r.tvalues[c]:+.1f})' for c in cols))
    r = glm(M, ['z'], 'mz')
    print(f'    + model prior z: {r.params.z:+.3f} (t {r.tvalues.z:+.1f})   [the v6 blend, for scale]')
    print('== 3. how far the prior and the market move between consecutive games (A = game k blue team)')
    S2 = S[S.k >= 2].copy()
    S2['dz_pre'] = S2.zp - S2.prevA_zp
    S2['same_side'] = (S2.prev_sideA > 0).astype(float)
    o = sm.OLS(S2.dz_pre, sm.add_constant(S2[['last_res', 'same_side']])).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(S2.match_id)[0]})
    print(f'  pre-draft model logit change: {o.params.last_res:+.3f} ({o.bse.last_res:.3f}) per unit surprise (y - p) of the previous game; '
          f'mean |change| after a win {S2.dz_pre[S2["last"] > 0].abs().mean():.3f}')
    Mm = S2[S2.mz.notna() & S2.prevA_m.notna()].copy()
    Mm['dm'] = Mm.mz - lg(Mm.prevA_m)
    o = sm.OLS(Mm.dm, sm.add_constant(Mm[['last_res', 'same_side']])).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(Mm.match_id)[0]})
    o2 = sm.OLS(Mm.dz_pre, sm.add_constant(Mm[['last_res', 'same_side']])).fit()
    print(f'  market logit change (same {len(Mm)} games): {o.params.last_res:+.3f} ({o.bse.last_res:.3f}) per unit surprise; model on these games {o2.params.last_res:+.3f}')
    for nm, m in (('cross-league', Mm.cross == 1), ('same-league', Mm.cross == 0)):
        if m.sum() > 50:
            a = sm.OLS(Mm.dm[m], sm.add_constant(Mm[['last_res', 'same_side']][m])).fit()
            b = sm.OLS(Mm.dz_pre[m], sm.add_constant(Mm[['last_res', 'same_side']][m])).fit()
            print(f'    {nm}: n {m.sum()} market {a.params.last_res:+.3f} ({a.bse.last_res:.3f})  model {b.params.last_res:+.3f} ({b.bse.last_res:.3f})')
    # calibration of the prior by series state
    print('== 4. calibration of the union prior on game k >= 2 by series state (A = blue)')
    S['state'] = np.select([S.sd >= 2, S.sd == 1, S.sd == 0, S.sd == -1, S.sd <= -2], ['A +2', 'A +1', 'level', 'A -1', 'A -2'], 'other')
    for st, g in S.groupby('state'):
        print(f'    {st:6s} n {len(g):5d}  mean P(A) {sig(g.z).mean():.3f}  A won {g.blue_win.mean():.3f}  '
              f'(diff {g.blue_win.mean() - sig(g.z).mean():+.3f} se {g.blue_win.std() / np.sqrt(len(g)):.3f})'
              + (f'  market {g.pm_pre.mean():.3f} on {g.pm_pre.notna().sum()}' if g.pm_pre.notna().sum() > 30 else ''))
    T.to_parquet(f'{D}/draftseries/series_test.parquet')
    part5(X)


def mkt_grid(X):
    import market_draft as MD
    Q = MD.priors()
    A = MD.archive_grid(Q)
    B = MD.books_grid(Q)
    G = pd.concat([A, B[~B.index.isin(A.index)]])
    return pd.DataFrame({'p_pre': G[-15], 'p_post': G[-1], 'p_p1': G[1]}).rename_axis('game_id').reset_index()


def part5(X):
    """Clean market prices: p_pre = t0-15 (after the previous game, before the draft), p_post = t0-1 (after the draft)."""
    print('== 5. market with clean pre-/post-draft prices (archive minute VWAP / captured mid)')
    X = X.merge(mkt_grid(X), on='game_id', how='left')
    X = X[(X.t_start >= SPLIT) & X.z.notna()].sort_values('t_start')
    # previous game's post-draft price from A's view, and the result surprise vs the market
    pr = X.set_index('game_id')
    X['mpost'] = lg(X.p_post)
    X['mpre'] = lg(X.p_pre)
    X['mp1'] = lg(X.p_p1)
    bym = {}
    rows = []
    for r in X.itertuples():
        prev = bym.get(r.match_id, [])
        if prev:
            sA = [1 if p[0] == r.blue_id else -1 for p in prev]
            yA = [p[1] if s > 0 else 1 - p[1] for p, s in zip(prev, sA)]
            mA = [(p[2] if s > 0 else 1 - p[2]) for p, s in zip(prev, sA)]
            rows.append(dict(game_id=r.game_id, res_mp=sum(y - m for y, m in zip(yA, mA)) if all(np.isfinite(mA)) else np.nan,
                             last_res_mp=yA[-1] - mA[-1] if np.isfinite(mA[-1]) else np.nan,
                             prev_mpost_A=lg(mA[-1]) if np.isfinite(mA[-1]) else np.nan))
        bym.setdefault(r.match_id, []).append((r.blue_id, r.blue_win, r.p_post if pd.notna(r.p_post) else np.nan))
    X = X.merge(pd.DataFrame(rows), on='game_id', how='left')
    S = X[X.k >= 2]
    for off in ('mpost', 'mp1'):
        M = S[S[off].notna()]
        print(f'  offset {off} ({"t0-1" if off == "mpost" else "t0+1"}): {len(M)} games k >= 2')
        for cols in (['sd'], ['res_mp'], ['res'], ['last'], ['gold'], ['dom'], ['event'], ['z']):
            Mc = M.dropna(subset=cols)
            r = glm(Mc, cols, off)
            print(f'    + {" + ".join(cols):8s} n {len(Mc):5d}  ' + ', '.join(f'{c} {r.params[c]:+.3f} ({r.bse[c]:.3f}, t {r.tvalues[c]:+.1f})' for c in cols))
        Mc = M.dropna(subset=['res_mp'])
        for nm, m in (('cross-league', Mc.cross == 1), ('same-league', Mc.cross == 0)):
            r = glm(Mc[m], ['res_mp'], off)
            print(f'      {nm}: n {m.sum()} res_mp {r.params.res_mp:+.3f} (t {r.tvalues.res_mp:+.1f})')
    M = S.dropna(subset=['mpre', 'prev_mpost_A', 'last_res_mp']).copy()
    M['same_side'] = (M.prev_sideA > 0).astype(float)
    M['dmk'] = M.mpre - M.prev_mpost_A
    M['dz'] = M.zp - M.prevA_zp
    g = pd.factorize(M.match_id)[0]
    a = sm.OLS(M.dmk, sm.add_constant(M[['last_res_mp', 'same_side']])).fit(cov_type='cluster', cov_kwds={'groups': g})
    b = sm.OLS(M.dz, sm.add_constant(M[['last_res', 'same_side']])).fit(cov_type='cluster', cov_kwds={'groups': g})
    print(f'  between-game update, {len(M)} games: market (t0-1 of game k-1 -> t0-15 of game k) {a.params.last_res_mp:+.3f} ({a.bse.last_res_mp:.3f}) '
          f'logit per unit surprise; model pre-draft prior {b.params.last_res:+.3f} ({b.bse.last_res:.3f})')
    # implied ideal update: regress y_k on the previous price (A view, at game k's side) + surprise
    M['zprevside'] = M.prev_mpost_A
    r = glm(M, ['last_res_mp', 'same_side'], 'zprevside')
    print(f'  outcome-implied update on top of the previous game\'s market price: {r.params.last_res_mp:+.3f} ({r.bse.last_res_mp:.3f})')
    r = glm(M.assign(zz=M.prevA_zp), ['last_res', 'same_side'], 'zz')
    print(f'  outcome-implied update on top of the previous game\'s pre-draft model prior: {r.params.last_res:+.3f} ({r.bse.last_res:.3f})')
    for nm, m in (('cross-league', M.cross == 1), ('same-league', M.cross == 0)):
        Mm = M[m]
        a = sm.OLS(Mm.dmk, sm.add_constant(Mm[['last_res_mp', 'same_side']])).fit()
        r1 = glm(Mm, ['last_res_mp', 'same_side'], 'zprevside')
        b = sm.OLS(Mm.dz, sm.add_constant(Mm[['last_res', 'same_side']])).fit()
        r2 = glm(Mm.assign(zz=Mm.prevA_zp), ['last_res', 'same_side'], 'zz')
        print(f'    {nm}: n {len(Mm)}  market moves {a.params.last_res_mp:+.3f}, outcome-implied {r1.params.last_res_mp:+.3f} ({r1.bse.last_res_mp:.3f});  '
              f'model moves {b.params.last_res:+.3f}, outcome-implied {r2.params.last_res:+.3f} ({r2.bse.last_res:.3f})')


if __name__ == '__main__':
    main()
