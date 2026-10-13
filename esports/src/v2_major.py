"""v2-major-leagues: v2 refit on major-league games only (2026-10-05).

Same design as `lr v2` (STATS + prior, each x (1, m, m^2), StandardScaler + logistic C=1), same split (train before 2025-07),
but the in-game fit uses only MAJOR train games. The prior is v2's all-league `prior_new` (ratings need every league); the
ablation `v2-major-leagues+prior` also refits the pre-game prior on major games (out-of-fold on train, as eval_pregame2 does).

Scored against v2 on:
  1. the test split (all / major / non-major games), per-game log loss
  2. model vs market at lag 0 and 22 s (last trade; two-sided frames vs the mid proxy), as in overfit_check
  3. the trading backtest (backtest.py's quotes/run/summarize; gate theta picked on the selection months, clip), all and major games
  4. the live tracker's resolved games, rescored offline from their saved feeds (lollive data/frames60) with the stored prior_new
Writes data/v2_major_leagues.pkl and data/ingame_test_preds_v2major.parquet; log in results/v2_major.log.
"""
import os, sys, pickle, sqlite3
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from eval_pregame import BURN, SPLIT, ll, boot_diff
from eval_ingame import STATS, design

D = '../data'
LL = os.path.expanduser('~/fyp-pmlive/lollive')
MAJOR = {'lck', 'lpl', 'lec', 'lcs', 'lta_n', 'lta_s', 'lta_cross', 'cblol-brazil', 'lcp', 'pcs', 'vcs', 'ljl-japan', 'lla',
         'worlds', 'msi', 'first_stand', 'ewc_lol', 'americas_cup'}
rng = np.random.default_rng(0)
mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def pg(df, col, y='y'):
    return pd.Series(ll(df[y].to_numpy(), df[col].clip(1e-4, 1 - 1e-4).to_numpy()), index=df.game_id).groupby(level=0).mean()


def compare(df, cols, base, label):
    g2m = df.drop_duplicates('game_id').set_index('game_id').match_id
    b = pg(df, base)
    print(f'  [{label}] games {df.game_id.nunique()} matches {df.match_id.nunique()}')
    for c in cols:
        a = pg(df, c)
        d, lo, hi = boot_diff(a.to_numpy(), b.reindex(a.index).to_numpy(), g2m.reindex(a.index).to_numpy())
        print(f'    {c:14s} {a.mean():.4f}   minus {base} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]')


def major_prior():
    """prior_new refit on major games: OOF (5 folds by match) on major train rows, train fit applied to test rows."""
    from eval_pregame2 import BASE, G
    p = pd.read_parquet(f'{D}/pregame.parquet').merge(pd.read_parquet(f'{D}/pregame2.parquet'), on='game_id')
    p['age_x_elo'] = p.elo_play_32 * np.exp(-p.patch_age / 7)
    p = p[p.t_start >= BURN].reset_index(drop=True)
    cols = BASE + sorted({c for k, v in G.items() if not k.startswith('h2h') for c in v})
    y = p.blue_win.astype(int).to_numpy()
    tr = ((p.t_start < SPLIT) & p.league.isin(MAJOR)).to_numpy()
    te = (p.t_start >= SPLIT).to_numpy()

    def fp(a, b):
        sc = p.loc[a, cols].std().replace(0, 1)
        q = LogisticRegression(C=0.03, max_iter=5000).fit(p.loc[a, cols] / sc, y[a]).predict_proba(p.loc[b, cols] / sc)[:, 1]
        return np.log(q / (1 - q))
    z = np.full(len(p), np.nan)
    mids = p.match_id[tr].unique()
    k = p.match_id.map(pd.Series(rng.integers(0, 5, len(mids)), index=mids)).to_numpy()
    for j in range(5):
        z[tr & (k == j)] = fp(tr & (k != j), tr & (k == j))
    z[te] = fp(tr, te)
    return p[['game_id']].assign(prior_major=z)


def fit_models():
    f = pd.read_parquet(f'{D}/ingame.parquet')
    f = f[(f.t_start >= BURN) & (f.state != 'finished')].reset_index(drop=True)
    f = f.merge(pd.read_parquet(f'{D}/prior2.parquet')[['game_id', 'prior_new']], on='game_id')
    f = f.merge(major_prior(), on='game_id', how='left')
    y = f.blue_win.astype(int).to_numpy()
    tr = (f.t_start < SPLIT).to_numpy()
    maj = f.league.isin(MAJOR).to_numpy()
    X = design(f.assign(prior_logit=f.prior_new), STATS + ['prior'])
    m_all = mk().fit(X[tr], y[tr])
    m_maj = mk().fit(X[tr & maj], y[tr & maj])
    Xp = design(f.assign(prior_logit=f.prior_major), STATS + ['prior'])
    m_majp = mk().fit(Xp[tr & maj], y[tr & maj])
    print(f'train games: all {f[tr].game_id.nunique()}, major {f[tr & maj].game_id.nunique()}')
    te = ~tr
    ft = f[te][['game_id', 'match_id', 'league', 'ts', 't_start', 'clock', 'blue_win']].copy()
    ft['y'] = y[te]
    ft['q_v2'] = m_all.predict_proba(X[te])[:, 1]
    ft['q_v2major'] = m_maj.predict_proba(X[te])[:, 1]
    ft['q_v2major_p'] = np.where(np.isnan(Xp[te]).any(axis=1), np.nan, m_majp.predict_proba(np.nan_to_num(Xp[te]))[:, 1])
    pickle.dump(dict(name='v2-major-leagues', model=m_maj, cols=STATS + ['prior'], prior='prior_new (all-league v2 prior)',
                     leagues=sorted(MAJOR), split=str(SPLIT), built='2026-10-05'), open(f'{D}/v2_major_leagues.pkl', 'wb'))
    return ft, m_maj


def part1(ft):
    print('\n== 1. test split, per-game log loss ==')
    cols = ['q_v2major', 'q_v2major_p']
    compare(ft, cols, 'q_v2', 'all test games')
    compare(ft[ft.league.isin(MAJOR)], cols, 'q_v2', 'major-league test games')
    compare(ft[~ft.league.isin(MAJOR)], cols, 'q_v2', 'non-major test games')
    compare(ft[ft.league == 'emea_masters'], cols, 'q_v2', 'emea_masters test games')


def part2(ft):
    import overfit_check as O
    from market_compare import market_at
    print('\n== 2. vs the market ==')
    pr = pd.read_parquet(f'{D}/ingame_test_preds.parquet')
    assert np.allclose(pr['q_lr v2'].to_numpy(), ft.q_v2.to_numpy())
    pr = pr.assign(q_v2major=ft.q_v2major.to_numpy(), q_v2major_p=ft.q_v2major_p.to_numpy())
    pr.to_parquet(f'{D}/ingame_test_preds_v2major.parquet')
    mkt = pd.read_parquet(f'{D}/pm_game_markets.parquet')
    fills = pd.read_parquet(f'{D}/pm_fills.parquet', columns=['condition_id', 'ts', 'p_event', 'D'])
    fills = fills[fills.condition_id.isin(mkt.condition_id)].merge(mkt[['condition_id', 'seq1_blue']], on='condition_id')
    fills['p_blue'] = np.where(fills.seq1_blue == 1, fills.p_event, 1 - fills.p_event)
    fills['Db'] = np.where(fills.seq1_blue == 1, fills.D, -fills.D)
    fills = {c: g.sort_values('ts') for c, g in fills.groupby('condition_id')}
    mkt = mkt.sort_values('vol', ascending=False).drop_duplicates('game_id')
    base = pr.merge(mkt[['game_id', 'condition_id']], on='game_id')
    for lag in (0, 22):
        fr = market_at(base, fills, lag)
        fr = fr[(fr.mkt_age <= 60) & (fr.mkt > 0) & (fr.mkt < 1)]
        fr = O.mid_proxy(fr, fills, lag)
        fr['y'] = fr.blue_win.astype(int)
        fr['q_v2'] = fr['q_lr v2']
        cols = ['q_v2', 'q_v2major']
        compare(fr, cols, 'mkt', f'lag {lag}s, all PM games, last trade')
        compare(fr[fr.league.isin(MAJOR)], cols, 'mkt', f'lag {lag}s, major PM games, last trade')
        two = fr[fr.mid.notna()]
        compare(two, cols, 'mid', f'lag {lag}s, all PM games, two-sided frames vs mid proxy')
        compare(two[two.league.isin(MAJOR)], cols, 'mid', f'lag {lag}s, major PM games, two-sided frames vs mid proxy')


def part3():
    os.environ['PREDS_SFX'] = '_v2major'
    import backtest as B
    print('\n== 3. trading backtest (eval months from 2026-02; gate theta chosen on selection months) ==')
    fr, fills = B.load()
    rows = []
    for L in (0, 5, 15, 30):
        q = B.quotes(fr, fills, L)
        for scope, qq in (('all', q), ('major', q[q.league.isin(MAJOR)] if 'league' in q else None)):
            if qq is None:
                continue
            sel, ev = qq[qq.t_start < B.SEL_END], qq[qq.t_start >= B.SEL_END]
            for sig in ('q_lr v2', 'q_v2major'):
                rows.append(dict(L=L, scope=scope, sig=sig, strat='clip', **B.summarize(B.run(ev, sig, 'A'), 'eval')))
                best = max(((B.summarize(B.run(sel, sig, 'B', th), 'sel').get('gross_pp', -99), th) for th in B.THETAS),
                           key=lambda z: z[0] if not np.isnan(z[0]) else -99)
                rows.append(dict(L=L, scope=scope, sig=sig, strat=f'gate th={best[1]}', **B.summarize(B.run(ev, sig, 'B', best[1]), 'eval')))
    r = pd.DataFrame(rows).drop(columns='name')
    print(r[['L', 'scope', 'sig', 'strat', 'n', 'games', 'gross_pp', 'ci_lo', 'ci_hi', 'fee_regime_net_pp']].to_string(
        index=False, float_format=lambda x: f'{x:.2f}'))
    r.to_csv(f'{D}/../results/backtest_v2major.csv', index=False)


def part4(m_maj):
    print('\n== 4. live tracker games, rescored from saved feeds ==')
    sys.path.insert(0, f'{LL}/src')
    import live
    M = pickle.load(open(f'{LL}/data/models.pkl', 'rb'))
    c = sqlite3.connect(f'file:{LL}/data/paper.sqlite?mode=ro', uri=True)
    g = pd.read_sql('select * from games where blue_win_rule is not null', c)
    t = pd.read_sql('select * from ticks', c).merge(
        g[['game_id', 'match_id', 'league', 'blue_win_rule', 'team_games_min', 'prior_new']], on='game_id')
    t = t[t.blue_mid.notna() & (t.blue_mid > 0) & (t.blue_mid < 1)].copy()
    out = []
    for gid, tk in t.groupby('game_id'):
        fp = f'{LL}/data/frames60/{gid}.parquet'
        if not os.path.exists(fp):
            continue
        o = live.features(pd.read_parquet(fp))
        if o is None:
            continue
        o = o.sort_values('ts')
        tk = tk.assign(_t=pd.to_datetime(tk.frame_ts, format='ISO8601')).sort_values('_t')
        j = pd.merge_asof(tk, o.drop(columns='state'), left_on='_t', right_on='ts', direction='backward', suffixes=('', '_f'))
        j = j[j.ts.notna()]
        X = design(j.assign(prior_logit=j.prior_new), STATS + ['prior'])
        j['q_v2re'] = M['v2']['model'].predict_proba(X)[:, 1]
        j['q_v2major'] = m_maj.predict_proba(X)[:, 1]
        out.append(j)
    j = pd.concat(out)
    err = (j.q_v2re - j.q_v2).abs()
    print(f'  rescored ticks {len(j)} of {len(t)}; recomputed v2 vs the stored q_v2: median |diff| {err.median():.4f}, '
          f'share > 0.02: {(err > 0.02).mean():.1%}')
    j['y'] = j.blue_win_rule.astype(int)
    j['mid'] = j.blue_mid
    low = j.team_games_min < 20
    cols = ['q_v2', 'q_v2re', 'q_v2major']
    for lab, d in [('all live games', j), ('real prior (team_games_min >= 20)', j[~low]), ('no real prior', j[low]),
                   ('real prior, major leagues', j[~low & j.league.isin(MAJOR)]),
                   ('real prior, EMEA Masters', j[~low & (j.league == 'emea_masters')]),
                   ('real prior, other non-major', j[~low & ~j.league.isin(MAJOR) & (j.league != 'emea_masters')])]:
        if d.game_id.nunique():
            compare(d, cols, 'mid', lab)
    compare(j[~low], ['q_v2major'], 'q_v2re', 'real prior: v2-major-leagues minus v2 (same features)')
    # the saved feed is coarser than what the trader saw: only ticks whose frame is in it reproduce the live features exactly
    ex = j[(j._t == j.ts) & ~low]
    print(f'  exact-frame ticks: {len(ex)}; recomputed v2 vs stored: median |diff| {(ex.q_v2re - ex.q_v2).abs().median():.4f}')
    compare(ex, ['q_v2', 'q_v2major'], 'mid', 'real prior, exact-frame ticks only')
    compare(ex, ['q_v2major'], 'q_v2', 'real prior, exact-frame ticks: v2-major-leagues minus stored v2')


if __name__ == '__main__':
    if sys.argv[1:] == ['live']:
        part4(pickle.load(open(f'{D}/v2_major_leagues.pkl', 'rb'))['model'])
        sys.exit()
    ft, m_maj = fit_models()
    part1(ft)
    part2(ft)
    part3()
    part4(m_maj)
